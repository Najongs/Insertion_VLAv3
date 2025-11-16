#!/usr/bin/env python3
"""
Optimized Async Real-time Inference Receiver

Optimizations Applied:
- Image resize: 640x360 (4x faster VLM)
- Sensor window: 65 samples (100ms @ 650Hz)
- VL feature reuse: 4x (VLM @ ~2.6Hz, Action Expert @ 10Hz)
- Background VLM update thread
- 10Hz action prediction loop

Architecture:
- VLM Thread: Updates VL features every ~380ms (2.6Hz) in background
- Action Thread: Predicts actions every 100ms (10Hz) using cached VL features
- Each VL feature is reused 4 times

Data Sources:
- Cameras: 5 views (ZED left x4 + OAK x1) via ZMQ PULL (port 5555)
- Robot State: Joint angles + EE pose via ZMQ SUB (port 5556)
- Sensor Data: Force + OCT A-scan via UDP (port 9999)

Usage:
    # Standard async inference (recommended)
    python Async_inference_receiver.py --checkpoint /home/najo/NAS/VLA/Insertion_VLAv3/checkpoints/backup/flow_matching_best.pt --auto-start --vl-reuse 2 --task-name Eye_trocar

    # Save data for debugging
    python Async_inference_receiver.py --checkpoint checkpoints/qwen_vla_sensor_best.pt --save-data

    # Custom VL reuse count (default: 4)
    python Async_inference_receiver.py --checkpoint checkpoints/qwen_vla_sensor_best.pt --vl-reuse 3

Performance:
    - VLM inference: ~381ms (2.6Hz) @ 640x360 with 5 views
    - Action Expert: ~20-30ms (10Hz capable)
    - VL feature reuse: 4x (fresh VL features every 400ms)
    - Total output: 80 actions/sec (10Hz × 8-horizon)
"""

import os, time, json, cv2, zmq, numpy as np, torch
import threading, argparse, signal, csv
from queue import Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import struct
from dataclasses import dataclass, asdict, field
from typing import List, Dict

# Import VLA model
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.unified_model import QwenVLAUnified


# ==============================
# Performance Monitoring
# ==============================
@dataclass
class InferenceTimings:
    """Single inference timing record"""
    timestamp: float
    action_id: int
    vl_update_number: int
    vl_update_time: float  # Time for VL update (ms)
    sensor_encoding_time: float = 0.0  # Sensor encoding time (ms)
    robot_encoding_time: float = 0.0  # Robot state encoding time (ms)
    action_prediction_time: float = 0.0  # Action expert time (ms)
    total_time: float = 0.0  # Total inference time (ms)
    sensor_buffer_size: int = 0
    robot_buffer_size: int = 0

    def to_dict(self):
        return asdict(self)


class PerformanceMonitor:
    """Monitor and track real-time inference performance with auto-save"""
    def __init__(self, output_dir: Path = None, session_time: str = None, auto_save_interval: int = 10):
        self.timings: List[InferenceTimings] = []
        self.start_time = time.time()
        self.lock = threading.Lock()

        # Auto-save settings
        self.output_dir = output_dir
        self.session_time = session_time
        self.auto_save_interval = auto_save_interval  # Save every N records
        self.last_saved_count = 0

        # Initialize file if auto-save enabled
        if output_dir and session_time:
            self.timings_path = output_dir / f"performance_timings_{session_time}.jsonl"
            self.summary_path = output_dir / f"performance_summary_{session_time}.json"
            # Create empty file
            self.timings_path.touch()
            print(f"✅ Auto-save enabled: {self.timings_path}")

    def add_timing(self, timing: InferenceTimings):
        """Add a timing record and auto-save if needed"""
        with self.lock:
            self.timings.append(timing)

            # Auto-save if interval reached
            if self.output_dir and self.session_time:
                unsaved_count = len(self.timings) - self.last_saved_count
                if unsaved_count >= self.auto_save_interval:
                    self._auto_save_append()

    def _auto_save_append(self):
        """Append new records to file (called with lock held)"""
        try:
            # Get unsaved records
            unsaved = self.timings[self.last_saved_count:]
            if not unsaved:
                return

            # Append to JSON Lines file (one JSON object per line)
            with open(self.timings_path, 'a') as f:
                for timing in unsaved:
                    json.dump(timing.to_dict(), f)
                    f.write('\n')

            self.last_saved_count = len(self.timings)

        except Exception as e:
            print(f"⚠️ Auto-save failed: {e}")

    def get_summary(self) -> Dict:
        """Get performance summary statistics"""
        with self.lock:
            if not self.timings:
                return {}

            vl_times = [t.vl_update_time for t in self.timings if t.vl_update_time > 0]
            sensor_times = [t.sensor_encoding_time for t in self.timings if t.sensor_encoding_time > 0]
            robot_times = [t.robot_encoding_time for t in self.timings if t.robot_encoding_time > 0]
            action_times = [t.action_prediction_time for t in self.timings if t.action_prediction_time > 0]
            total_times = [t.total_time for t in self.timings if t.total_time > 0]

            elapsed_time = time.time() - self.start_time

            summary = {
                'total_actions': len(self.timings),
                'elapsed_time_sec': elapsed_time,
                'average_fps': len(self.timings) / elapsed_time if elapsed_time > 0 else 0,
                'vl_encoding': {
                    'count': len(vl_times),
                    'mean_ms': np.mean(vl_times) if vl_times else 0,
                    'std_ms': np.std(vl_times) if vl_times else 0,
                    'min_ms': np.min(vl_times) if vl_times else 0,
                    'max_ms': np.max(vl_times) if vl_times else 0,
                    'p50_ms': np.percentile(vl_times, 50) if vl_times else 0,
                    'p95_ms': np.percentile(vl_times, 95) if vl_times else 0,
                    'p99_ms': np.percentile(vl_times, 99) if vl_times else 0,
                },
                'sensor_encoding': {
                    'mean_ms': np.mean(sensor_times) if sensor_times else 0,
                    'std_ms': np.std(sensor_times) if sensor_times else 0,
                },
                'robot_encoding': {
                    'mean_ms': np.mean(robot_times) if robot_times else 0,
                    'std_ms': np.std(robot_times) if robot_times else 0,
                },
                'action_prediction': {
                    'mean_ms': np.mean(action_times) if action_times else 0,
                    'std_ms': np.std(action_times) if action_times else 0,
                    'min_ms': np.min(action_times) if action_times else 0,
                    'max_ms': np.max(action_times) if action_times else 0,
                    'p50_ms': np.percentile(action_times, 50) if action_times else 0,
                    'p95_ms': np.percentile(action_times, 95) if action_times else 0,
                    'p99_ms': np.percentile(action_times, 99) if action_times else 0,
                },
                'total_inference': {
                    'mean_ms': np.mean(total_times) if total_times else 0,
                    'std_ms': np.std(total_times) if total_times else 0,
                    'min_ms': np.min(total_times) if total_times else 0,
                    'max_ms': np.max(total_times) if total_times else 0,
                }
            }

            return summary

    def save_results(self):
        """Save final summary (timings are auto-saved during execution)"""
        # Check if we have data (without holding lock)
        with self.lock:
            if not self.timings:
                print("⚠️ No timing data to save")
                return

            # Save any remaining unsaved records
            if self.output_dir and self.session_time:
                unsaved_count = len(self.timings) - self.last_saved_count
                if unsaved_count > 0:
                    print(f"💾 Saving final {unsaved_count} records...")
                    self._auto_save_append()

        # Get summary (releases lock before calling get_summary)
        summary = self.get_summary()

        # Save summary
        if self.output_dir and self.session_time:
            try:
                with open(self.summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"💾 Performance summary saved: {self.summary_path}")
                print(f"💾 Detailed timings saved: {self.timings_path} ({len(self.timings)} records)")

                # Print summary to console
                self._print_summary(summary)
            except Exception as e:
                print(f"⚠️ Failed to save summary: {e}")

    def _print_summary(self, summary: Dict):
        """Print performance summary to console"""
        print(f"\n{'='*80}")
        print("Performance Summary")
        print(f"{'='*80}")
        print(f"Total Actions: {summary['total_actions']}")
        print(f"Elapsed Time: {summary['elapsed_time_sec']:.1f}s")
        print(f"Average FPS: {summary['average_fps']:.2f} Hz")
        print(f"\nVL Encoding ({summary['vl_encoding']['count']} updates):")
        print(f"  Mean: {summary['vl_encoding']['mean_ms']:.1f}ms ± {summary['vl_encoding']['std_ms']:.1f}ms")
        print(f"  Range: [{summary['vl_encoding']['min_ms']:.1f}, {summary['vl_encoding']['max_ms']:.1f}]ms")
        print(f"  Percentiles: P50={summary['vl_encoding']['p50_ms']:.1f}ms, P95={summary['vl_encoding']['p95_ms']:.1f}ms, P99={summary['vl_encoding']['p99_ms']:.1f}ms")
        print(f"\nAction Prediction:")
        print(f"  Mean: {summary['action_prediction']['mean_ms']:.1f}ms ± {summary['action_prediction']['std_ms']:.1f}ms")
        print(f"  Range: [{summary['action_prediction']['min_ms']:.1f}, {summary['action_prediction']['max_ms']:.1f}]ms")
        print(f"  Percentiles: P50={summary['action_prediction']['p50_ms']:.1f}ms, P95={summary['action_prediction']['p95_ms']:.1f}ms, P99={summary['action_prediction']['p99_ms']:.1f}ms")
        print(f"\nTotal Inference:")
        print(f"  Mean: {summary['total_inference']['mean_ms']:.1f}ms ± {summary['total_inference']['std_ms']:.1f}ms")
        print(f"  Range: [{summary['total_inference']['min_ms']:.1f}, {summary['total_inference']['max_ms']:.1f}]ms")
        print(f"{'='*80}\n")


# ==============================
# Configuration
# ==============================
class Config:
    # Model settings (OPTIMIZED)
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    MODEL_TYPE = "flow_matching"  # "flow_matching" or "regression"
    ACTION_DIM = 7  # 6 joints + 1 gripper
    HORIZON = 8  # Action prediction horizon
    HIDDEN_DIM = 1024
    CACHE_DIR = str(Path(__file__).resolve().parent.parent / "cache" / "qwen_vl_features")

    # 🔥 OPTIMIZED: Image resize for faster VLM inference
    IMAGE_RESIZE_HEIGHT = 360
    IMAGE_RESIZE_WIDTH = 640

    # 🔥 OPTIMIZED: Sensor settings (100ms window)
    SENSOR_ENABLED = True
    SENSOR_TEMPORAL_LENGTH = 65  # 100ms at 650Hz (was 650)
    SENSOR_INPUT_CHANNELS = 1026  # 1 force + 1025 A-scan

    # 🔥 NEW: Robot state settings
    ROBOT_STATE_ENABLED = True
    ROBOT_STATE_DIM = 12  # 6 joints + 6 poses
    ROBOT_STATE_BUFFER_LENGTH = 100  # 100 samples @ 100Hz = 1 second window

    FUSION_STRATEGY = 'concat'

    # Flow matching settings
    FLOW_STEPS = 10  # ODE integration steps for flow matching
    FLOW_SOLVER = 'euler'  # 'euler' or 'rk4'

    # Network settings
    ZMQ_CAM_PULL_PORT = 5555
    ZMQ_ROBOT_PUB_ADDRESS = "127.0.0.1"  # Default: same machine as robot_command_receiver
    ZMQ_ROBOT_PUB_PORT = 5556
    ZMQ_ROBOT_TOPIC = b"robot_state"
    ZMQ_CMD_PUSH_ADDRESS = None  # Set from --robot-ip at runtime
    ZMQ_CMD_PUSH_PORT = 5000           # The port robot_command_receiver listens on
    SENSOR_UDP_PORT = 9999
    SENSOR_UDP_IP = "0.0.0.0"
    SENSOR_BUFFER_SIZE = 4 * 1024 * 1024

    # Sensor packet format
    SENSOR_NXZRt = 1025
    SENSOR_PACKET_HEADER_FORMAT = '<ddf'  # ts, send_ts, force
    SENSOR_PACKET_HEADER_SIZE = struct.calcsize(SENSOR_PACKET_HEADER_FORMAT)
    SENSOR_ALINE_FORMAT = f'<{SENSOR_NXZRt}f'
    SENSOR_ALINE_SIZE = struct.calcsize(SENSOR_ALINE_FORMAT)
    SENSOR_TOTAL_PACKET_SIZE = SENSOR_PACKET_HEADER_SIZE + SENSOR_ALINE_SIZE
    SENSOR_CALIBRATION_COUNT = 50

    # Robot packet format
    ROBOT_PAYLOAD_FORMAT = '<ddf12f'  # ts, send_ts, force, 6 joints, 6 pose
    ROBOT_PAYLOAD_SIZE = struct.calcsize(ROBOT_PAYLOAD_FORMAT)

    # Camera settings
    ZED_SERIAL_TO_VIEW = {
        "41182735": "View1",  # ZED 1 left
        "49429257": "View2",  # ZED 2 left
        "44377151": "View3",  # ZED 3 left
        "49045152": "View4"   # ZED 4 left
    }
    OAK_KEYWORD = "OAK"

    # 🔥 OPTIMIZED: Async inference settings
    ACTION_EXPERT_HZ = 10.0  # Action Expert runs at 10Hz
    VLM_REUSE_COUNT = 4  # Reuse VL features 4 times (VLM updates every 400ms)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging settings
    STATUS_PERIOD = 2.0
    STALL_SEC = 5.0


# ==============================
# Async Image Writer
# ==============================
class AsyncImageWriter(threading.Thread):
    """Asynchronous image writer to avoid blocking main loop"""
    def __init__(self, max_queue=5000):
        super().__init__(daemon=True)
        self.q = Queue(max_queue)
        self.stop_flag = threading.Event()
        self.written_count = 0

    def submit(self, path, img):
        if not self.stop_flag.is_set():
            try:
                self.q.put_nowait((path, img))
            except:
                pass  # Drop frame if queue full

    def run(self):
        while True:
            try:
                path, img = self.q.get(timeout=0.1)
                try:
                    cv2.imwrite(path, img)
                    self.written_count += 1
                except Exception as e:
                    print(f"[Writer] Error saving {path}: {e}")
                finally:
                    self.q.task_done()
            except Empty:
                if self.stop_flag.is_set() and self.q.empty():
                    break
                continue

    def stop(self):
        print(f"🕒 Flushing remaining {self.q.qsize()} images...")
        self.stop_flag.set()
        self.q.join()
        print(f"🛑 Writer thread stopped. Total written: {self.written_count}")


# ==============================
# Multi-View Image Buffer
# ==============================
class MultiViewImageBuffer:
    """
    Manages latest images from all camera views
    Thread-safe with automatic image resizing to 640x360
    """
    def __init__(self, required_views=None, save_dir=None, writer=None, resize_height=360, resize_width=640):
        self.required_views = required_views or ['View1', 'View2', 'View3', 'View4', 'View5']
        self.latest_images = {}  # view_name -> (img, timestamp)
        self.lock = threading.Lock()
        self.update_count = defaultdict(int)
        self.save_dir = save_dir
        self.writer = writer
        self.save_enabled = save_dir is not None and writer is not None
        self.resize_height = resize_height
        self.resize_width = resize_width

    def update(self, view_name: str, img: np.ndarray, timestamp: float, cam_name: str = ""):
        # Resize image to 640x360 for faster VLM inference
        img_resized = cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)

        with self.lock:
            self.latest_images[view_name] = (img_resized, timestamp)
            self.update_count[view_name] += 1

            # Save original size image if enabled
            if self.save_enabled:
                filename = f"{cam_name}_{timestamp:.3f}.jpg" if cam_name else f"{view_name}_{timestamp:.3f}.jpg"
                save_path = os.path.join(self.save_dir, view_name, filename)
                self.writer.submit(save_path, img)  # Save original

    def get_multi_view_set(self) -> dict:
        """Get latest image set from all views"""
        with self.lock:
            return {
                view: (img.copy(), ts)
                for view, (img, ts) in self.latest_images.items()
            }

    def is_ready(self) -> bool:
        """Check if we have at least one image from each required view"""
        with self.lock:
            return all(view in self.latest_images for view in self.required_views)

    def get_oldest_timestamp(self) -> float:
        """Get the oldest timestamp among current images"""
        with self.lock:
            if not self.latest_images:
                return 0.0
            return min(ts for _, ts in self.latest_images.values())


# ==============================
# Robot State Circular Buffer (joints + poses)
# ==============================
class RobotStateCircularBuffer:
    """
    Maintains a circular buffer of robot state data (joints + poses)
    Thread-safe operations for real-time updates
    """
    def __init__(self, max_length=65, state_dim=12, save_buffer=None):
        self.max_length = max_length
        self.state_dim = state_dim
        self.buffer = deque(maxlen=max_length)
        self.lock = threading.Lock()
        self.save_buffer = save_buffer  # Optional: list to save all data

    def add_state(self, joints, pose, timestamp):
        """Add single robot state (6 joints + 6 pose)"""
        with self.lock:
            # Combine joints + pose into (12,) vector
            state = np.concatenate([joints, pose], dtype=np.float32)
            self.buffer.append((state, timestamp))

            # Save to permanent buffer if enabled
            if self.save_buffer is not None:
                self.save_buffer.append({
                    'timestamp': timestamp,
                    'joints': joints.copy(),
                    'pose': pose.copy()
                })

    def get_tensor(self) -> torch.Tensor:
        """Get current buffer as torch tensor (T, 12) with padding if needed"""
        with self.lock:
            if len(self.buffer) == 0:
                # Return zeros if no data yet
                return torch.zeros(self.max_length, self.state_dim, dtype=torch.float32)

            # Extract states (ignore timestamps)
            states = [state for state, _ in self.buffer]
            data = np.array(states, dtype=np.float32)  # (current_len, 12)

            # Pad to max_length if needed
            if len(data) < self.max_length:
                pad_length = self.max_length - len(data)
                padding = np.zeros((pad_length, self.state_dim), dtype=np.float32)
                data = np.concatenate([padding, data], axis=0)

            return torch.from_numpy(data)  # (65, 12)

    def is_ready(self) -> bool:
        """Check if buffer has enough samples (at least 50% full)"""
        with self.lock:
            return len(self.buffer) >= self.max_length // 2

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)

    def get_latest_timestamp(self) -> float:
        """Get timestamp of most recent data in buffer"""
        with self.lock:
            if len(self.buffer) == 0:
                return 0.0
            return self.buffer[-1][1]  # Last element's timestamp

    def is_fresh(self, max_age_sec: float = 1.0) -> bool:
        """Check if buffer data is fresh (not stale)"""
        latest_ts = self.get_latest_timestamp()
        if latest_ts == 0.0:
            return False
        age = time.time() - latest_ts
        return age <= max_age_sec


# ==============================
# Sensor Data Circular Buffer (OPTIMIZED for 65 samples)
# ==============================
class SensorCircularBuffer:
    """
    Maintains a circular buffer of sensor data for 100ms window (65 samples @ 650Hz)
    Thread-safe operations for real-time updates
    """
    def __init__(self, max_length=65, channels=1026, save_buffer=None):
        self.max_length = max_length
        self.channels = channels
        self.buffer = deque(maxlen=max_length)
        self.lock = threading.Lock()
        self.save_buffer = save_buffer  # Optional: list to save all data

    def add_samples(self, samples: list):
        """Add multiple samples (from UDP batch)"""
        with self.lock:
            for sample in samples:
                # Combine force + aline into (1026,) vector
                force = np.array([sample['force']], dtype=np.float32)
                aline = sample['aline'].astype(np.float32)
                combined = np.concatenate([force, aline])  # (1026,)
                # Store with timestamp for freshness check
                self.buffer.append((combined, sample['timestamp']))

                # Save to permanent buffer if enabled
                if self.save_buffer is not None:
                    self.save_buffer.append({
                        'timestamp': sample['timestamp'],
                        'send_timestamp': sample['send_timestamp'],
                        'force': sample['force'],
                        'aline': sample['aline']
                    })

    def get_tensor(self) -> torch.Tensor:
        """Get current buffer as torch tensor (T, C) with padding if needed"""
        with self.lock:
            if len(self.buffer) == 0:
                # Return zeros if no data yet
                return torch.zeros(self.max_length, self.channels, dtype=torch.float32)

            # Extract sensor data (ignore timestamps)
            samples = [sample for sample, _ in self.buffer]
            data = np.array(samples, dtype=np.float32)  # (current_len, C)

            # Pad to max_length if needed
            if len(data) < self.max_length:
                pad_length = self.max_length - len(data)
                padding = np.zeros((pad_length, self.channels), dtype=np.float32)
                data = np.concatenate([padding, data], axis=0)

            return torch.from_numpy(data)  # (65, 1026)

    def is_ready(self) -> bool:
        """Check if buffer has enough samples (at least 50% full)"""
        with self.lock:
            return len(self.buffer) >= self.max_length // 2

    def size(self) -> int:
        with self.lock:
            return len(self.buffer)

    def get_latest_timestamp(self) -> float:
        """Get timestamp of most recent data in buffer"""
        with self.lock:
            if len(self.buffer) == 0:
                return 0.0
            return self.buffer[-1][1]  # Last element's timestamp

    def is_fresh(self, max_age_sec: float = 1.0) -> bool:
        """Check if buffer data is fresh (not stale)"""
        latest_ts = self.get_latest_timestamp()
        if latest_ts == 0.0:
            return False
        age = time.time() - latest_ts
        return age <= max_age_sec


# ==============================
# Async VLA Inference Engine
# ==============================
class AsyncVLAInferenceEngine:
    """
    Async VLA Inference Engine with VL feature reuse

    Architecture:
    - VLM Thread: Updates VL features in background (~381ms, 2.6Hz)
    - Action Thread: Predicts actions at 10Hz using cached VL features
    - VL features reused 4x before update
    """
    def __init__(self, config: Config, checkpoint_path: str = None, performance_monitor: PerformanceMonitor = None, verbose: bool = False, task_name: str = "eye"):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.performance_monitor = performance_monitor
        self.verbose = verbose
        self.task_name = task_name

        # Generate instruction prompt (same format as training dataset)
        self.text_prompt = (f"""Respond ONLY with the next action.
Environment Context:
- This is a Meca500 robot workspace.
- The end-effector holds a needle; the needle tip is the tool.
- The scene is an optical table with many holes, but these are NOT targets.
- The ONLY true insertion target is the {task_name}.

Task:
You must analyze the five camera views and determine the needle’s relative position to the {task_name}.
Identify:
1) needle tip location
2) alignment relative to the {task_name} center
3) required direction to align for insertion

Respond with:
- target visibility
- needle alignment
- required adjustment direction
- insertion readiness (yes/no)
"""
        )

        print(f"\n{'='*80}")
        print(f"Initializing Async VLA Inference Engine")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {config.MODEL_NAME}")
        print(f"Model Type: {config.MODEL_TYPE}")
        print(f"Task Name: {task_name}")
        print(f"Text Prompt: {self.text_prompt[:100]}...")  # Show first 100 chars
        print(f"Image Resize: {config.IMAGE_RESIZE_WIDTH}x{config.IMAGE_RESIZE_HEIGHT}")
        print(f"Sensor Window: {config.SENSOR_TEMPORAL_LENGTH} samples (100ms @ 650Hz)")
        print(f"Robot State: {config.ROBOT_STATE_ENABLED}")
        print(f"Action Expert: {config.ACTION_EXPERT_HZ} Hz")
        print(f"VL Reuse: {config.VLM_REUSE_COUNT}x")
        print(f"Fusion Strategy: {config.FUSION_STRATEGY}")
        if config.MODEL_TYPE == 'flow_matching':
            print(f"Flow Steps: {config.FLOW_STEPS}, Solver: {config.FLOW_SOLVER}")

        # Load model
        # IMPORTANT: Parameters must match training script (TRAIN_FlowMatching.py:804-816)
        self.model = QwenVLAUnified(
            model_type=config.MODEL_TYPE,
            vl_model_name=config.MODEL_NAME,
            action_dim=config.ACTION_DIM,
            horizon=config.HORIZON,
            hidden_dim=config.HIDDEN_DIM,
            cache_dir=config.CACHE_DIR,
            finetune_vl="none",
            sensor_enabled=config.SENSOR_ENABLED,
            sensor_encoder_type='force_aware',  # ✅ Match training script
            sensor_input_channels=config.SENSOR_INPUT_CHANNELS,
            sensor_temporal_length=config.SENSOR_TEMPORAL_LENGTH,
            sensor_output_dim=1024,  # ✅ Must match training script! (was 2048)
            robot_state_enabled=config.ROBOT_STATE_ENABLED,
            robot_state_temporal_length=config.ROBOT_STATE_BUFFER_LENGTH,  # 100 samples @ 100Hz
            robot_state_output_dim=512,  # ✅ Match training script
            image_resize_height=config.IMAGE_RESIZE_HEIGHT,
            image_resize_width=config.IMAGE_RESIZE_WIDTH,
            flow_steps=config.FLOW_STEPS if config.MODEL_TYPE == 'flow_matching' else 10,
            flow_solver=config.FLOW_SOLVER if config.MODEL_TYPE == 'flow_matching' else 'euler',
            # VL optimization: Parallel view encoding for 2-3x faster VL updates
            parallel_view_encoding=True,  # 🚀 Enable multi-view parallel encoding
            view_aggregation='mean',      # Aggregate multiple views
            device_map="cuda",  # Use single GPU for inference instead of "auto"
        )

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract state dict
            if 'model_state_dict' in checkpoint:
                checkpoint_state = checkpoint['model_state_dict']
            else:
                checkpoint_state = checkpoint

            # Get current model's state dict to check for size mismatches
            current_state = self.model.state_dict()

            # Filter out incompatible keys (size mismatch)
            compatible_state = {}
            incompatible_keys = []
            size_mismatches = []

            for key, value in checkpoint_state.items():
                if key in current_state:
                    if value.shape == current_state[key].shape:
                        compatible_state[key] = value
                    else:
                        incompatible_keys.append(key)
                        size_mismatches.append(f"{key}: {value.shape} -> {current_state[key].shape}")
                else:
                    # Key doesn't exist in current model
                    incompatible_keys.append(key)

            # Load only compatible weights
            missing_keys, unexpected_keys = self.model.load_state_dict(compatible_state, strict=False)

            # Report what was loaded/skipped
            print("✅ Checkpoint loaded with compatibility mode:")

            if size_mismatches:
                print(f"   ⚠️ Size mismatches (skipped): {len(size_mismatches)} keys")
                # Show first few mismatches
                for mismatch in size_mismatches[:3]:
                    print(f"      {mismatch}")
                if len(size_mismatches) > 3:
                    print(f"      ... and {len(size_mismatches) - 3} more")

            if missing_keys:
                print(f"   ⚠️ Missing keys (random init): {len(missing_keys)} keys")
                # Print only sensor/robot related missing keys for brevity
                relevant_missing = [k for k in missing_keys if 'sensor' in k or 'robot' in k]
                if relevant_missing:
                    print(f"      Sensor/Robot related: {relevant_missing[:5]}{'...' if len(relevant_missing) > 5 else ''}")

            if unexpected_keys:
                print(f"   ⚠️ Unexpected keys (ignored): {len(unexpected_keys)} keys")

            # Check if critical components were loaded
            vl_keys_loaded = sum(1 for k in compatible_state.keys() if 'vl_encoder' in k or 'vl_model' in k)
            action_keys_loaded = sum(1 for k in compatible_state.keys() if 'action_expert' in k)
            sensor_keys_loaded = sum(1 for k in compatible_state.keys() if 'sensor_encoder' in k or 'sensor_proj' in k)
            robot_keys_loaded = sum(1 for k in compatible_state.keys() if 'robot_state' in k)

            print(f"   ✅ VL Encoder: {vl_keys_loaded} parameters loaded")
            print(f"   ✅ Action Expert: {action_keys_loaded} parameters loaded")
            print(f"   ℹ️ Sensor Encoder: {sensor_keys_loaded} parameters loaded (may need fine-tuning)")
            print(f"   ℹ️ Robot Encoder: {robot_keys_loaded} parameters loaded (may need fine-tuning)")

            if len(incompatible_keys) > 0:
                print(f"   ⚠️ Note: {len(incompatible_keys)} incompatible parameters will use random initialization")

        self.model = self.model.to(self.device)
        self.model.eval()

        # VL feature cache (V2: stores tuple of (image_features, guidance_vectors))
        self.vl_image_features = None
        self.vl_guidance_vectors = None
        self.vl_features_lock = threading.Lock()
        self.vl_update_count = 0
        self.vl_update_times = deque(maxlen=10)

        # Action prediction stats
        self.action_count = 0
        self.action_times = deque(maxlen=20)
        self.stats_lock = threading.Lock()

        print(f"{'='*80}\n")

    @torch.no_grad()
    def update_vl_features(self, images_dict: dict):
        """
        Update VL features (runs in background thread)
        This is the slow operation (~381ms with 5 views @ 640x360)
        Uses self.text_prompt generated from task_name
        """
        start_time = time.time()
        if self.verbose:
            print(f"[VL] Encoding {len(images_dict)} views...")

        # Save images temporarily
        temp_dir = Path("/tmp/vla_inference")
        temp_dir.mkdir(exist_ok=True)

        image_paths = []
        sorted_views = sorted(images_dict.keys())
        for view in sorted_views:
            img, ts = images_dict[view]
            temp_path = temp_dir / f"{view}_{ts:.3f}.jpg"
            cv2.imwrite(str(temp_path), img)
            image_paths.append(str(temp_path))

        # Extract VL features only
        text_inputs = [self.text_prompt]
        image_inputs = [image_paths]

        t_encode_start = time.time()
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            # V2: Use model's VL encoder which returns (image_features, guidance_vectors) tuple
            image_features, guidance_vectors = self.model.vl_encoder.encode(
                text_inputs,
                image_inputs,
                cache_keys=[f"realtime_{time.time()}"],
                use_cache=False
            )
        t_encode_end = time.time()
        encode_time = (t_encode_end - t_encode_start) * 1000
        if self.verbose:
            print(f"[VL] Vision/Text encoding took {encode_time:.1f} ms")

        elapsed = (time.time() - start_time) * 1000
        self.vl_update_times.append(elapsed / 1000)

        # Update cache (V2: store both features)
        with self.vl_features_lock:
            self.vl_image_features = image_features
            self.vl_guidance_vectors = guidance_vectors
            self.vl_update_count += 1

        # Clean up temp files
        for path in image_paths:
            Path(path).unlink(missing_ok=True)

        # 🔹 PerformanceMonitor에도 기록 추가
        if self.performance_monitor:
            timing_record = InferenceTimings(
                timestamp=time.time(),
                action_id=self.action_count,
                vl_update_number=self.vl_update_count,
                vl_update_time=elapsed
            )
            self.performance_monitor.add_timing(timing_record)

        return elapsed / 1000

    @torch.no_grad()
    def predict_action(self, sensor_data: torch.Tensor = None, robot_states: torch.Tensor = None,
                      z_chunk: torch.Tensor = None, current_vl_update_number: int = 0,
                      sensor_buffer_size: int = 0, robot_buffer_size: int = 0) -> dict:
        """
        Predict actions using cached VL features (fast operation ~20-30ms)
        This runs at 10Hz

        Args:
            sensor_data: (65, 1026) - sensor tensor
            robot_states: (65, 12) - robot state tensor (joints + poses)
            z_chunk: (1, H, A) - action chunk (for regression mode)
            current_vl_update_number: Current VL update number
            sensor_buffer_size: Current sensor buffer size
            robot_buffer_size: Current robot buffer size
        """
        start_time = time.time()
        timings = {}

        # Get cached VL features (V2: both image features and guidance vectors)
        with self.vl_features_lock:
            if self.vl_image_features is None or self.vl_guidance_vectors is None:
                return None
            image_features = self.vl_image_features
            guidance_vectors = self.vl_guidance_vectors

        # Prepare sensor batch
        sensor_batch = None
        if self.config.SENSOR_ENABLED and sensor_data is not None:
            sensor_batch = sensor_data.unsqueeze(0).to(self.device)  # (1, 65, 1026)

        # Prepare robot state batch
        robot_batch = None
        if self.config.ROBOT_STATE_ENABLED and robot_states is not None:
            robot_batch = robot_states.unsqueeze(0).to(self.device)  # (1, 65, 12)

        # Model inference with detailed timing (V2 Architecture)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            # 1. Encode sensor features
            t_sensor_start = time.time()
            sensor_features_encoded = None
            if sensor_batch is not None:
                sensor_features_encoded = self.model.sensor_encoder(sensor_batch)
            timings['sensor_encoding'] = (time.time() - t_sensor_start) * 1000

            # 2. Encode robot state features
            t_robot_start = time.time()
            robot_state_features_encoded = None
            if robot_batch is not None:
                robot_state_features_encoded = self.model.robot_state_encoder(robot_batch)
            timings['robot_encoding'] = (time.time() - t_robot_start) * 1000

            # 3. Prepare vision context + concatenated sensor features for ActionExpert (V2)
            context_features = image_features
            if context_features.dim() == 2:
                context_features = context_features.unsqueeze(1)

            sensor_features_combined = None
            if sensor_features_encoded is not None and robot_state_features_encoded is not None:
                sensor_features_combined = torch.cat(
                    (sensor_features_encoded, robot_state_features_encoded), dim=-1
                )
            elif sensor_features_encoded is not None:
                sensor_features_combined = sensor_features_encoded
            elif robot_state_features_encoded is not None:
                sensor_features_combined = robot_state_features_encoded

            # 4. Action prediction with V2 architecture
            t_action_start = time.time()
            if self.config.MODEL_TYPE == 'flow_matching':
                pred_actions = self.model.action_expert.sample(
                    context_features,
                    guidance_vectors,
                    sensor_features=sensor_features_combined,
                    num_steps=self.config.FLOW_STEPS,
                    method=self.config.FLOW_SOLVER
                )
                delta = None
            else:  # regression
                # Prepare z_chunk
                if z_chunk is None:
                    z_chunk = torch.zeros(1, self.config.HORIZON, self.config.ACTION_DIM,
                                        dtype=torch.float32, device=self.device)

                pred_actions, delta = self.model.action_expert(
                    z_chunk, context_features, guidance_vectors,
                    sensor_features=sensor_features_combined
                )
            timings['action_prediction'] = (time.time() - t_action_start) * 1000

        elapsed = time.time() - start_time
        timings['total'] = elapsed * 1000

        # Update stats
        with self.stats_lock:
            self.action_count += 1
            self.action_times.append(elapsed)

        # Record performance timing if monitor is available
        if self.performance_monitor:
            timing_record = InferenceTimings(
                timestamp=time.time(),
                action_id=self.action_count,
                vl_update_number=current_vl_update_number,
                vl_update_time=0,  # Will be updated by VL thread
                sensor_encoding_time=timings.get('sensor_encoding', 0),
                robot_encoding_time=timings.get('robot_encoding', 0),
                action_prediction_time=timings.get('action_prediction', 0),
                total_time=timings['total'],
                sensor_buffer_size=sensor_buffer_size,
                robot_buffer_size=robot_buffer_size
            )
            self.performance_monitor.add_timing(timing_record)

        if self.verbose:
            print(f"[INFER] VLM Features       : Used update #{current_vl_update_number}")
            print(f"[INFER] Sensor Encoding    : {timings.get('sensor_encoding', 0):.1f} ms")
            print(f"[INFER] Robot State Encoding : {timings.get('robot_encoding', 0):.1f} ms")
            print(f"[INFER] Action Prediction  : {timings.get('action_prediction', 0):.1f} ms")
            print(f"[INFER] Total Inference Time : {timings.get('total', 0):.1f} ms")

        return {
            'actions': pred_actions[0].float().cpu().numpy(),  # (H, action_dim)
            'delta': delta[0].float().cpu().numpy() if delta is not None else None,
            'inference_time': elapsed,
            'timestamp': time.time(),
            'timings': timings
        }

    def get_stats(self) -> dict:
        with self.stats_lock:
            vl_avg = np.mean(self.vl_update_times) if self.vl_update_times else 0.0
            action_avg = np.mean(self.action_times) if self.action_times else 0.0

            return {
                'vl_update_count': self.vl_update_count,
                'vl_avg_time_ms': vl_avg * 1000,
                'action_count': self.action_count,
                'action_avg_time_ms': action_avg * 1000,
                'vl_features_cached': (self.vl_image_features is not None and self.vl_guidance_vectors is not None)
            }


# ==============================
# UDP Sensor Receiver (수정됨)
# ==============================
class SensorUDPReceiver(threading.Thread):
    """Receives sensor data via UDP and updates circular buffer"""
    def __init__(self, config: Config, sensor_buffer: SensorCircularBuffer, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.sensor_buffer = sensor_buffer
        self.stop_event = stop_event
        self.clock_offset = None
        self.calibration_samples = []
        self.packet_count = 0
        self.last_recv_time = 0  # Track last successful receive time

    def run(self):
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.SENSOR_BUFFER_SIZE)
            sock.bind((self.config.SENSOR_UDP_IP, self.config.SENSOR_UDP_PORT))
            sock.settimeout(1.0)
            print(f"✅ Sensor UDP Receiver started on port {self.config.SENSOR_UDP_PORT}")
        except Exception as e:
            print(f"[ERROR] Failed to bind UDP socket: {e}")
            return

        # ✅ 수정: 이제 '배치'가 아닌 '패킷' 기준입니다.
        print(f"⏳ Calibrating sensor clock offset (first {self.config.SENSOR_CALIBRATION_COUNT} packets)...")

        while not self.stop_event.is_set():
            try:
                data, addr = sock.recvfrom(self.config.SENSOR_BUFFER_SIZE)
            except socket.timeout:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[UDP Sensor] Receive error: {e}")
                continue

            recv_time = time.time()

            # ✅ 수정: 단일 패킷을 처리하는 로직으로 변경
            try:
                # 1. 수신한 데이터가 정확히 DataPacket 1개의 크기인지 확인
                if len(data) != self.config.SENSOR_TOTAL_PACKET_SIZE:
                    print(f"[UDP Sensor] Warning: Received packet of wrong size. Got {len(data)}, expected {self.config.SENSOR_TOTAL_PACKET_SIZE}")
                    continue

                # 2. 'num_packets' 헤더가 없으므로 바로 파싱 시작
                mv = memoryview(data)
                offset = 0

                # 3. 헤더 언패킹
                header = mv[offset:offset + self.config.SENSOR_PACKET_HEADER_SIZE]
                ts, send_ts, force = struct.unpack(self.config.SENSOR_PACKET_HEADER_FORMAT, header)
                offset += self.config.SENSOR_PACKET_HEADER_SIZE

                # 4. A-line 데이터 언패킹
                aline_bytes = mv[offset:offset + self.config.SENSOR_ALINE_SIZE]
                aline = np.frombuffer(aline_bytes, dtype=np.float32).copy()
                offset += self.config.SENSOR_ALINE_SIZE

                # 5. 단일 레코드 생성
                record = {
                    'timestamp': ts,
                    'send_timestamp': send_ts,
                    'force': force,
                    'aline': aline
                }

                # 6. 클럭 동기화 로직 (유지)
                #    (이제 이 단일 패킷의 send_ts를 기준으로 계산됩니다)
                if self.clock_offset is None:
                    net_plus_offset = recv_time - send_ts
                    self.calibration_samples.append(net_plus_offset)

                    if len(self.calibration_samples) >= self.config.SENSOR_CALIBRATION_COUNT:
                        self.clock_offset = np.mean(self.calibration_samples)
                        # ✅ 수정: '패킷' 기준으로 변경되었음을 명시
                        print(f"\n✅ Sensor Clock Offset Calibrated (from {self.config.SENSOR_CALIBRATION_COUNT} packets): {self.clock_offset * 1000:.1f} ms\n")

                # 7. 버퍼에 추가 (add_samples가 리스트를 받는다고 가정)
                self.sensor_buffer.add_samples([record])
                self.packet_count += 1  # ✅ 수정: 1씩 증가
                self.last_recv_time = recv_time

            except Exception as e:
                # ✅ 수정: 에러 출력 시 수신된 데이터 길이를 포함
                print(f"[ERROR] Sensor UDP unpack failed (Data len: {len(data)}): {e}")
                continue

        sock.close()
        print("🛑 Sensor UDP Receiver stopped")


# ==============================
# Data Saving Functions
# ==============================
def save_robot_data_to_csv(data_list, filepath):
    """Save robot data to CSV"""
    if not data_list:
        return

    print(f"💾 Saving {len(data_list)} robot states to {filepath}")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                "recv_timestamp", "origin_timestamp", "send_timestamp", "force_placeholder",
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                "pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"
            ])
            w.writerows(data_list)
        print(f"💾✅ Robot data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save robot data: {e}")


def save_sensor_data_to_npz(data_list, filepath):
    """Save sensor data to NPZ"""
    if not data_list:
        return

    print(f"💾 Saving {len(data_list)} sensor records to {filepath}")
    try:
        timestamps = np.array([d['timestamp'] for d in data_list], dtype=np.float64)
        send_timestamps = np.array([d['send_timestamp'] for d in data_list], dtype=np.float64)
        forces = np.array([d['force'] for d in data_list], dtype=np.float32)
        alines = np.array([d['aline'] for d in data_list], dtype=np.float32)

        np.savez(filepath, timestamps=timestamps, send_timestamps=send_timestamps,
                forces=forces, alines=alines)
        print(f"💾✅ Sensor data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save sensor data: {e}")


def save_inference_results(results_list, filepath):
    """Save inference results to JSON"""
    if not results_list:
        return

    print(f"💾 Saving {len(results_list)} inference results to {filepath}")
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results_list:
            serializable_results.append({
                'timestamp': result['timestamp'],
                'actions': result['actions'].tolist() if isinstance(result['actions'], np.ndarray) else result['actions'],
                'delta': result['delta'].tolist() if isinstance(result['delta'], np.ndarray) else result['delta'],
                'inference_time': result['inference_time'],
                'vl_update_number': result.get('vl_update_number', 0),
                'robot_state': result.get('robot_state')
            })

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"💾✅ Inference results saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save inference results: {e}")

# =======================================================================================
# Main Async Inference Loop
# =======================================================================================
def main():
    parser = argparse.ArgumentParser(description='Async Real-time VLA Inference Receiver')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--save-data', action='store_true', help='Save images, sensor data, robot state')
    parser.add_argument('--vl-reuse', type=int, default=4, help='VL feature reuse count (default: 4)')
    parser.add_argument('--model-type', type=str, default='flow_matching', choices=['flow_matching', 'regression'], help='Model type: flow_matching or regression (default: flow_matching)')
    parser.add_argument('--flow-steps', type=int, default=10, help='ODE integration steps for flow matching (default: 10)')
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for debugging.")
    parser.add_argument('--robot-ip', type=str, default='10.130.41.111', help='IP address of the robot state publisher (robot_command_receiver.py).')
    parser.add_argument('--auto-start', action='store_true', help='Automatically send START command to robot before inference')
    parser.add_argument('--start-joints', type=float, nargs=6, default=[130.0, 70.0, 265.0, -150.0, 25.0, 105.0], help='Start joint positions for auto-start (default: [0,0,0,0,0,0])')
                        #168.387659, 36.190885, 250.816119, 178.747768, 19.072951, 144.28431 # [169.055308, -22.135383, 232.007663, 177.66655, 19.996871, 144.314525],
    parser.add_argument('--task-name', type=str, default='eye', help='Task name for prompt generation (e.g., eye, yellow_point, blue_point)')
    args = parser.parse_args()

    config = Config()
    config.ZMQ_ROBOT_PUB_ADDRESS = args.robot_ip  # Set robot IP from args
    config.ZMQ_CMD_PUSH_ADDRESS = args.robot_ip   # Also use for command push
    config.VLM_REUSE_COUNT = args.vl_reuse
    config.MODEL_TYPE = args.model_type
    config.FLOW_STEPS = args.flow_steps

    # Setup output directory
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./async_inference_{session_time}")
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories if saving data
    image_writer = None
    sensor_save_buffer = None
    robot_save_buffer = None

    if args.save_data:
        print(f"\n{'='*80}")
        print(f"📁 Data saving enabled: {output_dir}")
        print(f"{'='*80}\n")

        # Create view directories
        for view in ['View1', 'View2', 'View3', 'View4', 'View5']:
            (output_dir / view).mkdir(exist_ok=True)

        # Initialize image writer
        image_writer = AsyncImageWriter(max_queue=5000)
        image_writer.start()

        # Initialize save buffers
        sensor_save_buffer = []
        robot_save_buffer = []
    else:
        print(f"\n{'='*80}")
        print(f"📁 Inference only mode (no data saving)")
        print(f"📁 Results will be saved to: {output_dir}")
        print(f"{'='*80}\n")

    # Initialize components
    stop_event = threading.Event()

    # Initialize performance monitor with auto-save
    performance_monitor = PerformanceMonitor(
        output_dir=output_dir,
        session_time=session_time,
        auto_save_interval=10  # Save every 10 records
    )
    print("✅ Performance monitoring enabled (auto-save every 10 records)")

    image_buffer = MultiViewImageBuffer(
        save_dir=str(output_dir) if args.save_data else None,
        writer=image_writer,
        resize_height=config.IMAGE_RESIZE_HEIGHT,
        resize_width=config.IMAGE_RESIZE_WIDTH
    )
    sensor_buffer = SensorCircularBuffer(
        max_length=config.SENSOR_TEMPORAL_LENGTH,
        channels=config.SENSOR_INPUT_CHANNELS,
        save_buffer=sensor_save_buffer
    )
    robot_state_buffer = RobotStateCircularBuffer(
        max_length=config.ROBOT_STATE_BUFFER_LENGTH,  # 100 samples @ 100Hz
        state_dim=config.ROBOT_STATE_DIM,
        save_buffer=None  # Robot state already saved separately
    )

    # Initialize inference engine
    inference_engine = AsyncVLAInferenceEngine(config, args.checkpoint, performance_monitor, verbose=args.verbose, task_name=args.task_name)

    # ZMQ Setup
    ctx = zmq.Context.instance()

    # Camera socket
    cam_sock = ctx.socket(zmq.PULL)
    cam_sock.setsockopt(zmq.RCVHWM, 5000)
    cam_sock.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
    cam_sock.bind(f"tcp://0.0.0.0:{config.ZMQ_CAM_PULL_PORT}")
    print(f"✅ Camera PULL listening on port {config.ZMQ_CAM_PULL_PORT}")

    # Robot socket
    robot_sock = ctx.socket(zmq.SUB)
    robot_sock.setsockopt(zmq.RCVHWM, 100)
    robot_sock.connect(f"tcp://{config.ZMQ_ROBOT_PUB_ADDRESS}:{config.ZMQ_ROBOT_PUB_PORT}")
    robot_sock.subscribe(config.ZMQ_ROBOT_TOPIC)
    print(f"✅ Robot SUB connected to {config.ZMQ_ROBOT_PUB_ADDRESS}:{config.ZMQ_ROBOT_PUB_PORT}")

    # Robot Command socket (PUSH)
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.connect(f"tcp://{config.ZMQ_CMD_PUSH_ADDRESS}:{config.ZMQ_CMD_PUSH_PORT}")
    print(f"✅ Robot Command PUSH connected to {config.ZMQ_CMD_PUSH_ADDRESS}:{config.ZMQ_CMD_PUSH_PORT}")

    # Poller
    poller = zmq.Poller()
    poller.register(cam_sock, zmq.POLLIN)
    poller.register(robot_sock, zmq.POLLIN)

    # Start sensor receiver
    sensor_thread = SensorUDPReceiver(config, sensor_buffer, stop_event)
    sensor_thread.start()

    # State tracking
    robot_state = None
    cam_recv_count = defaultdict(int)
    inference_results = []

    # Data freshness tracking
    last_robot_recv_time = 0
    last_sensor_recv_time = 0
    last_camera_recv_time = 0

    # Async VL update tracking
    vl_update_counter = 0
    last_vl_update_time = 0
    vl_update_thread = None
    current_vl_update_number = 0

    # Action prediction tracking
    last_action_time = time.time()
    action_period = 1.0 / config.ACTION_EXPERT_HZ  # 0.1s for 10Hz
    last_status_print = time.time()

    # Auto-start flag
    start_command_sent = False

    # Action sending statistics
    action_send_success = 0
    action_send_failed = 0

    # Signal handler
    def sigint_handler(sig, frame):
        print("\n🛑 Ctrl+C detected — Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, sigint_handler)

    print(f"\n{'='*80}")
    print(f"Async Real-time Inference Started")
    print(f"{'='*80}")
    print(f"Model Type: {config.MODEL_TYPE.upper()}")
    if config.MODEL_TYPE == 'flow_matching':
        print(f"Flow Steps: {config.FLOW_STEPS} (Solver: {config.FLOW_SOLVER})")
    print(f"Action Expert: {config.ACTION_EXPERT_HZ} Hz (every {action_period*1000:.0f}ms)")
    print(f"VL Update: ~2.6 Hz (VL features reused {config.VLM_REUSE_COUNT}x)")
    print(f"Image Resolution: {config.IMAGE_RESIZE_WIDTH}x{config.IMAGE_RESIZE_HEIGHT}")
    print(f"Sensor Window: {config.SENSOR_TEMPORAL_LENGTH} samples (100ms @ 650Hz)")
    print(f"Robot State: ENABLED ({config.ROBOT_STATE_BUFFER_LENGTH} samples @ 100Hz, 12 dims: 6 joints + 6 poses)")
    print(f"Device: {config.DEVICE}")
    print(f"Data Saving: {'Enabled' if args.save_data else 'Disabled'}")
    print(f"\nWaiting for data from all sources (images, sensor, robot state)...")
    print(f"Press Ctrl+C to stop\n")

    try:
        while not stop_event.is_set():
            # Poll for messages (non-blocking)
            try:
                socks = dict(poller.poll(timeout=10))
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[WARN] Poller error: {e}")
                time.sleep(0.01)
                continue

            now = time.time()

            # Process camera messages
            if cam_sock in socks and socks[cam_sock] == zmq.POLLIN:
                while True:
                    try:
                        parts = cam_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) < 2:
                            continue

                        meta_raw, jpg = parts[0], parts[1]
                        meta = json.loads(meta_raw.decode("utf-8"))

                        cam_name = meta.get("camera", "unknown")
                        timestamp = float(meta.get("timestamp", 0.0))
                        
                        if args.verbose:
                            recv_time = time.time()
                            net_delay = (recv_time - timestamp) * 1000  # ms 단위
                            print(f"[RECV] Camera: {cam_name} at ts {timestamp:.3f} (delay: {net_delay:.1f}ms)")
                            
                        # Decode image
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        # Determine view name
                        view_name = None
                        cam_lower = cam_name.lower()

                        if "left" in cam_lower:
                            for serial, view in config.ZED_SERIAL_TO_VIEW.items():
                                if serial in cam_name:
                                    view_name = view
                                    break

                        if config.OAK_KEYWORD.lower() in cam_lower:
                            view_name = "View5"

                        if view_name:
                            image_buffer.update(view_name, img, timestamp, cam_name)
                            cam_recv_count[view_name] += 1
                            last_camera_recv_time = now

                    except zmq.Again:
                        break
                    except Exception as e:
                        print(f"[ERROR] Camera processing: {e}")
                        break

            # Process robot messages
            if robot_sock in socks and socks[robot_sock] == zmq.POLLIN:
                while True:
                    try:
                        parts = robot_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) != 2:
                            continue

                        topic, payload = parts[0], parts[1]
                        if len(payload) != config.ROBOT_PAYLOAD_SIZE:
                            continue

                        unpacked = struct.unpack(config.ROBOT_PAYLOAD_FORMAT, payload)
                        origin_ts, send_ts, force = unpacked[0:3]

                        if args.verbose:
                            print(f"[RECV] Robot state at ts {origin_ts:.3f} (J1: {unpacked[3]:.2f}, Px: {unpacked[9]:.2f})")

                        joints = np.array(unpacked[3:9], dtype=np.float32)  # 6 joints
                        pose = np.array(unpacked[9:15], dtype=np.float32)   # 6 poses

                        robot_state = {
                            'timestamp': origin_ts,
                            'joints': joints,
                            'pose': pose,
                            'recv_time': now
                        }

                        # Add to robot state buffer for model input
                        robot_state_buffer.add_state(joints, pose, origin_ts)
                        last_robot_recv_time = now

                        # Save robot data if enabled
                        if args.save_data:
                            robot_save_buffer.append([now] + list(unpacked))

                    except zmq.Again:
                        break
                    except Exception as e:
                        print(f"[ERROR] Robot processing: {e}")
                        break

            # === ASYNC VLM UPDATE (Background Thread) ===
            # Start VL update if:
            # 1. Enough time has passed OR first update
            # 2. Previous update thread is done (or None)
            # 3. Data is ready
            if image_buffer.is_ready():
                should_update_vl = (vl_update_counter % config.VLM_REUSE_COUNT == 0) or (inference_engine.vl_image_features is None)

                if should_update_vl:
                    # Check if previous VL update thread is done
                    if vl_update_thread is None or not vl_update_thread.is_alive():
                        # Start new VL update in background
                        images_dict = image_buffer.get_multi_view_set()
                        print(f"\n[VL] Starting background VL feature update #{current_vl_update_number + 1}...")

                        def vl_update_worker():
                            nonlocal current_vl_update_number, vl_update_counter
                            elapsed = inference_engine.update_vl_features(images_dict)
                            current_vl_update_number += 1
                            vl_update_counter = 0  # Reset counter AFTER update is complete
                            print(f"✅ [VL] Update #{current_vl_update_number} complete in {elapsed*1000:.0f}ms.")

                        vl_update_thread = threading.Thread(target=vl_update_worker, daemon=True)
                        vl_update_thread.start()

            # === ACTION EXPERT PREDICTION (10Hz) ===
            time_since_last_action = now - last_action_time

            if time_since_last_action >= action_period:
                # Check if all data sources are ready AND fresh (not stale)
                sensor_fresh = sensor_buffer.is_fresh(max_age_sec=1.0)
                robot_fresh = robot_state_buffer.is_fresh(max_age_sec=1.0)

                data_ready = (
                    inference_engine.vl_image_features is not None and
                    inference_engine.vl_guidance_vectors is not None and
                    sensor_buffer.is_ready() and sensor_fresh and
                    robot_state_buffer.is_ready() and robot_fresh
                )

                if data_ready:
                    # Send START command once if auto-start is enabled
                    if args.auto_start and not start_command_sent:
                        print(f"\n🚀 Sending AUTO-START command to robot...")
                        start_cmd = {
                            "cmd": "start",
                            "start_joints": args.start_joints,
                            "lock_j6": False
                        }
                        try:
                            cmd_sock.send_json(start_cmd, zmq.DONTWAIT)
                            print(f"✅ START command sent: {start_cmd}")
                            start_command_sent = True
                            time.sleep(0.5)  # Give robot time to process start command
                        except zmq.Again:
                            print("⚠️ Failed to send START command (socket busy)")
                        except Exception as e:
                            print(f"❌ Error sending START command: {e}")

                    if args.verbose:
                        print(f"[INFER] Preparing to run inference #{len(inference_results) + 1}...")
                    # Get sensor data
                    sensor_tensor = sensor_buffer.get_tensor()

                    # Get robot state data
                    robot_tensor = robot_state_buffer.get_tensor()

                    # Predict action (fast, ~20-30ms with flow matching)
                    result = inference_engine.predict_action(
                        sensor_data=sensor_tensor,
                        robot_states=robot_tensor,
                        current_vl_update_number=current_vl_update_number,
                        sensor_buffer_size=sensor_buffer.size(),
                        robot_buffer_size=robot_state_buffer.size()
                    )

                    if result:
                        vl_update_counter += 1

                        # Log action prediction (moved below to show action values)

                        # Save result
                        inference_results.append({
                            'timestamp': result['timestamp'],
                            'actions': result['actions'],
                            'delta': result['delta'],
                            'inference_time': result['inference_time'],
                            'vl_update_number': current_vl_update_number,
                            'robot_state': {
                                'joints': robot_state['joints'].tolist(),
                                'pose': robot_state['pose'].tolist(),
                                'timestamp': robot_state['timestamp']
                            } if robot_state else None
                        })

                        # === SEND ACTION TO ROBOT CONTROLLER ===
                        # Use the first action from the horizon
                        action_to_send = result['actions'][0]

                        # The model outputs 7D action (6-DoF + gripper)
                        # We send the 6-DoF delta pose to the robot receiver
                        delta_pose = action_to_send[:6].tolist()
                        gripper = action_to_send[6]

                        # Format the command
                        robot_cmd = {
                            "cmd": "dpose",
                            "dp": delta_pose
                        }

                        # Display action being sent
                        if not args.verbose:
                            print(f"[INFER] Action #{len(inference_results)} | VL_reuse={vl_update_counter}/{config.VLM_REUSE_COUNT} | Time: {result['inference_time']*1000:.1f}ms")
                            print(f"[ACTION] dpose: [x:{delta_pose[0]:+.4f}, y:{delta_pose[1]:+.4f}, z:{delta_pose[2]:+.4f}, a:{delta_pose[3]:+.4f}, b:{delta_pose[4]:+.4f}, r:{delta_pose[5]:+.4f}] | gripper:{gripper:.3f}")
                        else:
                            print(f"[ACTION] 7D Action: [x:{delta_pose[0]:+.4f}, y:{delta_pose[1]:+.4f}, z:{delta_pose[2]:+.4f}, a:{delta_pose[3]:+.4f}, b:{delta_pose[4]:+.4f}, r:{delta_pose[5]:+.4f}, grip:{gripper:.3f}]")

                        try:
                            cmd_sock.send_json(robot_cmd, zmq.DONTWAIT)
                            action_send_success += 1
                            if args.verbose:
                                print(f"[ACTION] ✅ Command sent successfully")
                        except zmq.Again:
                            action_send_failed += 1
                            print(f"⚠️ [ACTION] Robot command socket busy, command dropped. (Failed: {action_send_failed})")
                        except Exception as e:
                            action_send_failed += 1
                            print(f"💥 [ACTION] Error sending robot command: {e} (Failed: {action_send_failed})")

                    last_action_time = now

                else:
                    # Data not ready - show why
                    if time_since_last_action >= 2.0:  # Print warning every 2s
                        sensor_age = time.time() - sensor_buffer.get_latest_timestamp() if sensor_buffer.get_latest_timestamp() > 0 else 999
                        robot_age = time.time() - robot_state_buffer.get_latest_timestamp() if robot_state_buffer.get_latest_timestamp() > 0 else 999

                        print(f"[WAIT] Waiting for fresh data:")
                        print(f"  VL Features: {inference_engine.vl_image_features is not None and inference_engine.vl_guidance_vectors is not None}")
                        print(f"  Images: {image_buffer.is_ready()} ({len(image_buffer.latest_images)}/5)")
                        print(f"  Sensor: ready={sensor_buffer.is_ready()}, fresh={sensor_fresh} ({sensor_buffer.size()}/{config.SENSOR_TEMPORAL_LENGTH}, age={sensor_age:.1f}s)")
                        print(f"  Robot: ready={robot_state_buffer.is_ready()}, fresh={robot_fresh} ({robot_state_buffer.size()}/{config.ROBOT_STATE_BUFFER_LENGTH}, age={robot_age:.1f}s)")
                        last_action_time = now

            # Status print
            if now - last_status_print >= config.STATUS_PERIOD:
                stats = inference_engine.get_stats()

                print(f"\n--- Status ({datetime.now().strftime('%H:%M:%S')}) ---")
                print(f"VL Updates: {stats['vl_update_count']} | VL avg: {stats['vl_avg_time_ms']:.0f}ms")
                print(f"Actions: {stats['action_count']} | Action avg: {stats['action_avg_time_ms']:.1f}ms")
                print(f"Actions Sent: {action_send_success} | Failed: {action_send_failed} | Success Rate: {100*action_send_success/(action_send_success+action_send_failed) if (action_send_success+action_send_failed)>0 else 0:.1f}%")

                # Data freshness check
                camera_stale = (now - last_camera_recv_time) > 1.0 if last_camera_recv_time > 0 else True
                sensor_stale = (now - sensor_thread.last_recv_time) > 1.0 if sensor_thread.last_recv_time > 0 else True
                robot_stale = (now - last_robot_recv_time) > 1.0 if last_robot_recv_time > 0 else True

                camera_age = f"{now - last_camera_recv_time:.1f}s ago" if last_camera_recv_time > 0 else "never"
                sensor_age = f"{now - sensor_thread.last_recv_time:.1f}s ago" if sensor_thread.last_recv_time > 0 else "never"
                robot_age = f"{now - last_robot_recv_time:.1f}s ago" if last_robot_recv_time > 0 else "never"

                print(f"Images recv: {', '.join([f'{v}:{cam_recv_count[v]}' for v in sorted(cam_recv_count.keys())])} {'⚠️ STALE' if camera_stale else '✅'} (last: {camera_age})")
                print(f"Sensor buffer: {sensor_buffer.size()}/{config.SENSOR_TEMPORAL_LENGTH} {'⚠️ STALE' if sensor_stale else '✅'} (last: {sensor_age})")
                print(f"Robot buffer: {robot_state_buffer.size()}/{config.ROBOT_STATE_BUFFER_LENGTH} {'⚠️ STALE' if robot_stale else '✅'} (last: {robot_age})")

                if robot_state:
                    print(f"Robot: J1={robot_state['joints'][0]:.2f}°, Px={robot_state['pose'][0]:.2f}mm")

                if args.save_data and image_writer:
                    print(f"Writer queue: {image_writer.q.qsize()} | Written: {image_writer.written_count}")

                last_status_print = now

    finally:
        print(f"\n{'='*80}")
        print("Cleanup and Data Saving")
        print(f"{'='*80}\n")
        stop_event.set()

        # Wait for threads
        print("⏳ Waiting for sensor thread...")
        sensor_thread.join(timeout=2.0)

        if vl_update_thread and vl_update_thread.is_alive():
            print("⏳ Waiting for VL update thread...")
            vl_update_thread.join(timeout=5.0)

        # Save all data
        if args.save_data:
            # Save robot data
            if robot_save_buffer:
                robot_csv = output_dir / f"robot_state_{session_time}.csv"
                save_robot_data_to_csv(robot_save_buffer, str(robot_csv))

            # Save sensor data
            if sensor_save_buffer:
                sensor_npz = output_dir / f"sensor_data_{session_time}.npz"
                save_sensor_data_to_npz(sensor_save_buffer, str(sensor_npz))

            # Stop image writer
            if image_writer:
                image_writer.stop()
                image_writer.join()

        # Save inference results
        if inference_results:
            inference_json = output_dir / f"inference_results_{session_time}.json"
            save_inference_results(inference_results, str(inference_json))

        # Save performance results (final summary only - data already auto-saved)
        print(f"\n{'='*80}")
        print("Saving Performance Results")
        print(f"{'='*80}")
        performance_monitor.save_results()

        # Print final stats
        stats = inference_engine.get_stats()
        print(f"\n{'='*80}")
        print("Final Statistics")
        print(f"{'='*80}")
        print(f"VL Updates: {stats['vl_update_count']}")
        if stats['vl_avg_time_ms'] > 0:
            print(f"VL Avg Time: {stats['vl_avg_time_ms']:.1f}ms (~{1000/stats['vl_avg_time_ms']:.1f}Hz)")
        print(f"Action Predictions: {stats['action_count']}")
        if stats['action_avg_time_ms'] > 0:
            print(f"Action Avg Time: {stats['action_avg_time_ms']:.1f}ms (~{1000/stats['action_avg_time_ms']:.1f}Hz)")
        else:
            print(f"Action Avg Time: N/A (no actions executed)")
        print(f"\nAction Commands Sent to Robot:")
        print(f"  Successful: {action_send_success}")
        print(f"  Failed: {action_send_failed}")
        if (action_send_success + action_send_failed) > 0:
            print(f"  Success Rate: {100*action_send_success/(action_send_success+action_send_failed):.1f}%")

        if args.save_data:
            print(f"\nData saved:")
            print(f"  Images: {image_writer.written_count if image_writer else 0}")
            print(f"  Robot states: {len(robot_save_buffer) if robot_save_buffer else 0}")
            print(f"  Sensor records: {len(sensor_save_buffer) if sensor_save_buffer else 0}")

        print(f"  Inference results: {len(inference_results)}")
        print(f"\n📁 Output directory: {output_dir}")
        print(f"{'='*80}\n")

        # Cleanup sockets
        cam_sock.close()
        robot_sock.close()
        cmd_sock.close()
        ctx.term()

        print("✅ Shutdown complete")


if __name__ == "__main__":
    main()
