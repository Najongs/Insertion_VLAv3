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
    python Async_inference_receiver.py --checkpoint checkpoints/qwen_vla_sensor_best.pt

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

# Import VLA model
import sys

# ==============================
# Configuration
# ==============================
class Config:
    # Model settings (OPTIMIZED)
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    ACTION_DIM = 7  # 6 joints + 1 gripper
    HORIZON = 8  # Action prediction horizon
    HIDDEN_DIM = 1024
    CACHE_DIR = "/home/najo/NAS/VLA/dataset/cache/qwen_vl_features"

    # 🔥 OPTIMIZED: Image resize for faster VLM inference
    IMAGE_RESIZE_HEIGHT = 360
    IMAGE_RESIZE_WIDTH = 640

    # 🔥 OPTIMIZED: Sensor settings (100ms window)
    SENSOR_ENABLED = True
    SENSOR_TEMPORAL_LENGTH = 65  # 100ms at 650Hz (was 650)
    SENSOR_INPUT_CHANNELS = 1026  # 1 force + 1025 A-scan
    FUSION_STRATEGY = 'concat'

    # Network settings
    ZMQ_CAM_PULL_PORT = 5555
    ZMQ_ROBOT_PUB_ADDRESS = "10.130.41.111"
    ZMQ_ROBOT_PUB_PORT = 5556
    ZMQ_ROBOT_TOPIC = b"robot_state"
    ZMQ_END_TOPIC = b"episode_end"
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
                self.buffer.append(combined)

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

            data = np.array(list(self.buffer), dtype=np.float32)  # (current_len, C)

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
    def __init__(self, config: Config, checkpoint_path: str = None):
        self.config = config
        self.device = torch.device(config.DEVICE)

        print(f"\n{'='*80}")
        print(f"Initializing Async VLA Inference Engine")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {config.MODEL_NAME}")
        print(f"Image Resize: {config.IMAGE_RESIZE_WIDTH}x{config.IMAGE_RESIZE_HEIGHT}")
        print(f"Sensor Window: {config.SENSOR_TEMPORAL_LENGTH} samples (100ms @ 650Hz)")
        print(f"Action Expert: {config.ACTION_EXPERT_HZ} Hz")
        print(f"VL Reuse: {config.VLM_REUSE_COUNT}x")
        print(f"Fusion Strategy: {config.FUSION_STRATEGY}")

        # Load model
        self.model = Not_freeze_QwenVLAWithSensor(
            vl_model_name=config.MODEL_NAME,
            action_dim=config.ACTION_DIM,
            horizon=config.HORIZON,
            hidden_dim=config.HIDDEN_DIM,
            cache_dir=config.CACHE_DIR,
            finetune_vl="none",
            sensor_enabled=config.SENSOR_ENABLED,
            sensor_input_channels=config.SENSOR_INPUT_CHANNELS,
            sensor_temporal_length=config.SENSOR_TEMPORAL_LENGTH,
            fusion_strategy=config.FUSION_STRATEGY,
            image_resize_height=config.IMAGE_RESIZE_HEIGHT,
            image_resize_width=config.IMAGE_RESIZE_WIDTH,
            device_map="cuda",  # Use single GPU for inference instead of "auto"
        )

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            print("✅ Checkpoint loaded successfully")

        self.model = self.model.to(self.device)
        self.model.eval()

        # VL feature cache
        self.vl_features = None
        self.vl_features_lock = threading.Lock()
        self.vl_update_count = 0
        self.vl_update_times = deque(maxlen=10)

        # Action prediction stats
        self.action_count = 0
        self.action_times = deque(maxlen=20)
        self.stats_lock = threading.Lock()

        print(f"{'='*80}\n")

    @torch.no_grad()
    def update_vl_features(self, images_dict: dict, text_prompt: str = "Perform needle insertion into the eye"):
        """
        Update VL features (runs in background thread)
        This is the slow operation (~381ms with 5 views @ 640x360)
        """
        start_time = time.time()

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
        text_inputs = [text_prompt]
        image_inputs = [image_paths]

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            # Use model's VL extraction
            vl_tokens = self.model._encode_vision_features(
                text_inputs,
                image_inputs,
                cache_keys=[f"realtime_{time.time()}"],
                use_cache=False,
                device=self.device
            )

        # Update cache
        with self.vl_features_lock:
            self.vl_features = vl_tokens
            self.vl_update_count += 1

        # Clean up temp files
        for path in image_paths:
            Path(path).unlink(missing_ok=True)

        elapsed = time.time() - start_time
        self.vl_update_times.append(elapsed)

        return elapsed

    @torch.no_grad()
    def predict_action(self, sensor_data: torch.Tensor, z_chunk: torch.Tensor = None) -> dict:
        """
        Predict actions using cached VL features (fast operation ~20-30ms)
        This runs at 10Hz
        """
        start_time = time.time()

        # Get cached VL features
        with self.vl_features_lock:
            if self.vl_features is None:
                return None
            vl_features = self.vl_features

        # Prepare inputs
        if z_chunk is None:
            z_chunk = torch.zeros(1, self.config.HORIZON, self.config.ACTION_DIM,
                                dtype=torch.float32, device=self.device)

        sensor_batch = sensor_data.unsqueeze(0).to(self.device)  # (1, 65, 1026)

        # Encode sensor features
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            sensor_features = None
            if self.config.SENSOR_ENABLED and sensor_batch is not None:
                sensor_features = self.model.sensor_encoder(sensor_batch)  # (1, sensor_dim)

            # Predict actions using Action Expert (it handles fusion internally)
            pred_actions, delta = self.model.action_expert(vl_features, z_chunk, sensor_features)

        elapsed = time.time() - start_time

        # Update stats
        with self.stats_lock:
            self.action_count += 1
            self.action_times.append(elapsed)

        return {
            'actions': pred_actions[0].float().cpu().numpy(),  # (H, action_dim)
            'delta': delta[0].float().cpu().numpy(),
            'inference_time': elapsed,
            'timestamp': time.time(),
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
                'vl_features_cached': self.vl_features is not None
            }

    def reset_stats(self):
        """Resets inference statistics for a new episode."""
        with self.stats_lock:
            self.vl_update_count = 0
            self.vl_update_times.clear()
            self.action_count = 0
            self.action_times.clear()
        print("📈 Inference engine stats reset.")


# ==============================
# UDP Sensor Receiver
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

        print(f"⏳ Calibrating sensor clock offset (first {self.config.SENSOR_CALIBRATION_COUNT} batches)...")

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

            if len(data) < self.config.SENSOR_TOTAL_PACKET_SIZE:
                continue

            try:
                # Parse batch header
                num_packets = struct.unpack('<I', data[:4])[0]
                expected_size = 4 + (num_packets * self.config.SENSOR_TOTAL_PACKET_SIZE)

                if len(data) != expected_size or num_packets == 0:
                    continue

                # Parse packets
                records = []
                mv = memoryview(data)[4:]
                offset = 0
                last_send_ts = 0.0

                for _ in range(num_packets):
                    header = mv[offset:offset + self.config.SENSOR_PACKET_HEADER_SIZE]
                    ts, send_ts, force = struct.unpack(self.config.SENSOR_PACKET_HEADER_FORMAT, header)
                    offset += self.config.SENSOR_PACKET_HEADER_SIZE

                    aline_bytes = mv[offset:offset + self.config.SENSOR_ALINE_SIZE]
                    aline = np.frombuffer(aline_bytes, dtype=np.float32).copy()
                    offset += self.config.SENSOR_ALINE_SIZE

                    records.append({
                        'timestamp': ts,
                        'send_timestamp': send_ts,
                        'force': force,
                        'aline': aline
                    })
                    last_send_ts = send_ts

                # Clock calibration
                if self.clock_offset is None:
                    net_plus_offset = recv_time - last_send_ts
                    self.calibration_samples.append(net_plus_offset)

                    if len(self.calibration_samples) >= self.config.SENSOR_CALIBRATION_COUNT:
                        self.clock_offset = np.mean(self.calibration_samples)
                        print(f"\n✅ Sensor Clock Offset Calibrated: {self.clock_offset * 1000:.1f} ms\n")

                # Add to circular buffer
                self.sensor_buffer.add_samples(records)
                self.packet_count += num_packets

            except Exception as e:
                print(f"[ERROR] Sensor UDP unpack failed: {e}")
                continue

        sock.close()
        print("🛑 Sensor UDP Receiver stopped")


# ==============================
# Data Saving Functions
# ==============================
def save_episode_data(args, output_dir, session_time, robot_save_buffer, sensor_save_buffer, image_writer, inference_results, inference_engine):
    """Saves all data for a completed episode."""
    if not output_dir:
        print("Output directory not set, skipping save.")
        return

    if not any([robot_save_buffer, sensor_save_buffer, inference_results]):
        print("No data collected in this episode, skipping save.")
        if image_writer:
            image_writer.stop()
            image_writer.join()
        return

    print(f"\n{'='*80}")
    print(f"Saving data for episode {session_time}")
    print(f"{'='*80}\n")

    num_robot_states = len(robot_save_buffer) if robot_save_buffer else 0
    num_sensor_records = len(sensor_save_buffer) if sensor_save_buffer else 0
    num_inference_results = len(inference_results) if inference_results else 0
    num_images_written = 0

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
            num_images_written = image_writer.written_count

    # Save inference results
    if inference_results:
        inference_json = output_dir / f"inference_results_{session_time}.json"
        save_inference_results(inference_results, str(inference_json))

    # Print final stats
    stats = inference_engine.get_stats()
    print(f"\n--- Episode {session_time} Statistics ---")
    print(f"VL Updates: {stats['vl_update_count']}")
    if stats['vl_avg_time_ms'] > 0:
        print(f"VL Avg Time: {stats['vl_avg_time_ms']:.1f}ms (~{1000/stats['vl_avg_time_ms']:.1f}Hz)")
    print(f"Action Predictions: {stats['action_count']}")
    if stats['action_avg_time_ms'] > 0:
        print(f"Action Avg Time: {stats['action_avg_time_ms']:.1f}ms (~{1000/stats['action_avg_time_ms']:.1f}Hz)")

    if args.save_data:
        print(f"\nData saved: Images: {num_images_written}, Robot states: {num_robot_states}, Sensor records: {num_sensor_records}")
    print(f"Inference results: {num_inference_results}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"{'='*80}\n")


# =======================================================================================
# Main Loop for Multi-Episode Collection
# =======================================================================================
def main():
    parser = argparse.ArgumentParser(description='Async Real-time VLA Inference Receiver')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--save-data', action='store_true', help='Save images, sensor data, robot state')
    parser.add_argument('--vl-reuse', type=int, default=4, help='VL feature reuse count (default: 4)')
    args = parser.parse_args()

    config = Config()
    config.VLM_REUSE_COUNT = args.vl_reuse

    # --- One-time setup ---
    stop_event = threading.Event()
    def sigint_handler(sig, frame):
        print("\n🛑 Ctrl+C detected — Shutting down...")
        stop_event.set()
    signal.signal(signal.SIGINT, sigint_handler)

    inference_engine = AsyncVLAInferenceEngine(config, args.checkpoint)

    ctx = zmq.Context.instance()
    cam_sock = ctx.socket(zmq.PULL)
    cam_sock.setsockopt(zmq.RCVHWM, 5000)
    cam_sock.setsockopt(zmq.RCVBUF, 8 * 1024 * 1024)
    cam_sock.bind(f"tcp://0.0.0.0:{config.ZMQ_CAM_PULL_PORT}")
    print(f"✅ Camera PULL listening on port {config.ZMQ_CAM_PULL_PORT}")

    robot_sock = ctx.socket(zmq.SUB)
    robot_sock.setsockopt(zmq.RCVHWM, 100)
    robot_sock.connect(f"tcp://{config.ZMQ_ROBOT_PUB_ADDRESS}:{config.ZMQ_ROBOT_PUB_PORT}")
    robot_sock.subscribe(config.ZMQ_ROBOT_TOPIC)
    robot_sock.subscribe(config.ZMQ_END_TOPIC)
    print(f"✅ Robot SUB connected to {config.ZMQ_ROBOT_PUB_ADDRESS}:{config.ZMQ_ROBOT_PUB_PORT}")

    poller = zmq.Poller()
    poller.register(cam_sock, zmq.POLLIN)
    poller.register(robot_sock, zmq.POLLIN)

    image_buffer = MultiViewImageBuffer(
        resize_height=config.IMAGE_RESIZE_HEIGHT,
        resize_width=config.IMAGE_RESIZE_WIDTH
    )
    sensor_buffer = SensorCircularBuffer(
        max_length=config.SENSOR_TEMPORAL_LENGTH,
        channels=config.SENSOR_INPUT_CHANNELS
    )
    sensor_thread = SensorUDPReceiver(config, sensor_buffer, stop_event)
    sensor_thread.start()

    # --- Episode State ---
    collecting = False
    session_time, output_dir = None, None
    image_writer, sensor_save_buffer, robot_save_buffer, inference_results = None, None, None, None
    cam_recv_count, robot_state, vl_update_thread = None, None, None
    vl_update_counter, current_vl_update_number = 0, 0
    last_action_time, last_status_print = time.time(), time.time()

    print(f"\n{'='*80}")
    print(f"Async Real-time Inference Started")
    print(f"Ready to receive multiple episodes. Press Ctrl+C to stop.")
    print(f"{'='*80}")
    print(f"\n--- Waiting for new episode... ---")

    try:
        while not stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=100))
            except KeyboardInterrupt:
                break

            # Always process camera messages to keep buffer fresh
            if cam_sock in socks and socks[cam_sock] == zmq.POLLIN:
                while True:
                    try:
                        parts = cam_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) < 2: continue
                        meta = json.loads(parts[0].decode("utf-8"))
                        img = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
                        if img is None: continue

                        view_name, cam_name = None, meta.get("camera", "unknown")
                        if "left" in cam_name.lower():
                            for serial, view in config.ZED_SERIAL_TO_VIEW.items():
                                if serial in cam_name: view_name = view; break
                        if config.OAK_KEYWORD.lower() in cam_name.lower(): view_name = "View5"

                        if view_name:
                            image_buffer.update(view_name, img, float(meta.get("timestamp", 0.0)), cam_name)
                            if collecting: cam_recv_count[view_name] += 1
                    except zmq.Again: break
                    except Exception as e: print(f"[ERROR] Camera processing: {e}"); break

            # Process robot messages (state and episode control)
            if robot_sock in socks and socks[robot_sock] == zmq.POLLIN:
                while True:
                    try:
                        parts = robot_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) != 2: continue
                        topic, payload = parts[0], parts[1]

                        if topic == config.ZMQ_ROBOT_TOPIC:
                            if not collecting:
                                # === START OF A NEW EPISODE ===
                                collecting = True
                                print("\n" + "="*80 + "\n🏁 NEW EPISODE STARTED 🏁\n" + "="*80)
                                session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_dir = Path(f"./async_inference_{session_time}")
                                output_dir.mkdir(exist_ok=True)

                                robot_save_buffer, sensor_save_buffer, inference_results = [], [], []
                                cam_recv_count = defaultdict(int)
                                vl_update_counter, current_vl_update_number = 0, 0
                                last_action_time, last_status_print = time.time(), time.time()
                                inference_engine.reset_stats()

                                if args.save_data:
                                    print(f"📁 Data saving enabled: {output_dir}")
                                    for view in ['View1', 'View2', 'View3', 'View4', 'View5']:
                                        (output_dir / view).mkdir(exist_ok=True)
                                    image_writer = AsyncImageWriter(max_queue=5000); image_writer.start()
                                else:
                                    print(f"📁 Inference only mode (no data saving)"); image_writer = None
                                
                                image_buffer.writer = image_writer
                                image_buffer.save_dir = str(output_dir) if args.save_data else None
                                image_buffer.save_enabled = args.save_data and image_writer is not None
                                sensor_buffer.save_buffer = sensor_save_buffer

                            # Process robot state
                            unpacked = struct.unpack(config.ROBOT_PAYLOAD_FORMAT, payload)
                            robot_state = {'timestamp': unpacked[0], 'joints': np.array(unpacked[3:9]), 'pose': np.array(unpacked[9:15]), 'recv_time': time.time()}
                            if args.save_data: robot_save_buffer.append([time.time()] + list(unpacked))

                        elif topic == config.ZMQ_END_TOPIC:
                            if collecting:
                                # === END OF EPISODE ===
                                print("\n" + "="*80 + "\n🏁 EPISODE FINISHED 🏁")
                                save_episode_data(args, output_dir, session_time, robot_save_buffer, sensor_save_buffer, image_writer, inference_results, inference_engine)
                                collecting = False
                                print("\n--- Waiting for new episode... ---")

                    except zmq.Again: break
                    except Exception as e: print(f"[ERROR] Robot processing: {e}"); break
            
            if not collecting:
                time.sleep(0.1) # Sleep briefly when not collecting
                continue

            now = time.time()
            # === ASYNC VLM UPDATE (Background Thread) ===
            if image_buffer.is_ready():
                should_update_vl = (vl_update_counter % config.VLM_REUSE_COUNT == 0) or (inference_engine.vl_features is None)
                if should_update_vl and (vl_update_thread is None or not vl_update_thread.is_alive()):
                    images_dict = image_buffer.get_multi_view_set()
                    def vl_update_worker():
                        nonlocal current_vl_update_number
                        elapsed = inference_engine.update_vl_features(images_dict)
                        current_vl_update_number += 1
                        print(f"🔄 [VL Update #{current_vl_update_number}] Completed in {elapsed*1000:.0f}ms")
                    vl_update_thread = threading.Thread(target=vl_update_worker, daemon=True); vl_update_thread.start()
                    vl_update_counter = 0

            # === ACTION EXPERT PREDICTION (10Hz) ===
            if now - last_action_time >= (1.0 / config.ACTION_EXPERT_HZ):
                if inference_engine.vl_features is not None and sensor_buffer.is_ready():
                    result = inference_engine.predict_action(sensor_buffer.get_tensor())
                    if result:
                        vl_update_counter += 1
                        print(f"[ACTION #{len(inference_results)+1}] VL_reuse={vl_update_counter}/{config.VLM_REUSE_COUNT} | Time: {result['inference_time']*1000:.1f}ms")
                        result.update({'vl_update_number': current_vl_update_number, 'robot_state': {'joints': robot_state['joints'].tolist(), 'pose': robot_state['pose'].tolist(), 'timestamp': robot_state['timestamp']} if robot_state else None})
                        inference_results.append(result)
                    last_action_time = now
                elif now - last_action_time >= 2.0:
                    print(f"[WAIT] VL Features: {inference_engine.vl_features is not None} | Images: {image_buffer.is_ready()} | Sensor: {sensor_buffer.is_ready()}")
                    last_action_time = now

            # Status print
            if now - last_status_print >= config.STATUS_PERIOD:
                stats = inference_engine.get_stats()
                print(f"\n--- Status ({datetime.now().strftime('%H:%M:%S')}) ---")
                print(f"VL Updates: {stats['vl_update_count']} | Action Predictions: {stats['action_count']}")
                if robot_state: print(f"Robot: J1={robot_state['joints'][0]:.2f}°, Px={robot_state['pose'][0]:.2f}mm")
                if args.save_data and image_writer: print(f"Writer queue: {image_writer.q.qsize()} | Written: {image_writer.written_count}")
                last_status_print = now

    finally:
        print(f"\n{'='*80}\nShutting down...")
        if collecting:
            print("An episode was running. Saving final data...")
            save_episode_data(args, output_dir, session_time, robot_save_buffer, sensor_save_buffer, image_writer, inference_results, inference_engine)
        
        stop_event.set()
        if sensor_thread.is_alive(): sensor_thread.join(timeout=2.0)
        if vl_update_thread and vl_update_thread.is_alive(): vl_update_thread.join(timeout=5.0)
        
        cam_sock.close(); robot_sock.close(); ctx.term()
        print("✅ Shutdown complete")


if __name__ == "__main__":
    main()
