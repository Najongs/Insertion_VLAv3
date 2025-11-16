#!/usr/bin/env python3
"""
Data Collector for VLA

Collects data from cameras, robot, and sensors and saves it into episode-based directories.
This script does NOT perform any model inference.

Data Sources:
- Cameras: 5 views (ZED left x4 + OAK x1) via ZMQ PULL (port 5555)
- Robot State: Joint angles + EE pose via ZMQ SUB (port 5556)
- Sensor Data: Force + OCT A-scan via UDP (port 9999)

Usage:
    python Data_collection/data_collector.py
"""

import os
import time
import json
import cv2
import zmq
import numpy as np
import threading
import argparse
import signal
import csv
from queue import Queue, Empty
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import struct

# ==============================
# Configuration
# ==============================
class Config:
    # Image settings
    IMAGE_RESIZE_HEIGHT = 480
    IMAGE_RESIZE_WIDTH = 640

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
        "41182735": "View1",
        "49429257": "View2",
        "44377151": "View3",
        "49045152": "View4"
    }
    OAK_KEYWORD = "OAK"

    # Logging settings
    STATUS_PERIOD = 2.0

# ==============================
# Async Image Writer
# ==============================
class AsyncImageWriter(threading.Thread):
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
                pass

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
    def __init__(self, required_views=None, save_dir=None, writer=None, resize_height=480, resize_width=640):
        self.required_views = required_views or ['View1', 'View2', 'View3', 'View4', 'View5']
        self.lock = threading.Lock()
        self.save_dir = save_dir
        self.writer = writer
        self.resize_height = resize_height
        self.resize_width = resize_width

    def update(self, view_name: str, img: np.ndarray, timestamp: float, cam_name: str = ""):
        img_resized = cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)
        with self.lock:
            filename = f"{cam_name}_{timestamp:.3f}.jpg" if cam_name else f"{view_name}_{timestamp:.3f}.jpg"
            save_path = os.path.join(self.save_dir, view_name, filename)
            self.writer.submit(save_path, img) # Save original image

# ==============================
# Sensor Data Buffer
# ==============================
class SensorBuffer:
    def __init__(self, save_buffer=None):
        self.lock = threading.Lock()
        self.save_buffer = save_buffer if save_buffer is not None else []

    def add_samples(self, samples: list):
        with self.lock:
            for sample in samples:
                self.save_buffer.append({
                    'timestamp': sample['timestamp'],
                    'send_timestamp': sample['send_timestamp'],
                    'force': sample['force'],
                    'aline': sample['aline']
                })


# ==============================
# UDP Sensor Receiver (수정됨)
# ==============================
class SensorUDPReceiver(threading.Thread):
    def __init__(self, config: Config, sensor_buffer: SensorBuffer, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.sensor_buffer = sensor_buffer
        self.stop_event = stop_event
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

        while not self.stop_event.is_set():
            try:
                data, addr = sock.recvfrom(self.config.SENSOR_BUFFER_SIZE)
            except socket.timeout:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[UDP Sensor] Receive error: {e}")
                continue

            # ✅ 수정된 로직: 
            # 이제 배치가 아닌 단일 패킷(DataPacket)을 수신합니다.
            try:
                # 1. 수신한 데이터가 정확히 DataPacket 1개의 크기인지 확인합니다.
                if len(data) != self.config.SENSOR_TOTAL_PACKET_SIZE:
                    # C++에서 보낸 4120 바이트와 크기가 다르면 버립니다.
                    print(f"[UDP Sensor] Warning: Received packet of wrong size. Got {len(data)}, expected {self.config.SENSOR_TOTAL_PACKET_SIZE}")
                    continue

                # 2. 'num_packets' 헤더가 없으므로 바로 DataPacket 파싱을 시작합니다.
                records = []
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

                # 5. 단일 레코드를 생성합니다.
                record = {'timestamp': ts, 'send_timestamp': send_ts, 'force': force, 'aline': aline}
                
                # 6. add_samples가 리스트를 받으므로 리스트에 담아 전달합니다.
                self.sensor_buffer.add_samples([record])
                self.packet_count += 1
                
            except Exception as e:
                print(f"[ERROR] Sensor UDP unpack failed (Data len: {len(data)}): {e}")
                continue
                
        sock.close()
        print("🛑 Sensor UDP Receiver stopped")

# ==============================
# Data Saving Functions
# ==============================
def save_robot_data_to_csv(data_list, filepath):
    if not data_list: return
    print(f"💾 Saving {len(data_list)} robot states to {filepath}")
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["recv_timestamp", "origin_timestamp", "send_timestamp", "force_placeholder", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"])
            w.writerows(data_list)
        print(f"💾✅ Robot data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save robot data: {e}")

def save_sensor_data_to_npz(data_list, filepath):
    if not data_list: return
    print(f"💾 Saving {len(data_list)} sensor records to {filepath}")
    try:
        timestamps = np.array([d['timestamp'] for d in data_list], dtype=np.float64)
        send_timestamps = np.array([d['send_timestamp'] for d in data_list], dtype=np.float64)
        forces = np.array([d['force'] for d in data_list], dtype=np.float32)
        alines = np.array([d['aline'] for d in data_list], dtype=np.float32)
        np.savez(filepath, timestamps=timestamps, send_timestamps=send_timestamps, forces=forces, alines=alines)
        print(f"💾✅ Sensor data saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save sensor data: {e}")

def save_episode_data(output_dir, session_time, robot_save_buffer, sensor_save_buffer, image_writer):
    if not output_dir:
        print("Output directory not set, skipping save.")
        return

    if not any([robot_save_buffer, sensor_save_buffer]):
        print("No data collected in this episode, skipping save.")
        if image_writer:
            image_writer.stop()
            image_writer.join()
        return

    print(f"\n{'='*80}")
    print(f"Saving data for episode {session_time}")
    print(f"{'='*80}\n")

    num_robot_states = len(robot_save_buffer)
    num_sensor_records = len(sensor_save_buffer)
    
    if robot_save_buffer:
        robot_csv = output_dir / f"robot_state_{session_time}.csv"
        save_robot_data_to_csv(robot_save_buffer, str(robot_csv))

    if sensor_save_buffer:
        sensor_npz = output_dir / f"sensor_data_{session_time}.npz"
        save_sensor_data_to_npz(sensor_save_buffer, str(sensor_npz))

    num_images_written = 0
    if image_writer:
        image_writer.stop()
        image_writer.join()
        num_images_written = image_writer.written_count

    print(f"\nData saved: Images: {num_images_written}, Robot states: {num_robot_states}, Sensor records: {num_sensor_records}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"{'='*80}\n")

# =======================================================================================
# Main Loop for Multi-Episode Data Collection
# =======================================================================================
def main():
    parser = argparse.ArgumentParser(description='VLA Data Collector')
    args = parser.parse_args()

    config = Config()

    stop_event = threading.Event()
    def sigint_handler(sig, frame):
        print("\n🛑 Ctrl+C detected — Shutting down...")
        stop_event.set()
    signal.signal(signal.SIGINT, sigint_handler)

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

    # --- Episode State ---
    collecting = False
    session_time, output_dir = None, None
    image_writer, sensor_save_buffer, robot_save_buffer = None, None, None
    cam_recv_count, robot_state_count = None, 0
    last_status_print = time.time()
    
    sensor_buffer = SensorBuffer()
    sensor_thread = SensorUDPReceiver(config, sensor_buffer, stop_event)
    sensor_thread.start()
    image_buffer = None

    print(f"\n{'='*80}")
    print(f"Data Collector Started")
    print(f"Ready to receive multiple episodes. Press Ctrl+C to stop.")
    print(f"{'='*80}")
    print(f"\n--- Waiting for new episode... ---")

    try:
        while not stop_event.is_set():
            try:
                socks = dict(poller.poll(timeout=100))
            except KeyboardInterrupt:
                break

            if cam_sock in socks and socks[cam_sock] == zmq.POLLIN:
                while True:
                    try:
                        parts = cam_sock.recv_multipart(zmq.DONTWAIT)
                        if len(parts) < 2: continue
                        meta = json.loads(parts[0].decode("utf-8"))
                        img = cv2.imdecode(np.frombuffer(parts[1], np.uint8), cv2.IMREAD_COLOR)
                        if img is None: continue

                        if collecting:
                            view_name, cam_name = None, meta.get("camera", "unknown")
                            if "left" in cam_name.lower():
                                for serial, view in config.ZED_SERIAL_TO_VIEW.items():
                                    if serial in cam_name: view_name = view; break
                            if config.OAK_KEYWORD.lower() in cam_name.lower(): view_name = "View5"

                            if view_name:
                                image_buffer.update(view_name, img, float(meta.get("timestamp", 0.0)), cam_name)
                                cam_recv_count[view_name] += 1
                    except zmq.Again: break
                    except Exception as e: print(f"[ERROR] Camera processing: {e}"); break

            if robot_sock in socks and socks[robot_sock] == zmq.POLLIN:
                while True:
                    try:
                        topic, payload = robot_sock.recv_multipart(zmq.DONTWAIT)
                        if topic == config.ZMQ_ROBOT_TOPIC:
                            if not collecting:
                                collecting = True
                                print("\n" + "="*80 + "\n🏁 NEW EPISODE STARTED 🏁\n" + "="*80)
                                session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_dir = Path(f"./data_collection_{session_time}")
                                output_dir.mkdir(exist_ok=True)

                                robot_save_buffer, sensor_save_buffer = [], []
                                cam_recv_count = defaultdict(int)
                                robot_state_count = 0
                                last_status_print = time.time()
                                
                                print(f"📁 Data saving to: {output_dir}")
                                for view in ['View1', 'View2', 'View3', 'View4', 'View5']:
                                    (output_dir / view).mkdir(exist_ok=True)
                                image_writer = AsyncImageWriter(max_queue=5000); image_writer.start()
                                
                                image_buffer = MultiViewImageBuffer(
                                    save_dir=str(output_dir),
                                    writer=image_writer,
                                    resize_height=config.IMAGE_RESIZE_HEIGHT,
                                    resize_width=config.IMAGE_RESIZE_WIDTH
                                )
                                sensor_buffer.save_buffer = sensor_save_buffer

                            unpacked = struct.unpack(config.ROBOT_PAYLOAD_FORMAT, payload)
                            robot_save_buffer.append([time.time()] + list(unpacked))
                            robot_state_count += 1

                        elif topic == config.ZMQ_END_TOPIC:
                            if collecting:
                                print("\n" + "="*80 + "\n🏁 EPISODE FINISHED 🏁")
                                save_episode_data(output_dir, session_time, robot_save_buffer, sensor_buffer.save_buffer, image_writer)
                                collecting = False
                                image_buffer = None
                                print("\n--- Waiting for new episode... ---")

                    except zmq.Again: break
                    except Exception as e: print(f"[ERROR] Robot processing: {e}"); break
            
            if not collecting:
                time.sleep(0.1)
                continue

            now = time.time()
            if now - last_status_print >= config.STATUS_PERIOD:
                print(f"\n--- Status ({datetime.now().strftime('%H:%M:%S')}) ---")
                print(f"Collecting: {collecting}")
                print(f"Robot States Received: {robot_state_count}")
                print(f"Sensor Records Received: {len(sensor_buffer.save_buffer)}")
                if image_writer:
                    print(f"Images Queued: {image_writer.q.qsize()} | Written: {image_writer.written_count}")
                    print(f"Images Received: {', '.join([f'{v}:{c}' for v,c in cam_recv_count.items()])}")
                last_status_print = now

    finally:
        print(f"\n{'='*80}\nShutting down...")
        if collecting:
            print("An episode was running. Saving final data...")
            save_episode_data(output_dir, session_time, robot_save_buffer, sensor_buffer.save_buffer, image_writer)
        
        stop_event.set()
        if sensor_thread.is_alive(): sensor_thread.join(timeout=2.0)
        
        cam_sock.close(); robot_sock.close(); ctx.term()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()
