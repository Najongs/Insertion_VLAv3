#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Camera Sender for Async VLA Inference

Optimizations:
- Send rate: 5 Hz (ensures fresh images for VLM @ ~2.6Hz)
- Image resize: 1280x720 (will be resized to 640x360 on receiver side)
- Lower bandwidth usage
- Synchronized capture across all cameras

Cameras:
- 4x ZED cameras (left view only)
- 1x OAK camera
- Total: 5 views

Data Format (ZMQ PUSH):
- Metadata: JSON {camera, timestamp, ...}
- Image: JPEG compressed

Usage:
    # Standard mode (5Hz)
    python Optimized_Camera_sender.py --server-ip 10.130.4.79

    # Custom rate
    python Optimized_Camera_sender.py --server-ip 10.130.4.79 --fps 3
"""

import os, time, json, cv2, zmq, numpy as np, threading
from queue import Queue, Empty
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
from collections import deque
import depthai as dai
import pyzed.sl as sl
import signal
import argparse

# ===================== 기본 설정 =====================
DEFAULT_SERVER_IP = "10.130.4.79"
DEFAULT_SERVER_PORT = 5555

# 🔥 OPTIMIZED: 5Hz 설정 (VLM @ 2.6Hz보다 약간 빠르게)
DEFAULT_CAPTURE_FPS = 4  # 5 frames per second
CAPTURE_INTERVAL = 1.0 / DEFAULT_CAPTURE_FPS  # 0.2초 간격
PULSE_WIDTH = 0.01  # 10ms 펄스

# JPEG 설정
JPEG_QUALITY = 75  # 고품질 유지
JPEG_OPTIMIZE = False
JPEG_PROGRESSIVE = False

# 🔥 OPTIMIZED: Left 카메라만 전송 (Right 비활성화)
SEND_ZED_RIGHT = False

# ZMQ 최적화
ZMQ_IO_THREADS = 4
CAMERA_SNDHWM = 5 # 5Hz이므로 버퍼 줄임
SNDBUF_SIZE = 32 * 1024 * 1024  # 32MB

# 인코딩 병렬화
NUM_ENCODER_PROCESSES = 4  # 5Hz이므로 프로세스 줄임
ENCODING_QUEUE_SIZE = 500
BATCH_SEND_SIZE = 2
BATCH_TIMEOUT = 0.02

# 프레임 전처리
RESIZE_BEFORE_ENCODE = True
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

# ===================== 전역 플래그 =====================
stop_flag = threading.Event()
encoder_stop_flag = MPEvent()


def handle_sigint(sig, frame):
    print("\n🛑 Ctrl+C detected — Shutting down...")
    stop_flag.set()
    encoder_stop_flag.set()


signal.signal(signal.SIGINT, handle_sigint)


def parse_args():
    parser = argparse.ArgumentParser(description='Optimized Camera Sender')
    parser.add_argument('--server-ip', type=str, default=DEFAULT_SERVER_IP,
                       help=f'Server IP address (default: {DEFAULT_SERVER_IP})')
    parser.add_argument('--server-port', type=int, default=DEFAULT_SERVER_PORT,
                       help=f'Server port (default: {DEFAULT_SERVER_PORT})')
    parser.add_argument('--fps', type=int, default=DEFAULT_CAPTURE_FPS,
                       help=f'Capture FPS (default: {DEFAULT_CAPTURE_FPS}Hz)')
    return parser.parse_args()


# ===================== 고정밀 트리거 =====================
class HighFreqTrigger(threading.Thread):
    """
    고정밀 트리거 (5Hz 기본)
    - 정확한 간격 유지
    - 드리프트 자동 보정
    """
    def __init__(self, interval=0.2, pulse_width=0.01):
        super().__init__(daemon=True)
        self.interval = float(interval)
        self.pulse_width = float(pulse_width)
        self.event = threading.Event()
        self.frame_count = 0

        # 타이밍 통계
        self.last_trigger_times = deque(maxlen=20)

        print(f"⏱  HighFreqTrigger: {1/interval:.1f} Hz ({interval*1000:.0f}ms interval)")

    def run(self):
        print("⏱  HighFreqTrigger started")
        next_trigger = time.time() + self.interval

        while not stop_flag.is_set():
            now = time.time()

            if now >= next_trigger:
                trigger_time = time.time()
                self.event.set()
                self.frame_count += 1

                # 타이밍 기록
                self.last_trigger_times.append(trigger_time)

                # 펄스 유지
                time.sleep(self.pulse_width)
                self.event.clear()

                # 다음 트리거 계산
                next_trigger += self.interval

                # 드리프트 보정
                if now - next_trigger > self.interval:
                    next_trigger = now + self.interval
                    print(f"⚠️  Trigger drift corrected at frame {self.frame_count}")
            else:
                # 정밀 대기
                sleep_time = next_trigger - now
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.7)
                else:
                    time.sleep(0.0001)

        # 통계 출력
        if len(self.last_trigger_times) >= 2:
            intervals = np.diff(list(self.last_trigger_times))
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            print(f"📊 Trigger Stats: avg={avg_interval*1000:.1f}ms, "
                  f"std={std_interval*1000:.2f}ms, frames={self.frame_count}")

        print(f"🛑 HighFreqTrigger stopped (triggers: {self.frame_count})")


# ===================== 고속 JPEG 인코더 =====================
def fast_jpeg_encoder_process(input_queue, output_queue, process_id, quality, resize_enabled):
    """최적화 JPEG 인코더"""
    print(f"🔧 FastEncoder-{process_id} started (Q={quality}, resize={resize_enabled})")

    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
        int(cv2.IMWRITE_JPEG_OPTIMIZE), 0,
        int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0
    ]

    encoded_count = 0
    total_encode_time = 0.0

    while not encoder_stop_flag.is_set():
        try:
            item = input_queue.get(timeout=0.1)
            if item is None:
                break

            cam_name, frame, timestamp = item
            t_start = time.time()

            # 리사이즈
            if resize_enabled and frame.shape[1] > RESIZE_WIDTH:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT),
                                 interpolation=cv2.INTER_AREA)

            # JPEG 인코딩
            ok, buf = cv2.imencode(".jpg", frame, encode_params)

            encode_time = (time.time() - t_start) * 1000
            total_encode_time += encode_time

            if ok:
                output_queue.put((cam_name, buf.tobytes(), timestamp, buf.nbytes, encode_time))
                encoded_count += 1

        except Empty:
            continue
        except Exception as e:
            print(f"[Encoder-{process_id}] Error: {e}")

    avg_encode_time = total_encode_time / encoded_count if encoded_count > 0 else 0
    print(f"🛑 Encoder-{process_id} stopped (frames: {encoded_count}, avg: {avg_encode_time:.1f}ms)")


# ===================== 고속 카메라 전송 =====================
class FastCameraSender(threading.Thread):
    """카메라 전송 (5Hz)"""
    def __init__(self, ip, port, quality=75, resize=True):
        super().__init__(daemon=True)

        # ZMQ 소켓
        self.ctx = zmq.Context.instance()
        self.ctx.setsockopt(zmq.IO_THREADS, ZMQ_IO_THREADS)

        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, CAMERA_SNDHWM)
        self.sock.setsockopt(zmq.SNDBUF, SNDBUF_SIZE)
        self.sock.setsockopt(zmq.SNDTIMEO, 5)
        self.sock.setsockopt(zmq.IMMEDIATE, 1)
        self.sock.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.sock.connect(f"tcp://{ip}:{port}")

        # 인코더 프로세스
        self.encode_input_queue = MPQueue(ENCODING_QUEUE_SIZE)
        self.encode_output_queue = MPQueue(ENCODING_QUEUE_SIZE)

        self.encoders = []
        for i in range(NUM_ENCODER_PROCESSES):
            p = Process(
                target=fast_jpeg_encoder_process,
                args=(self.encode_input_queue, self.encode_output_queue, i, quality, resize),
                daemon=True
            )
            p.start()
            self.encoders.append(p)

        # 통계
        self.stats = {
            'total_sent': 0,
            'total_bytes': 0,
            'encode_times': deque(maxlen=100),
            'send_times': deque(maxlen=100),
        }

        self.last_stats_print = time.time()
        self.stats_interval = 5.0  # 5초마다 통계 출력

        print(f"✅ FastCameraSender initialized ({NUM_ENCODER_PROCESSES} encoders)")

    def submit_frame(self, cam_name, frame, timestamp):
        """프레임 인코딩 요청"""
        try:
            self.encode_input_queue.put_nowait((cam_name, frame, timestamp))
        except:
            pass  # Queue full, drop frame

    def run(self):
            print("📡 FastCameraSender started")

            batch = []
            last_batch_time = time.time()

            while not stop_flag.is_set():
                try:
                    # 인코딩된 프레임 가져오기
                    item = self.encode_output_queue.get(timeout=BATCH_TIMEOUT)
                    cam_name, jpg_bytes, timestamp, jpg_size, encode_time = item

                    self.stats['encode_times'].append(encode_time)

                    # 메타데이터
                    meta = {
                        'camera': cam_name,
                        'timestamp': timestamp,
                        'size': jpg_size,
                    }

                    batch.append((meta, jpg_bytes))

                    # 배치 전송 조건
                    should_send = (
                        len(batch) >= BATCH_SEND_SIZE or
                        (time.time() - last_batch_time) >= BATCH_TIMEOUT
                    )

                    if should_send and batch:
                        t_send_start = time.time()

                        for meta, jpg in batch:
                            try:
                                meta_json = json.dumps(meta).encode('utf-8')
                                self.sock.send_multipart([meta_json, jpg], zmq.DONTWAIT)

                                self.stats['total_sent'] += 1
                                self.stats['total_bytes'] += len(jpg)

                            except zmq.Again:
                                pass # 버퍼가 꽉 찼으면 그냥 무시
                            except zmq.ZMQError as e:
                                if e.errno == zmq.ETERM:
                                    return # 컨텍스트 종료
                                print(f"[Sender] ZMQ error: {e}")
                            except Exception as e:
                                print(f"[Sender] Send error: {e}")

                        send_time = (time.time() - t_send_start) * 1000
                        self.stats['send_times'].append(send_time)

                        batch = []
                        last_batch_time = time.time()

                except Empty:
                    # 타임아웃 - 배치가 있으면 전송
                    if batch:
                        t_send_start = time.time()
                        
                        for meta, jpg in batch:
                            try:
                                meta_json = json.dumps(meta).encode('utf-8')
                                self.sock.send_multipart([meta_json, jpg], zmq.DONTWAIT)

                                self.stats['total_sent'] += 1
                                self.stats['total_bytes'] += len(jpg)

                            except:
                                pass # 오류 발생 시 무시

                        send_time = (time.time() - t_send_start) * 1000
                        self.stats['send_times'].append(send_time)

                        batch = []
                        last_batch_time = time.time()

                # 통계 출력
                now = time.time()
                if now - self.last_stats_print >= self.stats_interval:
                    self._print_stats()
                    self.last_stats_print = now

            # --- 📍 수정된 부분 시작 ---
            
            # 1. (사용자 요청) 남은 배치 버리기
            if batch:
                print(f"📡 Sender.run() loop stopped. Discarding {len(batch)} items from final batch.")

            # 2. (데드락 방지) 인코더 프로세스가 멈추지 않도록 출력 큐를 비웁니다.
            print(f"📡 Draining output queue to unblock encoders...")
            try:
                # 큐가 빌 때까지 모든 아이템을 강제로 꺼내서 버립니다.
                while True:
                    self.encode_output_queue.get(timeout=0.01)
            except Empty:
                print("📡 Output queue drained.")
            except Exception as e:
                print(f"[Sender Drain] Error draining queue: {e}")

            print("🛑 FastCameraSender.run() thread finished")
            # --- 📍 수정된 부분 끝 ---

    def _print_stats(self):
        """통계 출력"""
        total_sent = self.stats['total_sent']
        total_mb = self.stats['total_bytes'] / (1024 * 1024)

        avg_encode = np.mean(self.stats['encode_times']) if self.stats['encode_times'] else 0
        avg_send = np.mean(self.stats['send_times']) if self.stats['send_times'] else 0

        elapsed = time.time() - self.last_stats_print
        fps = total_sent / elapsed if elapsed > 0 else 0

        print(f"📊 [Sender Stats] Sent: {total_sent} frames ({fps:.1f} FPS), "
              f"Data: {total_mb:.1f} MB, Encode: {avg_encode:.1f}ms, Send: {avg_send:.1f}ms")

        # 통계 리셋
        self.stats['total_sent'] = 0
        self.stats['total_bytes'] = 0

    def stop(self):
            """종료"""
            print("⏳ Stopping encoders...")
            try:
                for _ in range(NUM_ENCODER_PROCESSES):
                    self.encode_input_queue.put(None)
            except Exception as e:
                print(f"[Sender Stop] Error putting None on queue: {e}")

            # --- BEGIN FIX (안전한 큐 종료) ---
            had_to_terminate = False
            for p in self.encoders:
                p.join(timeout=2.0) # run() 수정으로 이제 2초 안에 정상 종료되어야 합니다.
                if p.is_alive():
                    print(f"⚠️ Encoder {p.pid} did not terminate, killing...")
                    p.terminate()
                    p.join() # terminate()가 완료될 때까지 대기
                    had_to_terminate = True
            
            print("⏳ Closing queues...")
            try:
                self.encode_input_queue.close()
                self.encode_output_queue.close()

                if had_to_terminate:
                    print("⚠️ Terminated processes; force-cancelling queue threads.")
                    # '깨진' 큐에 대해 join_thread()를 호출하면 행이 걸리므로
                    # 대신 cancel_join_thread()를 호출합니다.
                    self.encode_input_queue.cancel_join_thread()
                    self.encode_output_queue.cancel_join_thread()
                else:
                    print("✅ Encoders joined gracefully; joining queue threads.")
                    # 정상 종료된 경우에만 안전하게 join_thread() 호출
                    self.encode_input_queue.join_thread()
                    self.encode_output_queue.join_thread()
            except Exception as e:
                print(f"[Sender Stop] Error closing queues: {e}")
            # --- END FIX ---

            print("⏳ Closing ZMQ socket...")
            self.sock.close(linger=0) 
            
            print("⏳ Terminating ZMQ context...")
            self.ctx.term()

            print("✅ FastCameraSender stopped")


# ===================== ZED 카메라 캡처 =====================
def zed_camera_process(serial, trigger_event, sender, stop_event):
    """ZED 카메라 캡처 (Left만) - Rising Edge 방식"""
    cam_name = f"ZED_{serial}_left"
    print(f"🎥 Starting {cam_name}...")

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.set_from_serial_number(int(serial))
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_disable_self_calib = True
    init_params.enable_image_enhancement = False

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"❌ {cam_name} failed to open: {status}")
        return

    print(f"✅ {cam_name} opened")

    mat_left = sl.Mat()
    runtime = sl.RuntimeParameters()
    runtime.enable_depth = False
    frame_count = 0

    # Rising edge 검출용
    last_trigger = False

    while not stop_event.is_set():
        # 현재 트리거 상태 확인
        current = trigger_event.is_set()

        # Rising edge 검출: 이전에 False였다가 True가 되는 순간만 캡처
        if current and not last_trigger:
            # 프레임 캡처
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                timestamp = time.time()
                zed.retrieve_image(mat_left, sl.VIEW.LEFT)

                # NumPy 변환 및 복사
                frame_left = mat_left.get_data()[:, :, :3].copy()  # BGRA → BGR

                # 전송
                sender.submit_frame(cam_name, frame_left, timestamp)
                frame_count += 1

        last_trigger = current
        time.sleep(0.0002)  # 0.2ms (고속 폴링)

    zed.close()
    print(f"🛑 {cam_name} stopped (frames: {frame_count})")


# ===================== OAK 카메라 캡처 =====================
def oak_camera_process(trigger_event, sender, stop_event):
    """OAK 카메라 캡처 - Rising Edge 방식"""
    cam_name = "OAK"
    print(f"🎥 Starting {cam_name}...")

    pipeline = dai.Pipeline()

    # RGB 카메라
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setFps(60)  # 명시적으로 FPS 설정
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.initialControl.setManualFocus(105)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    # 디바이스 시작
    try:
        device = dai.Device(pipeline)
        print(f"✅ {cam_name} opened")
    except Exception as e:
        print(f"❌ {cam_name} failed to open: {e}")
        return

    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    frame_count = 0

    # Rising edge 검출용
    last_trigger = False

    while not stop_event.is_set():
        # 현재 트리거 상태 확인
        current = trigger_event.is_set()

        # Rising edge 검출: 이전에 False였다가 True가 되는 순간만 캡처
        if current and not last_trigger:
            # 큐에 쌓인 모든 프레임을 버리고 최신 프레임만 가져오기
            frame = None
            while q_rgb.has():
                frame = q_rgb.get()

            # 최신 프레임이 있으면 전송
            if frame is not None:
                timestamp = time.time()
                img = frame.getCvFrame()
                sender.submit_frame(cam_name, img, timestamp)
                frame_count += 1

        last_trigger = current
        time.sleep(0.0002)  # 0.2ms (고속 폴링)

    device.close()
    print(f"🛑 {cam_name} stopped (frames: {frame_count})")


# ===================== Main =====================
def main():
    args = parse_args()

    global CAPTURE_INTERVAL
    CAPTURE_INTERVAL = 1.0 / args.fps

    print(f"\n{'='*80}")
    print(f"📷 Optimized Camera Sender")
    print(f"{'='*80}")
    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Capture Rate: {args.fps} Hz ({CAPTURE_INTERVAL*1000:.0f}ms interval)")
    print(f"Image Size: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    print(f"JPEG Quality: {JPEG_QUALITY}")
    print(f"Views: 5 (ZED left x4 + OAK x1)")
    print(f"{'='*80}\n")

    # ZED 시리얼 번호
    zed_serials = ["41182735", "49429257", "44377151", "49045152"]

    # 전송 스레드 시작
    sender = FastCameraSender(args.server_ip, args.server_port, quality=JPEG_QUALITY)
    sender.start()

    # 트리거 시작
    trigger = HighFreqTrigger(interval=CAPTURE_INTERVAL, pulse_width=PULSE_WIDTH)
    trigger.start()

    time.sleep(0.5)

    # 카메라 프로세스 시작
    camera_threads = []

    # ZED 카메라
    for serial in zed_serials:
        t = threading.Thread(
            target=zed_camera_process,
            args=(serial, trigger.event, sender, stop_flag),
            daemon=True
        )
        t.start()
        camera_threads.append(t)
        time.sleep(0.1)

    # OAK 카메라
    t = threading.Thread(
        target=oak_camera_process,
        args=(trigger.event, sender, stop_flag),
        daemon=True
    )
    t.start()
    camera_threads.append(t)

    print(f"\n{'='*80}")
    print(f"✅ Optimized Camera Sender Started")
    print(f"{'='*80}")
    print(f"Capturing at {args.fps} Hz")
    print(f"Press Ctrl+C to stop\n")

    # 대기 (수정된 부분)
    try:
        stop_flag.wait()
        print("\n🛑 Main thread received stop signal, initiating cleanup...")

    except KeyboardInterrupt:
        # signal 핸들러가 주 로직이지만, 만약을 대비한 fallback
        print("\n🛑 Shutting down (from KB Interrupt)...")
        stop_flag.set()
        encoder_stop_flag.set()

    # 종료
    print("⏳ Shutting down threads and processes...")
    
    # 플래그가 이미 핸들러에서 설정되었지만, 여기서 한 번 더 보장합니다.
    stop_flag.set()
    encoder_stop_flag.set()

    # 스레드 종료 대기
    for t in camera_threads:
        t.join(timeout=2.0)
    
    print("⏳ Stopping trigger...")
    trigger.join(timeout=1.0)

    # Sender와 인코더 종료
    sender.stop()
    sender.join(timeout=2.0)

    print("\n✅ Camera Sender stopped successfully")

if __name__ == "__main__":
    main()
