# robot_command_receiver.py (rev. Gemini-fix-2)
import socket, json, time, threading, atexit, signal, sys, struct, zmq, argparse
from math import copysign

# === Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Robot Command Receiver and State Publisher")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for debugging.")
    return parser.parse_args()
args = parse_args()

ROBOT_IP, CTRL_PORT = "192.168.0.100", 10000
LISTEN_PORT = 5000

# === ZMQ State Publisher Config ===
ZMQ_PUB_ADDRESS = "*"  # Bind to all interfaces
ZMQ_PUB_PORT = 5556
ZMQ_TOPIC = b"robot_state"
STATE_PUB_RATE_HZ = 100
# Format: <ddf12f (origin_ts, send_ts, force, 6 joints, 6 pose)
PAYLOAD_FORMAT = '<ddf12f'

# === 로봇 시작 위치 ===
HOME_JOINTS = [190, 1, 309, 1, 90, 0] # 안전한 시작을 위한 기본 홈 포지션 (관절 각도)
# === 주기/워치독 ===
CTRL_HZ   = 10.0
CTRL_DT   = 1.0 / CTRL_HZ
WATCHDOG_T = 5.0
KEEPALIVE_PERIOD = WATCHDOG_T * 0.5
VEL_TIMEOUT = 0.40  # (10Hz보다 크게)

# [제거] 불필요한 시작 속도 상수
# CART_LIN_V0 = 5
# CART_ANG_V0 = 10

# [수정] '액션 속도 1' (dpose) 설정
VEL_CLAMP_MM_S  = 1.0  # mm/s
VEL_CLAMP_DEG_S = 1.0  # deg/s

robot_sock = None
_keepalive_run = False

# 누적 Δpose 버퍼
_dp_lock = threading.Lock()
_robot_sock_lock = threading.Lock()
_dp_acc = [0.0]*6
_started = False
_stop_flag = threading.Event()
_dpose_enabled = False # dpose 루프 활성화 플래그

# 파일 상단 전역
_lock_j6 = False
_j6_target = None  # deg

shutdown_done = False

# ---------- TCP helpers ----------
def send_cmd(sock, cmd: str):
    sock.sendall((cmd + "\0").encode("ascii"))

def recv_line(sock, timeout=2.0):
    sock.settimeout(timeout)
    buf = bytearray()
    while True:
        b = sock.recv(1)
        if not b:
            raise ConnectionError("robot closed")
        if b == b"\x00":
            break
        buf.extend(b)
    return buf.decode("ascii", "ignore")

def try_recv_line(sock, timeout=0.5):
    try:
        return recv_line(sock, timeout=timeout)
    except Exception:
        return None

def send_and_get(sock, cmd, timeout=2.0):
    with _robot_sock_lock:
        send_cmd(sock, cmd)
        return recv_line(sock, timeout=timeout)

# ---------- Watchdog ----------
def watchdog_keepalive(sock):
    global _keepalive_run
    while _keepalive_run:
        try:
            with _robot_sock_lock:
                send_cmd(sock, f"ConnectionWatchdog({WATCHDOG_T})")
                _ = try_recv_line(sock, timeout=0.02)
        except Exception:
            break
        time.sleep(KEEPALIVE_PERIOD)

# ---------- Status / helpers ----------
def _parse_status(line):
    # [2007][as,hs,sm,es,pm,eob,eom]
    fields = line.split("[", 2)[2].split("]")[0].split(",")
    as_, hs, sm, es, pm, eob, eom = [int(x.strip()) for x in fields]
    return {"as": as_, "hs": hs, "sm": sm, "es": es, "pm": pm, "eob": eob, "eom": eom}

def get_status():
    try:
        with _robot_sock_lock:
            send_cmd(robot_sock, "GetStatusRobot()")
            line = try_recv_line(robot_sock, timeout=0.5)
        if not line or not line.startswith("[2007]"):
            return None
        return _parse_status(line)
    except Exception:
        return None

def wait_until(pred, timeout=60.0, poll=0.1, label=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = get_status()
        if st and pred(st):
            return True
        time.sleep(poll)
    print(f" -> [wait_until timeout] {label}")
    return False

def _clamp(v, lim):
    return v if abs(v) <= lim else copysign(lim, v)

# GetRtTargetCartPos (2201) 응답 파서
def _parse_pose_line(line):
    # [2201][t,x,y,z,a,b,g]
    try:
        if not line or not line.startswith("[2201]"):
            return None
        payload = line.split("][", 1)[1].split("]")[0]
        vals = [float(v.strip()) for v in payload.split(",")]
        if len(vals) == 7:
            return vals[1:]
    except Exception:
        pass
    return None

# GetRtTargetJointPos (2200) 응답 파서
def _parse_joints_line(line):
    # [2200][t,j1,j2,j3,j4,j5,j6]
    try:
        if not line or not line.startswith("[2200]"):
            return None
        payload = line.split("][", 1)[1].split("]")[0]
        vals = [float(v.strip()) for v in payload.split(",")]
        if len(vals) == 7:
            return vals[1:]
    except Exception:
        pass
    return None

# ---------- State Publisher Thread ----------
class StatePublisher(threading.Thread):
    """Periodically gets robot state and publishes it over ZMQ."""
    def __init__(self, robot_socket, stop_event):
        super().__init__(daemon=True)
        self.robot_sock = robot_socket
        self.stop_event = stop_event
        self.dt = 1.0 / float(STATE_PUB_RATE_HZ)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        bind_addr = f"tcp://{ZMQ_PUB_ADDRESS}:{ZMQ_PUB_PORT}"
        self.socket.set_hwm(10)
        self.socket.bind(bind_addr)
        print(f"✅ [PUB] ZMQ Publisher bound to {bind_addr} at {STATE_PUB_RATE_HZ} Hz.")

    def run(self):
        next_send_time = time.time()
        pub_count = 0
        last_log_time = time.time()

        while not self.stop_event.is_set():
            try:
                # Get latest robot state
                with _robot_sock_lock:
                    send_cmd(self.robot_sock, "GetRtTargetJointPos()")
                    joints_line = recv_line(self.robot_sock, timeout=0.1)
                    send_cmd(self.robot_sock, "GetRtTargetCartPos()")
                    pose_line = recv_line(self.robot_sock, timeout=0.1)

                joints = _parse_joints_line(joints_line)
                pose = _parse_pose_line(pose_line)

                if joints and pose:
                    origin_ts = time.time() # Or use robot timestamp if available
                    send_ts = time.time()
                    force = 0.0 # Placeholder

                    payload_bytes = struct.pack(PAYLOAD_FORMAT, origin_ts, send_ts, force, *joints, *pose)
                    self.socket.send_multipart([ZMQ_TOPIC, payload_bytes], zmq.DONTWAIT)
                    pub_count += 1

                    if args.verbose:
                        print(f"[PUB] Sent state packet ts:{origin_ts:.3f}")

            except zmq.Again:
                pass # Ignore if cannot send immediately
            except Exception as e:
                # Avoid spamming logs for common timeout errors during robot moves
                if not isinstance(e, socket.timeout) and args.verbose:
                    print(f"[PUB] Error: {e}")
            
            # Log status once per second
            if time.time() - last_log_time > 1.0:
                if pub_count > 0:
                    print(f"[PUB] Sent {pub_count} state packets in the last second.")
                else:
                    print("[PUB] No state packets sent in the last second.")
                pub_count = 0
                last_log_time = time.time()

            # Maintain rate
            next_send_time += self.dt
            sleep_duration = next_send_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        print("🛑 [PUB] StatePublisher Stopped.")

    def stop(self):
        self.socket.close()
        self.context.term()

# ---------- Robot session ----------
def robot_connect():
    print("[ROBOT] Connecting to robot...")
    s = socket.create_connection((ROBOT_IP, CTRL_PORT), timeout=3)
    print("[ROBOT] Connected.")
    banner = try_recv_line(s, timeout=1.0)
    if banner:
        print(f"[ROBOT] -> {banner}")
    return s

def robot_init():
    global robot_sock, _keepalive_run
    print("[ROBOT] Initializing...")

    # 워치독 + keepalive
    print("[ROBOT] ->", send_and_get(robot_sock, f"ConnectionWatchdog({WATCHDOG_T})", 1.0))
    _keepalive_run = True
    threading.Thread(target=watchdog_keepalive, args=(robot_sock,), daemon=True).start()

    # 에러 있으면 해제
    st = get_status()
    if st and st["es"] != 0:
        print("[ROBOT] ->", send_and_get(robot_sock, "ResetError()", 1.0))
        print("[ROBOT] ->", send_and_get(robot_sock, "ResumeMotion()", 1.0))
        wait_until(lambda s: s["es"] == 0, 5, label="error clear")

    # Activate
    st = get_status()
    if not st or st["as"] == 0:
        rep = send_and_get(robot_sock, "ActivateRobot()", 2.0)
        print("[ROBOT] ->", rep)
        if rep.startswith("[1013]"):
            print("[ROBOT] ->", send_and_get(robot_sock, "ResumeMotion()", 1.0))
            print("[ROBOT] ->", send_and_get(robot_sock, "ActivateRobot()", 2.0))
    wait_until(lambda s: s and s["as"] == 1 and s["es"] == 0, 10, label="activate")

    # Home
    st = get_status()
    if st and st["hs"] == 0:
        rep = send_and_get(robot_sock, "Home()", 4.0); print("[ROBOT] ->", rep)
        ok = wait_until(lambda s: s["hs"] == 1 and s["es"] == 0, 60, label="home")
        if not ok:
            print("[ROBOT] ->", send_and_get(robot_sock, "ResetError()", 1.0))
            print("[ROBOT] ->", send_and_get(robot_sock, "ResumeMotion()", 1.0))
            print("[ROBOT] ->", send_and_get(robot_sock, "Home()", 4.0))
            wait_until(lambda s: s["hs"] == 1 and s["es"] == 0, 60, label="home retry")

    # [수정] '처음 시작 속도 3' 설정
    print("[ROBOT] ->", send_and_get(robot_sock, "SetJointVel(3)", 1.0))
    
    # [제거] 불필요한 SetCart...Vel 호출
    # print(" ->", send_and_get(robot_sock, f"SetCartLinVel({CART_LIN_V0})", 1.0))
    # print(" ->", send_and_get(robot_sock, f"SetCartAngVel({CART_ANG_V0})", 1.0))

    # 완료 이벤트 on + 자동 구성
    print("[ROBOT] ->", send_and_get(robot_sock, "SetEom(1)", 1.0))
    print("[ROBOT] ->", send_and_get(robot_sock, "SetAutoConf(1)", 1.0))
    print("[ROBOT] ->", send_and_get(robot_sock, "SetAutoConfTurn(1)", 1.0))
    print("[ROBOT] ->", send_and_get(robot_sock, "SetConf(1, 1, 1)", 1.0))
    print("[ROBOT] ->", send_and_get(robot_sock, f"SetVelTimeout({VEL_TIMEOUT})", 1.0))

    print("✅ [ROBOT] Robot ready: homed & no error")

# 10Hz 실행 루프: dpose/apose 상태 관리 기능 추가
def exec_loop_10hz():
    global _dp_acc, _started, _dpose_enabled
    t0 = time.time()
    last_status = t0
    
    while not _stop_flag.is_set():
        t_next = t0 + CTRL_DT
        t0 = t_next

        st = get_status()
        if not st:
            dt = t_next - time.time()
            if dt > 0: time.sleep(dt)
            else: time.sleep(0.001)
            continue

        if _dpose_enabled:
            # --- 1. DPOSE 모드 (속도 스트리밍) ---
            with _dp_lock:
                dp = _dp_acc[:]
                _dp_acc = [0.0] * 6

            # [수정] '액션 속도 1' (dpose) 적용
            vx = _clamp(dp[0] / CTRL_DT, VEL_CLAMP_MM_S)
            vy = _clamp(dp[1] / CTRL_DT, VEL_CLAMP_MM_S)
            vz = _clamp(dp[2] / CTRL_DT, VEL_CLAMP_MM_S)
            va = _clamp(dp[3] / CTRL_DT, VEL_CLAMP_DEG_S)
            vb = _clamp(dp[4] / CTRL_DT, VEL_CLAMP_DEG_S)
            vg = _clamp(dp[5] / CTRL_DT, VEL_CLAMP_DEG_S)

            if st["es"] == 0 and _started:
                try:
                    with _robot_sock_lock:
                        cmd_str = f"MoveLinVelWrf({vx:.2f},{vy:.2f},{vz:.2f},{va:.2f},{vb:.2f},{vg:.2f})"
                        send_cmd(robot_sock, cmd_str)
                        _ = try_recv_line(robot_sock, timeout=0.02)
                    if args.verbose and any(abs(v) > 1e-3 for v in [vx,vy,vz,va,vb,vg]):
                        print(f"[ROBOT] -> {cmd_str}")
                except Exception as e:
                    print(f"[ROBOT] MoveLinVelWrf error: {e}")
        
        else:
            # --- 2. APOSE 모드 (dpose가 중지된 상태) ---
            if st["eom"] == 1 and st["pm"] == 0 and st["es"] == 0 and _started:
                print("[ROBOT] apose finished. Re-enabling dpose loop.")
                _dpose_enabled = True # dpose 루프 재시작

        # 1초 주기 상태 출력
        now = time.time()
        if now - last_status >= 1.0:
            mode = "DPOSE" if _dpose_enabled else "APOSE"
            print(f"[ROBOT] Status: mode={mode} as={st['as']} hs={st['hs']} es={st['es']} eom={st['eom']} pm={st['pm']}")
            last_status = now

        # 타이밍 정렬
        dt = t_next - time.time()
        if dt > 0:
            time.sleep(dt)
        else:
            time.sleep(0.001)



# ---------- Shutdown ----------
def graceful_shutdown():
    global robot_sock, _keepalive_run, shutdown_done, _dpose_enabled, state_publisher
    if shutdown_done:
        return
    shutdown_done = True
    print("\n[SYSTEM] Graceful shutdown...")
    _stop_flag.set()
    _dpose_enabled = False
    _keepalive_run = False
    time.sleep(0.1)

    if state_publisher:
        print("[SYSTEM] -> Stopping state publisher...")
        state_publisher.stop()
        state_publisher.join(timeout=1.0)

    es = 0
    try:
        st = get_status()
        if st:
            es = st["es"]
    except Exception:
        pass

    try:
        if es == 0:
            try:
                with _robot_sock_lock:
                    send_cmd(robot_sock, "MoveLinVelWrf(0,0,0,0,0,0)")
                    _ = try_recv_line(robot_sock, timeout=0.3)
            except Exception as e:
                print(f"[SYSTEM] -> MoveLinVelWrf(0) note: {e}")
            for cmd in ("ClearMotion()", "ResumeMotion()"): # Clear any lingering moves
                try:
                    print("[SYSTEM] ->", send_and_get(robot_sock, cmd, 0.5))
                except Exception as e:
                    print(f"[SYSTEM] -> {cmd} note: {e}")
        else:
            try:
                print("[SYSTEM] ->", send_and_get(robot_sock, "ResetError()", 0.5))
            except Exception as e:
                print(f"[SYSTEM] -> ResetError() note: {e}")

        try:
            print("[SYSTEM] ->", send_and_get(robot_sock, "DeactivateRobot()", 0.5))
        except Exception as e:
            print(f"[SYSTEM] -> DeactivateRobot() note: {e}")
        try:
            print("[SYSTEM] ->", send_and_get(robot_sock, "ConnectionWatchdog(0)", 0.5))
        except Exception as e:
            print(f"[SYSTEM] -> ConnectionWatchdog(0) note: {e}")
    finally:
        try:
            robot_sock.close()
        except Exception:
            pass
        print("✅ [SYSTEM] Shutdown complete.")

atexit.register(graceful_shutdown)
def _sig(signum, frame):
    graceful_shutdown(); sys.exit(0)
signal.signal(signal.SIGINT, _sig)
signal.signal(signal.SIGTERM, _sig)

# ---------- Main ----------
state_publisher = None

def main():
    global robot_sock, state_publisher
    global _started, _dpose_enabled, _lock_j6, _j6_target
    robot_sock = robot_connect()
    robot_init()

    # --- Go to a defined home position first ---
    print(f"[CMD] Moving to home position: {HOME_JOINTS}")
    with _robot_sock_lock:
        send_cmd(robot_sock, f"MovePose({HOME_JOINTS[0]},{HOME_JOINTS[1]},{HOME_JOINTS[2]},{HOME_JOINTS[3]},{HOME_JOINTS[4]},{HOME_JOINTS[5]})")
        _ = try_recv_line(robot_sock, timeout=0.1) # Consume response
    wait_until(lambda s: s["sm"] == 0 and s["eom"] == 1 and s["es"] == 0, 120, label="go home")
    print("[ROBOT] Reached home position.")
    # --- END ---

    # Start state publisher thread
    state_publisher = StatePublisher(robot_sock, _stop_flag)
    state_publisher.start()

    # 10 Hz 실행 스레드 시작
    th = threading.Thread(target=exec_loop_10hz, daemon=True)
    th.start()

    # ZMQ Command Receiver
    ctx = zmq.Context()
    cmd_sock = ctx.socket(zmq.PULL)
    cmd_sock.bind(f"tcp://*:{LISTEN_PORT}")
    print(f"✅ [SYSTEM] Waiting for inference commands on port {LISTEN_PORT}...")

    dpose_recv_count = 0

    try:
        while not _stop_flag.is_set():
            try:
                # Use polling with a timeout to prevent blocking indefinitely
                if cmd_sock.poll(timeout=100): # 100ms timeout
                    msg = cmd_sock.recv_json()
                else:
                    continue # No message, loop again
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM or _stop_flag.is_set():
                    break
                print(f"[CMD] ZMQ receive error: {e}")
                continue

            if args.verbose:
                print(f"[CMD] Received: {msg}")

            cmd = msg.get("cmd")

            if cmd == "start":
                if "start_joints" in msg:
                    j = msg["start_joints"]
                    print(f"[CMD] Executing start with MovePose: {j}")
                    with _robot_sock_lock:
                        send_cmd(robot_sock, f"MovePose({j[0]},{j[1]},{j[2]},{j[3]},{j[4]},{j[5]})")
                        _ = try_recv_line(robot_sock, timeout=0.1) # Consume response
                    ok = wait_until(lambda s: s["sm"] == 0 and s["eom"] == 1 and s["es"] == 0, 120, label="start joints")
                    print(f"[ROBOT] MovePose idle: {ok}")
                elif "start_pose" in msg:
                    x, y, z, a, b, g = msg["start_pose"]
                    print(f"[CMD] Executing start with MovePose(abs): {msg['start_pose']}")
                    with _robot_sock_lock:
                        send_cmd(robot_sock, f"MovePose({x},{y},{z},{a},{b},{g})")
                        _ = try_recv_line(robot_sock, timeout=0.1) # Consume response
                    ok = wait_until(lambda s: s["sm"] == 0 and s["eom"] == 1 and s["es"] == 0, 120, label="start pose")
                    print(f"[ROBOT] MovePose idle: {ok}")
                else:
                    print("[CMD] ERROR: start command requires 'start_joints' or 'start_pose'")
                    continue

                # J6 lock 처리
                if msg.get("lock_j6"):
                    print("[ROBOT] J6 lock enabled.")
                    _lock_j6 = True
                    if "start_joints" in msg:
                        _j6_target = msg["start_joints"][5]
                        print(f"[ROBOT] J6 target from start_joints: {_j6_target} deg")
                    else:
                        joints_line = send_and_get(robot_sock, "GetRtTargetJointPos()", 1.0)
                        joints = _parse_joints_line(joints_line)
                        if joints:
                            _j6_target = joints[5]
                            print(f"[ROBOT] J6 target from current joints: {_j6_target} deg")
                        else:
                            print("[ROBOT] WARN: Could not read joints for J6 lock. Lock may not work.")
                            _j6_target = None
                            _lock_j6 = False

                # 실제 시작 포즈 계산 (응답 대신 로그 출력)
                pose_line = send_and_get(robot_sock, "GetRtTargetCartPos()", 1.0)
                start_pose_actual = _parse_pose_line(pose_line)
                print(f"✅ [CMD] Start command complete. Robot ready at pose: {start_pose_actual}. D-pose loop enabled.")
                _started = True
                _dpose_enabled = True # dpose 루프 시작

            elif cmd == "dpose":
                dpose_recv_count += 1
                if dpose_recv_count % 10 == 0: # Print every 10 actions
                    print(f"✅ [CMD] Received action #{dpose_recv_count}.")

                if not _started or not _dpose_enabled: # apose 중에는 dpose 무시
                    continue
                dp = msg.get("dp", [0, 0, 0, 0, 0, 0])
                with _dp_lock:
                    for i in range(6):
                        _dp_acc[i] += float(dp[i])

            elif cmd == "apose":
                if not _started: continue
                
                print("[CMD] apose received, interrupting dpose.")
                _dpose_enabled = False # dpose 루프 중지
                
                try:
                    with _robot_sock_lock:
                        send_cmd(robot_sock, "SetJointVel(1)")
                        _ = try_recv_line(robot_sock, timeout=0.02)
                        send_cmd(robot_sock, "MoveLinVelWrf(0,0,0,0,0,0)")
                        _ = try_recv_line(robot_sock, timeout=0.02)
                        send_cmd(robot_sock, "ClearMotion()")
                        _ = try_recv_line(robot_sock, timeout=0.02)
                        send_cmd(robot_sock, "ResumeMotion()")
                        _ = try_recv_line(robot_sock, timeout=0.02)
                except Exception as e:
                    print(f"[ROBOT] ClearMotion/ResumeMotion error: {e}")

                x,y,z,a,b,g = msg["pose"]

                # --- J6 lock 로직 ---
                if _lock_j6 and _j6_target is not None:
                    try:
                        with _robot_sock_lock:
                            send_cmd(robot_sock, "GetRtTargetJointPos()")
                            jline = try_recv_line(robot_sock, timeout=0.2)
                        j = _parse_joints_line(jline)
                        if j:
                            delta = j[5] - _j6_target
                            g = g - delta
                    except Exception as e:
                        print(f"[ROBOT] J6 lock read/adjust note: {e}")
                # --- J6 lock 끝 ---

                try:
                    print(f"[ROBOT] Sending new MovePose: {x,y,z,a,b,g}")
                    with _robot_sock_lock:
                        send_cmd(robot_sock, f"MovePose({x},{y},{z},{a},{b},{g})")
                        _ = try_recv_line(robot_sock, timeout=0.02)
                except Exception as e:
                    print(f"[ROBOT] MovePose error: {e}")
                
            elif cmd == "stop":
                print("[CMD] Received STOP command.")
                _dpose_enabled = False
                _stop_flag.set()
                with _dp_lock:
                    for i in range(6):
                        _dp_acc[i] = 0.0
                break
        print("🛑 [CMD] Command listener loop stopped.")

    finally:
        print("[SYSTEM] Cleaning up ZMQ command socket...")
        cmd_sock.close()
        ctx.term()
        graceful_shutdown()

if __name__ == "__main__":
    main()