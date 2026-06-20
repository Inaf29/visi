#from ultralytics import YOLO
#from pymavlink import mavutil
#import numpy as np
#import cv2
#import time
#import sys
#import os
#import subprocess
#import threading
#import queue

# =========================================================
# KONFIGURASI UTAMA
# =========================================================

MODEL_PATH = "models/model_terbaru.tflite"

CONF_THRESHOLD = 0.80

# Lebih stabil pakai path langsung daripada index 0 di Orange Pi
CAMERA_CANDIDATES = [
    "/dev/video0",
    "/dev/video1",
    0,
    1
]

# Kamera Logitech C270 mendukung MJPG 640x480 30 FPS
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# YOLO
YOLO_IMGSZ = 416

# 1 = deteksi setiap frame
# 2 = deteksi tiap 2 frame agar CPU lebih ringan
DETECT_EVERY_N_FRAME = 1

# Suhu CPU
TEMP_WARNING = 75.0
TEMP_CUTOFF = 82.0
TEMP_CHECK_INTERVAL = 2.0

# Benchmark
FPS_REPORT_EVERY = 30

# =========================================================
# MAVLINK CONFIG
# =========================================================
# =========================================================
# MAVLINK CONFIG
# =========================================================

ENABLE_MAVLINK = True

# Gunakan 'udpout' untuk mengirim data via LAN ke GCS.
# 192.168.144.255 adalah alamat broadcast untuk subnet SIYI HM30.
# 14550 adalah port default yang didengarkan oleh Mission Planner / QGroundControl.
CONNECTION_STRING = "udpout:192.168.144.255:14550" 

# Opsional: Jika broadcast tidak masuk, ganti IP di atas dengan IP spesifik PC/Laptop GCS Anda.
# Contoh: "udpout:192.168.144.11:14550"

BAUDRATE = 115200 # Angka ini akan diabaikan oleh koneksi UDP, biarkan saja.

MAV_SOURCE_SYSTEM = 1
MAV_SOURCE_COMPONENT = 191

MAVLINK_SEND_INTERVAL = 1.0

MAV_SEVERITY_CRITICAL = 2
MAV_SEVERITY_ERROR = 3
MAV_SEVERITY_WARNING = 4
MAV_SEVERITY_INFO = 6
# =========================================================
# RTSP HM30 CONFIG
# =========================================================

ENABLE_RTSP_STREAM = True

# Resolusi QVGA: Video akan terlihat sedikit kotak-kotak (pixelated), 
# namun sangat ringan dan cukup untuk melihat bounding box AI.
RTSP_WIDTH = 240
RTSP_HEIGHT = 144

# 15 FPS: Standar kamera keamanan/CCTV. Sedikit patah-patah tapi tidak delay.
RTSP_FPS = 20 

# Bitrate 250k: Sangat kecil, memastikan pipa sinyal radio HM30 tidak pernah penuh.
RTSP_BITRATE = "800k"
# Pilihan:
# "listen"   = FFmpeg menjadi RTSP server langsung. Tidak perlu MediaMTX.
# "mediamtx" = FFmpeg publish ke MediaMTX. MediaMTX harus jalan dulu.
RTSP_MODE = "listen"

# URL yang dibuka dari laptop/GCS/SIYI FPV
RTSP_URL_GCS = "rtsp://192.168.144.30:8554/webcam"

# Kalau pakai MediaMTX, FFmpeg publish ke URL lokal ini
RTSP_URL_MEDIAMTX_LOCAL = "rtsp://127.0.0.1:8554/webcam"

# Kalau pakai listen, FFmpeg listen pada alamat ini
RTSP_URL_LISTEN_LOCAL = "rtsp://0.0.0.0:8554/webcam"

# Agar error FFmpeg tidak spam terus
RTSP_RESTART_ON_FAIL = False
RTSP_ERROR_PRINT_INTERVAL = 3.0

# =========================================================


class MavlinkMissionPlanner:
    def __init__(self, connection_string, baudrate):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.master = None
        self.connected = False

    def connect(self):
        print(f"[MAVLINK] Connecting to Pixhawk at {self.connection_string}...")

        try:
            self.master = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baudrate,
                source_system=MAV_SOURCE_SYSTEM,
                source_component=MAV_SOURCE_COMPONENT
            )

            print(f"[MAVLINK] Menyiapkan koneksi UDP ke {self.connection_string}...")
            
            # Kita set timeout lebih singkat (misal 3 detik)
            msg = self.master.wait_heartbeat(timeout=3)
            
            if msg:
                print(f"[MAVLINK] Heartbeat diterima dari GCS (System {self.master.target_system})")
            else:
                print("[MAVLINK] Timeout tunggu heartbeat, namun data akan tetap di-broadcast ke LAN.")
                
            # Tetap set True agar program terus mencoba mengirim (broadcast) data teks via LAN
            # kapanpun Mission Planner dibuka, pesan teks akan otomatis muncul di layar.
            self.connected = True

            self.send_text("VISION MAVLINK CONNECTED", MAV_SEVERITY_INFO)
            print(f"[MAVLINK] Connected to System {self.master.target_system}")
            self.connected = True

            self.send_text("VISION MAVLINK CONNECTED", MAV_SEVERITY_INFO)

        except Exception as e:
            print(f"[MAVLINK ERROR] Gagal konek ke Pixhawk: {e}")
            self.connected = False

    def send_text(self, text, severity=MAV_SEVERITY_INFO):
        if not self.connected or self.master is None:
            return

        try:
            text = str(text)
            if len(text) > 50:
                text = text[:50]

            self.master.mav.statustext_send(severity, text.encode())

        except Exception as e:
            print(f"[MAVLINK ERROR] Gagal mengirim pesan: {e}")
            self.connected = False

    def close(self):
        try:
            if self.connected:
                self.send_text("VISION MAVLINK STOPPED", MAV_SEVERITY_WARNING)

            if self.master is not None:
                self.master.close()

        except Exception:
            pass


def get_cpu_temperature():
    temps = []
    thermal_base = "/sys/class/thermal"

    try:
        for name in os.listdir(thermal_base):
            if not name.startswith("thermal_zone"):
                continue

            temp_path = os.path.join(thermal_base, name, "temp")

            if os.path.exists(temp_path):
                with open(temp_path, "r") as f:
                    raw = f.read().strip()

                if raw:
                    temp = float(raw) / 1000.0
                    if 0.0 < temp < 150.0:
                        temps.append(temp)

        if len(temps) > 0:
            return max(temps)

    except Exception as e:
        print(f"[TEMP ERROR] Gagal membaca suhu CPU: {e}")

    return 0.0


def open_camera():
    """
    Membuka kamera hanya satu kali.
    Tidak boleh ada program lain yang membuka /dev/video0 bersamaan.
    """

    for cam in CAMERA_CANDIDATES:
        print(f"[CAMERA] Mencoba membuka kamera: {cam}")

        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"[CAMERA] Gagal membuka: {cam}")
            cap.release()
            continue

        # Set MJPG dulu sebelum resolusi/FPS
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(0.5)

        ret, frame = cap.read()

        if ret and frame is not None:
            print(f"[CAMERA] Kamera aktif: {cam}")
            print(f"[CAMERA] Frame actual: {frame.shape[1]}x{frame.shape[0]}")
            return cap

        print(f"[CAMERA] Kamera terbuka tapi frame kosong: {cam}")
        cap.release()

    return None


def enqueue_stderr(pipe, q):
    """
    Membaca stderr FFmpeg di thread terpisah agar tidak blocking.
    """
    try:
        for line in iter(pipe.readline, b""):
            if not line:
                break
            q.put(line.decode(errors="ignore").strip())
    except Exception:
        pass


def build_ffmpeg_cmd():
    base_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",

        # Input raw frame dari Python
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{RTSP_WIDTH}x{RTSP_HEIGHT}",
        "-r", str(RTSP_FPS),
        "-i", "-",

        "-an",

        # H264 low latency
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",

        "-b:v", RTSP_BITRATE,
        "-maxrate", RTSP_BITRATE,
        "-bufsize", "300k",
        "-g", str(RTSP_FPS),
    ]

    if RTSP_MODE == "listen":
        # FFmpeg menjadi RTSP server langsung.
        # Tidak perlu MediaMTX.
        return base_cmd + [
            "-f", "rtsp",
            "-rtsp_flags", "listen",
            RTSP_URL_LISTEN_LOCAL
        ]

    if RTSP_MODE == "mediamtx":
        # Perlu menjalankan:
        # cd ~/mediamtx_rtsp
        # ./mediamtx
        return base_cmd + [
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            RTSP_URL_MEDIAMTX_LOCAL
        ]

    raise ValueError("RTSP_MODE harus 'listen' atau 'mediamtx'")


def start_rtsp_ffmpeg():
    print("[RTSP] Starting FFmpeg stream...")
    print("[RTSP] Mode   :", RTSP_MODE)
    print("[RTSP] GCS URL:", RTSP_URL_GCS)

    if RTSP_MODE == "mediamtx":
        print("[RTSP] Pastikan MediaMTX sudah jalan di terminal lain.")
        print("[RTSP] Command: cd ~/mediamtx_rtsp && ./mediamtx")

    try:
        cmd = build_ffmpeg_cmd()

        print("[RTSP] FFmpeg command:")
        print(" ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        err_queue = queue.Queue()
        err_thread = threading.Thread(
            target=enqueue_stderr,
            args=(proc.stderr, err_queue),
            daemon=True
        )
        err_thread.start()

        return {
            "proc": proc,
            "err_queue": err_queue,
            "last_error_print": 0.0,
            "dead_logged": False
        }

    except FileNotFoundError:
        print("[RTSP ERROR] ffmpeg belum terinstall.")
        print("[RTSP ERROR] Install: sudo apt install ffmpeg -y")
        return None

    except Exception as e:
        print(f"[RTSP ERROR] Gagal menjalankan FFmpeg: {e}")
        return None


def print_ffmpeg_errors(rtsp_state, force=False):
    if rtsp_state is None:
        return

    now = time.perf_counter()

    if not force and now - rtsp_state["last_error_print"] < RTSP_ERROR_PRINT_INTERVAL:
        return

    rtsp_state["last_error_print"] = now

    lines = []
    q = rtsp_state["err_queue"]

    while not q.empty():
        try:
            line = q.get_nowait()
            if line:
                lines.append(line)
        except Exception:
            break

    if lines:
        print("[RTSP FFMPEG LOG]")
        for line in lines[-10:]:
            print(line)


def send_frame_to_rtsp(rtsp_state, frame):
    if rtsp_state is None:
        return False

    proc = rtsp_state["proc"]

    if proc.poll() is not None:
        if not rtsp_state["dead_logged"]:
            print("[RTSP ERROR] FFmpeg sudah berhenti.")
            print_ffmpeg_errors(rtsp_state, force=True)
            rtsp_state["dead_logged"] = True

        return False

    try:
        if frame.shape[1] != RTSP_WIDTH or frame.shape[0] != RTSP_HEIGHT:
            frame = cv2.resize(frame, (RTSP_WIDTH, RTSP_HEIGHT))

        proc.stdin.write(frame.tobytes())
        return True

    except BrokenPipeError:
        print("[RTSP ERROR] FFmpeg pipe putus.")
        print_ffmpeg_errors(rtsp_state, force=True)
        return False

    except Exception as e:
        print(f"[RTSP ERROR] Gagal kirim frame: {e}")
        print_ffmpeg_errors(rtsp_state, force=True)
        return False


def stop_rtsp_ffmpeg(rtsp_state):
    if rtsp_state is None:
        return

    proc = rtsp_state["proc"]

    try:
        if proc.stdin:
            proc.stdin.close()

        proc.terminate()
        proc.wait(timeout=3)

    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    print(">>> MEMULAI VISION + MAVLINK + RTSP HM30 <<<")
    print(f"[INFO] Model        : {MODEL_PATH}")
    print(f"[INFO] Camera Size  : {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")
    print(f"[INFO] YOLO imgsz   : {YOLO_IMGSZ}")
    print(f"[INFO] Conf         : {CONF_THRESHOLD}")
    print(f"[INFO] Cutoff temp  : {TEMP_CUTOFF:.1f} °C")
    print(f"[INFO] MAVLink      : {ENABLE_MAVLINK}")
    print(f"[INFO] MAVLink Port : {CONNECTION_STRING}")
    print(f"[INFO] RTSP Stream  : {RTSP_URL_GCS}")
    print(f"[INFO] RTSP Mode    : {RTSP_MODE}")
    print("-" * 60)

    mavlink = None
    rtsp_state = None
    cap = None

    # ================= MAVLINK INIT =================

    if ENABLE_MAVLINK:
        mavlink = MavlinkMissionPlanner(CONNECTION_STRING, BAUDRATE)
        mavlink.connect()

    # ================= OPENCV OPTIMIZATION =================

    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # ================= LOAD MODEL =================

    print("[INIT] Loading Model AI...")

    try:
        # Explicit task agar warning Ultralytics berkurang
        model = YOLO(MODEL_PATH, task="detect")

        if mavlink and mavlink.connected:
            mavlink.send_text("YOLO MODEL LOADED", MAV_SEVERITY_INFO)

    except Exception as e:
        print(f"[ERROR] GAGAL LOAD MODEL! Pastikan path benar. Error: {e}")

        if mavlink and mavlink.connected:
            mavlink.send_text("ERROR LOAD YOLO MODEL", MAV_SEVERITY_CRITICAL)

        sys.exit(1)

    # ================= WARM-UP =================

    print("[INIT] Warm-up AI...")

    try:
        dummy_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

        model.predict(
            dummy_frame,
            imgsz=YOLO_IMGSZ,
            conf=CONF_THRESHOLD,
            device="cpu",
            verbose=False,
            max_det=1
        )

        print("[INIT] AI selesai warm-up.")

        if mavlink and mavlink.connected:
            mavlink.send_text("YOLO WARMUP DONE", MAV_SEVERITY_INFO)

    except Exception as e:
        print(f"[ERROR] Warm-up AI gagal: {e}")

        if mavlink and mavlink.connected:
            mavlink.send_text("ERROR YOLO WARMUP", MAV_SEVERITY_WARNING)

    # ================= CAMERA INIT =================

    print("[INIT] Menyalakan kamera...")

    cap = open_camera()

    if cap is None or not cap.isOpened():
        print("[ERROR] Kamera gagal dibuka.")
        print("[CEK] Jalankan: sudo fuser -v /dev/video0")
        print("[CEK] Kalau busy: sudo fuser -k /dev/video0")

        if mavlink and mavlink.connected:
            mavlink.send_text("ERROR CAMERA FAILED", MAV_SEVERITY_CRITICAL)

        sys.exit(1)

    print("[INIT] Kamera aktif. Memulai deteksi...\n")

    if mavlink and mavlink.connected:
        mavlink.send_text("VISION STARTED", MAV_SEVERITY_INFO)

    # ================= RTSP INIT =================

    if ENABLE_RTSP_STREAM:
        rtsp_state = start_rtsp_ffmpeg()

        if rtsp_state is not None:
            print("[RTSP] Stream aktif.")
            print(f"[RTSP] Buka di GCS/SIYI FPV/VLC: {RTSP_URL_GCS}")

            if mavlink and mavlink.connected:
                mavlink.send_text("RTSP STREAM STARTED", MAV_SEVERITY_INFO)
        else:
            print("[RTSP WARNING] RTSP tidak aktif, vision tetap berjalan.")

    # ================= LOOP VARIABLE =================

    frame_count = 0
    detect_count = 0

    fps_timer = time.perf_counter()
    last_report_time = 0
    last_temp_check = 0
    last_mavlink_send = 0
    last_rtsp_restart = 0

    cpu_temp = get_cpu_temperature()

    loop_fps = 0.0
    infer_fps = 0.0

    latest_object = "NONE"
    latest_conf = 0.0

    print(f"[TEMP] Suhu awal CPU: {cpu_temp:.1f} °C")

    if mavlink and mavlink.connected:
        mavlink.send_text(f"CPU START {cpu_temp:.1f}C", MAV_SEVERITY_INFO)

    # ================= MAIN LOOP =================

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[WARNING] Frame kosong. Cek kamera/kabel USB.")

                if mavlink and mavlink.connected:
                    mavlink.send_text("WARNING FRAME EMPTY", MAV_SEVERITY_WARNING)

                time.sleep(0.2)
                continue

            frame_count += 1
            now = time.perf_counter()

            # ================= RTSP STREAM =================

            if ENABLE_RTSP_STREAM and rtsp_state is not None:
                ok_rtsp = send_frame_to_rtsp(rtsp_state, frame)

                if not ok_rtsp and RTSP_RESTART_ON_FAIL:
                    if now - last_rtsp_restart > 5.0:
                        print("[RTSP] Mencoba restart FFmpeg...")
                        stop_rtsp_ffmpeg(rtsp_state)
                        rtsp_state = start_rtsp_ffmpeg()
                        last_rtsp_restart = now

                # Cetak error FFmpeg periodik jika ada
                print_ffmpeg_errors(rtsp_state)

            # ================= CEK SUHU CPU =================

            if now - last_temp_check >= TEMP_CHECK_INTERVAL:
                cpu_temp = get_cpu_temperature()
                last_temp_check = now

                if cpu_temp >= TEMP_WARNING:
                    print(f"[TEMP WARNING] CPU panas: {cpu_temp:.1f} °C")
                else:
                    print(f"[TEMP] CPU: {cpu_temp:.1f} °C")

            # ================= DETEKSI YOLO =================

            if frame_count % DETECT_EVERY_N_FRAME == 0:
                results = model.predict(
                    frame,
                    imgsz=YOLO_IMGSZ,
                    conf=CONF_THRESHOLD,
                    device="cpu",
                    verbose=False,
                    max_det=1
                )

                detect_count += 1

                if (
                    len(results) > 0
                    and results[0].boxes is not None
                    and len(results[0].boxes) > 0
                ):
                    boxes = results[0].boxes

                    best_box_idx = int(boxes.conf.argmax())
                    best_box = boxes[best_box_idx]

                    conf = float(best_box.conf[0])
                    cls_id = int(best_box.cls[0])
                    class_name = model.names[cls_id]

                    latest_object = class_name
                    latest_conf = conf

                    if time.perf_counter() - last_report_time > 2.0:
                        print(
                            f"[TARGET DETECTED] {class_name.upper()} | "
                            f"Akurasi: {conf * 100:.1f}% | "
                            f"CPU: {cpu_temp:.1f} °C"
                        )

                        if mavlink and mavlink.connected:
                            mavlink.send_text(
                                f"DETECT {class_name} {conf * 100:.0f}%",
                                MAV_SEVERITY_INFO
                            )

                        last_report_time = time.perf_counter()

                    # Cut-off hanya ketika objek terdeteksi dan suhu tinggi
                    if cpu_temp >= TEMP_CUTOFF:
                        print("\n" + "=" * 60)
                        print("[CUT-OFF] OBJEK TERDETEKSI SAAT CPU OVERHEAT!")
                        print(f"[CUT-OFF] Suhu CPU: {cpu_temp:.1f} °C")
                        print(f"[CUT-OFF] Batas    : {TEMP_CUTOFF:.1f} °C")
                        print("[CUT-OFF] Program vision dihentikan untuk melindungi hardware.")
                        print("=" * 60)

                        if mavlink and mavlink.connected:
                            mavlink.send_text(
                                f"CUTOFF CPU {cpu_temp:.1f}C",
                                MAV_SEVERITY_CRITICAL
                            )

                        break

                else:
                    latest_object = "NONE"
                    latest_conf = 0.0

            # ================= BENCHMARK FPS =================

            if frame_count % FPS_REPORT_EVERY == 0:
                elapsed = time.perf_counter() - fps_timer

                if elapsed > 0:
                    loop_fps = FPS_REPORT_EVERY / elapsed
                else:
                    loop_fps = 0.0

                if DETECT_EVERY_N_FRAME == 1:
                    infer_fps = loop_fps
                else:
                    if elapsed > 0:
                        infer_fps = detect_count / elapsed
                    else:
                        infer_fps = 0.0

                rtsp_status = "OFF"
                if rtsp_state is not None:
                    if rtsp_state["proc"].poll() is None:
                        rtsp_status = "ON"
                    else:
                        rtsp_status = "DEAD"

                print(
                    f"[BENCHMARK] Loop FPS: {loop_fps:.2f} | "
                    f"Inference FPS: {infer_fps:.2f} | "
                    f"CPU: {cpu_temp:.1f} °C | "
                    f"OBJ: {latest_object} {latest_conf * 100:.0f}% | "
                    f"RTSP: {rtsp_status} | "
                    f"{RTSP_URL_GCS}"
                )

                fps_timer = time.perf_counter()
                detect_count = 0

            # ================= MAVLINK PERIODIC SEND =================

            if mavlink and mavlink.connected:
                now_mavlink = time.perf_counter()

                if now_mavlink - last_mavlink_send >= MAVLINK_SEND_INTERVAL:
                    msg_text = (
                        f"FPS:{loop_fps:.1f} "
                        f"INF:{infer_fps:.1f} "
                        f"CPU:{cpu_temp:.1f}C"
                    )

                    if cpu_temp >= TEMP_WARNING:
                        severity = MAV_SEVERITY_WARNING
                    else:
                        severity = MAV_SEVERITY_INFO

                    mavlink.send_text(msg_text, severity)
                    print(f"[MAVLINK SEND] {msg_text}")

                    last_mavlink_send = now_mavlink

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan manual oleh user.")

        if mavlink and mavlink.connected:
            mavlink.send_text("VISION STOPPED BY USER", MAV_SEVERITY_WARNING)

    finally:
        print("[INFO] Membersihkan resource hardware...")

        stop_rtsp_ffmpeg(rtsp_state)

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        if mavlink:
            mavlink.close()

        print("[INFO] Sistem shutdown mulus.")


if __name__ == "__main__":
    main()
