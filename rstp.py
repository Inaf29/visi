# from ultralytics import YOLO
# from pymavlink import mavutil
import numpy as np
import cv2  # Diaktifkan kembali untuk membaca kamera
import time
import sys
import os
import subprocess
import threading
import queue

# =========================================================
# KONFIGURASI UTAMA
# =========================================================

# MODEL_PATH = "models/model_terbaru.tflite"
# CONF_THRESHOLD = 0.80

# Jalur device video di Orange Pi / Linux
CAMERA_CANDIDATES = [
    "/dev/video0",
    "/dev/video1",
    0,
    1
]

# Kamera Logitech C270 / USB Cam standard
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# YOLO_IMGSZ = 416
# DETECT_EVERY_N_FRAME = 1
# TEMP_WARNING = 75.0
# TEMP_CUTOFF = 82.0
# TEMP_CHECK_INTERVAL = 2.0
FPS_REPORT_EVERY = 30

# =========================================================
# MAVLINK CONFIG (TETAP NONAKTIF)
# =========================================================
# ENABLE_MAVLINK = True
CONNECTION_STRING = "udpout:192.168.144.255:14550" 
# BAUDRATE = 115200 

# =========================================================
# RTSP HM30 CONFIG
# =========================================================

ENABLE_RTSP_STREAM = True

# Resolusi streaming diperkecil agar sangat ringan di radio HM30
RTSP_WIDTH = 240
RTSP_HEIGHT = 144
RTSP_FPS = 20 
RTSP_BITRATE = "800k"
RTSP_MODE = "listen"

RTSP_URL_GCS = "rtsp://192.168.144.30:8554/webcam"
RTSP_URL_MEDIAMTX_LOCAL = "rtsp://127.0.0.1:8554/webcam"
RTSP_URL_LISTEN_LOCAL = "rtsp://0.0.0.0:8554/webcam"

RTSP_RESTART_ON_FAIL = False
RTSP_ERROR_PRINT_INTERVAL = 3.0

# =========================================================
# FUNGSI KAMERA (DIAKTIFKAN KEMBALI)
# =========================================================

def open_camera():
    """
    Membuka kamera menggunakan backend V4L2 (standar Linux/Orange Pi)
    """
    for cam in CAMERA_CANDIDATES:
        print(f"[CAMERA] Mencoba membuka kamera: {cam}")

        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"[CAMERA] Gagal membuka: {cam}")
            cap.release()
            continue

        # Set format MJPG & resolusi hardware kamera
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
            print(f"[CAMERA] Resolusi asli: {frame.shape[1]}x{frame.shape[0]}")
            return cap

        print(f"[CAMERA] Kamera terbuka tapi frame kosong: {cam}")
        cap.release()

    return None

# =========================================================
# FUNGSI RTSP & FFMPEG
# =========================================================

def enqueue_stderr(pipe, q):
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
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{RTSP_WIDTH}x{RTSP_HEIGHT}",
        "-r", str(RTSP_FPS),
        "-i", "-",
        "-an",
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
        return base_cmd + [
            "-f", "rtsp",
            "-rtsp_flags", "listen",
            RTSP_URL_LISTEN_LOCAL
        ]

    if RTSP_MODE == "mediamtx":
        return base_cmd + [
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            RTSP_URL_MEDIAMTX_LOCAL
        ]

    raise ValueError("RTSP_MODE harus 'listen' atau 'mediamtx'")


def start_rtsp_ffmpeg():
    print("[RTSP] Starting FFmpeg stream...")
    try:
        cmd = build_ffmpeg_cmd()
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
        # DIAKTIFKAN KEMBALI: Perkecil resolusi dari 640x480 ke resolusi RTSP (240x144)
        if frame.shape[1] != RTSP_WIDTH or frame.shape[0] != RTSP_HEIGHT:
            frame = cv2.resize(frame, (RTSP_WIDTH, RTSP_HEIGHT))

        proc.stdin.write(frame.tobytes())
        return True

    except BrokenPipeError:
        print("[RTSP ERROR] FFmpeg pipe putus.")
        return False
    except Exception as e:
        print(f"[RTSP ERROR] Gagal kirim frame: {e}")
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


# =========================================================
# MAIN ROUTINE
# =========================================================

def main():
    print(">>> MEMULAI RTSP STREAM DENGAN KAMERA (NO AI/MAVLINK) <<<")
    
    rtsp_state = None
    cap = None

    # ================= CAMERA INIT =================
    print("[INIT] Menyalakan kamera...")
    cap = open_camera()

    if cap is None or not cap.isOpened():
        print("[ERROR] Kamera gagal dibuka. Pastikan tidak sedang dibuka oleh program lain.")
        sys.exit(1)

    # ================= RTSP INIT =================
    if ENABLE_RTSP_STREAM:
        rtsp_state = start_rtsp_ffmpeg()
        if rtsp_state is not None:
            print("[RTSP] Stream aktif.")
            print(f"[RTSP] Buka di GCS/SIYI FPV/VLC: {RTSP_URL_GCS}")

    # ================= LOOP VARIABLE =================
    frame_count = 0
    fps_timer = time.perf_counter()
    last_rtsp_restart = 0
    loop_fps = 0.0

    # ================= MAIN LOOP =================
    try:
        while True:
            # Membaca gambar asli dari kamera
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame kosong. Cek kabel kamera.")
                time.sleep(0.1)
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

                print_ffmpeg_errors(rtsp_state)

            # ================= BENCHMARK FPS =================
            if frame_count % FPS_REPORT_EVERY == 0:
                elapsed = time.perf_counter() - fps_timer
                loop_fps = FPS_REPORT_EVERY / elapsed if elapsed > 0 else 0.0

                rtsp_status = "ON" if (rtsp_state and rtsp_state["proc"].poll() is None) else "DEAD"

                print(
                    f"[BENCHMARK] Streaming FPS: {loop_fps:.2f} | "
                    f"RTSP Status: {rtsp_status} | "
                    f"URL: {RTSP_URL_GCS}"
                )
                fps_timer = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan manual oleh user.")
    finally:
        print("[INFO] Membersihkan resource...")
        stop_rtsp_ffmpeg(rtsp_state)
        if cap is not None: 
            cap.release()
        print("[INFO] Sistem shutdown mulus.")


if __name__ == "__main__":
    main()
