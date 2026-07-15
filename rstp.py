import cv2
import time
import sys
import threading
import queue
import subprocess

# =========================================================
# PRESET JARAK — pilih salah satu sesuai kondisi terbang
# =========================================================
# "dekat"   : sinyal kuat, <1km, bandwidth HM30 masih tinggi
# "menengah": 1-5km, bandwidth sedang
# "jauh"    : 5km+, PRIORITAS UTAMA = tidak boleh putus, kualitas nomor dua
#
ACTIVE_PRESET = "jauh"

PRESETS = {
    "dekat": {
        "width": 640, "height": 480, "fps": 20,
        "bitrate": "800k", "maxrate": "800k", "bufsize": "400k",
    },
    "menengah": {
        "width": 480, "height": 360, "fps": 15,
        "bitrate": "350k", "maxrate": "350k", "bufsize": "175k",
    },
    "jauh": {
        "width": 320, "height": 240, "fps": 10,
        "bitrate": "150k", "maxrate": "150k", "bufsize": "75k",
    },
}

CFG = PRESETS[ACTIVE_PRESET]
RTSP_WIDTH = CFG["width"]
RTSP_HEIGHT = CFG["height"]
RTSP_FPS = CFG["fps"]
RTSP_BITRATE = CFG["bitrate"]
RTSP_MAXRATE = CFG["maxrate"]
RTSP_BUFSIZE = CFG["bufsize"]

# =========================================================
# KONFIGURASI KAMERA
# =========================================================
CAMERA_CANDIDATES = ["/dev/video0", "/dev/video1", 0, 1]
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
FPS_REPORT_EVERY = 30

# =========================================================
# KONFIGURASI TRANSMISI — UDP PUSH LANGSUNG (bukan RTSP pull)
# =========================================================
# PENTING: UDP dipilih karena TIDAK ADA retransmit/handshake.
# Di link radio long-range yang lossy, RTSP/TCP akan menumpuk
# buffer saat retransmit paket hilang -> video lag lalu freeze.
# UDP: paket hilang ya sudah dilewati, video tetap jalan.
#
# GANTI IP INI sesuai alamat network HM30 di sisi GCS/ground unit
UDP_TARGET_IP = "192.168.144.30"
UDP_TARGET_PORT = 5600  # pastikan player (VLC/QGC/ffplay) dengar di port yang sama

# Watchdog: restart FFmpeg cepat kalau prosesnya mati
FFMPEG_WATCHDOG_INTERVAL = 2.0

# =========================================================
# FUNGSI KAMERA
# =========================================================

def open_camera():
    for cam in CAMERA_CANDIDATES:
        print(f"[CAMERA] Mencoba membuka kamera: {cam}")
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)

        if not cap.isOpened():
            cap.release()
            continue

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(0.5)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[CAMERA] Kamera aktif: {cam} ({frame.shape[1]}x{frame.shape[0]})")
            return cap

        cap.release()

    return None

# =========================================================
# FFMPEG: UDP PUSH + INTRA-REFRESH
# =========================================================

def build_ffmpeg_cmd():
    # Ekstrak angka bitrate/bufsize (tanpa 'k') untuk vbv params
    br_num = RTSP_BITRATE.replace("k", "")
    buf_num = RTSP_BUFSIZE.replace("k", "")

    # --- INTRA-REFRESH: pengganti keyframe (IDR) besar tiap N detik.
    #     Refresh disebar sedikit-sedikit tiap frame, jadi kalau ada
    #     paket hilang di radio, kerusakan gambar cuma sebagian kecil
    #     dan langsung pulih di frame berikutnya -- bukan freeze/blocky
    #     sampai keyframe besar berikutnya berhasil terkirim utuh.
    x264_params = (
        "intra-refresh=1:"
        "scenecut=0:"
        f"vbv-maxrate={br_num}:"
        f"vbv-bufsize={buf_num}:"
        "aud=1"  # access unit delimiter, bantu decoder re-sync tiap frame
    )

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{RTSP_WIDTH}x{RTSP_HEIGHT}",
        "-r", str(RTSP_FPS),
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-profile:v", "baseline",   # paling ringan & paling kompatibel
        "-level", "3.0",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-b:v", RTSP_BITRATE,
        "-maxrate", RTSP_MAXRATE,
        "-bufsize", RTSP_BUFSIZE,
        "-x264-params", x264_params,
        # --- MPEG-TS di atas UDP: tidak perlu client "connect" dulu
        #     seperti RTSP. Mini PC langsung "siar" ke IP:PORT ground unit.
        "-f", "mpegts",
        f"udp://{UDP_TARGET_IP}:{UDP_TARGET_PORT}?pkt_size=1316",
    ]


def start_ffmpeg():
    print(f"[STREAM] Menjalankan FFmpeg -> udp://{UDP_TARGET_IP}:{UDP_TARGET_PORT}")
    try:
        proc = subprocess.Popen(
            build_ffmpeg_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        return proc
    except Exception as e:
        print(f"[STREAM ERROR] Gagal menjalankan FFmpeg: {e}")
        return None


def stop_ffmpeg(proc):
    if proc is None:
        return
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
# PRODUCER-CONSUMER: kamera tidak pernah nge-block
# =========================================================
# frame_queue hanya menampung 1 frame TERBARU. Kalau consumer
# (pengirim ke ffmpeg) sempat telat, frame lama otomatis dibuang,
# bukan diantre -- supaya video yang tampil selalu real-time,
# tidak lag mengejar ketinggalan.

frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()


def camera_capture_worker(cap):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        if frame.shape[1] != RTSP_WIDTH or frame.shape[0] != RTSP_HEIGHT:
            frame = cv2.resize(
                frame, (RTSP_WIDTH, RTSP_HEIGHT), interpolation=cv2.INTER_AREA
            )

        # Buang frame lama kalau consumer belum sempat ambil, lalu isi yang baru
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)


def ffmpeg_sender_worker(state):
    frame_interval = 1.0 / RTSP_FPS
    last_send = 0.0

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        now = time.perf_counter()
        if now - last_send < frame_interval:
            continue
        last_send = now

        proc = state["proc"]
        if proc is None or proc.poll() is not None:
            continue

        try:
            proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            pass  # watchdog thread yang akan restart ffmpeg


def watchdog_worker(state):
    while not stop_event.is_set():
        time.sleep(FFMPEG_WATCHDOG_INTERVAL)
        proc = state["proc"]
        if proc is None or proc.poll() is not None:
            print("[WATCHDOG] FFmpeg mati/tidak jalan, restart...")
            stop_ffmpeg(proc)
            state["proc"] = start_ffmpeg()


# =========================================================
# MAIN
# =========================================================

def main():
    print(">>> HM30 LONG-RANGE VIDEO STREAM (UDP + intra-refresh, anti-putus) <<<")
    print(f"[CONFIG] Preset aktif: {ACTIVE_PRESET} | "
          f"{RTSP_WIDTH}x{RTSP_HEIGHT} @ {RTSP_FPS}fps | {RTSP_BITRATE}")

    cap = open_camera()
    if cap is None:
        print("[ERROR] Kamera gagal dibuka.")
        sys.exit(1)

    state = {"proc": start_ffmpeg()}

    threads = [
        threading.Thread(target=camera_capture_worker, args=(cap,), daemon=True),
        threading.Thread(target=ffmpeg_sender_worker, args=(state,), daemon=True),
        threading.Thread(target=watchdog_worker, args=(state,), daemon=True),
    ]
    for t in threads:
        t.start()

    frame_count = 0
    fps_timer = time.perf_counter()

    try:
        while True:
            time.sleep(1.0)
            frame_count += RTSP_FPS  # perkiraan kasar untuk laporan berkala
            if time.perf_counter() - fps_timer >= 5.0:
                proc = state["proc"]
                status = "ON" if (proc and proc.poll() is None) else "DEAD"
                print(f"[STATUS] FFmpeg: {status} | Target: udp://{UDP_TARGET_IP}:{UDP_TARGET_PORT}")
                fps_timer = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan manual oleh user.")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
        stop_ffmpeg(state["proc"])
        cap.release()
        print("[INFO] Shutdown selesai.")


if __name__ == "__main__":
    main()
