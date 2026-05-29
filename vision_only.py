from ultralytics import YOLO
from imutils.video import VideoStream
import numpy as np
import cv2
import time
import sys

# ================= K O N F I G U R A S I =================
# PILIH SALAH SATU MODEL UNTUK DI-TEST (Beri tanda # pada yang tidak dipakai)

# Path model yolo11s tflite
MODEL_PATH = "models/yolo11n_416_int8_update.tflite"

CONF_THRESHOLD = 0.60
CAMERA_INDEX = 0
# =========================================================

def main():
    print(">>> MEMULAI PURE VISION BENCHMARK (NCNN vs TFLITE) <<<")
    print(f"[INFO] Model yang digunakan: {MODEL_PATH}")

    # --- 1. PERSIAPAN VISION & WARM-UP AI ---
    print("[INIT] Loading Model AI...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] GAGAL LOAD MODEL! Pastikan path benar. Error: {e}")
        sys.exit()

    print("[INIT] Memanaskan Otak AI (Tunggu 10-60 detik)...")
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        model(dummy_frame, imgsz=416, device='cpu', verbose=False)
        print("[INIT] AI Selesai Pemanasan. Siap Tempur!")
    except Exception as e:
        print(f"[ERROR] Pemanasan AI Gagal: {e}")

    # --- 2. PERSIAPAN KAMERA ---
    print("[INIT] Menyalakan Kamera...")
    vs = VideoStream(src=CAMERA_INDEX, resolution=(640, 480)).start()
    time.sleep(2.0)
    print("[INIT] Kamera Aktif! Memulai Deteksi...\n")
    print("-" * 50)

    # Variabel Pelacak untuk Benchmark
    last_report_time = 0
    frame_count = 0
    start_time_benchmark = time.time()

    try:
        while True:
            # Catat waktu mulai per frame
            frame_start_time = time.time()
            
            frame = vs.read()

            if frame is None:
                print("WARNING: Frame kosong! Mengecek ulang kabel USB kamera...")
                time.sleep(1)
                continue

            # MIKIR: Deteksi YOLO (Sistem akan otomatis mengenali NCNN atau TFLite)
            results = model(frame, conf=CONF_THRESHOLD, imgsz=416, device='cpu', verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                best_box_idx = np.argmax(boxes.conf.cpu().numpy())
                best_box = boxes[best_box_idx]

                conf = float(best_box.conf[0])
                cls_id = int(best_box.cls[0])
                class_name = model.names[cls_id]

                # ANTI-SPAM DETEKSI: Print hasil maksimal 1 kali per 2 detik
                if time.time() - last_report_time > 2.0:
                    print(f"[TARGET DETECTED] {class_name.upper()} | Akurasi: {conf*100:.1f}%")
                    last_report_time = time.time()

            # --- PENGHITUNG FPS PRESISI ---
            frame_count += 1
            if frame_count % 30 == 0: # Hitung rata-rata setiap 30 frame
                elapsed_time = time.time() - start_time_benchmark
                fps = 30 / elapsed_time
                print(f"[BENCHMARK] Kecepatan Rata-rata: {fps:.2f} FPS")
                
                # Reset waktu untuk 30 frame berikutnya
                start_time_benchmark = time.time()


            # Istirahatkan CPU (Dikecilkan jadi 0.01 detik agar tidak terlalu membatasi FPS maksimal)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan Manual oleh User.")
    finally:
        print("[INFO] Membersihkan resource hardware...")
        vs.stop()
        print("[INFO] Sistem Shutdown Mulus.")

if __name__ == "__main__":
    main()