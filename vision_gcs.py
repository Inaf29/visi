from ultralytics import YOLO
from pymavlink import mavutil
from imutils.video import VideoStream
import numpy as np
import cv2
import time
import sys

# ================= K O N F I G U R A S I =================
# KUNCI PERUBAHAN: Koneksi via TCP lokal ke MAVLink-Router
CONNECTION_STRING = 'tcp:127.0.0.1:5760'

# Path model yolo11s tflite
MODEL_PATH = "models/yolo11n_416_int8_update.tflite"

CONF_THRESHOLD = 0.60
CAMERA_INDEX = "/dev/video99" # Set video 99 klo sekalian streaming HUD Mission Planner
#CAMERA_INDEX = 0 # set 0 klo langsung kamera USB tanpa streaming HUD

HEADLESS_MODE = True # Wajib True kalau dijalankan tanpa monitor di Raspi
# =========================================================

# [UPDATE BARU] Fungsi pembaca sensor suhu hardware Raspi
def get_cpu_temperature():
    """Membaca suhu CPU Raspi langsung dari sensor hardware Linux"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.read()
        return float(temp_str) / 1000.0
    except Exception as e:
        print(f"[ERROR] Gagal membaca suhu: {e}")
        return 0.0

def main():
    print(">>> MEMULAI VISION VIA MAVLINK-ROUTER (DETEKSI & LAPOR) <<<")

    # --- 1. KONEKSI KE MAVLINK-ROUTER ---
    print(f"[INIT] Menghubungkan ke Pintu Virtual {CONNECTION_STRING}...")
    try:
        # Kita pakai ID komponen 191 agar GCS tahu ini pesan dari Companion Computer
        master = mavutil.mavlink_connection(CONNECTION_STRING, source_system=1, source_component=191)

        # MAVLink via TCP/UDP butuh "pancingan" detak jantung
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        master.wait_heartbeat(timeout=5)
        print(f"[INIT]  TERHUBUNG ke Jaringan MAVLink!")
    except Exception as e:
        print(f"[ERROR] Gagal Konek MAVLink: {e}")
        sys.exit()

    def send_telemetry(text, severity=6):
        """ Fungsi untuk mengirim teks ke layar Mission Planner """
        try:
            master.mav.statustext_send(severity, text.encode())
            print(f"[TELEM] -> {text}")
        except Exception as e:
            print(f"[ERROR] Gagal kirim telemetri: {e}")

    # --- 2. PERSIAPAN VISION & WARM-UP AI ---
    print(f"[INIT] Loading Model AI {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"GAGAL LOAD MODEL AI! Pastikan file ada di folder models/. Error: {e}")
        sys.exit()

    # TRIK WARM-UP AI (PEMANASAN CPU)
    # Mencegah Raspi nge-hang di frame pertama saat sedang terbang
    print("[INIT] Memanaskan Otak AI (Proses ini wajar, tunggu 10-60 detik)...")
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Bikin gambar hitam kosong
        model(dummy_frame, imgsz=416, device='cpu', verbose=False)
        print("[INIT]  AI Selesai Pemanasan. Siap Tempur!")
    except Exception as e:
        print(f"[ERROR] Pemanasan AI Gagal: {e}")

    print("[INIT] Menyalakan Kamera...")
    vs = VideoStream(src=CAMERA_INDEX, resolution=(640, 480)).start()
    time.sleep(2.0) # Tunggu sensor kamera stabil

    send_telemetry("Mata Pesawat (Vision) AKTIF & TERKONEKSI!", 6)

    # --- 3. LOOPING UTAMA ---
    last_report_time = 0
    frame_count = 0 
    
    # [UPDATE BARU] Variabel kontrol suhu
    last_temp_time = 0      
    TEMP_INTERVAL = 30.0    # Cek dan lapor suhu setiap 10 detik

    try:
        while True:
            # Baca frame
            frame = vs.read()
            
            # Indikator jika kamera diam-diam terputus
            if frame is None:
                print("WARNING: Frame kosong! Mengecek ulang kabel USB kamera...")
                time.sleep(1)
                continue

            # Tanda detak jantung program (muncul setiap 30 frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[DEBUG] Loop AI berjalan lancar... (Telah memproses {frame_count} frame)")

            # Deteksi YOLO (Resolusi 416, CPU)
            results = model(frame, conf=CONF_THRESHOLD, imgsz=416, device='cpu', verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                # Cari deteksi dengan keyakinan tertinggi
                best_box_idx = np.argmax(boxes.conf.cpu().numpy())
                best_box = boxes[best_box_idx]

                conf = float(best_box.conf[0])
                cls_id = int(best_box.cls[0])
                class_name = model.names[cls_id]

                # FITUR ANTI-SPAM (1 pesan per 3 detik via GSM agar bandwidth hemat)
                if time.time() - last_report_time > 3.0:
                    pesan = f"TARGET DETECTED: {class_name} ({conf:.2f})"
                    send_telemetry(pesan, 2) # Severity 2 = Merah/Kuning di MP
                    last_report_time = time.time()

            # =======================================================
            # [UPDATE BARU] FITUR MONITORING SUHU SMART ALERT
            # =======================================================
            if time.time() - last_temp_time > TEMP_INTERVAL:
                suhu = get_cpu_temperature()
                
                # Print log suhu di terminal SSH Raspi
                print(f"[HW STATS] Suhu CPU Raspi: {suhu:.1f} °C")
                
                # Logic warna peringatan untuk Mission Planner
                if suhu >= 80.0:
                    send_telemetry(f"CRITICAL! RASPI OVERHEAT: {suhu:.1f}C", 2)
                elif suhu >= 70.0:
                    send_telemetry(f"WARNING! Raspi Hot: {suhu:.1f}C", 4)
                else:
                    send_telemetry(f"Sys Temp: {suhu:.1f}C", 6)
                
                last_temp_time = time.time()
            # =======================================================

            # (Opsional) Tampilkan gambar jika tes darat pakai monitor
            if not HEADLESS_MODE:
                annotated_frame = results[0].plot()
                cv2.imshow("Raspi View", annotated_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            # Istirahatkan CPU sedikit agar suhu terjaga
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan Manual oleh User.")
    finally:
        print("[INFO] Membersihkan resource hardware...")
        vs.stop()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        print("[INFO] Sistem Shutdown Mulus.")

if __name__ == "__main__":
    main()
