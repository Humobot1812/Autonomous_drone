import cv2
import time
import socket
import struct
import numpy as np
import threading
import asyncio

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# =====================================================
# ================= SHARED STATE ======================
# =====================================================
vision_lock = threading.Lock()

error_x = 0.0
error_y = 0.0
object_detected = False
centered = False
object_engaged = False   # 🔒 NEW: latch flag

# =====================================================
# ================= CRC ===============================
# =====================================================
def crc16(data: bytes, poly=0x1021, crc=0x0000):
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc

# =====================================================
# ================= SIYI GIMBAL =======================
# =====================================================
GIMBAL_IP = "192.168.144.25"
GIMBAL_PORT = 37260

def cam_control(yaw_angle, pitch_angle):
    yaw = int(yaw_angle * 10)
    pitch = int(pitch_angle * 10)

    frame_wo_crc = struct.pack(
        "<H B H H B hh",
        0x6655, 0x01, 4, 0, 0x0E, yaw, pitch
    )
    frame = frame_wo_crc + struct.pack("<H", crc16(frame_wo_crc))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(frame, (GIMBAL_IP, GIMBAL_PORT))
    sock.close()
    print(f"[GIMBAL] Yaw={yaw_angle:.2f}°, Pitch={pitch_angle:.2f}°")

# =====================================================
# ================= VISION THREAD =====================
# =====================================================
def vision_thread():

    global error_x, error_y, object_detected, centered, object_engaged

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    yaw_angle = 0.0
    pitch_angle = -45.0
    cam_control(yaw_angle, pitch_angle)

    YAW_MIN, YAW_MAX = -90, 90
    PITCH_MIN, PITCH_MAX = -90, -45

    KP_YAW = 0.05
    KP_PITCH = 0.05
    DEADZONE = 5

    lower_g1 = np.array([35, 80, 80])
    upper_g1 = np.array([55, 255, 255])
    lower_g2 = np.array([56, 80, 80])
    upper_g2 = np.array([75, 255, 255])

    MIN_AREA = 3000
    LOST_TIMEOUT = 1.5
    last_seen_time = time.time()
    local_object_present = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cx_f, cy_f = w // 2, h // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_g1, upper_g1),
            cv2.inRange(hsv, lower_g2, upper_g2)
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False

        for c in contours:
            if cv2.contourArea(c) > MIN_AREA:
                x, y, bw, bh = cv2.boundingRect(c)
                cx = x + bw // 2
                cy = y + bh // 2

                ex = cx - cx_f
                ey = cy_f - cy

                if abs(ex) > DEADZONE:
                    yaw_angle += KP_YAW * ex
                if abs(ey) > DEADZONE:
                    pitch_angle += KP_PITCH * ey

                yaw_angle = np.clip(yaw_angle, YAW_MIN, YAW_MAX)
                pitch_angle = np.clip(pitch_angle, PITCH_MIN, PITCH_MAX)

                cam_control(yaw_angle, pitch_angle)

                with vision_lock:
                    error_x = ex
                    error_y = ey

                    if not object_engaged:
                        object_detected = True
                        centered = abs(ex) < DEADZONE and abs(ey) < DEADZONE
                    else:
                        object_detected = False
                        centered = False

                last_seen_time = time.time()
                local_object_present = True
                found = True

                cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
                cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
                cv2.putText(frame, f"err_x={ex}, err_y={ey}",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                break

        if not found and local_object_present and (time.time() - last_seen_time) > LOST_TIMEOUT:
            yaw_angle = 0.0
            pitch_angle = -45.0
            cam_control(yaw_angle, pitch_angle)

            with vision_lock:
                object_detected = False
                centered = False
                object_engaged = False   # 🔓 RESET ONLY AFTER OBJECT DISAPPEARS

            local_object_present = False

        cv2.line(frame, (cx_f,0), (cx_f,h), (255,0,0), 1)
        cv2.line(frame, (0,cy_f), (w,cy_f), (255,0,0), 1)

        cv2.imshow("Vision", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# ================= MAVSDK CONTROL ====================
# =====================================================
async def drone_control():

    global object_detected, centered, object_engaged, error_x, error_y

    drone = System()
    await drone.connect(system_address="udp://:14550")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break

    mission_active = True
    offboard_active = False

    while True:
        await asyncio.sleep(0.1)

        with vision_lock:
            detected = object_detected
            ex = error_x
            ey = error_y
            is_centered = centered

        if detected and mission_active:
            print("➡ OFFBOARD")
            await drone.mission.pause_mission()

            try:
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0,0,0,0)
                )
                await drone.offboard.start()
                offboard_active = True
                mission_active = False
            except OffboardError as e:
                print("Offboard failed:", e)

        if offboard_active:
            vx = ey * 0.002
            vy = ex * 0.002

            vx = np.clip(vx, -0.5, 0.5)
            vy = np.clip(vy, -0.5, 0.5)

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(vx, vy, 0, 0)
            )

            if is_centered:
                print("✔ Centered → Resume mission")

                await drone.offboard.stop()
                await drone.mission.start_mission()

                with vision_lock:
                    object_engaged = True   # 🔒 LATCH ENGAGED
                    object_detected = False
                    centered = False

                offboard_active = False
                mission_active = True

# =====================================================
# ================= MAIN ==============================
# =====================================================
if __name__ == "__main__":

    t = threading.Thread(target=vision_thread, daemon=True)
    t.start()

    asyncio.run(drone_control())
