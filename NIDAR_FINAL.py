# MAVSDK-Python version of aerothon drone survey code
import cv2
import socket
import struct
import asyncio
import time
import math
import numpy as np
import threading
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from mavsdk.mission import MissionItem, MissionPlan
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData
from shapely.geometry import Polygon as ShapelyPolygon , Point as ShapelyPoint, LineString
from fastkml import kml
from ultralytics import YOLO


# Load YOLOv11s model (person class included)
yolo_model = YOLO("/home/abhinav/Downloads/best.pt")  


# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

vision_lock = threading.Lock()

error_x = 0.0
error_y = 0.0
object_detected = False
centered = False
object_engaged = False   # 🔒 NEW: latch flag

# ==== PARAMETERS ====
ALTITUDE = 15            # meters
FENCE_ALT=30.0       # meters
ZIGZAG_SPACING = 25     # meters 64
CONNECTION_STRING = '/dev/ttyACM0'  # Serial connection for Cube Orange Plus

OFFBOARD_TIMEOUT = 40.0  # seconds


KML_FILE_PATH = f"/home/abhinav/Documents/Documents/Drone/Main/surveY.kml"  # Path to KML file


# polygon_coords=[(23.176860779846585, 80.0221183906635),
#                 (23.176497691583677, 80.02220781636967),
#                 (23.17648399012051, 80.02179546450226),
#                 (23.17682652627896, 80.02172591117522)]

# polygon_coords=[(23.179217035123223, 80.02054977343217),
#                 (23.17428032306531, 80.02117363534155),
#                 (23.175067510311976, 80.02647034527803),
#                 (23.179947967899658, 80.02546727318845)]      # College IIITDM Jabalpur           

# polygon_coords=[(28.412776953440858, 77.52158654535216),
#                 (28.40822683205534, 77.5237654260262),
#                 (28.410742944008803, 77.52865788201825),
#                 (28.414293617131516, 77.53028870068225)]      # Gautam Buddha University



def load_polygon_from_kml(kml_path):
    """
    Robust KML polygon loader.
    Returns: [(lat, lon), ...]
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # KML namespace
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    # Find all <coordinates> tags
    coord_elements = root.findall(".//kml:coordinates", ns)

    if not coord_elements:
        raise ValueError("❌ No <coordinates> found in KML file")

    polygon_coords = []

    # Take the FIRST polygon's coordinates
    coord_text = coord_elements[0].text.strip()

    for coord in coord_text.split():
        parts = coord.split(",")
        if len(parts) < 2:
            continue

        lon = float(parts[0])
        lat = float(parts[1])

        polygon_coords.append((lat, lon))

    # Remove duplicate last point if polygon is closed
    if len(polygon_coords) > 1 and polygon_coords[0] == polygon_coords[-1]:
        polygon_coords.pop()

    if len(polygon_coords) < 3:
        raise ValueError("❌ Invalid polygon in KML file")

    print(f"✅ Loaded {len(polygon_coords)} polygon points from KML")
    return polygon_coords


def crop_center_square(frame):
    """
    Crop the largest possible center square from the frame.
    """
    h, w = frame.shape[:2]
    side = min(h, w)

    x1 = (w - side) // 2
    y1 = (h - side) // 2

    return frame[y1:y1+side, x1:x1+side]

def vision_thread():

    global error_x, error_y, object_detected, centered, object_engaged

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    DEADZONE = 15                # pixels
    CONF_THRESH = 0.2            # YOLO confidence
    LOST_TIMEOUT = 1.5

    last_seen_time = time.time()
    local_object_present = False

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = crop_center_square(frame)
        h, w = frame.shape[:2]
        cx_f, cy_f = w // 2, h // 2

        # ================= YOLO INFERENCE =================
        results = yolo_model(frame, verbose=False)[0]

        found = False

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO class 0 = person
            if cls == 0 and conf >= CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                ex = cx - cx_f
                ey = cy_f - cy

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

                # ---------- Visualization ----------
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
                cv2.putText(
                    frame,
                    f"Person {conf:.2f} | ex={ex}, ey={ey}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )
                break   # track ONLY one human

        # ---------- Lost target ----------
        if not found and local_object_present and (time.time() - last_seen_time) > LOST_TIMEOUT:
            with vision_lock:
                object_detected = False
                centered = False
                object_engaged = False

            local_object_present = False

        # ---------- Crosshair ----------
        cv2.line(frame, (cx_f,0), (cx_f,h), (255,0,0), 1)
        cv2.line(frame, (0,cy_f), (w,cy_f), (255,0,0), 1)

        cv2.imshow("YOLO Human Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
async def offboard_controller(drone):

    
    global object_detected, centered, object_engaged, error_x, error_y, ALTITUDE

    mission_active = True
    offboard_active = False
    offboard_start_time = None 
    
    try:
        while True:
            await asyncio.sleep(0.1)

            with vision_lock:
                detected = object_detected
                ex = error_x
                ey = error_y
                is_centered = centered

            if detected and mission_active and not offboard_active:
                await drone.mission.pause_mission()
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
                await drone.offboard.start()
                offboard_active = True
                mission_active = False
                offboard_start_time = time.time()   # ⏱️ start timer

            if offboard_active:
                
                if not detected and (time.time() - offboard_start_time) > OFFBOARD_TIMEOUT:
                    print(f"No detection for {OFFBOARD_TIMEOUT}s → resuming mission")

                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0, 0, 0, 0)
                    )
                    await drone.offboard.stop()
                    await asyncio.sleep(1.0)

                    await drone.mission.start_mission()

                    with vision_lock:
                        object_detected = False
                        centered = False
                        object_engaged = False

                    offboard_active = False
                    mission_active = True
                    offboard_start_time = None
                    continue
                
                vx = ey * 0.015
                vy = ex * 0.015
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(vx, vy, 0, 0)
                )

                if is_centered:
                    await drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
                    await asyncio.sleep(0.2)
                    await drone.offboard.stop()
                    # await drone.action.set_flight_mode("HOLD")
                    await asyncio.sleep(1.5)
                    # time.sleep(1.5)  # brief pause
                    await drone.mission.start_mission()
                    async for pos in drone.telemetry.position():
                        lat = pos.latitude_deg
                        lon = pos.longitude_deg
                        print(f"Human_detected at GPS → Lat: {lat:.7f}, Lon: {lon:.7f} , Alt: {ALTITUDE} m")
                        break   # VERY IMPORTANT: exit after first reading
                    with vision_lock:
                        object_engaged = True
                        object_detected = False
                        centered = False
                    offboard_active = False
                    mission_active = True
                    offboard_start_time = None

    except OffboardError as e:
        print(f"❌ Offboard error: {e}")
        try:
            await drone.offboard.stop()
        except:
            pass
        
class NidarDrone:
    def __init__(self):
        self.drone = System()
        self.home_position = None

    async def connect_vehicle(self):
        """Connect to the vehicle via serial connection"""
        print("Connecting to vehicle...")
        connection_string = f"serial://{CONNECTION_STRING}:57600"
        await self.drone.connect(system_address=connection_string)
        
        # await self.drone.connect(system_address="udpin://0.0.0.0:14550")
        
        
        print("Waiting for drone to connect...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected to vehicle!")
                break

        print("Waiting for GPS fix...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("GPS fix obtained!")
                break

        async for position in self.drone.telemetry.home():
            self.home_position = position
            break

    async def clear_existing_geofence(self):
        """Clear any existing geofence"""
        print("Clearing existing geofence...")
        try:
            await self.drone.geofence.clear_geofence()
            print("Existing geofence cleared successfully.")
        except Exception as e:
            print(f"Note: {e}")


    async def upload_geofence(self, fence_coords,FENCE_ALT):
        
        """Upload polygon geofence with RTL action"""
        print("Uploading new geofence...")
        try:
            # Clear existing geofence
            await self.drone.geofence.clear_geofence()
            
            # Set fence parameters (ArduPilot/PX4 specific)
            await self.drone.param.set_param_int("FENCE_ENABLE", 1)
            await self.drone.param.set_param_int("FENCE_ACTION", 1)
            await self.drone.param.set_param_float("FENCE_ALT_MAX", FENCE_ALT)
            
            # Create Point objects from coordinates
            points = []
            for lat, lon in fence_coords:
                points.append(Point(lat, lon))
            
            # Create Polygon with INCLUSION fence type
            polygon = Polygon(points, FenceType.INCLUSION)
            circles = []
            # Create GeofenceData with polygon list and circle list
            geofence_data = GeofenceData([polygon],circles)
            
            # Upload the geofence
            await self.drone.geofence.upload_geofence(geofence_data)
            
            print(f"Geofence uploaded for {len(points)} points.")
        except Exception as e:
            print(f"Geofence upload error: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without geofence protection...")



    def shrink_polygon(self, polygon_coords, inset_m=1):
        """Shrink polygon by specified meters inward"""
        poly = ShapelyPolygon([(lon, lat) for lat, lon in polygon_coords])
        inset_deg = inset_m / 111139
        inner_poly = poly.buffer(-inset_deg)
        if inner_poly.is_empty:
            raise ValueError("Inset too large, no area left.")
        if inner_poly.geom_type == "MultiPolygon":
            inner_poly = max(inner_poly.geoms, key=lambda p: p.area)
        return [(lat, lon) for lon, lat in inner_poly.exterior.coords]

    def generate_zigzag_waypoints(self, polygon, spacing_m, home_lat, home_lon):

        poly = ShapelyPolygon([(lon, lat) for lat, lon in polygon])
        min_lon, min_lat, max_lon, max_lat = poly.bounds

        # --- Degree to meter conversions ---
        mean_lat = (min_lat + max_lat) / 2.0
        lat_spacing = spacing_m / 111139.0
        lon_spacing = spacing_m / (111139.0 * math.cos(math.radians(mean_lat)))

        lat_range_deg = max_lat - min_lat
        lon_range_deg = max_lon - min_lon

        lat_range_m = lat_range_deg * 111139.0
        lon_range_m = lon_range_deg * 111139.0 * math.cos(math.radians(mean_lat))

        # --- Estimate total path lengths ---
        sweeps_lat = lat_range_deg / lat_spacing
        sweeps_lon = lon_range_deg / lon_spacing

        total_dist_lat_sweep = sweeps_lat * lon_range_m
        total_dist_lon_sweep = sweeps_lon * lat_range_m

        waypoints = []
        toggle = False

        # =================================================
        # Choose the sweep that gives LESS total distance
        # =================================================
        if total_dist_lat_sweep <= total_dist_lon_sweep:
            # -------- Sweep along LATITUDE --------
            if abs(home_lat - max_lat) < abs(home_lat - min_lat):
                current_lat = max_lat
                step = -lat_spacing
            else:
                current_lat = min_lat
                step = lat_spacing

            while (current_lat >= min_lat if step < 0 else current_lat <= max_lat):
                line = LineString([(min_lon, current_lat), (max_lon, current_lat)])
                inter = poly.intersection(line)

                if not inter.is_empty:
                    segments = inter.geoms if inter.geom_type == 'MultiLineString' else [inter]
                    for seg in segments:
                        coords = list(seg.coords)
                        if toggle:
                            coords.reverse()
                        for lon_pt, lat_pt in coords:
                            waypoints.append((lat_pt, lon_pt))
                    toggle = not toggle

                current_lat += step

        else:
            # -------- Sweep along LONGITUDE --------
            if abs(home_lon - max_lon) < abs(home_lon - min_lon):
                current_lon = max_lon
                step = -lon_spacing
            else:
                current_lon = min_lon
                step = lon_spacing

            while (current_lon >= min_lon if step < 0 else current_lon <= max_lon):
                line = LineString([(current_lon, min_lat), (current_lon, max_lat)])
                inter = poly.intersection(line)

                if not inter.is_empty:
                    segments = inter.geoms if inter.geom_type == 'MultiLineString' else [inter]
                    for seg in segments:
                        coords = list(seg.coords)
                        if toggle:
                            coords.reverse()
                        for lon_pt, lat_pt in coords:
                            waypoints.append((lat_pt, lon_pt))
                    toggle = not toggle

                current_lon += step

        return waypoints


    async def upload_mission(self, waypoints, altitude):
        """Upload survey mission with waypoints"""
        print("Uploading mission...")
        mission_items = []
        for lat, lon in waypoints:
            mission_items.append(MissionItem(
                latitude_deg=lat,
                longitude_deg=lon,
                relative_altitude_m=altitude,
                speed_m_s=5.0,
                is_fly_through=True,
                gimbal_pitch_deg=float('nan'),
                gimbal_yaw_deg=float('nan'),
                camera_action=MissionItem.CameraAction.NONE,
                loiter_time_s=float('nan'),
                camera_photo_interval_s=float('nan'),
                camera_photo_distance_m=float('nan'),
                acceptance_radius_m=5.0,
                yaw_deg=float('nan'),
                vehicle_action=MissionItem.VehicleAction.NONE
            ))
        await self.drone.mission.upload_mission(MissionPlan(mission_items))
        await self.drone.mission.set_return_to_launch_after_mission(True)
        print(f"Mission uploaded with {len(waypoints)} waypoints.")

    def is_inside_polygon(self, polygon, lat, lon):
        """Check if point is inside polygon"""
        return polygon.contains(ShapelyPoint(lat, lon))

    async def arm_and_takeoff(self, target_alt):
        """Arm vehicle, takeoff, and start vision after reaching altitude"""

        print("Waiting for drone to be armable...")
        async for health in self.drone.telemetry.health():
            if health.is_armable:
                print("Drone is armable!")
                break

        print("Arming drone...")
        await self.drone.action.arm()

        print("Taking off...")
        await self.drone.action.set_takeoff_altitude(target_alt)
        await self.drone.action.takeoff()

        print("Monitoring takeoff progress...")
        async for position in self.drone.telemetry.position():
            altitude = position.relative_altitude_m
            print(f"Altitude: {altitude:.2f} m")

            if altitude >= target_alt * 0.95:
                print("✅ Reached target altitude")

                 # 🚀 START CAMERA + YOLO HERE
                 print("🎥 Starting vision system...")
                 threading.Thread(
                     target=vision_thread,
                     daemon=True
                 ).start()

                break

    async def execute_mission(self):
        """Execute the uploaded mission"""
        print("Starting mission...")
        await self.drone.mission.start_mission()
        print("Monitoring mission progress...")
        async for prog in self.drone.mission.mission_progress():
            print(f"Mission progress: {prog.current}/{prog.total}")
            if prog.current >= prog.total and prog.total > 0:
                print("Mission completed!")
                break 
        print("Mission completed! returning to launch position...")

    async def return_to_launch(self):
        """Return to Launch with monitoring"""
        print("Survey completed! Returning to launch...")
        await self.drone.action.return_to_launch()
        await asyncio.sleep(30)
        print("RTL command sent. Monitoring return...")
    
        # Wait for landing completion
        was_in_air = False
        async for in_air in self.drone.telemetry.in_air():
            if in_air:
                was_in_air = True
                print("Returning to launch...")
        
            if was_in_air and not in_air:
                print("Drone has landed safely!")
                break
            
            await asyncio.sleep(1)

 # ---------- Visualization ----------
    def plot_waypoints_map(self, polygon_coords, inner_polygon, waypoints, title="Survey plan"):
        """
        Plot outer polygon (fence), shrunk inner polygon, and zigzag waypoints
        on lon/lat axes using Matplotlib, numbering each waypoint.
        polygon_coords, inner_polygon, waypoints are lists of (lat, lon).
        """
        # Build Shapely polygons in (x, y) = (lon, lat)
        outer_poly = ShapelyPolygon([(lon, lat) for lat, lon in polygon_coords])
        inner_poly = ShapelyPolygon([(lon, lat) for lat, lon in inner_polygon])

        # Extract exterior rings as arrays for Matplotlib
        ox, oy = outer_poly.exterior.xy
        ix, iy = inner_poly.exterior.xy

        # Waypoints arrays (x=lon, y=lat)
        wp_lons = [lon for lat, lon in waypoints]
        wp_lats = [lat for lat, lon in waypoints]

        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        # Plot boundaries
        ax.plot(ox, oy, linestyle="--", color="k", linewidth=1.5, label="Fence (outer)")
        ax.plot(ix, iy, linestyle="-", color="g", linewidth=2.0, label="Shrunk polygon")

        # Plot path and points
        if len(waypoints) > 0:
            ax.plot(wp_lons, wp_lats, color="tab:blue", linewidth=1.2, label="Zigzag path")
            ax.scatter(wp_lons, wp_lats, s=12, color="tab:blue")

            # Number each waypoint with a small offset to avoid overlap
            for idx, (x, y) in enumerate(zip(wp_lons, wp_lats), start=1):
                ax.annotate(
                    str(idx),
                    xy=(x, y), xycoords='data',
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=16, color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7)
                )

            # Mark start and end
            ax.scatter([wp_lons[0]], [wp_lats[0]], s=70, color="limegreen", marker="o", label="Start")
            ax.scatter([wp_lons[-1]], [wp_lats[-1]], s=70, color="crimson", marker="x", label="End")

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)

        # Equal aspect so lon/lat scales match visually
        ax.set_aspect("equal", adjustable="datalim")

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        plt.tight_layout()
        # plt.show()
        plt.savefig("survey_plan.png", dpi=200)
        plt.close()
        print("📌 Survey plan saved as survey_plan.png")



    async def run_survey_mission(self):
        """Main function to run the complete survey mission"""
        try:
            # global polygon_coords
            global KML_FILE_PATH , FENCE_ALT
            await self.connect_vehicle()
            
            polygon_coords = load_polygon_from_kml(KML_FILE_PATH)
            print(f"Polygon coordinates: {polygon_coords}")
            
            await self.clear_existing_geofence()

            await self.upload_geofence(polygon_coords,FENCE_ALT)
            
            inner_polygon = self.shrink_polygon(polygon_coords, 1)
            print(f"Generated inner polygon with {len(inner_polygon)} points")

            waypoints = self.generate_zigzag_waypoints(inner_polygon, ZIGZAG_SPACING,self.home_position.latitude_deg, self.home_position.longitude_deg)
            print(f"Generated {len(waypoints)} survey waypoints")

            #Visualize the final plan before uploading/starting the mission
            
            
            # self.plot_waypoints_map(
            #     polygon_coords,
            #     inner_polygon,
            #     waypoints,
            #     title="Aerothon Survey Plan (Fence, Shrunk, Waypoints)" 
            # )
            
            
            await self.upload_mission(waypoints, ALTITUDE)
            await self.arm_and_takeoff(ALTITUDE)
            await self.execute_mission()

            # Simple RTL after mission
            # await self.return_to_launch()
            print("Survey completed! Returning to launch...")
            await self.drone.action.return_to_launch()
            await asyncio.sleep(30)
            
            
        except Exception as e:
            print("Error striked!!!!")
        finally:
            print("Mission and RTL sequence completed.")

# ==== MAIN ====
async def main():
    """Main entry point"""
    drone_survey = NidarDrone()
    asyncio.create_task(offboard_controller(drone_survey.drone))
    await drone_survey.run_survey_mission()

if __name__ == "__main__":
    asyncio.run(main())
