# MAVSDK-Python version of aerothon drone survey code
import asyncio
import time
import math
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from mavsdk.geofence import Point, Polygon, FenceType, GeofenceData, Circle
from shapely.geometry import Polygon as ShapelyPolygon , Point as ShapelyPoint, LineString
import logging
import matplotlib.pyplot as plt

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# ==== PARAMETERS ====
ALTITUDE = 10            # meters
ZIGZAG_SPACING = 10      # meters
CONNECTION_STRING = '/dev/ttyACM0'  # Serial connection for Cube Orange Plus

# ==== USER-DEFINED POLYGON ====                             
polygon_coords = [
    (23.176972017877027, 80.0214979056397),
    (23.177003843167036, 80.02157435296434),
    (23.176516325907812, 80.02129451811668),
    (23.176215899258498, 80.02133952651874),
    (23.175877693600867, 80.02141388822652),
    (23.17554986871377, 80.02171611689656),
    (23.175670556270056, 80.02263863615299),
    (23.176094592767043, 80.0226084768696),
    (23.17663279101092, 80.02242397304711),
    (23.177035622818686, 80.0222199543654),
    (23.177117167571478, 80.02189352447466),
    (23.177105751309075, 80.02191481338059)
    ]

waypoints=[(23.17657991601625, 80.02222342579057),
           (23.17657991601625, 80.02222342579057),
           (23.176801217534045, 80.0221594820308),
           (23.176624867916676, 80.02200526472784),
           (23.176716500589993, 80.02207296988523),
           (23.17668365114828, 80.02220273810359),
           (23.17657991601625, 80.0222253064894),
           (23.176626596835614, 80.02200902612546),
           (23.176652530617115, 80.02192251397989),
           (23.176851356108326, 80.022003384029),
           (23.176901494663813, 80.02186045091892),
           (23.176851356108326, 80.022003384029),
           (23.17676491027887, 80.02197141214913),
           (23.176816777783237, 80.02181531414733),
           (23.17676491027887, 80.02197141214913),
           (23.176652530617115, 80.02192251397989),
           (23.176707856000853, 80.02177581947217),
           (23.176759723527326, 80.0215896302893),
           (23.1767960307839, 80.02155389701178),
           (23.176849627192272, 80.0215557777106),
           (23.176880747677604, 80.02158398819286),
           (23.176849627192272, 80.0215557777106),
           (23.1767960307839, 80.02155389701178),
           (23.176759723527326, 80.0215896302893),
           (23.176761452444527, 80.02168554592896),
           (23.176761452444527, 80.02168742662778),
           (23.176801217534045, 80.02174948968873),
           (23.176877289846264, 80.02177581947217),
           (23.1769412597117, 80.02175325108638),
           (23.17694990428577, 80.02168742662778),
           (23.17694298862656, 80.02164040915736)
           ]
           
           
           

class AerothonDrone:
    def __init__(self):
        self.drone = System()
        self.home_position = None

    async def connect_vehicle(self):
        """Connect to the vehicle via serial connection"""
        print("Connecting to vehicle...")
        connection_string = f"serial://{CONNECTION_STRING}:57600"
        #await self.drone.connect(system_address=connection_string)
        
        await self.drone.connect(system_address="udpin://0.0.0.0:14550")
        
        
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


    async def upload_geofence(self, fence_coords):
        """Upload polygon geofence with RTL action"""
        print("Uploading new geofence...")
        try:
            # Clear existing geofence
            await self.drone.geofence.clear_geofence()
            
            # Set fence parameters (ArduPilot/PX4 specific)
            await self.drone.param.set_param_int("FENCE_ENABLE", 1)
            await self.drone.param.set_param_int("FENCE_ACTION", 1)
            await self.drone.param.set_param_float("FENCE_ALT_MAX", 15.0)
            
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

    def generate_zigzag_waypoints(self, polygon, spacing_m):
        """Generate zigzag survey pattern within polygon"""
        poly = ShapelyPolygon([(lon, lat) for lat, lon in polygon])
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        lat_spacing = spacing_m / 111139
        waypoints, current_lat, toggle = [], min_lat, False
        while current_lat <= max_lat:
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
            current_lat += lat_spacing
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
                speed_m_s=2.0,
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
        """Arm vehicle and takeoff to target altitude"""
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
            print(f"Altitude: {altitude:.2f}m")
            if altitude >= target_alt * 0.95:
                print("Reached target altitude!")
                break
            #await asyncio.sleep(1)

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
        plt.show()


    async def run_survey_mission(self):
        """Main function to run the complete survey mission"""
        try:
            await self.connect_vehicle()
            
            await self.clear_existing_geofence()

            #await self.upload_geofence(polygon_coords)
            
            #inner_polygon = self.shrink_polygon(polygon_coords, 1)
            #print(f"Generated inner polygon with {len(inner_polygon)} points")

            #waypoints = self.generate_zigzag_waypoints(inner_polygon, ZIGZAG_SPACING)
            #print(f"Generated {len(waypoints)} survey waypoints")

            #Visualize the final plan before uploading/starting the mission
            
            
            #self.plot_waypoints_map(
            #   polygon_coords,
            #   polygob_coords,
            #   waypoints,
            #   title="Aerothon Survey Plan (Fence, Shrunk, Waypoints)" 
            #)
            
            
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
    drone_survey = AerothonDrone()
    await drone_survey.run_survey_mission()

if __name__ == "__main__":
    asyncio.run(main())
