# 🚁 Autonomous Drone System

An **AI-powered autonomous drone system** designed for **human detection in disaster scenarios**, integrating **computer vision, GPS mapping, and autonomous navigation** using MAVSDK and simulation tools.

---

## 📌 Overview

This project focuses on building an intelligent drone capable of:

* 🔍 Detecting humans in disaster zones using computer vision
* 📍 Converting detections into GPS coordinates
* 🧭 Performing autonomous navigation and survey missions
* 🎯 Controlling camera orientation based on detected targets
* 🛩️ Running both **simulation (SITL)** and **real hardware implementations**

---

## 🧠 Key Features

* **Human Detection System**

  * AI-based detection (likely YOLO or similar model)
  * Works on live camera feed

* **Autonomous Navigation**

  * Waypoint-based mission execution
  * Survey area coverage using `.kml` files

* **Camera Tracking**

  * Dynamic camera control based on detection

* **Simulation + Hardware Support**

  * SITL (Software-in-the-loop) simulation
  * Real drone hardware integration using MAVSDK

---

## 📁 Project Structure

```
Autonomous_drone/
│
├── Drone_show/
│   ├── AFC.py
│   ├── Image_1.jpeg ... Image_4.jpeg
│
├── Human_detection_in Disaster/
│   ├── Autonomous_drone_simulation(mavsdk_version).py
│   ├── Autonomous_drone_hardware(mavsdk_version).py
│   ├── Cam_control_according_to_detection.py
│   ├── survey_area.kml
│   ├── surveY.kml
│   │
│   ├── Hardware_videos/
│   ├── Model_testing/
│   └── Simulation_video/
│
└── README.md
```

---

## ⚙️ Technologies Used

* 🐍 Python
* ✈️ MAVSDK
* 📡 MAVLink
* 🧠 Computer Vision (YOLO / Detection Model)
* 🗺️ KML-based GPS survey planning
* 🧪 SITL Simulation

---

## 🚀 How It Works

1. **Define Survey Area**

   * Use `.kml` files to define the region

2. **Drone Mission Start**

   * Drone follows waypoints autonomously

3. **Real-Time Detection**

   * Camera feed processed for human detection

4. **Target Localization**

   * Detected objects converted into GPS coordinates

5. **Camera Adjustment**

   * Camera aligns dynamically toward detected targets

6. **Logging / Output**

   * Results stored or used for further mission decisions

---

## ▶️ How to Run

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/Humobot1812/Autonomous_drone.git
cd Autonomous_drone
```

---

### 🔹 2. Install Dependencies

```bash
pip install mavsdk opencv-python numpy
```

*(Add other dependencies based on your environment)*

---

### 🔹 3. Run Simulation

```bash
cd "Human_detection_in Disaster"
python Autonomous_drone_simulation(mavsdk_version).py
```

---

### 🔹 4. Run on Hardware

```bash
python Autonomous_drone_hardware(mavsdk_version).py
```

---

### 🔹 5. Camera Control Module

```bash
python Cam_control_according_to_detection.py
```

---

## 🎥 Results

The repository includes:

* 📹 Simulation test videos
* 📹 Hardware testing videos
* 🧪 Model testing outputs

These demonstrate:

* Detection accuracy
* Autonomous navigation
* Real-world deployment capability

---

---
📺 YouTube playlist:
👉 https://youtube.com/playlist?list=PLQcgql__dXrco9mI7xbGSEP1FIZtFDI6o&si=sUMdqm7IwxunV-0C


---

## 🧩 Future Improvements

* 🔄 Multi-drone swarm coordination
* 🧠 Improved detection models (YOLOv8 optimization)
* 📡 Real-time ground station integration
* 🗺️ Advanced path planning (ROS 2 + Nav stack)
* 🎯 Precision landing using vision

---

## 📜 License

This project is licensed under the terms of the **MIT License**.

---

## 👨‍💻 Author

**Abhinav Goel**
Robotics & Autonomous Systems Developer 🚀

---
