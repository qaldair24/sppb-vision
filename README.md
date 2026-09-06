# SPPB Vision — 5xSTS Assessment System

A Python-based computer vision and IMU system designed to automate the **Five Times Sit-to-Stand (5xSTS)** assessment using a monocular camera and a lumbar inertial measurement unit (IMU).

The system combines real-time human pose estimation, movement detection, sensor data acquisition, signal processing, and event fusion to identify the different phases of the Sit-to-Stand movement and calculate performance metrics.

> **Note:** This version evaluates only the chair-stand component. It does not calculate a complete SPPB score.

---

## Overview

The Five Times Sit-to-Stand test measures how long it takes a participant to stand up and sit down five times.

This project was developed to automate the collection and analysis of the test using two independent data sources:

- A monocular camera for human pose and joint-angle analysis
- A lumbar IMU positioned approximately at the L5 region

The system processes both sources and compares their detected movement events to generate structured results for further analysis.

---

## Main Features

- Real-time video capture with OpenCV
- Human pose estimation using MediaPipe
- Knee-angle based posture detection
- Automatic detection of Sit-to-Stand repetitions
- Lumbar IMU data acquisition from an ESP32
- IMU signal processing and movement-event detection
- Camera and IMU event comparison
- Multimodal event fusion
- Automatic timing of the five repetitions
- Camera quality metrics
- IMU data quality metrics
- Input validation and error handling
- Spanish voice instructions
- Automated generation of CSV, JSON, graphs, and annotated video outputs
- Utilities for testing the camera and IMU
- Version validation for reproducible data collection

---

## Technologies

The project is primarily written in **Python**.

Main technologies and libraries include:

- Python 3.12
- OpenCV
- MediaPipe
- NumPy
- Pandas
- SciPy
- Matplotlib
- ReportLab
- Pillow
- Tkinter
- ESP32
- MPU6050 IMU
- UDP communication

---

## System Architecture

The application follows a multimodal processing workflow:

```text
               ┌─────────────────┐
               │   Participant   │
               └────────┬────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      Monocular Camera        Lumbar IMU
             │                     │
             ▼                     ▼
         OpenCV +              ESP32 +
         MediaPipe             MPU6050
             │                     │
             ▼                     ▼
      Pose estimation          IMU samples
             │                     │
             ▼                     ▼
      Joint-angle analysis    Signal processing
             │                     │
             ▼                     ▼
       Camera events           IMU events
             │                     │
             └──────────┬──────────┘
                        ▼
                   Event Fusion
                        │
                        ▼
               Performance Metrics
                        │
                        ▼
           CSV / JSON / Graphs / Video
```

---

## Detected Events

The 5xSTS movement is represented using a sequence of events.

| Event | Description |
|---|---|
| `GO` | Start signal |
| `MO` | Beginning of an upward movement |
| `R` | Standing position confirmed |
| `D` | Beginning of a downward movement |
| `S` | Seated position confirmed |
| `ABORT` | Test manually terminated before completion |

The expected sequence is approximately:

```text
GO
MO1 → R1 → D1 → S1
MO2 → R2 → D2 → S2
MO3 → R3 → D3 → S3
MO4 → R4 → D4 → S4
MO5 → R5
```

The fifth repetition ends when the final standing position (`R5`) is confirmed.

---

## Camera Analysis

The camera module uses MediaPipe Pose to estimate body landmarks.

Joint positions are used to calculate movement-related angles and determine transitions between seated and standing positions.

The system also records technical quality information such as:

- Effective camera frame rate
- Maximum gap between frames
- Percentage of frames where a pose was detected

---

## IMU Analysis

An MPU6050 inertial sensor is positioned around the lumbar L5 region.

The IMU communicates through an ESP32 and transmits measurements to the computer over UDP.

Default acquisition configuration:

```text
Nominal IMU frequency: 50 Hz
UDP port: 4212
Sensor: MPU6050
Board: ESP32-S3
SDA: GPIO 8
SCL: GPIO 9
```

The IMU processing pipeline analyzes lumbar motion to independently detect Sit-to-Stand events.

---

## Camera + IMU Fusion

Camera and IMU events are compared using a fixed temporal tolerance.

Events may be classified as:

- `acuerdo_camara_imu` — camera and IMU detected compatible events
- `solo_camara` — detected only by the camera
- `solo_imu` — detected only by the IMU
- `conflicto_rechazado` — both detected an event but with excessive temporal difference
- `no_detectado` — neither system detected the event

This allows the system to analyze agreement between two independent sensing methods.

---

## Metrics

For a completed five-repetition test, the software can calculate metrics including:

- Total clinical time
- Effective movement time
- Reaction time
- Duration of each upward movement
- Duration of each downward movement
- Standing pauses
- Seated pauses
- Number of detected repetitions
- Camera/IMU agreement
- Technical data-quality measurements

The system can also calculate the chair-stand subscore from **0 to 4** when the test is successfully completed.

It does **not** calculate a complete 0–12 SPPB score.

---

## Project Structure

```text
.
├── main.py
├── app_recoleccion.py
├── configuracion_v4.py
├── analisis_imu_sts.py
├── imu_lumbar.py
├── metricas_sts.py
├── monitor_imu_en_vivo.py
├── voz_espanol.py
│
├── PROBAR_CAMARA_MAC.py
├── PROBAR_IMU_L5.py
├── VALIDAR_ULTIMA_PRUEBA_IMU.py
├── VALIDAR_VERSION_V4.py
│
├── MAESTRO_L5_STS.ino
│
├── requirements.txt
├── VERSION.txt
├── INSTRUCCIONES.md
├── CHECKLIST_PRECOLECCION.md
├── DICCIONARIO_DATOS_V4.md
└── README.md
```

### Important Files

**`main.py`**  
Main processing pipeline for camera acquisition, pose estimation, movement detection, IMU integration, and result generation.

**`app_recoleccion.py`**  
Graphical interface used to configure and start a 5xSTS data collection session.

**`analisis_imu_sts.py`**  
Processes lumbar IMU signals and detects movement events.

**`imu_lumbar.py`**  
Handles communication and data acquisition from the lumbar IMU.

**`metricas_sts.py`**  
Calculates repetition-level metrics, camera quality measurements, event fusion, and chair-stand scoring.

**`configuracion_v4.py`**  
Contains version-controlled parameters and thresholds used by the movement detection algorithm.

**`monitor_imu_en_vivo.py`**  
Provides live monitoring of IMU communication.

**`MAESTRO_L5_STS.ino`**  
ESP32 firmware used for MPU6050 acquisition and communication.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The current frozen environment uses:

```text
mediapipe==0.10.21
numpy==1.26.4
opencv-contrib-python==4.11.0.86
pandas==2.2.3
matplotlib==3.10.1
scipy==1.15.2
reportlab==4.3.1
Pillow==11.1.0
```

---

## Running the Application

Start the collection interface with:

```bash
python app_recoleccion.py
```

The interface allows the user to configure:

- Anonymous participant ID
- Trial identifier
- Chair height
- IMU acquisition
- Spanish voice instructions

Before collecting data, the camera and IMU can be checked using the included validation utilities.

### Test the camera

```bash
python PROBAR_CAMARA_MAC.py
```

### Test the IMU

```bash
python PROBAR_IMU_L5.py
```

### Validate the software version

```bash
python VALIDAR_VERSION_V4.py
```

---

## Hardware Setup

The intended setup includes:

- Computer running Python
- Monocular camera
- ESP32-S3
- MPU6050 IMU
- Stable chair
- Lumbar placement of the IMU approximately at L5

For camera-based analysis, the participant should remain visible from a lateral view, including the head, shoulder, hip, knee, and ankle landmarks.

---

## Output

Each trial can generate several output files containing the raw and processed information.

Examples include:

```text
video_original_*.mp4
video_anotado_*.mp4
eventos_camara_*.csv
timeline_video_*.csv
imu_l5_*.csv
eventos_imu_l5_*.csv
eventos_fusion_*.csv
concordancia_camara_imu_*.csv
repeticiones_*.csv
resumen_sts_*.csv
metadata_*.json
grafica_*.png
```

The metadata file stores information about the software environment to improve reproducibility.

---

## Data Privacy

Participant identifiers should be anonymous and should not contain personally identifiable information.

Generated participant recordings and study results are intentionally not included in this public repository.

---

## Reproducibility

The project uses version-controlled algorithm parameters and fixed dependency versions.

Current system version:

```text
5xSTS-VISION 4.1.0-MAC-M1
```

Algorithm version:

```text
5xSTS-CAM-IMU-L5-2026.07
```

The frozen version was prepared for:

```text
MacBook Air M1
macOS Sonoma
Python 3.12
Apple Silicon / ARM64
```

Changing detection thresholds, fusion rules, or event definitions should result in a new algorithm version to preserve reproducibility.

---

## My Contributions

> **IMPORTANT: Replace this section with your actual contributions before submitting this repository as a code sample.**

This project was developed collaboratively. My contributions focused on:

- [Describe the Python modules or features you personally worked on]
- [Describe any computer vision work you contributed]
- [Describe any IMU, ESP32, or data-processing work you contributed]
- [Describe debugging, testing, validation, or integration work you performed]

Through this project, I gained experience integrating software, computer vision, sensor data, and real-time processing into a complete application.

---

## What I Learned

Working on this project strengthened my understanding of software engineering beyond writing individual scripts.

I gained experience with:

- Structuring a multi-module Python application
- Integrating third-party libraries
- Processing real-time camera data
- Working with hardware and sensor communication
- Processing and validating time-series data
- Implementing error handling and input validation
- Debugging interactions between hardware and software
- Designing reproducible data-processing workflows
- Working with Git and version-controlled software
- Turning technical requirements into a functioning end-to-end system

One of the most valuable lessons was learning how different components — computer vision, sensor acquisition, signal processing, user interfaces, and data storage — need to work together reliably in a real application.

---

## Scope and Disclaimer

This repository is a software and research-oriented implementation of an automated 5xSTS measurement system.

It should not be considered a standalone medical diagnostic system or a substitute for professional clinical assessment.

The camera–IMU comparison implemented here measures technical agreement between the two sensing methods and should not by itself be interpreted as clinical validation.

---

## License

No open-source license has currently been assigned to this repository.
