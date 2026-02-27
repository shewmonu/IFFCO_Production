🚂 IFFCO Cement Loading Monitoring System












An advanced industrial-grade computer vision pipeline designed to monitor, count, classify, and log cement bags being loaded into railway wagons using a Dual-Model YOLOv8 architecture.

📌 Project Highlights

🟣 Dual-Model Detection + Segmentation Architecture

🚪 Dynamic Door-wise Bag Tracking

🎨 HSV-Based Real-Time Product Classification (NP vs DAP)

🔍 Competitive Dual-Pipeline OCR

🧠 BoT-SORT Multi-Object Tracking

🗄 Automated PostgreSQL Logging

🏭 Designed for Industrial Deployment

🏗 System Architecture
🔷 High-Level Architecture Diagram
<p align="center"> <img src="assets/architecture-diagram.png" alt="IFFCO Cement Monitoring Architecture Diagram" width="750"/> </p>
🔷 Processing Pipeline
Video Input
    ↓
Wagon Segmentation Model (YOLOv8 - Instance Segmentation)
    ↓
Polygon Filtering (Noise Removal)
    ↓
Main Tracking Model (YOLOv8 - Object Detection)
    ↓
BoT-SORT Tracking
    ↓
Door Assignment (D1 / D2)
    ↓
Bag Classification (HSV Color Detection)
    ↓
Dual-Pipeline OCR (Wagon Number Extraction)
    ↓
Final Summary
    ↓
PostgreSQL Database Logging
🧠 Model Architecture

The system uses two custom YOLOv8 models working in tandem.

1️⃣ Main Tracking Model — best (3).pt

Object Detection Model

Class ID	Label
0	cement_bag
1	wagon_door
2	wagon_number

Responsibilities:

Detects bags, doors, and number plates

Provides tracking IDs

Enables door-based counting logic

2️⃣ Wagon Segmentation Model — best (4).pt

Instance Segmentation Model

Class ID	Label
1	wagon_body

Responsibilities:

Generates precise wagon polygon

Eliminates background interference

Restricts detection strictly inside wagon boundary

🔍 Core Feature Breakdown
🟣 Dual-Model Polygon Filtering

A segmentation model isolates the active wagon body.
All detection, tracking, and OCR operations are strictly restricted inside this polygon.

✅ Removes adjacent bay interference
✅ Eliminates background noise
✅ Reduces false positives

🚪 Dynamic Door Assignment

Detects wagon doors automatically

Sorts them left → right

Assigns:

D1

D2

Door-wise bag counting is handled independently.

🎨 Real-Time Product Classification (HSV Based)

Detected bag crops are analyzed in HSV color space:

Product	Detection Logic
🔵 NP	Blue threshold masking
🟢 DAP	Green threshold masking

This allows real-time product classification without retraining the detection model.

🔢 Smart Bag Counting Logic

Uses persistent tracking IDs (BoT-SORT)

Each bag counted only once

Bag considered “loaded” when:

Bag center intersects door bounding box

Prevents:

Double counting

False loading events

🔍 Robust Dual-Pipeline OCR

Designed for industrial lighting variation (dust, haze, glare).

🔹 Detail Pipeline

Upscaling

CLAHE contrast enhancement

Optimized for low-light text

🔹 Morphology Pipeline

Downscaling

Gaussian blur

Otsu inverse thresholding

Prevents character merging

The system dynamically selects the highest-confidence output.

🗄 Automated PostgreSQL Logging

At completion:

Date

Time

Wagon number

Door 1 count

Door 2 count

NP / DAP breakdown

Total bags

All records are securely committed to PostgreSQL.

📂 Project Structure
IFFCO-Cement-Monitoring/
│
├── assets/
│   └── architecture-diagram.png
│
├── count_bags3.py
├── best (3).pt
├── best (4).pt
├── clipped_wagon_video.mp4
├── requirements.txt
└── README.md
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/your-username/IFFCO-Cement-Monitoring.git
cd IFFCO-Cement-Monitoring
2️⃣ Install Dependencies
pip install ultralytics
pip install opencv-python
pip install numpy
pip install paddleocr
pip install paddlepaddle
pip install psycopg2

⚠ CUDA-capable NVIDIA GPU recommended for real-time performance.

🗄 Database Setup

Create database:

CREATE DATABASE wagon_monitoring;

Create table:

CREATE TABLE wagon_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    time TIME NOT NULL,
    wagon_number VARCHAR(50) NOT NULL,
    door1_count INTEGER DEFAULT 0,
    door2_count INTEGER DEFAULT 0,
    np_bags INTEGER DEFAULT 0,
    dap_bags INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0
);

Update credentials in script:

DB_CONFIG = {
    "host": "localhost",
    "database": "wagon_monitoring",
    "user": "postgres",
    "password": "YourPasswordHere",
    "port": "5432"
}
🚀 Usage

Ensure:

Model paths are correct

Video path is correct

PostgreSQL is running

Run:

python count_bags3.py

Press q to safely terminate processing.

📊 Live HUD Display

During processing, the system displays:

🟢 Door 1 Count

🟠 Door 2 Count

🔴 Total Bags

🔵 NP Count

🟢 DAP Count

🔢 Locked Wagon Number

📈 Performance Notes

Optimized for NVIDIA GPU (CUDA)

Robust against haze, glare, dust

Designed for industrial-scale monitoring

Production-ready architecture

👨‍💻 Author

Shew Narayan Ray
Computer Vision & Industrial AI Systems