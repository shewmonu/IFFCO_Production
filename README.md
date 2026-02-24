# IFFCO Cement Loading Monitoring System

An automated computer vision pipeline designed to monitor, count, and log cement bags being loaded into railway wagons. This system uses advanced tracking and OCR techniques to operate effectively in industrial environments with variable lighting and hazy conditions.

## 🚀 Overview

The system uses a custom-trained **YOLOv8** model integrated with the **BoT-SORT** tracking algorithm to detect moving cement bags, railway wagon doors, and wagon number plates. It automatically classifies bags by product type (NP vs. DAP) based on color, tallies them by which door they enter, and extracts the wagon's identification number using a competitive dual-pipeline OCR strategy.

## ✨ Key Features

* **Dynamic Door Assignment:** Automatically detects wagon doors and spatially sorts them from left to right (assigning them as `D1` and `D2`) to track exactly where bags are being loaded.
* **HSV Color Classification:** Extracts crops of detected bags and uses precise HSV color-space masking to classify products on the fly:
  * **NP Bags:** Detected via blue thresholds.
  * **DAP Bags:** Detected via green thresholds.
* **Smart Bag Counting:** Uses tracking IDs to ensure bags are only counted once. A bag is mathematically verified as "loaded" when its center coordinate intersects the bounding box of a detected door.
* **Robust Dual-Pipeline OCR:** To combat difficult environmental conditions (glare, dust, haze), the system runs two parallel image-processing pipelines for text extraction:
  * **Detail Pipeline:** Upscales and applies CLAHE contrast enhancement for reading thin text clearly.
  * **Aggressive Pipeline:** Downscales, blurs, and applies Otsu's Inverse thresholding to prevent letters from morphing into blobs in hazy videos.
  * The system dynamically compares the confidence scores from both pipelines and seamlessly outputs the most accurate result.
* **OCR Stabilization & Formatting:** Buffers the OCR output to prevent flickering, filters out hallucinations, and automatically structures the locked text into a clean `[Code] [Number]` format (e.g., `BCNA 310925`).
* **Automated Data Export:** Automatically generates and updates `wagon_data.xlsx` with timestamped logs of wagon numbers and exact bag counts.

## 🧠 System Architecture & Classes

The system relies on a custom YOLOv8 model (`best (3).pt`) trained to detect three specific classes:
* `0`: `cement_bag` 
* `1`: `wagon_door` 
* `2`: `wagon_number` (number plate)

## ⚙️ Prerequisites & Installation

**1. Install Python Dependencies** Ensure you have Python 3.8+ installed. Install the required libraries using pip:
```bash
pip install ultralytics opencv-python numpy paddleocr paddlepaddle openpyxl
