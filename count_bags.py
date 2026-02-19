import os
os.environ["FLAGS_use_mkldnn"] = "False"

from ultralytics import YOLO
import cv2
import re
import numpy as np
from collections import Counter
from paddleocr import PaddleOCR
from openpyxl import Workbook, load_workbook
from datetime import datetime

# ----------------------------
# PATHS
# ----------------------------
BASE_PATH = r"D:\Projects\Age group\IFFCO"
MODEL_PATH = os.path.join(BASE_PATH, "best (3).pt")
VIDEO_PATH = os.path.join(BASE_PATH, "clipped_wagon_video.avi")
EXCEL_PATH = os.path.join(BASE_PATH, "wagon_data.xlsx")

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow("IFFCO Cement Monitoring", cv2.WINDOW_NORMAL)

# ----------------------------
# OCR
# ----------------------------
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# ----------------------------
# EXCEL SETUP
# ----------------------------
if not os.path.exists(EXCEL_PATH):
    wb = Workbook()
    ws = wb.active
    ws.append([
        "Date", "Time", "Wagon Number",
        "Door1 Count", "Door2 Count",
        "NP Bags", "DAP Bags",
        "Total"
    ])
    wb.save(EXCEL_PATH)

# ----------------------------
# COLOR DETECTION
# ----------------------------
def detect_product_color(crop):

    h, w = crop.shape[:2]
    crop = crop[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation_mask = hsv[:, :, 1] > 20

    lower_green = np.array([40, 60, 50])
    upper_green = np.array([85, 255, 255])

    lower_blue = np.array([90, 20, 40])
    upper_blue = np.array([145, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    green_pixels = cv2.countNonZero(green_mask & saturation_mask)
    blue_pixels = cv2.countNonZero(blue_mask & saturation_mask)

    total_pixels = crop.shape[0] * crop.shape[1]

    if green_pixels / total_pixels > 0.20:
        return "DAP"
    elif blue_pixels / total_pixels > 0.05:
        return "NP"
    else:
        return "Unknown"

# ----------------------------
# COUNTERS
# ----------------------------
door1_count = 0
door2_count = 0
np_count = 0
dap_count = 0

counted_d1 = set()
counted_d2 = set()

wagon_number_text = ""
wagon_detected = False
wagon_candidates = []

print("Processing...")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.30,
        imgsz=960
    )

    doors = []

    if results[0].boxes.id is not None:

        # ---------------- PASS 1: COLLECT DOORS ----------------
        for box, track_id, cls in zip(
            results[0].boxes.xyxy,
            results[0].boxes.id,
            results[0].boxes.cls
        ):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)

            if class_id == 1:  # wagon_door
                doors.append((x1, y1, x2, y2))

        # Sort doors left → right
        doors = sorted(doors, key=lambda d: d[0])

        # Draw door labels
        for i, (dx1, dy1, dx2, dy2) in enumerate(doors):

            if i == 0:
                color = (0, 255, 0)       # D1 green
            else:
                color = (0, 165, 255)     # D2 orange

            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 3)

            cv2.putText(
                frame,
                f"D{i+1}",
                (dx1, dy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

        # ---------------- PASS 2: PROCESS OBJECTS ----------------
        for box, track_id, cls in zip(
            results[0].boxes.xyxy,
            results[0].boxes.id,
            results[0].boxes.cls
        ):

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            class_id = int(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)

            # ---------------- BAG LOGIC ----------------
            if class_id == 0:

                center = ((x1 + x2)//2, (y1 + y2)//2)
                bag_crop = frame[y1:y2, x1:x2]
                product_type = detect_product_color(bag_crop)

                # Door 1
                if len(doors) > 0:
                    dx1, dy1, dx2, dy2 = doors[0]
                    if dx1 <= center[0] <= dx2 and dy1 <= center[1] <= dy2:
                        if track_id not in counted_d1:
                            counted_d1.add(track_id)
                            door1_count += 1
                            if product_type == "NP":
                                np_count += 1
                            elif product_type == "DAP":
                                dap_count += 1

                # Door 2
                if len(doors) > 1:
                    dx1, dy1, dx2, dy2 = doors[1]
                    if dx1 <= center[0] <= dx2 and dy1 <= center[1] <= dy2:
                        if track_id not in counted_d2:
                            counted_d2.add(track_id)
                            door2_count += 1
                            if product_type == "NP":
                                np_count += 1
                            elif product_type == "DAP":
                                dap_count += 1

            # ---------------- ROBUST WAGON OCR (DUAL-PIPELINE) ----------------
            if class_id == 2 and not wagon_detected:

                # Run every 5th frame to save FPS (running 2 OCRs is expensive)
                if frame_count % 5 != 0:
                    continue

                pad = 70
                # Ensure crop is safe
                y_min, y_max = max(0, y1-pad), min(frame.shape[0], y2+pad)
                x_min, x_max = max(0, x1-pad), min(frame.shape[1], x2+pad)
                crop = frame[y_min:y_max, x_min:x_max]

                if crop.size == 0: continue

                # --- PIPELINE A: DETAIL ORIENTED (Best for Video 2 / Thin Text) ---
                # Strategy: Upscale + CLAHE to enhance local contrast of thin lines
                img_A = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                input_A = clahe.apply(gray_A)

                # --- PIPELINE B: SHAPE ORIENTED (Best for Video 1 / Thick Hazy Text) ---
                # Strategy: Downscale (0.8) to prevent blobbing + Invert + Otsu
                img_B = cv2.resize(crop, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
                gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
                # Blur slightly to smooth edges before thresholding
                blurred_B = cv2.GaussianBlur(gray_B, (3,3), 0)
                _, input_B = cv2.threshold(blurred_B, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # --- COMPETITION: Run Both & Pick the Winner ---
                attempts = [("Detail", input_A), ("Aggressive", input_B)]
                
                best_conf = 0.0
                best_tokens = []

                for name, img_input in attempts:
                    # Optional: View what the AI sees
                    # cv2.imshow(f"Pipeline {name}", img_input)

                    result = ocr.ocr(img_input, cls=True)
                    
                    current_conf = 0.0
                    current_tokens = []
                    count = 0

                    if result and result[0]:
                        for line in result[0]:
                            text = line[1][0].upper()
                            conf = line[1][1]
                            
                            # Clean special chars
                            text = re.sub(r'[^A-Z0-9 ]', '', text)
                            tokens = text.split()
                            
                            for token in tokens:
                                # SMART FILTER: Ignore Hindi hallucinations
                                if len(token) >= 3:
                                    is_valid = False
                                    if any(c.isdigit() for c in token): # It's a number
                                        is_valid = True
                                    elif token in ["BCNA", "BOXN", "BHL", "HSM"]: # It's a code
                                        is_valid = True
                                    
                                    if is_valid:
                                        current_tokens.append(token)
                                        current_conf += conf
                                        count += 1
                    
                    # Calculate score for this pipeline
                    if count > 0:
                        avg_conf = current_conf / count
                        # Check if this pipeline is better than the previous best
                        if avg_conf > best_conf:
                            best_conf = avg_conf
                            best_tokens = current_tokens
                
                # --- PROCESS WINNER ---
                if best_tokens:
                    # Add winning tokens to the candidate list
                    for token in best_tokens:
                        wagon_candidates.append(token)

                # --- STABILIZATION ---
                buffer_size = 60
                if len(wagon_candidates) > buffer_size:
                    recent = wagon_candidates[-buffer_size:]
                    counter = Counter(recent)
                    
                    # 1. Gather all tokens that meet the stability threshold
                    stable_tokens = [token for token, count in counter.items() if count >= 5]

                    # Lock if we have at least 2 parts
                    if len(stable_tokens) >= 2:
                        
                        # 2. Sort by LAST appearance in the buffer.
                        # This overrides early blurry mistakes and trusts the spatial order of the most recent frame.
                        stable_tokens.sort(key=lambda x: len(recent) - recent[::-1].index(x))
                        
                        # 3. SMART FORMATTER: Separate into Codes (contains letters) and Numbers (pure digits)
                        codes = [t for t in stable_tokens if any(c.isalpha() for c in t)]
                        nums = [t for t in stable_tokens if not any(c.isalpha() for c in t)]
                        
                        # 4. Interleave them to force the "BCNA 310925 HSM1 17725" pattern
                        formatted_parts = []
                        for i in range(max(len(codes), len(nums))):
                            if i < len(codes):
                                formatted_parts.append(codes[i])
                            if i < len(nums):
                                formatted_parts.append(nums[i])
                                
                        wagon_number_text = " ".join(formatted_parts)
                        wagon_detected = True
                        print(f"Wagon Locked: {wagon_number_text} (Conf: {best_conf:.2f})")
                        
    total = door1_count + door2_count

    # ---------------- DISPLAY ----------------
    cv2.putText(frame, f"D1: {door1_count}", (40,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.putText(frame, f"D2: {door2_count}", (40,100),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),3)

    cv2.putText(frame, f"Total: {total}", (40,150),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.putText(frame, f"NP: {np_count}", (40,200),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)

    cv2.putText(frame, f"DAP: {dap_count}", (40,250),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.putText(frame, f"Wagon: {wagon_number_text}", (40,300),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("IFFCO Cement Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- SAVE FINAL DATA ----------------
if wagon_number_text != "":
    now = datetime.now()
    total = door1_count + door2_count

    wb = load_workbook(EXCEL_PATH)
    ws = wb.active

    ws.append([
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        wagon_number_text,
        door1_count,
        door2_count,
        np_count,
        dap_count,
        total
    ])

    wb.save(EXCEL_PATH)
    print("Final Data Saved to Excel ✔")

print("Final Summary")
print("D1:", door1_count)
print("D2:", door2_count)
print("NP:", np_count)
print("DAP:", dap_count)
print("Total:", total)
print("Wagon:", wagon_number_text)
