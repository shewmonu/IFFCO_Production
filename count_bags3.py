import os
os.environ["FLAGS_use_mkldnn"] = "False"

from ultralytics import YOLO
import cv2
import re
import numpy as np
from collections import Counter
from paddleocr import PaddleOCR
from datetime import datetime
import psycopg2

# ----------------------------
# PATHS
# ----------------------------
BASE_PATH = r"D:\Projects\Age group\IFFCO"
MODEL_PATH_MAIN = os.path.join(BASE_PATH, "best (3).pt") # Box model (Bags, Doors, Numbers)
MODEL_PATH_WAGON = os.path.join(BASE_PATH, "best (4).pt") # Seg model (Wagon Polygon)
VIDEO_PATH = os.path.join(BASE_PATH, "clipped_wagon_video.mp4")

# ----------------------------
# LOAD MODELS
# ----------------------------
model_main = YOLO(MODEL_PATH_MAIN)
model_wagon = YOLO(MODEL_PATH_WAGON)
cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow("IFFCO Cement Monitoring", cv2.WINDOW_NORMAL)

# ----------------------------
# OCR & DATABASE SETUP
# ----------------------------
ocr = PaddleOCR(use_angle_cls=True, lang='en')

DB_CONFIG = {
    "host": "localhost",
    "database": "wagon_monitoring",
    "user": "postgres",
    "password": "Shew1252",
    "port": "5432"
}

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

    if green_pixels / total_pixels > 0.20: return "DAP"
    elif blue_pixels / total_pixels > 0.05: return "NP"
    else: return "Unknown"

# ----------------------------
# COUNTERS & STATE
# ----------------------------
door1_count, door2_count, np_count, dap_count = 0, 0, 0, 0
counted_d1, counted_d2 = set(), set()
door_assignments = {}

wagon_number_text = ""
wagon_detected = False
wagon_candidates = []

print("Processing with Optimized Dual-Model Architecture...")

frame_count = 0
wagon_poly = None  # INITIALIZE CACHE OUTSIDE LOOP

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # ========================================================================
    # PASS 0: WAGON POLYGON (Run Segmentation ONLY every 5 frames)
    # ========================================================================
    if frame_count % 5 == 1 or wagon_poly is None:
        results_wagon = model_wagon.predict(frame, conf=0.30, imgsz=640, verbose=False)
        wagon_poly_temp = None
        
        if results_wagon[0].masks is not None:
            for mask_xy, cls in zip(results_wagon[0].masks.xy, results_wagon[0].boxes.cls):
                if int(cls) == 1:  # Class 1 is Wagon in best(4).pt
                    if len(mask_xy) > 0:
                        wagon_poly_temp = np.array(mask_xy, dtype=np.int32)
                        break 
        
        wagon_poly = wagon_poly_temp

    # Draw the cached polygon
    if wagon_poly is not None:
        cv2.polylines(frame, [wagon_poly], isClosed=True, color=(255, 0, 255), thickness=4)
        cv2.putText(frame, "WAGON BOUNDARY", tuple(wagon_poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    # ========================================================================
    # PASS 1 & 2: DOORS, BAGS, AND OCR (Using best(3).pt)
    # ========================================================================
    results = model_main.track(frame, persist=True, tracker="botsort.yaml", conf=0.30, imgsz=640, verbose=False)

    if results[0].boxes.id is not None:
        
        # ---------------- PASS 1: PERSISTENT DOOR ASSIGNMENT ----------------
        current_frame_unassigned = []
        active_doors_this_frame = {}

        for box, track_id, cls in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
            tid, class_id = int(track_id), int(cls)

            if class_id == 1:  # Class 1 is Door in best(3).pt
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # POLYGON FILTER: Only track doors INSIDE the cached wagon
                if wagon_poly is not None and cv2.pointPolygonTest(wagon_poly, center, False) >= 0:
                    if tid not in door_assignments:
                        current_frame_unassigned.append({'tid': tid, 'x1': x1, 'box': (x1, y1, x2, y2)})
                    else:
                        active_doors_this_frame[tid] = {'label': door_assignments[tid], 'box': (x1, y1, x2, y2)}

        if current_frame_unassigned:
            current_frame_unassigned.sort(key=lambda d: d['x1'])
            for d in current_frame_unassigned:
                next_label = f"D{len(door_assignments) + 1}"
                door_assignments[d['tid']] = next_label
                active_doors_this_frame[d['tid']] = {'label': next_label, 'box': d['box']}

        for tid, data in active_doors_this_frame.items():
            dx1, dy1, dx2, dy2 = data['box']
            label = data['label']
            color = (0, 255, 0) if label == "D1" else (0, 165, 255)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 3)
            cv2.putText(frame, label, (dx1, dy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # ---------------- PASS 2: PROCESS BAGS & WAGON OCR ----------------
        for box, track_id, cls in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            tid, class_id = int(track_id), int(cls)

            if class_id != 1: 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)

            # BAG LOGIC
            if class_id == 0:
                center = ((x1 + x2)//2, (y1 + y2)//2)
                bag_crop = frame[y1:y2, x1:x2]
                product_type = detect_product_color(bag_crop) if bag_crop.size > 0 else "Unknown"

                for door_tid, d_data in active_doors_this_frame.items():
                    dx1, dy1, dx2, dy2 = d_data['box']
                    if dx1 <= center[0] <= dx2 and dy1 <= center[1] <= dy2:
                        label = d_data['label']
                        if label == "D1" and tid not in counted_d1:
                            counted_d1.add(tid); door1_count += 1
                            if product_type == "NP": np_count += 1
                            elif product_type == "DAP": dap_count += 1
                        elif label == "D2" and tid not in counted_d2:
                            counted_d2.add(tid); door2_count += 1
                            if product_type == "NP": np_count += 1
                            elif product_type == "DAP": dap_count += 1

            # ROBUST WAGON OCR 
            if class_id == 2 and not wagon_detected: 
                center = ((x1 + x2)//2, (y1 + y2)//2)
                
                # POLYGON FILTER: Only read plates INSIDE the cached wagon
                if wagon_poly is not None and cv2.pointPolygonTest(wagon_poly, center, False) >= 0:
                    
                    if frame_count % 5 != 0: continue

                    pad = 70
                    y_min, y_max = max(0, y1-pad), min(frame.shape[0], y2+pad)
                    x_min, x_max = max(0, x1-pad), min(frame.shape[1], x2+pad)
                    crop = frame[y_min:y_max, x_min:x_max]
                    if crop.size == 0: continue

                    # --- PIPELINE A: DETAIL (Best for Night Video) ---
                    img_A = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                    gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    input_A = clahe.apply(gray_A)

                    # --- PIPELINE B: AGGRESSIVE (Best for Daytime Video - NO DILATION) ---
                    img_B = cv2.resize(crop, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
                    gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
                    blurred_B = cv2.GaussianBlur(gray_B, (3,3), 0)
                    _, input_B = cv2.threshold(blurred_B, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    attempts = [("Detail", input_A), ("Aggressive", input_B)]
                    
                    best_score = 0.0  # ACCUMULATED SCORE LOGIC
                    best_tokens = []

                    for name, img_input in attempts:
                        result = ocr.ocr(img_input, cls=True)
                        current_conf, current_tokens = 0.0, []

                        if result and result[0]:
                            for line in result[0]:
                                text = re.sub(r'[^A-Z0-9 ]', '', line[1][0].upper())
                                conf = line[1][1]
                                for token in text.split():
                                    if len(token) >= 3 and (any(c.isdigit() for c in token) or token in ["BCNA", "BOXN", "BHL", "HSM", "MAC"]):
                                        current_tokens.append(token)
                                        current_conf += conf  # ACCUMULATE
                        
                        # Pipeline with the most valid text wins
                        if current_conf > best_score:
                            best_score = current_conf
                            best_tokens = current_tokens
                    
                    if best_tokens: wagon_candidates.extend(best_tokens)

                    buffer_size = 15 
                    if len(wagon_candidates) > buffer_size:
                        recent = wagon_candidates[-buffer_size:]
                        counter = Counter(recent)
                        stable_tokens = [token for token, count in counter.items() if count >= 3]

                        codes = [t for t in stable_tokens if any(c.isalpha() for c in t)]
                        nums = [t for t in stable_tokens if not any(c.isalpha() for c in t)]

                        standard_lock = len(stable_tokens) >= 2
                        fallback_lock = (len(nums) >= 1) and (len(wagon_candidates) > buffer_size + 10)

                        if standard_lock or fallback_lock:
                            stable_tokens.sort(key=lambda x: len(recent) - recent[::-1].index(x))
                            codes = [t for t in stable_tokens if any(c.isalpha() for c in t)]
                            nums = [t for t in stable_tokens if not any(c.isalpha() for c in t)]
                            
                            primary_codes = ["BCNA", "BOXN", "BHL", "MAC"]
                            codes.sort(key=lambda x: 0 if any(p in x for p in primary_codes) else 1)
                            nums.sort(key=lambda x: len(x), reverse=True)
                            
                            formatted_parts = []
                            for i in range(max(len(codes), len(nums))):
                                if i < len(codes): formatted_parts.append(codes[i])
                                if i < len(nums): formatted_parts.append(nums[i])
                                    
                            wagon_number_text = " ".join(formatted_parts)
                            wagon_detected = True
                            print(f"Wagon Locked: {wagon_number_text} (Score: {best_score:.2f} | Mode: {'Fallback' if fallback_lock and not standard_lock else 'Standard'})")

    # HUD & DISPLAY
    total = door1_count + door2_count
    cv2.putText(frame, f"D1: {door1_count}", (40,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.putText(frame, f"D2: {door2_count}", (40,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),3)
    cv2.putText(frame, f"Total: {total}", (40,150), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.putText(frame, f"NP: {np_count}", (40,200), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
    cv2.putText(frame, f"DAP: {dap_count}", (40,250), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    cv2.putText(frame, f"Wagon: {wagon_number_text}", (40,300), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("IFFCO Cement Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# ---------------- SAVE FINAL DATA TO POSTGRES ----------------
now = datetime.now()
total = door1_count + door2_count

if wagon_number_text == "":
    wagon_number_text = "UNKNOWN"

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    insert_query = """
    INSERT INTO wagon_data 
    (date, time, wagon_number, door1_count, door2_count, np_bags, dap_bags, total)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    cur.execute(insert_query, (
        now.date(),
        now.time(),
        wagon_number_text,
        door1_count,
        door2_count,
        np_count,
        dap_count,
        total
    ))

    conn.commit()
    cur.close()
    conn.close()

    print("\n✅ Data saved to PostgreSQL successfully")

except Exception as e:
    print("\n❌ Database Error:", e)

print("\nFinal Summary")
print("D1:", door1_count)
print("D2:", door2_count)
print("NP:", np_count)
print("DAP:", dap_count)
print("Total:", total)
print("Wagon:", wagon_number_text)