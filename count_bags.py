from ultralytics import YOLO
import cv2
import os
import re
import numpy as np
from collections import Counter
from paddleocr import PaddleOCR

# ----------------------------
# PATHS
# ----------------------------
BASE_PATH = r"D:\Projects\Age group\IFFCO"
MODEL_PATH = os.path.join(BASE_PATH, "best_goods_model.pt")
VIDEO_PATH = os.path.join(BASE_PATH, "clipped_wagon_video.mp4")

# ----------------------------
# LOAD MODEL & VIDEO
# ----------------------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

window_name = "IFFCO Cement Monitoring"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# ----------------------------
# PADDLE OCR INITIALIZATION
# ----------------------------
ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=True  # replaces deprecated use_angle_cls
)

# ----------------------------
# COUNTERS
# ----------------------------
bag_count_d1 = 0
bag_count_d2 = 0

wagon_number_text = ""
wagon_detected = False
wagon_candidates = []

# ----------------------------
# DOOR POLYGONS
# ----------------------------
door1_polygon = np.array([
    (1559, 564),
    (1903, 646),
    (1841, 1041),
    (1562, 948)
], np.int32)

door2_polygon = np.array([
    (446, 411),
    (570, 420),
    (674, 677),
    (557, 644)
], np.int32)

prev_inside_d1 = {}
prev_inside_d2 = {}

print("Processing video...")

# ----------------------------
# MAIN LOOP
# ----------------------------
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
        conf=0.25,
        imgsz=960
    )

    # Draw door polygons
    cv2.polylines(frame, [door1_polygon], True, (0, 255, 0), 3)
    cv2.polylines(frame, [door2_polygon], True, (255, 0, 0), 3)

    if results[0].boxes.id is not None:

        for box, track_id, cls in zip(
            results[0].boxes.xyxy,
            results[0].boxes.id,
            results[0].boxes.cls
        ):

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            class_id = int(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # ==================================================
            # BAG COUNTING
            # ==================================================
            if class_id == 0:

                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                inside_d1 = cv2.pointPolygonTest(door1_polygon, center, False) >= 0
                if track_id in prev_inside_d1:
                    if not prev_inside_d1[track_id] and inside_d1:
                        bag_count_d1 += 1
                        print(f"Door1 Count: {bag_count_d1}")
                prev_inside_d1[track_id] = inside_d1

                inside_d2 = cv2.pointPolygonTest(door2_polygon, center, False) >= 0
                if track_id in prev_inside_d2:
                    if not prev_inside_d2[track_id] and inside_d2:
                        bag_count_d2 += 1
                        print(f"Door2 Count: {bag_count_d2}")
                prev_inside_d2[track_id] = inside_d2

            # ==================================================
            # WAGON OCR (PADDLEOCR VERSION)
            # ==================================================
            if class_id == 1 and not wagon_detected:

                # Run OCR every 5 frames for stability
                if frame_count % 5 != 0:
                    continue

                pad = 40
                crop = frame[
                    max(0, y1 - pad):min(frame.shape[0], y2 + pad),
                    max(0, x1 - pad):min(frame.shape[1], x2 + pad)
                ]

                # Upscale for better recognition
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                result = ocr.ocr(crop)

                if result and result[0]:
                    detected_texts = []

                    for line in result[0]:
                        text = line[1][0].upper()
                        detected_texts.append(text)

                    raw_text = " ".join(detected_texts)
                    print("OCR RAW:", raw_text)

                    # Keep only A-Z and 0-9
                    clean_text = re.sub(r'[^A-Z0-9]', ' ', raw_text)
                    tokens = clean_text.split()

                    for token in tokens:
                        if len(token) >= 3:
                            wagon_candidates.append(token)

                # Lock wagon when stable
                if len(wagon_candidates) > 25:

                    counter = Counter(wagon_candidates)

                    stable_tokens = [
                        token for token, count in counter.items()
                        if count > 3
                    ]

                    if stable_tokens:
                        wagon_number_text = " ".join(stable_tokens)
                        wagon_detected = True
                        print("Final Wagon Detected:", wagon_number_text)

    # ==================================================
    # DISPLAY
    # ==================================================
    total = bag_count_d1 + bag_count_d2

    cv2.putText(frame, f"Door1: {bag_count_d1}", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    cv2.putText(frame, f"Door2: {bag_count_d2}", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)

    cv2.putText(frame, f"Total: {total}", (40, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.putText(frame, f"Wagon: {wagon_number_text}", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow(window_name, frame)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()

print("Door1:", bag_count_d1)
print("Door2:", bag_count_d2)
print("Total:", total)
print("Wagon:", wagon_number_text)
