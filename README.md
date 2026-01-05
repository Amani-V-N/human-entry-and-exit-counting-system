# human-entry-and-exit-counting-system
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# ----------------- MODEL & TRACKER -----------------
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=45)

# ----------------- VIDEO SOURCE -----------------
cap = cv2.VideoCapture(0)   # 0 = default webcam

# ----------------- LINE & COUNTER -----------------
LINE_X = 320                # vertical line x-position (tune as needed)

# Net number of people on the "inside" side
people_count = 0            

# Track which side of the line each ID was last fully on
# Values: "left", "right"
last_side = {}              # track_id -> "left"/"right"

# For heuristic: track typical single-person width
person_widths = []          # recent widths of normal person boxes
MAX_WIDTH_MEMORY = 50       # how many widths to remember


def estimate_person_factor(width: int) -> int:
    """
    Heuristic: if this box is much wider than typical person width,
    we guess it may contain 2 people. Otherwise, 1.
    """
    if len(person_widths) < 10:
        # Not enough data yet, assume 1 person
        return 1

    avg_width = np.mean(person_widths)
    # If box is at least 1.8x wider than average, guess 2 people
    if width > 1.8 * avg_width:
        return 2
    return 1


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Run YOLOv8 on the frame
    results = model(frame)[0]
    detections = []

    # 2) Build detection list for DeepSORT
    for box in results.boxes:
        cls = int(box.cls[0])       # class index
        conf = float(box.conf[0])   # confidence

        if cls == 0 and conf > 0.5:   # 0 = "person" in COCO
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w = x2 - x1
            h = y2 - y1

            # Save width for heuristic (single-person stats)
            person_widths.append(w)
            if len(person_widths) > MAX_WIDTH_MEMORY:
                person_widths.pop(0)

            # DeepSORT format: ([x, y, w, h], confidence, class_name)
            detections.append(([x1, y1, w, h], conf, "person"))

    # 3) Update tracker with current detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # 4) Loop through tracked objects
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        l, t, r, b = int(l), int(t), int(r), int(b)

        # Draw bounding box & ID
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---------- FULL-CROSS + ENTRY/RETURN WITH HEURISTIC ----------
        # Determine which side this person is FULLY on now:
        #   - fully left of the line:  r < LINE_X
        #   - fully right of the line: l > LINE_X
        #   - overlapping the line:    ignore for side update
        current_side = None
        if r < LINE_X:
            current_side = "left"
        elif l > LINE_X:
            current_side = "right"

        prev_side = last_side.get(track_id)

        # If we have a previous side and a new full side, check transitions
        if prev_side is not None and current_side is not None:
            bbox_width = r - l
            # Heuristic: how many persons might be in this one box?
            person_factor = estimate_person_factor(bbox_width)

            # ENTRY: right -> left  → count += person_factor
            if prev_side == "right" and current_side == "left":
                people_count += person_factor

            # RETURN: left -> right → count -= person_factor (clamp at 0)
            elif prev_side == "left" and current_side == "right":
                people_count -= person_factor
                if people_count < 0:
                    people_count = 0

        # Update last_side only when fully on one side
        if current_side in ("left", "right"):
            last_side[track_id] = current_side

    # 5) Draw vertical counting line
    cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 0, 255), 2)
    cv2.putText(frame, "ENTRY LINE", (LINE_X + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 6) Show counter on screen
    cv2.putText(frame, f"Estimated Count: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 7) Show camera window
    cv2.imshow("Entry / Return Counter (Heuristic)", frame)

    # 8) Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9) Cleanup
cap.release()
cv2.destroyAllWindows()
