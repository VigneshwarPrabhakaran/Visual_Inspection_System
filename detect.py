import cv2
from ultralytics import YOLO

# =========================
# LOAD BOTH MODELS
# =========================
rgb_model = YOLO("models/RGB/best.pt")
canny_model = YOLO("models/Canny/best.pt")

print("Models loaded successfully!")

# =========================
# FUNCTIONS
# =========================
def get_boxes(results):
    detections = []

    if results[0].boxes is None:
        return detections

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append({
            "box": [x1, y1, x2, y2],
            "conf": conf,
            "cls": cls
        })

    return detections


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0

    return inter / union


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(1)      # Use 0 if using laptop camera

# Set resolution to 640x360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # CREATE CANNY IMAGE
    # =========================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    canny = cv2.Canny(blur, 60, 150)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    # =========================
    # RUN BOTH MODELS
    # =========================
    rgb_results = rgb_model(frame, verbose=False)
    canny_results = canny_model(canny, verbose=False)

    # =========================
    # GET DETECTIONS
    # =========================
    rgb_boxes = get_boxes(rgb_results)
    canny_boxes = get_boxes(canny_results)

    # =========================
    # FUSION
    # =========================
    matched_canny = set()
    final_boxes = []

    for rgb in rgb_boxes:

        best_iou = 0
        best_index = -1

        for i, canny_box in enumerate(canny_boxes):

            if i in matched_canny:
                continue

            score = calculate_iou(rgb["box"], canny_box["box"])

            if score > best_iou:
                best_iou = score
                best_index = i

        # Same defect detected by both models
        if best_iou >= 0.5:

            matched_canny.add(best_index)
            canny_box = canny_boxes[best_index]

            final_boxes.append({
                "box": rgb["box"],
                "rgb_conf": rgb["conf"],
                "canny_conf": canny_box["conf"],
                "iou": best_iou,
                "source": "both"
            })

        else:
            # Only RGB found it
            final_boxes.append({
                "box": rgb["box"],
                "rgb_conf": rgb["conf"],
                "canny_conf": None,
                "iou": None,
                "source": "rgb_only"
            })

    # Add unmatched Canny detections
    for i, canny_box in enumerate(canny_boxes):

        if i not in matched_canny:
            # Only Canny found it
            final_boxes.append({
                "box": canny_box["box"],
                "rgb_conf": None,
                "canny_conf": canny_box["conf"],
                "iou": None,
                "source": "canny_only"
            })

    # =========================
    # DRAW VISUALIZATIONS
    # =========================
    output = frame.copy()
    
    # Starting vertical position for our Top-Left Stats Panel
    panel_y = 30  

    for idx, det in enumerate(final_boxes):
        x1, y1, x2, y2 = map(int, det["box"])
        
        # 1. Bounding box stays strictly on the defect (No text attached)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add a tiny index label on the box corner so you know which stats match which box
        cv2.putText(output, f"#{idx+1}", (x1 + 5, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 2. Format string scores for Top-Left display
        if det["source"] == "both":
            canny_str = f"{det['canny_conf']:.2f}"
            rgb_str = f"{det['rgb_conf']:.2f}"
            iou_str = f"{det['iou']:.2f}"
        elif det["source"] == "rgb_only":
            canny_str = "N/A"
            rgb_str = f"{det['rgb_conf']:.2f}"
            iou_str = "N/A"
        else:  # canny_only
            canny_str = f"{det['canny_conf']:.2f}"
            rgb_str = "N/A"
            iou_str = "N/A"

        # 3. Print metrics stack in the uniform Top-Left corner
        stats_lines = [
            f"[Defect #{idx+1}]",
            f"  Canny : {canny_str}",
            f"  RGB   : {rgb_str}",
            f"  IOU   : {iou_str}"
        ]

        for line in stats_lines:
            # Draw text shadow for better readability against light backgrounds
            cv2.putText(output, line, (16, panel_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            # Draw main text
            cv2.putText(output, line, (15, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            panel_y += 20  # Move line down
        
        panel_y += 10 # Extra spacing between multiple defects

    # If no defects are found, show clear status
    if not final_boxes:
        cv2.putText(output, "System Status: No Defects", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("Final Detection", output)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()