
import cv2
import os
from ultralytics import YOLO

# =========================
# Configuration
# =========================
MODEL_PATH = r"F:\snake\detect\snake.pt"
SOURCE = r"F:\snake\detect\vdo2.mp4"  # 0 = Webcam, or use "input.mp4"
OUTPUT_DIR = r"F:\snake\detect"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load custom YOLO model
model = YOLO(MODEL_PATH)

# Open webcam or video
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Error: Could not open camera/video.")
    exit()

# Output video settings
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 20.0

output_path = os.path.join(OUTPUT_DIR, "_Res_snake_detected.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path, fourcc, fps, (width, height)
)

print("Snake detection started...")
print("Press 'q' to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run snake detection
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Save annotated frame to output video
    writer.write(annotated_frame)

    # Display output
    cv2.imshow("Snake Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Output saved at: {output_path}")
