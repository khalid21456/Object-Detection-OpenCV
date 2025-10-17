import cv2
import numpy as np

# Configuration
MODEL_PATH = "frozen_inference_graph.pb"
CONFIG_PATH = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
CLASSES_FILE = "coco.names"
CONFIDENCE_THRESHOLD = 0.6  # Increased threshold to reduce false positives
NMS_THRESHOLD = 0.4


# Load class names
def load_classes(classes_file):
    """Load class names from file"""
    try:
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except FileNotFoundError:
        print(f"Warning: {classes_file} not found. Using default COCO classes.")
        # Default COCO classes
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]


# Generate random colors for each class
def generate_colors(num_classes):
    """Generate random colors for bounding boxes"""
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        colors.append((
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255))
        ))
    return colors


# Draw detection results
def draw_predictions(frame, class_id, confidence, x1, y1, x2, y2, classes, colors):
    """Draw bounding box and label on frame"""
    if class_id >= len(classes):
        return

    label = f"{classes[class_id]}: {confidence:.0%}"
    color = colors[class_id]

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Prepare label
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    # Draw label background
    y1_label = max(y1, label_height + 10)
    cv2.rectangle(frame, (x1, y1_label - label_height - 10),
                  (x1 + label_width + 10, y1_label), color, -1)

    # Draw label text
    cv2.putText(frame, label, (x1 + 5, y1_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    """Main function for real-time object detection"""

    print("=" * 60)
    print("🎥 Real-time Object Detection with SSD MobileNet")
    print("=" * 60)
    print("\n🔄 Loading model...")

    # Load class names
    classes = load_classes(CLASSES_FILE)
    print(f"✅ Loaded {len(classes)} classes")

    # Generate colors
    colors = generate_colors(len(classes))

    # Load the model
    try:
        net = cv2.dnn.readNetFromTensorflow(MODEL_PATH, CONFIG_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you have:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {CONFIG_PATH}")
        return

    # Set DNN backend and target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        print("Tip: Try cv2.VideoCapture(1) or another camera index")
        return

    # Set camera resolution (optional, for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ Webcam opened successfully!")
    print("\n" + "=" * 60)
    print("📹 Starting detection...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  '+' - Increase confidence threshold")
    print("  '-' - Decrease confidence threshold")
    print("=" * 60 + "\n")

    frame_count = 0
    confidence_threshold = CONFIDENCE_THRESHOLD

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("❌ Error: Could not read frame!")
            break

        frame_count += 1
        height, width = frame.shape[:2]

        # Prepare frame for detection (MobileNet SSD expects 300x300)
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 127.5,  # Scale pixel values to [-1, 1]
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),  # Subtract mean
            swapRB=True,
            crop=False
        )

        # Set input and run detection
        net.setInput(blob)
        detections = net.forward()

        # Process detections
        detected_objects = []

        # detections shape: [1, 1, N, 7]
        # Each detection: [image_id, class_id, confidence, x1, y1, x2, y2]
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence > confidence_threshold:
                class_id = int(detections[0, 0, i, 1])

                # COCO dataset uses 1-90 class IDs, but with gaps
                # Map to 0-79 range for our class list
                class_id = class_id - 1

                # Check if class_id is valid
                if class_id < 0 or class_id >= len(classes):
                    continue

                # Get bounding box coordinates (normalized to [0, 1])
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)

                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Draw predictions
                draw_predictions(frame, class_id, confidence,
                                 x1, y1, x2, y2, classes, colors)

                detected_objects.append(classes[class_id])

        # Display info overlay
        info_bg = frame.copy()
        cv2.rectangle(info_bg, (0, 0), (width, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, info_bg, 0.3, 0)

        # Display stats
        fps_text = f"Frame: {frame_count}"
        obj_count_text = f"Objects: {len(detected_objects)}"
        threshold_text = f"Threshold: {confidence_threshold:.2f}"

        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, obj_count_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, threshold_text, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Show detected objects list
        if detected_objects:
            unique_objects = list(set(detected_objects))
            objects_text = "Detected: " + ", ".join(unique_objects[:4])
            cv2.putText(frame, objects_text, (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Display frame
        cv2.imshow('Object Detection - Press Q to quit', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n🛑 Quitting...")
            break
        elif key == ord('s'):
            filename = f"detection_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Threshold increased to: {confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"Threshold decreased to: {confidence_threshold:.2f}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Done!")
    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()