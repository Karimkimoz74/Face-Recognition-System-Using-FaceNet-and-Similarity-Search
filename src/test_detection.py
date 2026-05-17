"""
Phase 1 - Face detection test.

This script verifies that MTCNN (the face detector bundled with
facenet-pytorch) works on your machine. MTCNN finds faces in an image and
returns a bounding box + a confidence probability for each one. In later
phases the cropped face it produces is what gets fed into FaceNet.

It does NOT recognize anyone yet - it only proves "the detector can find a
face." That is the goal of Phase 1.

Usage:
    python src/test_detection.py --image photo.jpg     # detect in an image file
    python src/test_detection.py --webcam 1            # detect live from a camera

Inside the webcam window:
    q  -> quit

Image mode saves an annotated copy next to the input as <name>_detected.jpg
"""

import argparse
import os
import cv2

# MTCNN is imported lazily inside main() so that --help still works even
# before the dependencies are installed.

DETECT_WIDTH = 640    # frames are downscaled to this width before detection


def downscale(frame, target_width=DETECT_WIDTH):
    """Resize down to target_width, keeping aspect ratio. Returns frame + scale."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame, 1.0
    scale = target_width / w
    resized = cv2.resize(frame, (target_width, int(h * scale)))
    return resized, scale


def draw_detections(frame, boxes, probs):
    """Draw a green box + confidence label for every detected face."""
    if boxes is None:
        return frame
    for box, prob in zip(boxes, probs):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{prob * 100:.1f}%"
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def detect_image(detector, image_path):
    """Run MTCNN on a single image file and save an annotated copy."""
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Could not read image: {image_path}")
        return

    # MTCNN expects RGB; OpenCV loads BGR.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = detector.detect(rgb)

    if boxes is None:
        print("No face detected.")
    else:
        print(f"Detected {len(boxes)} face(s):")
        for i, prob in enumerate(probs, start=1):
            print(f"  face {i}: confidence {prob * 100:.1f}%")

    annotated = draw_detections(bgr, boxes, probs if boxes is not None else [])
    root, ext = os.path.splitext(image_path)
    out_path = f"{root}_detected{ext}"
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated image to: {out_path}")

    cv2.imshow("Detection result (press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_webcam(detector, index):
    """Run MTCNN live on a webcam feed."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera index {index}.")
        print("Is Iriun running and connected? Run check_camera.py to list indices.")
        return

    print(f"Detecting faces from camera index {index}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Lost the camera feed.")
            break

        small, _ = downscale(frame)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes, probs = detector.detect(rgb)
        small = draw_detections(small, boxes, probs if boxes is not None else [])

        count = 0 if boxes is None else len(boxes)
        cv2.putText(small, f"faces: {count}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("MTCNN detection  (q=quit)", small)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Phase 1 MTCNN face detection test.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="detect faces in an image file")
    group.add_argument("--webcam", type=int, metavar="INDEX",
                       help="detect faces live from this camera index")
    args = parser.parse_args()

    # Imported here so --help works without the packages installed.
    import torch
    from facenet_pytorch import MTCNN

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MTCNN on device: {device}")
    # keep_all=True so it reports every face in the frame, not just one.
    detector = MTCNN(keep_all=True, device=device)

    if args.image:
        detect_image(detector, args.image)
    else:
        detect_webcam(detector, args.webcam)


if __name__ == "__main__":
    main()
