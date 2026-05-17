"""
Phase 1 - Camera check utility.

The Iriun Webcam app makes a phone act as a virtual webcam. It does NOT always
appear as camera index 0 - if the laptop has a built-in webcam, that is usually
index 0 and Iriun becomes index 1 or 2.

This script:
  1. Probes camera indices 0..MAX and reports which ones actually work.
  2. Optionally opens a live preview of a chosen index so you can confirm the
     Iriun feed and pick the right index for the rest of the project.

Usage:
    python src/check_camera.py                 # list all working cameras
    python src/check_camera.py --preview 1     # live preview of index 1
    python src/check_camera.py --preview 1 --rotate 90

Inside the preview window:
    q  -> quit
    r  -> rotate the frame 90 degrees (useful if the phone is in portrait)
"""

import argparse
import cv2

MAX_INDEX = 5          # highest camera index to probe
PREVIEW_WIDTH = 640    # downscale width for the preview (keeps it fast)


def open_camera(index):
    """Open a camera by index using the DirectShow backend (best on Windows
    for virtual cameras like Iriun). Returns the VideoCapture or None."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def list_cameras():
    """Probe indices 0..MAX_INDEX and print which ones return a frame."""
    print(f"Probing camera indices 0..{MAX_INDEX} ...\n")
    working = []
    for index in range(MAX_INDEX + 1):
        cap = open_camera(index)
        if cap is None:
            print(f"  [index {index}]  not available")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  [index {index}]  OK    resolution {w}x{h}")
            working.append(index)
        else:
            print(f"  [index {index}]  opened but returned no frame")
        cap.release()

    print()
    if working:
        print(f"Working camera indices: {working}")
        print("Tip: the high-resolution one is usually your Iriun phone camera.")
        print(f"Preview it with:  python src/check_camera.py --preview {working[-1]}")
    else:
        print("No cameras found.")
        print("If you are using Iriun: make sure the Iriun desktop app is running")
        print("and the phone app is connected (same Wi-Fi or USB) BEFORE running this.")
    return working


def downscale(frame, target_width=PREVIEW_WIDTH):
    """Resize a frame down to target_width while keeping the aspect ratio."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)))


def rotate(frame, degrees):
    """Rotate a frame by 0/90/180/270 degrees."""
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def preview(index, rotate_degrees=0):
    """Open a live preview window for the given camera index."""
    cap = open_camera(index)
    if cap is None:
        print(f"Could not open camera index {index}.")
        print("Is Iriun running and connected? Try a different index "
              "(run without --preview to list them).")
        return

    print(f"Previewing camera index {index}. Press 'q' to quit, 'r' to rotate.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Lost the camera feed.")
            break

        frame = rotate(frame, rotate_degrees)
        frame = downscale(frame)
        cv2.imshow(f"Camera index {index}  (q=quit, r=rotate)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            rotate_degrees = (rotate_degrees + 90) % 360
            print(f"Rotation set to {rotate_degrees} degrees.")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Phase 1 camera check utility.")
    parser.add_argument("--preview", type=int, metavar="INDEX",
                        help="open a live preview of this camera index")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="rotate the preview by this many degrees")
    args = parser.parse_args()

    if args.preview is not None:
        preview(args.preview, args.rotate)
    else:
        list_cameras()


if __name__ == "__main__":
    main()
