"""
Phase 4 - Recognition logic.

This module makes the final decision: given a face, is it someone we know,
or a stranger ("Unknown")?

It ties together the two earlier phases:
    embedder.py (Phase 2)  ->  face image  -> 512-d embedding
    database.py (Phase 3)  ->  embedding   -> nearest enrolled person + distance

and then applies THE THRESHOLD: if the nearest person is close enough, return
their name; otherwise return "Unknown".

The FaceRecognizer class is imported by the web app and the apps. This file can
also be run directly to test recognition:

    python src/recognize.py --image  photo.jpg     # recognize faces in an image
    python src/recognize.py --webcam 1             # live recognition from a camera
"""

import argparse
import warnings

import cv2

from embedder import FaceEmbedder
from database import FaceDatabase

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================================================
#  THE THRESHOLD - the single most important number in the project.
# --------------------------------------------------------------------------
#  Recognition is just "distance + a cut-off line":
#
#      distance <= THRESHOLD  ->  accept the match, return the person's name
#      distance >  THRESHOLD  ->  reject it, return "Unknown"
#
#  FaceNet embeddings are L2-normalized, so the Euclidean distance between any
#  two of them is always between 0.0 (identical) and 2.0 (opposite).
#
#  Picking the value:
#    - too LOW  -> the system is too strict; it calls real people "Unknown"
#    - too HIGH -> the system is too loose; it confuses different people
#
#  1.0 is a sensible starting value. Phase 6 measures the BEST value using
#  real photos of the team, so treat this as a tunable setting, not a fact.
# ==========================================================================
RECOGNITION_THRESHOLD = 1.0

UNKNOWN_LABEL = "Unknown"


def distance_to_confidence(distance, threshold=RECOGNITION_THRESHOLD):
    """
    Convert a raw distance into a friendly 0-100% confidence score.

    The raw distance is not intuitive ("is 0.74 good?"), so we map it:
        distance 0            -> 100%   (identical faces)
        distance = threshold  ->  50%   (right on the borderline)
        distance >= 2*threshold -> 0%   (completely different)

    This is only for display - the accept/reject decision still uses the
    raw distance vs. the threshold.
    """
    confidence = 1.0 - distance / (2.0 * threshold)
    confidence = max(0.0, min(1.0, confidence))     # clamp into 0..1
    return confidence * 100.0


class FaceRecognizer:
    """Identifies faces by embedding them and searching the database."""

    def __init__(self, embedder=None, database=None,
                 threshold=RECOGNITION_THRESHOLD):
        # Reuse passed-in objects if given (lets the web app share one copy),
        # otherwise create them.
        self.embedder = embedder if embedder is not None else FaceEmbedder()
        self.database = database if database is not None else FaceDatabase.load()
        self.threshold = threshold

    def identify_embedding(self, embedding):
        """
        Decide who one 512-d embedding belongs to.
        Returns a result dictionary describing the decision.
        """
        name, distance = self.database.find_nearest(embedding)

        if name is None:                            # nobody is enrolled yet
            return {
                "recognized": False,
                "name": UNKNOWN_LABEL,
                "nearest": None,
                "distance": None,
                "confidence": 0.0,
                "threshold": self.threshold,
            }

        recognized = distance <= self.threshold
        return {
            "recognized": recognized,
            "name": name if recognized else UNKNOWN_LABEL,
            "nearest": name,                        # closest match, even if "Unknown"
            "distance": round(float(distance), 4),
            "confidence": round(distance_to_confidence(distance, self.threshold), 1),
            "threshold": self.threshold,
        }

    def recognize_image(self, image):
        """
        Detect and identify EVERY face in an image.
        Returns a list of results - each result is the identity info plus the
        face's bounding box and detection confidence.
        """
        results = []
        for face in self.embedder.embed(image):
            info = self.identify_embedding(face["embedding"])
            info["box"] = face["box"]
            info["detection_prob"] = round(face["prob"], 4)
            results.append(info)
        return results

    def recognize_single(self, image):
        """
        Identify just the single most prominent face in an image.
        Returns one result dictionary, or None if no face was found.
        """
        embedding = self.embedder.embed_single(image)
        if embedding is None:
            return None
        return self.identify_embedding(embedding)


# --------------------------------------------------------------------------
# Below: command-line tools for testing recognition.
# --------------------------------------------------------------------------

def _label_for(result):
    """Build the text drawn next to a face, e.g. 'Karim 87%' or 'Unknown'."""
    if result["recognized"]:
        return f"{result['name']} {result['confidence']:.0f}%"
    return UNKNOWN_LABEL


def _run_image(recognizer, path):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"Could not read image: {path}")
        return

    results = recognizer.recognize_image(bgr)
    if not results:
        print("No face detected.")
        return

    print(f"Found {len(results)} face(s):")
    for i, r in enumerate(results, start=1):
        if r["recognized"]:
            print(f"  face {i}: {r['name']}  (distance {r['distance']}, "
                  f"confidence {r['confidence']}%)")
        else:
            print(f"  face {i}: Unknown  (closest '{r['nearest']}' at "
                  f"distance {r['distance']})")
        # draw the box + label
        x1, y1, x2, y2 = r["box"]
        color = (0, 255, 0) if r["recognized"] else (0, 200, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, _label_for(r), (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Recognition result (press any key to close)", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _run_webcam(recognizer, index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera index {index}. Is Iriun running?")
        return

    print(f"Live recognition from camera {index}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

        for r in recognizer.recognize_image(frame):
            x1, y1, x2, y2 = r["box"]
            color = (0, 255, 0) if r["recognized"] else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, _label_for(r), (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Phase 4 - live recognition  (q=quit)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Phase 4 face recognition test.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="recognize faces in an image file")
    group.add_argument("--webcam", type=int, metavar="INDEX",
                       help="live recognition from a camera index")
    args = parser.parse_args()

    recognizer = FaceRecognizer()
    if len(recognizer.database) == 0:
        print("Warning: the database is empty - enroll people first "
              "(run the web app).")

    if args.image:
        _run_image(recognizer, args.image)
    else:
        _run_webcam(recognizer, args.webcam)


if __name__ == "__main__":
    main()
