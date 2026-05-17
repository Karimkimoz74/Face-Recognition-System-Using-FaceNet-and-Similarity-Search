"""
Phase 2 - Face embedding module.

This is the CORE of the project. It turns a face image into a 512-number
"embedding" vector using the pre-trained FaceNet model. Two photos of the SAME
person produce vectors that are close together; DIFFERENT people produce
vectors that are far apart. Every later phase (database, recognition, the apps)
uses this module.

Pipeline:
    image  ->  MTCNN (detect + crop + align to 160x160)  ->  FaceNet  ->  512-d embedding

The FaceEmbedder class is meant to be imported by the rest of the project.
It can also be run directly to test embeddings:

    python src/embedder.py --image  photo.jpg            # embed one image, show info
    python src/embedder.py --compare a.jpg b.jpg         # distance between two faces
    python src/embedder.py --webcam 1                    # live: SPACE to embed, q to quit
"""

import argparse
import warnings

import cv2
import numpy as np
from PIL import Image

# facenet-pytorch prints a harmless FutureWarning when it loads its weights.
warnings.filterwarnings("ignore", category=FutureWarning)

IMAGE_SIZE = 160     # FaceNet expects face crops of exactly 160x160 pixels
MARGIN = 20          # extra pixels kept around the detected face when cropping
MIN_PROB = 0.90      # ignore detections less confident than 90%


class FaceEmbedder:
    """Loads MTCNN + FaceNet once, then converts faces into 512-d embeddings."""

    def __init__(self, device=None, keep_all=True):
        # Imported here so the file's --help still works before install.
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self._torch = torch
        # Use the GPU if one is available, otherwise the CPU.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # MTCNN: finds faces AND returns them already cropped and aligned to
        # 160x160 - exactly the size FaceNet needs.
        #   keep_all=True  -> handle EVERY face in the image (not just one)
        self.mtcnn = MTCNN(
            image_size=IMAGE_SIZE,
            margin=MARGIN,
            keep_all=keep_all,
            min_face_size=40,
            post_process=True,
            device=self.device,
        )

        # FaceNet (InceptionResnetV1): the network that converts a 160x160 face
        # crop into a 512-number embedding.
        #   pretrained='vggface2' -> load weights learned on the VGGFace2 dataset
        #   .eval()               -> inference mode (we are NOT training)
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        print(f"FaceEmbedder ready on device: {self.device}")

    @staticmethod
    def _to_pil(image):
        """Accept a file path, an OpenCV BGR array, or a PIL image; return an RGB PIL image."""
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            # OpenCV stores images as BGR; PIL and MTCNN expect RGB.
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError("image must be a file path, a numpy array, or a PIL image")

    def embed(self, image):
        """
        Detect every face in 'image' and return a list of results, one per face:
            {"box": [x1, y1, x2, y2], "prob": confidence, "embedding": 512-d np.array}
        Returns an empty list [] if no face is found.
        """
        torch = self._torch
        pil = self._to_pil(image)

        # Step 1 - DETECT: find where the faces are (boxes + confidence).
        boxes, probs = self.mtcnn.detect(pil)
        if boxes is None:
            return []

        # Step 2 - CROP + ALIGN: cut each face out and resize it to 160x160.
        # Passing the boxes we already found means no second detection is run.
        face_tensors = self.mtcnn.extract(pil, boxes, save_path=None)
        if face_tensors is None:
            return []
        if face_tensors.ndim == 3:                  # a single face -> make it a batch of 1
            face_tensors = face_tensors.unsqueeze(0)

        # Step 3 - EMBED: run every face crop through FaceNet in one batch.
        # torch.no_grad() = "don't track gradients" - faster, we are not training.
        with torch.no_grad():
            embeddings = self.facenet(face_tensors.to(self.device))
        embeddings = embeddings.cpu().numpy()       # torch tensor -> numpy array

        # Pair each box/prob with its embedding; drop low-confidence detections.
        results = []
        for box, prob, emb in zip(boxes, probs, embeddings):
            if prob is None or prob < MIN_PROB:
                continue
            results.append({
                "box": [int(v) for v in box],
                "prob": float(prob),
                "embedding": emb,        # already L2-normalized by FaceNet (length = 1)
            })
        return results

    def embed_single(self, image):
        """Return ONE embedding - the most confident face - or None if no face."""
        faces = self.embed(image)
        if not faces:
            return None
        best = max(faces, key=lambda f: f["prob"])
        return best["embedding"]


def euclidean_distance(a, b):
    """Straight-line distance between two embedding vectors. Smaller = more similar."""
    return float(np.linalg.norm(a - b))


# --------------------------------------------------------------------------
# Below: command-line testing helpers. Not used when this file is imported.
# --------------------------------------------------------------------------

def _print_embedding_info(emb, prob=None):
    """Pretty-print facts about one embedding so we can sanity-check it."""
    if prob is not None:
        print(f"  detection confidence : {prob * 100:.1f}%")
    print(f"  embedding shape      : {emb.shape}        (should be (512,))")
    print(f"  embedding length(L2) : {np.linalg.norm(emb):.4f}   (should be ~1.0)")
    preview = ", ".join(f"{v:+.3f}" for v in emb[:5])
    print(f"  first 5 numbers      : [{preview}, ...]")


def _run_image(embedder, path):
    faces = embedder.embed(path)
    if not faces:
        print(f"No face detected in: {path}")
        return
    print(f"Detected {len(faces)} face(s) in: {path}")
    for i, face in enumerate(faces, start=1):
        print(f"\nFace {i}:")
        _print_embedding_info(face["embedding"], face["prob"])


def _run_compare(embedder, path_a, path_b):
    emb_a = embedder.embed_single(path_a)
    emb_b = embedder.embed_single(path_b)
    if emb_a is None or emb_b is None:
        print("Could not find a face in one of the images.")
        return
    dist = euclidean_distance(emb_a, emb_b)
    print(f"\nEuclidean distance between the two faces: {dist:.4f}")
    print("Rough guide (tune the exact threshold later):")
    print("  ~0.0 - 0.9  -> very likely the SAME person")
    print("  ~0.9 - 1.1  -> borderline")
    print("  ~1.1 - 2.0  -> very likely DIFFERENT people")
    verdict = "SAME person" if dist < 1.0 else "DIFFERENT people"
    print(f"=> Guess: {verdict}")


def _run_webcam(embedder, index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera index {index}. Is Iriun running?")
        return
    print(f"Live embedding from camera {index}. SPACE = embed, q = quit.")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # Downscale for speed (phone cameras are large).
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

        faces = embedder.embed(frame)
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"faces: {len(faces)}  SPACE=embed  q=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Phase 2 - live embedding", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if faces:
                print("\n[SPACE] embedded the most confident face:")
                best = max(faces, key=lambda f: f["prob"])
                _print_embedding_info(best["embedding"], best["prob"])
            else:
                print("\n[SPACE] no face in frame - nothing to embed.")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Phase 2 face embedding test.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="embed one image and show info")
    group.add_argument("--compare", nargs=2, metavar=("A", "B"),
                       help="show the distance between the faces in two images")
    group.add_argument("--webcam", type=int, metavar="INDEX",
                       help="live embedding from a camera index")
    args = parser.parse_args()

    embedder = FaceEmbedder()

    if args.image:
        _run_image(embedder, args.image)
    elif args.compare:
        _run_compare(embedder, args.compare[0], args.compare[1])
    else:
        _run_webcam(embedder, args.webcam)


if __name__ == "__main__":
    main()
