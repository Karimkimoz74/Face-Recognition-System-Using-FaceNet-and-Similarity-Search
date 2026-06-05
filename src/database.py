"""
Phase 3 - Face database / enrollment module.

This module stores the people the system KNOWS. For each person it keeps one
512-number embedding (their "face signature"). Recognition (Phase 4) will
compare a new face against everything stored here.

How a person is enrolled:
    several photos  ->  one embedding each  ->  average them  ->  one signature

Averaging several photos makes the signature more stable than a single photo
(different angles / lighting average out).

The database is saved to disk as 'embeddings.pkl' so it survives between runs.

The FaceDatabase class is imported by Phase 4 and the apps. This file can also
be run directly to enroll people and manage the database:

    python src/database.py --capture "Karim" --webcam 1   # enroll via webcam
    python src/database.py --enroll-folder                 # enroll from data/faces/
    python src/database.py --list                          # show enrolled people
    python src/database.py --remove "Karim"                # delete a person
"""

import argparse
import os
import pickle

import cv2
import numpy as np

# When this file is run as a script, its own folder (src/) is on the import
# path, so 'embedder' can be imported directly.
from embedder import FaceEmbedder

# Paths are built relative to the project root so they work from any folder.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
FACES_DIR = os.path.join(PROJECT_ROOT, "data", "faces")
# DB_PATH can be overridden via env var (used by the Modal deployment to point
# the database at a persistent Volume mounted at /data).
DB_PATH = os.environ.get("FACEREC_DB_PATH",
                         os.path.join(PROJECT_ROOT, "embeddings.pkl"))

MIN_PHOTOS = 3      # minimum photos needed to enroll a person well


class FaceDatabase:
    """Stores {name -> averaged 512-d embedding} and saves/loads it to disk."""

    def __init__(self):
        self.people = {}        # name (str) -> embedding (np.array of 512 floats)

    # ---- enrollment -------------------------------------------------------

    def enroll_person(self, name, embeddings):
        """
        Add (or replace) a person from a list of their face embeddings.
        The embeddings are averaged into a single signature.
        """
        if len(embeddings) == 0:
            raise ValueError(f"No embeddings given for '{name}'.")

        # Average all the photos' embeddings element by element.
        average = np.mean(embeddings, axis=0)
        # The average of unit vectors is usually NOT unit length, so we
        # L2-normalize it again to keep distance comparisons consistent.
        average = average / np.linalg.norm(average)

        self.people[name] = average
        print(f"Enrolled '{name}' from {len(embeddings)} photo(s).")

    def enroll_from_folder(self, embedder, faces_dir=FACES_DIR):
        """
        Enroll everyone found under data/faces/. Expected layout:
            data/faces/Karim/photo1.jpg, photo2.jpg, ...
            data/faces/Rawan/...
        Each sub-folder name becomes a person's name.
        """
        if not os.path.isdir(faces_dir):
            print(f"Folder not found: {faces_dir}")
            return

        person_folders = [d for d in sorted(os.listdir(faces_dir))
                           if os.path.isdir(os.path.join(faces_dir, d))]
        if not person_folders:
            print(f"No person folders inside {faces_dir}.")
            print("Create one folder per person and put their photos inside.")
            return

        for person in person_folders:
            folder = os.path.join(faces_dir, person)
            image_files = [f for f in sorted(os.listdir(folder))
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            embeddings = []
            for fname in image_files:
                emb = embedder.embed_single(os.path.join(folder, fname))
                if emb is None:
                    print(f"  '{person}/{fname}': no face found - skipped")
                else:
                    embeddings.append(emb)

            if len(embeddings) < MIN_PHOTOS:
                print(f"  '{person}': only {len(embeddings)} usable photo(s) "
                      f"(need {MIN_PHOTOS}) - skipped")
                continue
            self.enroll_person(person, embeddings)

    # ---- managing the database -------------------------------------------

    def remove(self, name):
        """Delete a person. Returns True if they existed."""
        if name in self.people:
            del self.people[name]
            print(f"Removed '{name}'.")
            return True
        print(f"'{name}' is not in the database.")
        return False

    def list_people(self):
        """Return a sorted list of enrolled names."""
        return sorted(self.people.keys())

    def __len__(self):
        return len(self.people)

    # ---- lookup (used by Phase 4) ----------------------------------------

    def find_nearest(self, embedding):
        """
        Compare 'embedding' to every stored person and return the closest one:
            (name, distance)   or   (None, None) if the database is empty.
        Smaller distance = more similar. Phase 4 adds the threshold decision.
        """
        if not self.people:
            return None, None
        best_name, best_distance = None, float("inf")
        for name, stored in self.people.items():
            distance = float(np.linalg.norm(embedding - stored))
            if distance < best_distance:
                best_name, best_distance = name, distance
        return best_name, best_distance

    # ---- saving / loading -------------------------------------------------

    def save(self, path=DB_PATH):
        """Write the database to disk as a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.people, f)
        print(f"Database saved to {path}  ({len(self.people)} people).")

    @classmethod
    def load(cls, path=DB_PATH):
        """Load a database from disk. Returns an empty database if none exists."""
        db = cls()
        if os.path.exists(path):
            with open(path, "rb") as f:
                db.people = pickle.load(f)
            print(f"Loaded database from {path}  ({len(db.people)} people).")
        else:
            print(f"No database file at {path} yet - starting empty.")
        return db


# --------------------------------------------------------------------------
# Below: command-line tools for enrolling and managing the database.
# --------------------------------------------------------------------------

def capture_and_enroll(db, embedder, name, cam_index, num_photos=5):
    """
    Open the webcam, let the user capture several photos of one person,
    then enroll that person and save the database.
    Captured photos are also saved under data/faces/<name>/ as a dataset.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera index {cam_index}. Is Iriun running?")
        return

    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"\nEnrolling '{name}'. Aim the camera at the face.")
    print(f"  SPACE = capture a photo (need at least {MIN_PHOTOS}, {num_photos} recommended)")
    print("  q     = finish")

    embeddings = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

        faces = embedder.embed(frame)
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        status = f"{name}: captured {len(embeddings)}  SPACE=capture  q=finish"
        cv2.putText(frame, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Phase 3 - enroll person", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            # Capture only when EXACTLY ONE face is visible, so we know who
            # the photo belongs to.
            if len(faces) == 1:
                embeddings.append(faces[0]["embedding"])
                idx = len(embeddings)
                cv2.imwrite(os.path.join(person_dir, f"{name}_{idx}.jpg"), frame)
                print(f"  captured photo {idx}")
            elif len(faces) == 0:
                print("  no face in frame - not captured")
            else:
                print("  more than one face in frame - not captured")

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < MIN_PHOTOS:
        print(f"\nOnly {len(embeddings)} photo(s) captured - need at least "
              f"{MIN_PHOTOS}. '{name}' was NOT enrolled.")
        return

    db.enroll_person(name, embeddings)
    db.save()


def main():
    parser = argparse.ArgumentParser(description="Phase 3 face database / enrollment.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture", metavar="NAME",
                       help="enroll one person by capturing photos from the webcam")
    group.add_argument("--enroll-folder", action="store_true",
                       help="enroll everyone from the data/faces/ folders")
    group.add_argument("--list", action="store_true",
                       help="list the people currently in the database")
    group.add_argument("--remove", metavar="NAME", help="remove a person from the database")
    parser.add_argument("--webcam", type=int, default=1, metavar="INDEX",
                        help="camera index for --capture (default 1)")
    args = parser.parse_args()

    # Always start from whatever is already saved on disk.
    db = FaceDatabase.load()

    if args.list:
        people = db.list_people()
        if people:
            print(f"\n{len(people)} person(s) enrolled:")
            for i, name in enumerate(people, start=1):
                print(f"  {i}. {name}")
        else:
            print("\nThe database is empty.")
        return

    if args.remove:
        if db.remove(args.remove):
            db.save()
        return

    # The remaining actions need the FaceNet model.
    embedder = FaceEmbedder()

    if args.capture:
        capture_and_enroll(db, embedder, args.capture, args.webcam)
    elif args.enroll_folder:
        db.enroll_from_folder(embedder)
        db.save()


if __name__ == "__main__":
    main()
