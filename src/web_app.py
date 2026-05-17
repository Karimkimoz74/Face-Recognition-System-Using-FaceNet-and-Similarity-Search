"""
Phase 3 - Web app for enrollment and testing.

A small Flask web server that puts the project in the browser. It has two parts:

  ENROLLMENT  - type a name and provide several face photos (upload or webcam);
                the photos become one averaged "signature" stored in the database.
  TESTING     - provide a face (upload or webcam) and the system says who it is.

The actual brains are reused from earlier phases:
  embedder.py  -> turns a face image into a 512-d embedding
  database.py  -> stores / loads / searches the enrolled people
  recognize.py -> applies the threshold to decide name vs. "Unknown" (Phase 4)

Run it with:
    python src/web_app.py
Then open  http://127.0.0.1:5000  in a browser.
"""

import os
import sys

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file

# Make sure sibling modules in src/ can be imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import MIN_PHOTOS
from recognize import FaceRecognizer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

app = Flask(__name__)

print("Loading the FaceNet model, please wait ...")
# FaceRecognizer builds the embedder and loads the database in one step.
recognizer = FaceRecognizer()
embedder = recognizer.embedder      # shared with the recognizer (one copy)
db = recognizer.database            # shared with the recognizer (one copy)


def _read_image(file_storage):
    """Turn an uploaded file (or webcam blob) into an OpenCV BGR image."""
    raw = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    """Serve the single-page web interface."""
    return send_file(os.path.join(WEB_DIR, "index.html"))


@app.route("/people")
def people():
    """Return the list of enrolled people (used to refresh the UI)."""
    return jsonify({"people": db.list_people(), "count": len(db)})


@app.route("/enroll", methods=["POST"])
def enroll():
    """Enroll one person from a name + several face images."""
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "message": "Please enter a name."}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"ok": False, "message": "No images were received."}), 400

    embeddings = []
    skipped = 0
    for f in files:
        img = _read_image(f)
        if img is None:
            skipped += 1
            continue
        emb = embedder.embed_single(img)        # one face -> 512-d vector
        if emb is None:
            skipped += 1                        # no face found in this photo
        else:
            embeddings.append(emb)

    if len(embeddings) < MIN_PHOTOS:
        return jsonify({
            "ok": False,
            "message": (f"Only {len(embeddings)} usable photo(s); need at least "
                        f"{MIN_PHOTOS}. {skipped} image(s) had no detectable face."),
        }), 400

    db.enroll_person(name, embeddings)          # average + store the signature
    db.save()                                   # persist to embeddings.pkl

    return jsonify({
        "ok": True,
        "message": (f"Enrolled '{name}' from {len(embeddings)} photo(s)"
                    + (f"; {skipped} skipped (no face)." if skipped else ".")),
        "people": db.list_people(),
        "count": len(db),
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """Identify the face in one uploaded/webcam image."""
    if "image" not in request.files:
        return jsonify({"ok": False, "message": "No image was received."}), 400
    img = _read_image(request.files["image"])
    if img is None:
        return jsonify({"ok": False, "message": "Could not read the image."}), 400

    if len(db) == 0:
        return jsonify({"ok": False, "message": "No one is enrolled yet."}), 400

    # Phase 4: the FaceRecognizer does embed -> nearest -> threshold decision.
    result = recognizer.recognize_single(img)
    if result is None:
        return jsonify({"ok": True, "found": False, "message": "No face detected."})

    result["ok"] = True
    result["found"] = True
    return jsonify(result)


@app.route("/remove", methods=["POST"])
def remove():
    """Delete one person from the database."""
    name = (request.form.get("name") or "").strip()
    if db.remove(name):
        db.save()
        return jsonify({"ok": True, "people": db.list_people(), "count": len(db)})
    return jsonify({"ok": False, "message": f"'{name}' is not in the database."}), 400


if __name__ == "__main__":
    print("\n  Web app ready.  Open this in your browser:\n")
    print("      http://127.0.0.1:5000\n")
    # use_reloader=False so the heavy FaceNet model is not loaded twice.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
