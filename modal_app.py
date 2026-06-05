"""
Modal deployment of the Face Recognition system.

Build + deploy from the project root with:
    modal deploy modal_app.py

Modal will:
  1) build the container image (first time: ~5-10 min; cached after)
  2) start the function on its servers
  3) print a public *.modal.run URL

The embeddings database is stored in a persistent Modal Volume named
'face-recognition-db', so enrolled people SURVIVE container restarts.
"""

import modal

# ---------------------------------------------------------------------------
# 1. Persistent volume - one shared file system that outlives the container.
#    embeddings.pkl is stored here, mounted at /data inside the container.
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("face-recognition-db", create_if_missing=True)


# ---------------------------------------------------------------------------
# 2. Container image (Modal's Python equivalent of a Dockerfile).
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    # System libraries OpenCV / PyTorch need at runtime.
    .apt_install("libglib2.0-0", "libgomp1")
    # CPU-only PyTorch wheel (the CUDA wheel is ~2 GB and useless on CPU Modal).
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    # The rest of the project dependencies.
    .pip_install(
        "facenet-pytorch",
        "opencv-python-headless",
        "numpy",
        "Pillow",
        "flask",
    )
    # Pre-download the FaceNet (vggface2) weights INTO the image so the first
    # request after a cold start is fast.
    .run_commands(
        "python -c \"from facenet_pytorch import InceptionResnetV1; "
        "InceptionResnetV1(pretrained='vggface2').eval()\""
    )
    # Bring our project code into the image at /code/src and /code/web.
    .add_local_dir("src", "/code/src")
    .add_local_dir("web", "/code/web")
)


# ---------------------------------------------------------------------------
# 3. The Modal App and the WSGI function that serves the Flask app.
# ---------------------------------------------------------------------------
app = modal.App("face-recognition", image=image)


@app.function(
    volumes={"/data": volume},      # mount the persistent volume at /data
    memory=2048,                    # 2 GB RAM (FaceNet uses ~1.5 GB)
    cpu=2.0,                        # 2 vCPUs
    timeout=120,                    # per-request timeout
    scaledown_window=300,           # keep warm 5 min after the last request
)
@modal.wsgi_app()
def web():
    """Returns the Flask WSGI app. Modal serves it on a public *.modal.run URL."""
    import os
    import sys

    # Point the database at the persistent volume BEFORE importing the modules.
    # database.py reads FACEREC_DB_PATH from env at import time.
    os.environ["FACEREC_DB_PATH"] = "/data/embeddings.pkl"

    sys.path.insert(0, "/code/src")

    # Patch FaceDatabase.save() so that every save also commits the Volume.
    # Without commit(), writes stay inside the running container and are lost
    # when Modal scales the container down.
    from database import FaceDatabase
    _original_save = FaceDatabase.save
    def save_and_commit(self, *args, **kwargs):
        _original_save(self, *args, **kwargs)
        volume.commit()
    FaceDatabase.save = save_and_commit

    # Import the existing Flask app and hand it to Modal.
    from web_app import app as flask_app
    return flask_app
