# ============================================================================
#  Face Recognition - Hugging Face Spaces Dockerfile
# ----------------------------------------------------------------------------
#  Builds a self-contained image that runs the Flask web app and includes the
#  pre-downloaded FaceNet weights, so cold starts are fast on the free CPU
#  Space (~1-2 GB RAM at runtime).
# ============================================================================

FROM python:3.12-slim

# System libraries that NumPy / Torch / opencv-python-headless need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# 1) CPU-only PyTorch.  The default CUDA wheel is ~2 GB and useless on a
#    free Space; the CPU wheel is ~200 MB.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# 2) Project libraries.  Use opencv-python-headless because there is no
#    display server inside the container (saves the X11 dependencies).
RUN pip install --no-cache-dir \
        facenet-pytorch \
        opencv-python-headless \
        numpy \
        Pillow \
        flask \
        gunicorn

# 3) Pre-download the FaceNet (vggface2) weights into the image so the first
#    request after startup doesn't pay the download cost.
RUN python -c "from facenet_pytorch import InceptionResnetV1; \
               InceptionResnetV1(pretrained='vggface2').eval()"

# 4) Copy the application source code.  The data/ and evaluation/ folders are
#    excluded by .dockerignore.
COPY src /code/src
COPY web /code/web

# Hugging Face Spaces expects the app on port 7860.
EXPOSE 7860

# Production WSGI server.  ONE worker because FaceNet eats ~1.5 GB of RAM
# and the free Space has 16 GB to spare but only one CPU.
CMD ["gunicorn", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:7860", \
     "--chdir", "/code/src", \
     "web_app:app"]
