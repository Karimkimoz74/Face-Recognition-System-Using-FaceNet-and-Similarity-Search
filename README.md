# Face Recognition System Using FaceNet and Similarity Search

Neural Networks Project — Semester 8.

A face recognition system that uses a pre-trained **FaceNet** model
(`InceptionResnetV1`) to convert faces into 512-dimensional embeddings, then
recognizes people with **similarity search** (Euclidean distance + threshold).

## Project structure

```
Project/
├── requirements.txt        # Python dependencies
├── src/
│   ├── check_camera.py     # Phase 1: find & test the Iriun webcam
│   ├── test_detection.py   # Phase 1: verify MTCNN face detection
│   ├── embedder.py         # (Phase 2) image -> 512-d embedding
│   ├── database.py         # (Phase 3) enroll / save / search
│   ├── recognize.py        # (Phase 4) distance + threshold
│   ├── app_image.py        # (Phase 5) static-image mode
│   └── app_webcam.py       # (Phase 5) real-time webcam mode
├── data/faces/<person>/    # enrollment photos, one folder per person
└── evaluation/             # metrics + results
```

## Setup

```powershell
pip install -r requirements.txt
```

## Phase 1 — Setup & exploration

1. **Find your camera.** Start the Iriun desktop app and connect your phone, then:
   ```powershell
   python src/check_camera.py
   ```
   It lists every working camera index. Note the index for Iriun.

2. **Preview a camera** to confirm the live feed works:
   ```powershell
   python src/check_camera.py --preview 1
   ```
   (replace `1` with your Iriun index; press `q` to quit)

3. **Test face detection** with MTCNN:
   ```powershell
   python src/test_detection.py --webcam 1        # detect from webcam
   python src/test_detection.py --image photo.jpg # or from an image file
   ```

> Iriun note: the Iriun desktop app must be running and the phone connected
> **before** launching these scripts, or the camera will not be found.

## Phase 2 — Face embedding

`src/embedder.py` turns a face into a 512-number embedding vector
(image → MTCNN crop → FaceNet → embedding). It is imported by later phases,
and can also be run directly to test it:

```powershell
python src/embedder.py --image photo.jpg          # embed one image, show info
python src/embedder.py --compare a.jpg b.jpg       # distance between two faces
python src/embedder.py --webcam 1                  # live: SPACE = embed, q = quit
```

Same person → small distance (< ~1.0). Different people → large distance (> ~1.1).

> The first run downloads the FaceNet weights (~110 MB) automatically.

## Team

Rawan Mohamed · Malak Wael · Omar Mohamed · Karim Mohamed · Mohamed Gasser · Ziad Amr
