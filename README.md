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

## Team

Rawan Mohamed · Malak Wael · Omar Mohamed · Karim Mohamed · Mohamed Gasser · Ziad Amr
