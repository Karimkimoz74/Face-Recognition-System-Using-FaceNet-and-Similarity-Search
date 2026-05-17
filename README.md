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

## Phase 3 — Face database / enrollment

`src/database.py` stores the people the system knows. Each person is saved as
one averaged 512-d embedding ("face signature") in `embeddings.pkl`.

```powershell
python src/database.py --capture "Karim" --webcam 1   # enroll a person via webcam
python src/database.py --enroll-folder                # enroll from data/faces/ folders
python src/database.py --list                         # list enrolled people
python src/database.py --remove "Karim"               # remove a person
```

Webcam enrollment: press `SPACE` to capture each photo (at least 3, ~5 recommended),
`q` to finish. To enroll from folders instead, put photos in
`data/faces/<PersonName>/`.

## Web app (recommended) — enrollment & testing in the browser

`src/web_app.py` runs a small website with two sections:

- **Enrollment** — type a name, add face photos (upload *or* webcam capture).
- **Testing** — identify a face (upload *or* live webcam recognition).

```powershell
python src/web_app.py
```

Then open **http://127.0.0.1:5000** in your browser. The web app reuses
`embedder.py`, `database.py` and `recognize.py`, and saves to `embeddings.pkl`.

## Phase 4 — Recognition logic

`src/recognize.py` makes the "name vs Unknown" decision. The `FaceRecognizer`
class embeds a face, finds the nearest enrolled person, and applies the
`RECOGNITION_THRESHOLD` (currently `1.0`, tuned in Phase 6). It also reports a
0-100% confidence score.

```powershell
python src/recognize.py --image photo.jpg   # recognize faces in an image
python src/recognize.py --webcam 1           # live recognition from a camera
```

The web app and the apps all use this one module for recognition.

## Team

Rawan Mohamed · Malak Wael · Omar Mohamed · Karim Mohamed · Mohamed Gasser · Ziad Amr
