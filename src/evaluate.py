"""
Phase 6 - Evaluation.

Measures how well the recognition system works and produces the numbers for
the project report:

  - Accuracy        : % of test faces handled correctly
  - Threshold sweep : tries many threshold values and finds the best one
  - Confusion matrix: shows which people get mixed up
  - Speed (FPS)     : how fast the system processes a face

Test data lives in  data/test/  with one sub-folder per label:

    data/test/Karim/      <- photos of the enrolled person "Karim"
    data/test/Rawan/      <- photos of the enrolled person "Rawan"
    data/test/Unknown/    <- photos of strangers (NOT enrolled)

A folder named after an enrolled person  -> those photos SHOULD be recognised
as that person. The 'Unknown' folder     -> those photos SHOULD be rejected.

Run it with:
    python src/evaluate.py
"""

import os
import time
import warnings

import cv2

from embedder import FaceEmbedder
from database import FaceDatabase
from recognize import RECOGNITION_THRESHOLD

warnings.filterwarnings("ignore", category=FutureWarning)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
REPORT_PATH = os.path.join(EVAL_DIR, "report.txt")
PLOT_PATH = os.path.join(EVAL_DIR, "threshold_accuracy.png")

UNKNOWN = "Unknown"
IMAGE_EXT = (".jpg", ".jpeg", ".png")

# The range of threshold values to try in the sweep.
SWEEP_LO, SWEEP_HI, SWEEP_STEP = 0.40, 1.60, 0.02


# ==========================================================================
#  Step 1 - turn every test image into a (true label, nearest, distance) sample
# ==========================================================================
def collect_samples(embedder, db):
    """
    Embed every test image once. Returns (samples, timings):
      samples - list of {"true": label, "nearest": name, "distance": float}
      timings - seconds taken per embedding (for the speed measurement)
    """
    samples, timings = [], []
    folders = sorted(d for d in os.listdir(TEST_DIR)
                     if os.path.isdir(os.path.join(TEST_DIR, d)))

    for folder in folders:
        is_unknown = folder.lower() == "unknown"
        if not is_unknown and folder not in db.people:
            print(f"  note: '{folder}' is not an enrolled person and is not "
                  f"'Unknown' - skipped")
            continue

        true_label = UNKNOWN if is_unknown else folder
        folder_path = os.path.join(TEST_DIR, folder)
        images = [f for f in sorted(os.listdir(folder_path))
                  if f.lower().endswith(IMAGE_EXT)]

        used = 0
        for fname in images:
            bgr = cv2.imread(os.path.join(folder_path, fname))
            if bgr is None:
                continue
            start = time.perf_counter()
            emb = embedder.embed_single(bgr)
            timings.append(time.perf_counter() - start)
            if emb is None:
                continue                        # no face detected - skip
            name, distance = db.find_nearest(emb)
            samples.append({"true": true_label, "nearest": name,
                            "distance": float(distance)})
            used += 1
        print(f"  {folder}: {used} usable test image(s)")

    return samples, timings


# ==========================================================================
#  Step 2 - scoring helpers
# ==========================================================================
def predict(sample, threshold):
    """The system's answer for one sample at a given threshold."""
    return sample["nearest"] if sample["distance"] <= threshold else UNKNOWN


def accuracy_at(samples, threshold):
    """Return (overall, known_acc, unknown_acc, balanced) as fractions 0..1."""
    known = [s for s in samples if s["true"] != UNKNOWN]
    strangers = [s for s in samples if s["true"] == UNKNOWN]

    def rate(group):
        if not group:
            return None
        correct = sum(1 for s in group if predict(s, threshold) == s["true"])
        return correct / len(group)

    overall = (sum(1 for s in samples if predict(s, threshold) == s["true"])
               / len(samples))
    known_acc = rate(known)
    unknown_acc = rate(strangers)
    # "balanced" = average of the two groups, so a lopsided test set (e.g. many
    # strangers, few known) cannot hide a bad score in one group.
    parts = [a for a in (known_acc, unknown_acc) if a is not None]
    balanced = sum(parts) / len(parts) if parts else overall
    return overall, known_acc, unknown_acc, balanced


def sweep(samples):
    """Try every threshold in the range; return (rows, best_threshold)."""
    rows = []
    best_t, best_balanced = SWEEP_LO, -1.0
    t = SWEEP_LO
    while t <= SWEEP_HI + 1e-9:
        overall, known, unknown, balanced = accuracy_at(samples, t)
        rows.append((round(t, 2), overall, known, unknown, balanced))
        if balanced > best_balanced:
            best_balanced, best_t = balanced, round(t, 2)
        t += SWEEP_STEP
    return rows, best_t


def confusion_matrix(samples, threshold):
    """Build a {actual -> {predicted -> count}} table at one threshold."""
    labels = sorted(set(s["true"] for s in samples)
                    | set(predict(s, threshold) for s in samples))
    matrix = {a: {p: 0 for p in labels} for a in labels}
    for s in samples:
        matrix[s["true"]][predict(s, threshold)] += 1
    return labels, matrix


def pct(value):
    """Format a 0..1 fraction as a percentage string."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ==========================================================================
#  Step 3 - the threshold-vs-accuracy plot (optional, needs matplotlib)
# ==========================================================================
def save_plot(rows, best_t):
    try:
        import matplotlib
        matplotlib.use("Agg")               # no window - just save a file
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the plot.")
        return

    thresholds = [r[0] for r in rows]
    overall = [r[1] * 100 for r in rows]
    known = [(r[2] * 100 if r[2] is not None else float("nan")) for r in rows]
    strangers = [(r[3] * 100 if r[3] is not None else float("nan")) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, overall, linewidth=2, label="overall")
    plt.plot(thresholds, known, label="known faces recognised")
    plt.plot(thresholds, strangers, label="strangers rejected")
    plt.axvline(best_t, color="red", linestyle="--", label=f"best = {best_t}")
    plt.xlabel("threshold")
    plt.ylabel("accuracy (%)")
    plt.title("Recognition accuracy vs. threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(EVAL_DIR, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {PLOT_PATH}")


# ==========================================================================
#  Step 4 - tie it all together
# ==========================================================================
def main():
    print("Loading the model and the database ...")
    embedder = FaceEmbedder()
    db = FaceDatabase.load()

    if len(db) == 0:
        print("\nThe database is empty - enroll people first (use the web app).")
        return
    if not os.path.isdir(TEST_DIR):
        print(f"\nNo test folder found at: {TEST_DIR}")
        print("Create it with one sub-folder per label, for example:")
        print("  data/test/Karim/      (photos of the enrolled person Karim)")
        print("  data/test/Unknown/    (photos of strangers)")
        return

    print("\nEmbedding the test images ...")
    samples, timings = collect_samples(embedder, db)
    if not samples:
        print("\nNo usable test images were found in data/test/.")
        return

    # Collect every report line so we can print AND save it.
    lines = []
    def out(text=""):
        print(text)
        lines.append(text)

    n_known = sum(1 for s in samples if s["true"] != UNKNOWN)
    n_strangers = len(samples) - n_known

    out("=" * 62)
    out(" FACE RECOGNITION  -  EVALUATION REPORT")
    out("=" * 62)
    out(f"Enrolled people : {len(db)}  ({', '.join(db.list_people())})")
    out(f"Test images     : {len(samples)}  "
        f"({n_known} known, {n_strangers} stranger)")
    out("")

    # ---- accuracy at the current (un-tuned) threshold --------------------
    o, k, u, b = accuracy_at(samples, RECOGNITION_THRESHOLD)
    out(f"-- At the CURRENT threshold ({RECOGNITION_THRESHOLD}) --")
    out(f"   overall accuracy : {pct(o)}")
    out(f"   known faces      : {pct(k)}   (recognised correctly)")
    out(f"   stranger faces   : {pct(u)}   (rejected as Unknown)")
    out("")

    # ---- threshold sweep -------------------------------------------------
    rows, best_t = sweep(samples)
    o2, k2, u2, b2 = accuracy_at(samples, best_t)
    out(f"-- Threshold SWEEP ({SWEEP_LO} to {SWEEP_HI}) --")
    out(f"   >>> BEST threshold : {best_t} <<<")
    out(f"   overall accuracy   : {pct(o2)}")
    out(f"   known faces        : {pct(k2)}")
    out(f"   stranger faces     : {pct(u2)}")
    out(f"   balanced accuracy  : {pct(b2)}")
    out("")
    out("   threshold | overall | known  | stranger | balanced")
    out("   ----------+---------+--------+----------+---------")
    for (t, ov, kn, un, ba) in rows:
        if round((t * 100)) % 10 == 0:          # print every 0.10 step
            out(f"     {t:.2f}    | {pct(ov):>7} | {pct(kn):>6} | "
                f"{pct(un):>8} | {pct(ba):>7}")
    out("")

    # ---- confusion matrix at the best threshold --------------------------
    out(f"-- Confusion matrix (at best threshold {best_t}) --")
    out("   rows = actual person, columns = system's answer")
    labels, matrix = confusion_matrix(samples, best_t)
    for i, label in enumerate(labels):
        out(f"   [{i}] {label}")
    out("       " + "".join(f"{i:>5}" for i in range(len(labels))))
    for i, actual in enumerate(labels):
        cells = "".join(f"{matrix[actual][p]:>5}" for p in labels)
        out(f"   [{i}]{cells}")
    out("")

    # ---- speed -----------------------------------------------------------
    if timings:
        avg = sum(timings) / len(timings)
        out("-- Speed --")
        out(f"   average time per face : {avg * 1000:.1f} ms")
        out(f"   throughput            : {1.0 / avg:.1f} faces / second")
        out("")

    out("=" * 62)
    out("Recommendation: set RECOGNITION_THRESHOLD in src/recognize.py")
    out(f"to {best_t} (the best value found above).")
    out("=" * 62)

    # ---- save the report + plot -----------------------------------------
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {REPORT_PATH}")
    save_plot(rows, best_t)


if __name__ == "__main__":
    main()
