"""Step 4: scenario application — flag PPE violations on images, a folder, a video or a webcam.

Boxes are drawn at --conf; a frame raises an alarm when a no_helmet / no_gloves / no_goggle / no_boots
box reaches the alarm threshold. By default that threshold is the one chosen on the validation split
by alarm_analysis.py (results/alarm/alarm_summary.csv). Annotated outputs are saved to results/demo.

Usage:
    python course_code/ppe_demo.py --source datasets/construction-ppe/images/test
    python course_code/ppe_demo.py --source 0          # webcam
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from project_paths import prepare_data
prepare_data()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIOLATIONS = {"no_helmet", "no_gloves", "no_goggle", "no_boots"}


def tuned_threshold(run_name, default=0.25):
    """Alarm threshold selected on the val split for this run, if alarm_analysis.py has been run."""
    p = ROOT / "results" / "alarm" / "alarm_summary.csv"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["model"] == run_name and r["threshold_rule"] == "val-tuned":
                    return float(r["threshold"])
    return default


def find_violations(result, alarm_thr):
    """Return {violation class: highest confidence} for violation boxes at or above the alarm threshold."""
    hits = {}
    for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        name = result.names[int(c)]
        if name in VIOLATIONS and s >= alarm_thr:
            hits[name] = max(s, hits.get(name, 0.0))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "results" / "ppe" / "YOLO-Master-v0.1-N" / "weights" / "best.pt"))
    ap.add_argument("--source", default=str(ROOT / "datasets" / "construction-ppe" / "images" / "test"))
    ap.add_argument("--conf", type=float, default=0.25, help="box display threshold")
    ap.add_argument("--alarm-thr", type=float, default=None, help="alarm threshold (default: val-tuned)")
    args = ap.parse_args()

    run_name = Path(args.weights).parents[1].name
    alarm_thr = args.alarm_thr if args.alarm_thr is not None else tuned_threshold(run_name)
    print(f"model {run_name}, alarm threshold {alarm_thr:.2f}")

    model = YOLO(args.weights)
    total = Counter()
    flagged = 0
    # Predict down to the alarm threshold so violation boxes between it and --conf can still raise an alarm.
    for r in model.predict(args.source, conf=min(args.conf, alarm_thr), stream=True, save=True,
                           project=str(ROOT / "results"), name="demo", exist_ok=True):
        total.update(r.names[int(c)] for c in r.boxes.cls)
        hits = find_violations(r, alarm_thr)
        if hits:
            flagged += 1
            print(f"[ALERT] {Path(r.path).name}: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(hits.items())))
    print(f"frames with violations: {flagged}; detections: {dict(total)}")


if __name__ == "__main__":
    main()
