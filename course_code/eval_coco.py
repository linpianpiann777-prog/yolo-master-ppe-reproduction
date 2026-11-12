"""Step 1: reproduce the paper's COCO val2017 numbers with the official YOLO-Master weights.

Usage:
    python course_code/eval_coco.py                      # all downloaded models
    python course_code/eval_coco.py --models YOLO-Master-EsMoE-N
"""

import argparse
import csv
import json
from pathlib import Path

from project_paths import prepare_data
prepare_data()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "course_code" / "configs" / "coco_val.yaml"
OUT = ROOT / "results" / "coco"
ANNO = ROOT / "datasets" / "coco" / "annotations" / "instances_val2017.json"

# Reported numbers from the official README (mAP50-95 / mAP50, COCO val2017, 640px).
# The paper abstract reports 42.4 AP for YOLO-Master-N.
REPORTED = {
    "YOLO-Master-EsMoE-N": (42.7, 58.7),
    "YOLO-Master-EsMoE-S": (48.9, 60.3),
    "YOLO-Master-v0.1-N": (42.9, 59.2),
    "YOLO-Master-v0.1-S": (48.9, 66.2),
}


def coco_official(pred_json):
    """Official COCO metrics (pycocotools), the protocol used in the paper."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(ANNO))
    e = COCOeval(gt, gt.loadRes(str(pred_json)), "bbox")
    e.evaluate()
    e.accumulate()
    e.summarize()
    keys = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
    return {k: round(float(v) * 100, 1) for k, v in zip(keys, e.stats[:6])}


def main():
    ap = argparse.ArgumentParser()
    # Nano models only by default (lightweight, matches the paper's headline YOLO-Master-N result).
    ap.add_argument("--models", nargs="*", default=["YOLO-Master-EsMoE-N", "YOLO-Master-v0.1-N"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--reuse", action="store_true", help="reuse existing predictions.json, skip inference")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.models:
        run = OUT / name
        model = YOLO(str(ROOT / "weights" / f"{name}.pt"))
        if not (args.reuse and (run / "predictions.json").exists() and (run / "ultra_metrics.json").exists()):
            m = model.val(
                data=str(DATA),
                imgsz=args.imgsz,
                batch=args.batch,
                conf=0.001,
                iou=0.7,
                save_json=True,  # writes predictions.json in COCO format
                project=str(OUT),
                name=name,
                exist_ok=True,
                plots=True,
            )
            ultra = {
                "ultra_mAP50-95": round(m.box.map * 100, 1),
                "ultra_mAP50": round(m.box.map50 * 100, 1),
                "precision": round(m.box.mp * 100, 1),
                "recall": round(m.box.mr * 100, 1),
                "inference_ms": round(m.speed["inference"], 2),
            }
            (run / "ultra_metrics.json").write_text(json.dumps(ultra), encoding="utf-8")
        ultra = json.loads((run / "ultra_metrics.json").read_text(encoding="utf-8"))
        rep_ap, rep_ap50 = REPORTED.get(name, (None, None))
        row = {
            "model": name,
            "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
            **coco_official(run / "predictions.json"),
            "reported_AP": rep_ap,
            "reported_AP50": rep_ap50,
            **ultra,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
