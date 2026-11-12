"""Ablation: ES-MoE sparse (official default) vs dense inference routing.

Evaluates the same weights twice, flipping ES_MOE.use_sparse_inference:
  - official COCO weights YOLO-Master-EsMoE-N on COCO val2017
  - our PPE fine-tuned EsMoE-N on the PPE test split
mAP comes from the same (Ultralytics) evaluator for both modes, so the comparison is like-for-like.

Usage:
    python course_code/ablation_routing.py
"""

import csv
import json
from pathlib import Path

from project_paths import prepare_data
prepare_data()

from ultralytics import YOLO
from ultralytics.nn.modules.moe.modules import ES_MOE

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "course_code" / "configs"
OUT = ROOT / "results" / "ablation"

CASES = [
    ("COCO / official EsMoE-N", ROOT / "weights" / "YOLO-Master-EsMoE-N.pt", CFG / "coco_val.yaml", "val"),
    ("PPE / fine-tuned EsMoE-N", ROOT / "results" / "ppe" / "YOLO-Master-EsMoE-N" / "weights" / "best.pt",
     CFG / "construction_ppe.yaml", "test"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, weights, data, split in CASES:
        if not weights.exists():
            print(f"skip {case}: {weights} not found")
            continue
        for mode, sparse in (("sparse (official default)", True), ("dense (train-consistent)", False)):
            model = YOLO(str(weights))
            n = 0
            for m in model.model.modules():
                if isinstance(m, ES_MOE):
                    m.use_sparse_inference = sparse
                    n += 1
            tag = f"{'coco' if 'COCO' in case else 'ppe'}_{'sparse' if sparse else 'dense'}"
            r = model.val(data=str(data), split=split, imgsz=640, batch=16, conf=0.001, iou=0.7,
                          project=str(OUT), name=tag, exist_ok=True, plots=False)
            row = {"case": case, "mode": mode, "es_moe_blocks": n,
                   "mAP50-95": round(r.box.map * 100, 1), "mAP50": round(r.box.map50 * 100, 1),
                   "precision": round(r.box.mp * 100, 1), "recall": round(r.box.mr * 100, 1),
                   "inference_ms": round(r.speed["inference"], 2)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    with open(OUT / "routing.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {OUT / 'routing.csv'}")


if __name__ == "__main__":
    main()
