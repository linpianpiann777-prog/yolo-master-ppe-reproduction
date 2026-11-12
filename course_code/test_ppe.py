"""Step 3: evaluate the fine-tuned PPE models on the held-out test split (141 images).

Usage:
    python course_code/test_ppe.py
"""

import argparse
import csv
import json
from pathlib import Path

from project_paths import prepare_data
prepare_data()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "course_code" / "configs" / "construction_ppe.yaml"
RUNS = ROOT / "results" / "ppe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS), help="folder holding <model>/weights/best.pt training runs")
    runs = Path(ap.parse_args().runs).resolve()  # ultralytics nests relative projects under runs/detect

    rows = []
    for best in sorted(runs.glob("*/weights/best.pt")):
        name = best.parents[1].name
        model = YOLO(str(best))
        if "EsMoE" in name:
            from ultralytics.nn.modules.moe.modules import ES_MOE

            for module in model.model.modules():
                if isinstance(module, ES_MOE):
                    module.use_sparse_inference = False
        m = model.val(data=str(DATA), split="test", imgsz=640, batch=16, project=str(runs), name=f"{name}_test", exist_ok=True)
        row = {
            "model": name,
            "mAP50-95": round(m.box.map * 100, 1),
            "mAP50": round(m.box.map50 * 100, 1),
            "precision": round(m.box.mp * 100, 1),
            "recall": round(m.box.mr * 100, 1),
            "inference_ms": round(m.speed["inference"], 2),
        }
        # per-class AP50 for the report
        for i, c in enumerate(m.box.ap_class_index):
            row[f"AP50_{m.names[int(c)]}"] = round(m.box.ap50[i] * 100, 1)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k[0].isalpha() or k.startswith("AP50_"), k))
    with open(runs / "test_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {runs / 'test_summary.csv'}")


if __name__ == "__main__":
    main()
