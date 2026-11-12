"""Scenario-level evaluation: image-level PPE-violation alarm (ROC / AUC / threshold selection).

The site manager does not care about every box, only "does this frame contain a violation?".
For each image:  y = 1 if its ground truth contains any no_* box,
                 score = highest confidence among predicted no_* boxes (0 if none).
The alarm threshold is chosen on the VAL split (max F1) and then reported on the TEST split.

Usage:
    python course_code/alarm_analysis.py
"""

import csv
import json
from pathlib import Path

import numpy as np

from project_paths import prepare_data
prepare_data()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
PPE = ROOT / "datasets" / "construction-ppe"
RUNS = ROOT / "results" / "ppe"
OUT = ROOT / "results" / "alarm"
VIOL = {7: "no_helmet", 8: "no_goggle", 9: "no_gloves", 10: "no_boots"}
MODELS = ["YOLO-Master-v0.1-N", "YOLO-Master-v0.1-N-clspw0.5", "YOLO-Master-EsMoE-N"]
DEFAULT_CONF = 0.25  # Ultralytics / official predict() default


def image_scores(model, split):
    """Per image: ground-truth violation flags and max predicted confidence per violation class."""
    rows = []
    imgs = sorted((PPE / "images" / split).iterdir())
    # Results come back in input order; r.path is renamed to image0, image1, ... for list sources.
    preds = model.predict([str(p) for p in imgs], conf=0.001, iou=0.7, stream=True, verbose=False, batch=16)
    for img, r in zip(imgs, preds):
        stem = img.stem
        lab = PPE / "labels" / split / f"{stem}.txt"
        gt = {int(l.split()[0]) for l in lab.read_text().splitlines() if l.strip()} if lab.exists() else set()
        row = {"image": stem}
        cls, conf = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
        for k, n in VIOL.items():
            row[f"gt_{n}"] = int(k in gt)
            row[f"score_{n}"] = float(conf[cls == k].max()) if (cls == k).any() else 0.0
        row["gt_any"] = int(any(row[f"gt_{n}"] for n in VIOL.values()))
        row["score_any"] = max(row[f"score_{n}"] for n in VIOL.values())
        rows.append(row)
    return rows


def roc(y, s):
    """ROC points and AUC (trapezoid) without sklearn."""
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp, fp = np.cumsum(y)[distinct], np.cumsum(1 - y)[distinct]
    tpr = np.r_[0, tp / max(y.sum(), 1)]
    fpr = np.r_[0, fp / max((1 - y).sum(), 1)]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def prf(y, s, t):
    y, p = np.asarray(y), np.asarray(s) >= t
    tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, curves = [], {}
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    for name in MODELS:
        w = RUNS / name / "weights" / "best.pt"
        if not w.exists():
            continue
        model = YOLO(str(w))
        if "EsMoE" in name:
            from ultralytics.nn.modules.moe.modules import ES_MOE

            for m in model.model.modules():
                if isinstance(m, ES_MOE):
                    m.use_sparse_inference = False
        val, test = image_scores(model, "val"), image_scores(model, "test")
        for split, rows in (("val", val), ("test", test)):
            with open(OUT / f"scores_{name}_{split}.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
        yv, sv = [r["gt_any"] for r in val], [r["score_any"] for r in val]
        yt, st = [r["gt_any"] for r in test], [r["score_any"] for r in test]
        f1s = [prf(yv, sv, t)[2] for t in grid]
        t_best = float(grid[int(np.argmax(f1s))])
        fpr, tpr, auc = roc(yt, st)
        per_class = {}
        for n in VIOL.values():
            y = [r[f"gt_{n}"] for r in test]
            if sum(y):
                f_, t_, a_ = roc(y, [r[f"score_{n}"] for r in test])
                per_class[n] = {"fpr": f_.tolist(), "tpr": t_.tolist(), "auc": a_}
        curves[name] = {
            "any": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc},
            "per_class": per_class,
            "threshold_sweep": {"t": grid.tolist(), "test": [prf(yt, st, t) for t in grid], "val_f1": f1s},
        }
        for label, t in (("default conf 0.25", DEFAULT_CONF), ("val-tuned", t_best)):
            P, R, F = prf(yt, st, t)
            summary.append({"model": name, "threshold_rule": label, "threshold": t, "test_images": len(yt),
                            "positives": int(sum(yt)), "precision": round(P * 100, 1), "recall": round(R * 100, 1),
                            "F1": round(F * 100, 1), "AUC": round(auc * 100, 1)})
            print(json.dumps(summary[-1]))
    (OUT / "curves.json").write_text(json.dumps(curves), encoding="utf-8")
    with open(OUT / "alarm_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
