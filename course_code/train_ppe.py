"""Step 2: transfer YOLO-Master to the construction-site PPE scenario.

Fine-tunes the COCO-pretrained official weights on Construction-PPE (train split),
selects the best epoch on the val split. The held-out test split is evaluated
separately by test_ppe.py.

Usage:
    python course_code/train_ppe.py --model YOLO-Master-v0.1-N --epochs 100
    python course_code/train_ppe.py --model YOLO-Master-EsMoE-N --epochs 100
    python course_code/train_ppe.py --model YOLO-Master-v0.1-N --resume   # continue an interrupted run

Creating the file <run>/STOP pauses training right after the next epoch's checkpoint is saved
(used by train_gui.py).
"""

import argparse
import os
from pathlib import Path

from project_paths import prepare_data
prepare_data()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "course_code" / "configs" / "construction_ppe.yaml"


def dense_esmoe(trainer):
    """ES_MOE prunes to a single expert in eval mode by default, which does not match the
    dense training forward and collapses validation mAP after fine-tuning (documented in the
    official scripts/reproduce/_reproduce_common.py). Use the dense path for val and checkpoints."""
    from ultralytics.nn.modules.moe.modules import ES_MOE

    for target in (trainer.model, getattr(trainer.ema, "ema", None)):
        if target is None:
            continue
        for module in target.modules():
            if isinstance(module, ES_MOE):
                module.use_sparse_inference = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="YOLO-Master-v0.1-N")
    ap.add_argument("--hours", type=float, default=1.0, help="Ultralytics training budget in hours per run")
    ap.add_argument("--epochs", type=int, default=100)
    # 16 overflows the 8 GB RTX 4060 into shared system memory (~8 s/it); 8 uses ~5 GB (~3.5 it/s)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--cache", default="ram", help="ram | disk | none (1416 images fit easily in 16 GB RAM)")
    ap.add_argument("--fraction", type=float, default=1.0, help="use <1.0 only for a quick smoke test")
    ap.add_argument("--name", default=None, help="run name (default: model name)")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/weights/last.pt if it exists")
    ap.add_argument("--cls-pw", type=float, default=0.0,
                    help="inverse class-frequency loss weighting power (0=off, 1=full); targets rare no_* classes")
    args = ap.parse_args()
    if not 0 < args.hours <= 4:
        ap.error("--hours must be in (0, 4]")

    project = ROOT / "results" / ("smoke" if args.fraction < 1 else "ppe")
    run_dir = project / (args.name or (args.model + "-retrain"))
    last = run_dir / "weights" / "last.pt"
    stop_flag = run_dir / "STOP"
    stop_flag.unlink(missing_ok=True)
    running_flag = run_dir / "RUNNING"  # lets train_gui.py see runs started outside the GUI
    run_dir.mkdir(parents=True, exist_ok=True)
    running_flag.write_text(str(os.getpid()))

    resuming = args.resume and last.exists()
    model = YOLO(str(last if resuming else ROOT / "weights" / f"{args.model}.pt"))
    if "EsMoE" in args.model:
        model.add_callback("on_pretrain_routine_end", dense_esmoe)

    def pause_after_epoch(trainer):
        # last.pt (with optimizer state) has just been written; exiting here keeps it resumable.
        if stop_flag.exists():
            stop_flag.unlink(missing_ok=True)
            running_flag.unlink(missing_ok=True)
            print(f"[PAUSED] checkpoint saved after epoch {trainer.epoch + 1}: {last}", flush=True)
            os._exit(0)

    model.add_callback("on_model_save", pause_after_epoch)

    try:
        if resuming:
            print(f"[RESUME] continuing from {last}", flush=True)
            model.train(resume=True)
            return
        model.train(
            data=str(DATA),
            epochs=args.epochs,
            time=args.hours,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            cache=False if args.cache == "none" else args.cache,
            amp=True,
            seed=0,
            deterministic=False,  # faster cuDNN kernels on a laptop GPU; seed still fixed
            patience=50,
            cls_pw=args.cls_pw,
            fraction=args.fraction,
            project=str(project),
            name=run_dir.name,
            exist_ok=True,
        )
    finally:
        running_flag.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
