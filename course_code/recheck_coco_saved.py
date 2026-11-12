"""Recompute official COCO AP from all saved predictions and official annotations."""
import argparse
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotations', type=Path, default=ROOT/'datasets/coco/annotations/instances_val2017.json')
    args = ap.parse_args()
    gt = COCO(str(args.annotations))
    out = {}
    for name in ('YOLO-Master-EsMoE-N','YOLO-Master-v0.1-N'):
        pred = ROOT/'results/coco'/name/'predictions.json'
        data = json.loads(pred.read_text())
        ids = {row['image_id'] for row in data}
        assert ids <= set(gt.getImgIds())
        ev = COCOeval(gt,gt.loadRes(data),'bbox')
        ev.params.imgIds = gt.getImgIds()  # Include images with zero predictions.
        ev.evaluate(); ev.accumulate(); ev.summarize()
        out[name] = {'evaluated_images':len(ev.params.imgIds),'images_with_predictions':len(ids),
                     'prediction_boxes':len(data),'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),
                     'AP75':float(ev.stats[2]),'source':'saved official-weight predictions; not a new training run'}
    (ROOT/'verification/coco_recheck.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
