"""Portable dataset configurations; generated files are not committed."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def prepare_data():
    cfg = ROOT/'course_code/configs'
    for filename, dataset in [('construction_ppe.yaml','construction-ppe'), ('coco_val.yaml','coco')]:
        p = cfg/filename
        data = yaml.safe_load(p.read_text(encoding='utf-8'))
        data['path'] = str(ROOT/'datasets'/dataset)
        p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
