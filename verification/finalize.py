"""Verify deliverables and create a portable submission archive."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
report = ROOT/'report/YOLO-Master_PPE_Course_Design.docx'
pdf = report.with_suffix('.pdf')
assert len(PdfReader(pdf).pages) == 30
with zipfile.ZipFile(report) as z:
    assert z.testzip() is None
    xml = z.read('word/document.xml').decode()
    assert '143.77' in xml and '8626.22' in xml
    assert '深度学习在施工现场' in xml
rows = list(csv.DictReader((ROOT/'results/ppe/test_summary.csv').open()))
assert {r['model']:(r['mAP50'],r['mAP50-95']) for r in rows} == {
    'YOLO-Master-EsMoE-N':('55.9','27.6'),
    'YOLO-Master-v0.1-N':('56.8','27.6'),
    'YOLO-Master-v0.1-N-clspw0.5':('54.0','26.9')}
co = json.loads((ROOT/'verification/coco_recheck.json').read_text())
assert all(r['evaluated_images']==5000 for r in co.values())
assert 'frames with violations: 27' in (ROOT/'verification/demo_recheck.log').read_text(encoding='utf-8',errors='replace')
meta = json.loads((ROOT/'release_assets.json').read_text())
assets = json.loads((ROOT/'verification/github_assets.json').read_text())
for a in assets:
    assert a['digest']=='sha256:'+meta[a['name']]['sha256']
assert '1 epochs completed' in (ROOT/'verification/training_smoke.log').read_text(encoding='utf-8',errors='replace')
check = {'passed':True,'report_pages':30,'dataset_images':1416,'formal_training_minutes':8626.22/60,
         'ppe_test_models_reexecuted':3,'coco_images_recomputed':5000,
         'training_smoke_completed':True,'demo_images_reexecuted':141,'demo_alert_images':27,
         'release_assets_digest_verified':True,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),
         'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
         'rendering':'Word COM PDF export; PDFium render; all 30 pages visually reviewed. Bundled LibreOffice renderer unavailable.'}
(ROOT/'verification/final_checks.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))

# Git supplies the public source-file list; include the offline images and saved COCO predictions.
tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
files = {ROOT/p for p in tracked if p and (ROOT/p).is_file()}
for folder in ('datasets/construction-ppe','results/coco','results/ablation'):
    files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix!='.cache')
files.add(ROOT/'verification/final_checks.json')
target = ROOT.parent/'YOLO-Master_PPE_submission.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted(files):
        if p.name=='last_healthy.pt': continue
        z.write(p,'yolo-master-ppe/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
print('Submission archive:',target,target.stat().st_size,'bytes')
