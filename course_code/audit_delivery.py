"""Check real dataset files, labels, split leakage and experiment timing."""
from pathlib import Path
import hashlib
import csv
import json
import platform
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ds = ROOT/'datasets/construction-ppe'
    split_hashes = {}
    manifest = []
    stats = {}
    for split in ('train','val','test'):
        images = sorted((ds/'images'/split).glob('*'))
        images = [p for p in images if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
        boxes = Counter()
        hashes = set()
        for image in images:
            with Image.open(image) as im:
                im.verify()
            digest = sha(image)
            hashes.add(digest)
            label = ds/'labels'/split/(image.stem+'.txt')
            assert label.exists(), image
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                cls, *coords = map(float, line.split())
                assert len(coords) == 4 and cls.is_integer() and 0 <= cls <= 10
                assert all(0 <= x <= 1 for x in coords), label
                boxes[int(cls)] += 1
            manifest.append({'split':split,'image':image.relative_to(ROOT).as_posix(),
                             'image_sha256':digest,'label_sha256':sha(label)})
        split_hashes[split] = hashes
        orphan = [p.name for p in (ds/'labels'/split).glob('*.txt') if p.stem not in {i.stem for i in images}]
        stats[split] = {'images':len(images),'boxes':sum(boxes.values()),'per_class':dict(boxes),'orphan_labels':orphan}
    overlaps = {f'{a}/{b}':len(split_hashes[a] & split_hashes[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    runs = {}
    for p in (ROOT/'results/ppe').glob('*/results.csv'):
        rows = list(csv.DictReader(p.open(encoding='utf-8-sig')))
        rows = [{k.strip():v.strip() for k,v in r.items()} for r in rows]
        runs[p.parent.name] = {'epochs':len(rows),'training_seconds':float(rows[-1]['time'])}
    out = ROOT/'verification'
    out.mkdir(exist_ok=True)
    with (out/'dataset_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
    evidence = {}
    for folder in ('logs','weights','results'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.pt','.csv','.json','.log'}:
                evidence[p.relative_to(ROOT).as_posix()] = sha(p)
    (out/'evidence_sha256.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    report = {'python':platform.python_version(),'dataset':stats,'exact_split_overlap':overlaps,
              'training':runs,'total_training_seconds':sum(r['training_seconds'] for r in runs.values()),
              'provenance':'Existing local experiments from 2026-11-02 and 2026-11-03; re-audited 2026-11-12',
              'note':'Exact SHA256 separation does not establish person/site/video independence.'}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    assert not any(overlaps.values()), 'Exact image leakage across provided splits'
    assert report['total_training_seconds'] < 14400

if __name__ == '__main__':
    main()
