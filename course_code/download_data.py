"""Download the GitHub release's real PPE data and verify the archive SHA256."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    meta = json.loads((ROOT/'release_assets.json').read_text())['construction-ppe-dataset.zip']
    target = ROOT/'release_assets/construction-ppe-dataset.zip'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(meta['url'],target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('SHA256 mismatch; remove archive and retry')
    with zipfile.ZipFile(target) as z:
        for n in z.namelist():
            dest = (ROOT/n).resolve()
            if not dest.is_relative_to(ROOT.resolve()) or PurePosixPath(n).is_absolute():
                raise ValueError('Invalid archive path')
        z.extractall(ROOT)
    print('Verified and extracted real PPE images and labels')
