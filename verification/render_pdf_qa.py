from pathlib import Path
import json
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
pdf = ROOT/'report/YOLO-Master_PPE_Course_Design.pdf'
out = ROOT/'verification/pages'
out.mkdir(exist_ok=True)
for stale in out.glob('page-*.png'):
    stale.unlink()
d = pdfium.PdfDocument(pdf)
for i in range(len(d)):
    im = d[i].render(scale=1.5).to_pil().convert('RGB')
    im.save(out/f'page-{i+1}.png')
for group in range((len(d)+3)//4):
    sheet = Image.new('RGB',(1300,1880),'#cccccc')
    for k in range(4):
        n = group*4+k+1
        p = out/f'page-{n}.png'
        if not p.exists(): continue
        im = Image.open(p); im.thumbnail((640,910))
        x=(k%2)*650; y=(k//2)*940
        sheet.paste(im,(x,y+24)); ImageDraw.Draw(sheet).text((x+10,y+5),str(n),fill='black')
    sheet.save(ROOT/f'verification/contact-{group+1}.jpg',quality=90)
texts = ['\n'.join(line.rstrip() for line in (p.extract_text() or '').splitlines()) for p in PdfReader(pdf).pages]
(ROOT/'verification/report_text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
print(json.dumps({'pages':len(d),'text_chars':[len(t) for t in texts],'contacts':(len(d)+3)//4}))
