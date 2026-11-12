"""Finalize Word typography, three-line tables and clickable internal references.
Run after refreshing the Word table of contents, before final PDF export.
"""
from pathlib import Path
import re,json,zipfile,copy
from lxml import etree as E
R=Path(__file__).resolve().parents[1];P=R/'report/YOLO-Master_PPE_Course_Design.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main';N={'w':W};q=lambda n:'{'+W+'}'+n

def child(parent,name):
 x=parent.find(q(name))
 if x is None:
  x=E.Element(q(name))
  if name in ['rPr','pPr','tblPr','tcPr']:parent.insert(0,x)
  else:parent.append(x)
 return x

def fmt(rpr,size=21):
 fonts=child(rpr,'rFonts');fonts.attrib.clear()
 for k in ['ascii','eastAsia','hAnsi','cs']:fonts.set(q(k),'宋体')
 color=child(rpr,'color');color.attrib.clear();color.set(q('val'),'000000')
 for name in ['sz','szCs']:child(rpr,name).set(q('val'),str(size))

def text(p):return ''.join(p.xpath('.//w:t/text()',namespaces=N))
with zipfile.ZipFile(P) as z:data={n:z.read(n) for n in z.namelist()}
root=E.fromstring(data['word/document.xml']);styles=E.fromstring(data['word/styles.xml'])
style_names={s.get(q('styleId')):s.find(q('name')).get(q('val'),'').lower() for s in styles.findall(q('style'))}
for s in styles.findall(q('style')):
 name=style_names[s.get(q('styleId'))];size=28 if name in ['heading 1','title'] else 24 if name in ['heading 2','heading 3'] else 21
 fmt(child(s,'rPr'),size)
fmt(child(child(child(styles,'docDefaults'),'rPrDefault'),'rPr'))
paras=root.findall('.//w:p',N);body=root.find(q('body'));prose=0
cover=True
for p in paras:
 t=text(p).strip();pr=child(p,'pPr');sid=pr.find(q('pStyle'));name=style_names.get(sid.get(q('val')),'') if sid is not None else ''
 size=(84 if cover and t in ['深度学习在施工现场','个人防护装备检测场景','的应用实验报告'] else 28 if cover and t else 28 if name in ['heading 1','title'] or t in ['摘  要','目　录'] else 24 if name in ['heading 2','heading 3'] else 21)
 for run in p.findall('.//w:r',N):fmt(child(run,'rPr'),size)
 if not cover and t and p.getparent() is body:
  centered=(pr.find(q('jc')) is not None and pr.find(q('jc')).get(q('val'))=='center')
  special=name.startswith('heading') or name.startswith('toc') or centered or bool(re.match(r'^\[\d+\]',t))
  if not special:
   ind=child(pr,'ind');ind.attrib.pop(q('hanging'),None);ind.attrib.pop(q('firstLineChars'),None);ind.set(q('firstLine'),'420');prose+=1
 if pr.find(q('sectPr')) is not None:cover=False
# Right-aligned page numbers use real tab stops with dot leaders, not typed dots.
toc_count=0
toc_levels={}
def toc_layout(pr,name):
 match=re.search(r'(\d+)$',name);level=int(match.group(1)) if match else 1
 indent=child(pr,'ind');indent.attrib.clear();indent.set(q('left'),str((level-1)*420));indent.set(q('firstLine'),'0')
 spacing=child(pr,'spacing');spacing.set(q('before'),'0');spacing.set(q('after'),'0');spacing.set(q('line'),'240');spacing.set(q('lineRule'),'exact')
 return level
for p in paras:
 pr=p.find(q('pPr'));sid=pr.find(q('pStyle')) if pr is not None else None
 name=style_names.get(sid.get(q('val')),'') if sid is not None else ''
 if name.startswith('toc') or name.startswith('目录'):
  level=toc_layout(pr,name);toc_levels[level]=toc_levels.get(level,0)+1
  tabs=child(pr,'tabs')
  for old in list(tabs):tabs.remove(old)
  tab=E.SubElement(tabs,q('tab'));tab.set(q('val'),'right');tab.set(q('leader'),'dot');tab.set(q('pos'),'9070');toc_count+=1
for st in styles.findall(q('style')):
 if style_names[st.get(q('styleId'))].startswith(('toc','目录')):
  toc_layout(child(st,'pPr'),style_names[st.get(q('styleId'))])
  tabs=child(child(st,'pPr'),'tabs')
  for old in list(tabs):tabs.remove(old)
  tab=E.SubElement(tabs,q('tab'));tab.set(q('val'),'right');tab.set(q('leader'),'dot');tab.set(q('pos'),'9070')
# All data tables are three-line tables; the first table is the reference cover form.
def edge(borders,name,kind='nil',size=0):
 b=child(borders,name);b.attrib.clear();b.set(q('val'),kind)
 if kind!='nil':b.set(q('sz'),str(size));b.set(q('color'),'000000')
alltables=root.findall('.//w:tbl',N)
for table in alltables[1:]:
 borders=child(child(table,'tblPr'),'tblBorders')
 for e in ['top','left','bottom','right','insideH','insideV']:edge(borders,e)
 rows=table.findall(q('tr'))
 for i,row in enumerate(rows):
  for cell in row.findall(q('tc')):
   b=child(child(cell,'tcPr'),'tcBorders')
   for e in ['top','left','bottom','right','insideH','insideV']:edge(b,e)
   if i==0:edge(b,'top','single',12);edge(b,'bottom','single',6)
   if i==len(rows)-1:edge(b,'bottom','single',12)
# Stable target bookmarks for all captions, chapter headings, subsections and bibliography.
# Replace only this script's bookmarks on repeat runs; preserve Word TOC targets.
old_ids=set()
for b in list(root.findall('.//w:bookmarkStart',N)):
 if b.get(q('name'),'').startswith('Course_'):old_ids.add(b.get(q('id')));b.getparent().remove(b)
for b in list(root.findall('.//w:bookmarkEnd',N)):
 if b.get(q('id')) in old_ids:b.getparent().remove(b)
existing_ids=[int(x.get(q('id'))) for x in root.findall('.//w:bookmarkStart',N)];bookmark_id=max(existing_ids+[0])+1
targets={};target_paras={};counts={'figure':0,'table':0,'section':0,'chapter':0,'reference':0}
def anchor(p,key,kind):
 global bookmark_id
 name='Course_'+kind+'_'+re.sub(r'[^A-Za-z0-9]','_',key)
 if key in targets:return
 start=E.Element(q('bookmarkStart'));start.set(q('id'),str(bookmark_id));start.set(q('name'),name)
 end=E.Element(q('bookmarkEnd'));end.set(q('id'),str(bookmark_id));bookmark_id+=1
 p.insert(1 if p.find(q('pPr')) is not None else 0,start);p.append(end);targets[key]=name;target_paras[key]=p;counts[kind]+=1
for p in paras:
 t=text(p).strip();pr=p.find(q('pPr'));sid=pr.find(q('pStyle')) if pr is not None else None;name=style_names.get(sid.get(q('val')),'') if sid is not None else ''
 centered=pr is not None and pr.find(q('jc')) is not None and pr.find(q('jc')).get(q('val'))=='center'
 m=re.match(r'^(图|表)\s*([A-Z\d]+)[-－–](\d+)',t)
 if m and centered:anchor(p,m[1]+m[2]+'-'+m[3],'figure' if m[1]=='图' else 'table')
 if name.startswith('heading'):
  m=re.match(r'第\s*(\d+)\s*章',t)
  if m:anchor(p,'章'+m[1],'chapter')
  m=re.match(r'([A-Z\d]+(?:\.\d+)+)',t)
  if m:anchor(p,'节'+m[1],'section')
 m=re.match(r'^\[(\d+)\]',t)
 if m:anchor(p,'文'+m[1],'reference')
pattern=re.compile(r'(图|表)\s*([A-Z\d]+)[-－–](\d+)|第\s*(\d+)\s*章|((?:\d+|[A-Z])(?:\.\d+)+)\s*节|\[(\d+)\]')
links=0
for p in paras:
 if any(p is target for target in target_paras.values()):continue
 allowed={'pPr','r','proofErr','bookmarkStart','bookmarkEnd'}
 if any(E.QName(c).localname not in allowed for c in p):continue
 runs=p.findall(q('r'));spans=[];value=''
 valid=True
 for run in runs:
  ts=run.findall(q('t'))
  if len(ts)!=1 or any(run.find(q(tag)) is not None for tag in ['fldChar','instrText','drawing','br']):valid=False;break
  fragment=ts[0].text or '';spans.append((len(value),len(value)+len(fragment),run));value+=fragment
 if not valid:continue
 matches=[]
 for m in pattern.finditer(value):
  if m.group(1):key=m.group(1)+m.group(2)+'-'+m.group(3)
  elif m.group(4):key='章'+m.group(4)
  elif m.group(5):key='节'+m.group(5)
  else:key='文'+m.group(6)
  if key in targets:matches.append((m,targets[key]))
 if not matches:continue
 def slices(start,end):
  result=[]
  for lo,hi,run in spans:
   a=max(start,lo);b=min(end,hi)
   if a<b:
    r=copy.deepcopy(run);t=r.find(q('t'));t.text=value[a:b];t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');result.append(r)
  return result
 insert=p.index(runs[0]);new=[];offset=0
 for m,destination in matches:
  new.extend(slices(offset,m.start()))
  link=E.Element(q('hyperlink'));link.set(q('anchor'),destination);link.set(q('history'),'1');link.set(q('tooltip'),'跳转至 '+m.group())
  for r in slices(m.start(),m.end()):child(child(r,'rPr'),'u').set(q('val'),'single');link.append(r)
  new.append(link);offset=m.end();links+=1
 new.extend(slices(offset,len(value)))
 for run in runs:p.remove(run)
 for i,node in enumerate(new):p.insert(insert+i,node)
# Normalize footer fonts and lock final field output until the next intentional refresh.
for filename,raw in list(data.items()):
 if re.match(r'word/(footer|header)\d+\.xml$',filename):
  part=E.fromstring(raw)
  for run in part.findall('.//w:r',N):fmt(child(run,'rPr'))
  data[filename]=E.tostring(part,xml_declaration=True,encoding='UTF-8',standalone=True)
settings=E.fromstring(data['word/settings.xml']);child(settings,'updateFields').set(q('val'),'false');data['word/settings.xml']=E.tostring(settings,xml_declaration=True,encoding='UTF-8',standalone=True)
data['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
data['word/styles.xml']=E.tostring(styles,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(P,'w',zipfile.ZIP_DEFLATED) as z:
 for n,v in data.items():z.writestr(n,v)
# Audit every text run and all internal link destinations.
names={x.get(q('name')) for x in root.findall('.//w:bookmarkStart',N)}
internal=root.findall('.//w:hyperlink[@w:anchor]',N);assert all(x.get(q('anchor')) in names for x in internal)
for p in paras:
 for run in p.findall('.//w:r',N):
  rp=run.find(q('rPr'));assert rp.find(q('color')).get(q('val'))=='000000';assert rp.find(q('rFonts')).get(q('eastAsia'))=='宋体'
course_links=sum(x.get(q('anchor'),'').startswith('Course_') for x in internal)
assert course_links>=41 and len(alltables)>5
result={'toc_indentation_twips_by_level':{1:0,2:420,3:840},'toc_entries_by_level':toc_levels,'toc_dot_leader_entries':toc_count,'cover_title_pt':42,'cover_fields_pt':14,'font':'宋体','text_color':'000000','heading_1_pt':14,'heading_2_3_pt':12,'body_pt':10.5,'prose_first_line_indent_twips':420,'indented_paragraphs':prose,'three_line_data_tables':len(alltables)-1,'cover_form_preserved':True,'new_cross_reference_links':course_links,'total_internal_links':len(internal),'bookmark_targets':counts,'all_internal_destinations_valid':True}
(R/'verification/report_format_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=True))
