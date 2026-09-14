from pathlib import Path
import html, re
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W,H=1600,1540
im=Image.new('RGB',(W,H),'#F5F3EE'); draw=ImageDraw.Draw(im)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc"><title id="title">RAG first. Fine-tune when the evidence says so.</title><desc id="desc">Start with base plus RAG, build a golden set, and add fine-tuning only if a measured behavior gap and held-out improvement justify it. Otherwise continue improving RAG. A four-alert catalog-change test illustrates why changing facts need current evidence.</desc>']
ink='#172B39'; muted='#52616A'; teal='#076C66'; amber='#88571B'; line='#D4DAD7'
fontbase='/System/Library/Fonts/Supplemental/Arial'
def font(size,bold=False): return ImageFont.truetype(fontbase+(' Bold.ttf' if bold else '.ttf'),size)
def rect(x,y,w,h,fill,stroke=None,r=18):
 draw.rounded_rectangle((x,y,x+w,y+h),radius=r,fill=fill,outline=stroke,width=2)
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="2"' if stroke else '')+'/>')
def text(x,y,s,size=26,color=ink,bold=False):
 draw.text((x,y),s,font=font(size,bold),fill=color,anchor='lt')
 svg.append(f'<text x="{x}" y="{y}" dominant-baseline="text-before-edge" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(s)}</text>')
def path(points,color=teal,arrow=True):
 draw.line(points,fill=color,width=3)
 svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="3"/>')
 if arrow:
  x,y=points[-1]; px,py=points[-2]
  pts=[(x,y),(x-8,y-13),(x+8,y-13)] if y>py else ([(x,y),(x-13,y-8),(x-13,y+8)] if x>px else [(x,y),(x+13,y-8),(x+13,y+8)])
  draw.polygon(pts,fill=color);svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a,b in pts)}" fill="{color}"/>')
rect(0,0,W,H,'#F5F3EE',r=0)
rect(64,60,42,6,teal,r=0);text(122,48,'MLX LAB  /  EPISODE 04',23,teal,True)
text(790,48,'Qwen3.5-4B · M5 Max · 24-alert grid',23,muted)
text(64,110,'RAG first.',78,ink,True)
text(64,204,'Fine-tune when the evidence says so.', 50,ink,True)
text(66,284,'Production may need both. Let evaluation decide when.',31,muted)
# Start and gate
rect(64,364,1472,124,ink)
text(96,387,'01  START WITH BASE + RAG',29,'#FFFFFF',True)
text(96,434,'Current sources + clear instructions. Establish the baseline.',27,'#CFDFE4')
path([(800,488),(800,526)])
rect(64,530,1472,116,'#FFFFFF',line)
text(96,552,'02  BUILD A GOLDEN EVALUATION SET',29,ink,True)
text(96,598,'Representative, reviewed cases with trusted outcomes. Keep them separate from training.',25,muted)
path([(420,646),(420,699)]);path([(1160,646),(1160,699)])
text(440,660,'NOT YET',20,muted,True);text(1180,660,'READY',20,teal,True)
rect(64,704,712,290,'#E9EDEA',line)
text(96,735,'Stay with base + RAG',35,ink,True)
text(96,796,'Review errors. Improve sources and retrieval.',25,muted)
text(96,840,'Collect and label representative cases.',25,muted)
text(96,904,'No golden set? Stick with RAG.',25,teal,True)
rect(808,704,728,290,'#FFFFFF',line)
text(840,735,'Diagnose the remaining gap',35,ink,True)
text(840,796,'Missing evidence? Fix retrieval.',25,muted)
text(840,840,'Behavior errors? Test prompts and rules first.',25,muted)
text(840,904,'Persistent gap? Trial fine-tuning + RAG.',25,teal,True)
path([(1172,994),(1172,1040)])
rect(64,1044,1472,116,'#DCECE6')
text(96,1067,'03  ADD FINE-TUNING ONLY IF IT EARNS ITS PLACE',29,teal,True)
text(96,1111,'Require held-out gains, acceptable regressions, cost and latency. Otherwise keep base + RAG.',25,ink)
# evidence strip
text(64,1200,'AFTER ONE CATALOG EDIT, NO RETRAINING  /  NEW DESTINATION CORRECT',22,muted,True)
for x,label,num,col in [(64,'Fine-tuned','0/4',amber),(564,'Base + RAG','4/4',teal),(1064,'Fine-tuned + RAG','4/4',teal)]:
 rect(x,1244,472,144,'#FFFFFF',line)
 text(x+24,1266,label,25,ink,True);text(x+24,1310,num,49,col,True)
text(64,1414,'4 synthetic alerts. The combined setup still missed 1 severity label.',25,ink,True)
text(64,1458,'One model · small synthetic test · proposed decision path, not proof that both always win',22,muted)
svg.append('</svg>');(ROOT/'theme-diagram.svg').write_text('\n'.join(svg));im.save(ROOT/'theme-diagram.png')
# Minimal rendering for this document's Markdown subset.
def inline(s):
 s=html.escape(s)
 s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
 return re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
lines=(ROOT/'article.md').read_text().splitlines();blocks=[];i=0
while i<len(lines):
 s=lines[i]
 if not s: i+=1;continue
 if s.startswith('# '): blocks.append('<h1>'+inline(s[2:])+'</h1>')
 elif s.startswith('## '): blocks.append('<h2>'+inline(s[3:])+'</h2>')
 elif s.startswith('> '): blocks.append('<blockquote>'+inline(s[2:])+'</blockquote>')
 elif s.startswith('!['):
  alt,src=re.match(r'!\[(.*?)\]\((.*?)\)',s).groups();blocks.append(f'<figure><a href="theme-diagram.png"><img src="{src}" alt="{html.escape(alt)}" width="1600" height="1540"></a><figcaption>A proposed production path, informed by the experiment. <a href="theme-diagram.png">Download PNG</a> · <a href="theme-diagram.svg">Vector SVG</a></figcaption></figure>')
 elif s.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].startswith('|'):
   cells=[x.strip() for x in lines[i].strip('|').split('|')]
   if not all(re.fullmatch('[-: ]+',c) for c in cells): rows.append(cells)
   i+=1
  blocks.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in rows[1:])+'</tbody></table></div>');continue
 else: blocks.append('<p>'+inline(s)+'</p>')
 i+=1
template=(ROOT/'article-template.html').read_text()
body='\n'.join(blocks[1:])
# Use the editable vector in the article so it inherits the site's type and palette.
vector=(ROOT/'theme-diagram.svg').read_text()
vector=vector.replace('<svg ', '<svg class="ep04-diagram" ')
body=re.sub(r'<img[^>]+>', lambda match: vector, body, count=1)
(ROOT/'article.html').write_text(template.replace('{{BODY}}',body))
print('Built article.html, theme-diagram.svg, theme-diagram.png')
