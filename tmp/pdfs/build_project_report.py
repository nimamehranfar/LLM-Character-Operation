from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/LLM_Character_Operations_Project_Report.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
W,H = 960,540
NAVY='#12243A'; TEAL='#087E8B'; BLUE='#356AE6'; MUTED='#53647A'; BG='#F4F7FB'; GOLD='#E5A437'
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle('LLM Character Operations | Project Progress and System Overview')
c.setAuthor('LLM Character Operations Project')

def rect(x,y,w,h,color,r=12):
    c.setFillColor(HexColor(color)); c.roundRect(x,y,w,h,r,fill=1,stroke=0)
def text(s,x,y,size=18,color=NAVY,bold=False):
    c.setFillColor(HexColor(color)); c.setFont('Helvetica-Bold' if bold else 'Helvetica',size); c.drawString(x,y,s)
def para(s,x,top,w,size=17,color=MUTED):
    p=Paragraph(s,ParagraphStyle('p',fontName='Helvetica',fontSize=size,leading=size*1.35,textColor=HexColor(color)))
    _,h=p.wrap(w,500); p.drawOn(c,x,top-h); return h
def arrow(x1,y1,x2,y2,color=TEAL):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(2); c.line(x1,y1,x2,y2)
    a=math.atan2(y2-y1,x2-x1)
    p=c.beginPath(); p.moveTo(x2,y2); p.lineTo(x2-9*math.cos(a-.45),y2-9*math.sin(a-.45)); p.lineTo(x2-9*math.cos(a+.45),y2-9*math.sin(a+.45)); p.close()
    c.setFillColor(HexColor(color)); c.drawPath(p,fill=1,stroke=0)
def icon(kind,x,y,color=TEAL,scale=1):
    c.saveState(); c.translate(x,y); c.scale(scale,scale); c.setStrokeColor(HexColor(color)); c.setFillColor(HexColor(color)); c.setLineWidth(2)
    if kind=='model':
        pts=[(0,18),(20,32),(42,18),(8,-5),(33,-7)]
        for a,b in [(0,1),(1,2),(0,3),(3,4),(2,4),(1,4),(0,2)]: c.line(*pts[a],*pts[b])
        for a,b in pts: c.circle(a,b,5,fill=1,stroke=0)
    elif kind=='tool':
        c.roundRect(0,-12,46,46,7,fill=0,stroke=1); text('{ }',7,2,23,color,True)
    elif kind=='data':
        for dy in [0,12,24]:
            c.ellipse(0,dy-8,46,dy+6,fill=0,stroke=1)
        c.line(0,-1,0,31); c.line(46,-1,46,31)
    elif kind=='chat':
        c.roundRect(0,-5,48,36,7,fill=0,stroke=1); c.line(7,-5,7,-15); c.line(7,-15,19,-5)
        for dy in [6,15,24]: c.line(10,dy,38,dy)
    elif kind=='check':
        c.circle(22,12,24,fill=0,stroke=1); c.line(9,12,19,2); c.line(19,2,36,23)
    c.restoreState()
def page(n,title,sub):
    rect(0,0,W,H,BG,0); rect(0,H-8,W,8,TEAL,0)
    text('LLM CHARACTER OPERATIONS',40,503,11,TEAL,True)
    text(title,40,455,30,NAVY,True); para(sub,40,430,880,16)
    text('Project progress | 14 October 2026',40,21,10,MUTED)
    text(f'{n} / 6',888,21,10,MUTED)
def box(x,y,w,h,title,body,color=TEAL,kind=None):
    rect(x,y,w,h,'#FFFFFF')
    if kind: icon(kind,x+20,y+h-52,color,.75)
    tx=x+20+(48 if kind else 0)
    text(title,tx,y+h-32,18,color,True)
    para(body,x+20,y+h-65,w-40,16)
def note(s,y=55):
    rect(40,y,880,49,'#E5EEF6',8); para(s,56,y+34,848,15,NAVY)

# 1: title and illustrated identity
rect(0,0,W,H,NAVY,0); rect(0,0,18,H,TEAL,0)
text('RESEARCH PROJECT | PROGRESS REPORT',54,477,12,'#67D5CE',True)
text('LLM Character',54,378,48,'#FFFFFF',True)
text('Operations',54,319,48,'#FFFFFF',True)
para('Connecting language understanding<br/>with exact character and string tools.',56,273,485,24,'#D9E6F4')
text('System overview and work completed',56,124,17,'#D9E6F4')
text('Report to Professor  |  14 October 2026',56,86,13,'#96AFCB')
for kind,y,label in [('chat',389,'Understand'),('tool',268,'Execute'),('model',147,'Integrate')]:
    rect(640,y-25,252,82,'#203C55'); icon(kind,665,y+6,'#67D5CE',.8); text(label,725,y+7,22,'#FFFFFF',True)
arrow(762,363,762,326,'#67D5CE'); arrow(762,242,762,205,'#67D5CE')
c.showPage()

# 2: purpose and simple example
page(2,'Why we are building this system','Goal: let the model understand the task, then use exact code for the character-level work.')
box(40,200,405,184,'Language model','Interprets the wording, identifies the required operation and extracts its inputs.',BLUE,'model')
box(485,200,435,184,'Deterministic tools','Carry out counting, lookup and text editing using fixed, repeatable rules.',TEAL,'tool')
arrow(449,290,480,290)
rect(40,113,880,64,NAVY)
text('Example',58,150,13,'#67D5CE',True)
text('How many "a" characters are in "banana"?',58,126,19,'#FFFFFF')
text('COUNT_CHAR',540,136,17,'#67D5CE',True); arrow(705,143,752,143,'#67D5CE'); text('3',783,130,32,'#FFFFFF',True)
note('Research question: can this combination improve reliability and handle unfamiliar wording?',49)
c.showPage()

# 3: actual branching routing
page(3,'How a request moves through the system','The model selects an operation; the executor runs the predicted call; the output follows its result type.')
for x,t,b,k in [(40,'1. Request','User asks a question','chat'),(270,'2. Selector','Choose tool + inputs','model'),(500,'3. Executor','Compute exact result','tool')]:
    box(x,274,200,105,t,b,TEAL,k)
arrow(242,326,263,326); arrow(472,326,493,326)
rect(747,277,173,98,'#E4EBFC'); text('Result type',765,333,19,BLUE,True)
arrow(703,326,740,326)
box(420,119,275,111,'Number / character','Learned mapper feeds the result into the model; model answers.',BLUE)
box(720,119,200,111,'Edited string','Return the executor output directly.',TEAL)
arrow(785,275,558,233,BLUE); arrow(835,275,819,234)
rect(40,119,341,111,'#E8F1F5'); text('No supported tool needed?',59,199,18,NAVY,True)
para('Selector emits NO_CALL. The evaluation checks that unnecessary tool use is avoided.',59,183,301,15)
arrow(368,270,212,234,MUTED)
note('An exact executor still needs the correct call. Wrong tool choices or inputs can produce wrong answers.')
c.showPage()

# 4: training two stages and mapper image
page(4,'What the system learns','Two separate learning stages around a Qwen3-8B backbone; Qwen3-4B is available for replication.')
box(40,213,416,169,'Stage 1: choose the tool','A small trainable adapter learns the operation, exact inputs and when to make no call.',TEAL,'model')
box(486,213,434,169,'Stage 2: use the result','A small mapper learns how to feed a number or single character into the model\'s internal state.',BLUE,'tool')
rect(40,117,880,72,NAVY)
text('Exact result',60,148,18,'#FFFFFF',True); arrow(180,155,232,155,'#67D5CE')
text('Learned mapper',250,148,18,'#67D5CE',True); arrow(413,155,465,155,'#67D5CE')
for x in [482,505,528,551,574]: rect(x,132,14,46,'#356AE6',3)
text('Model state',606,148,18,'#FFFFFF',True); arrow(725,155,775,155,'#67D5CE'); text('Answer',798,148,18,'#FFFFFF',True)
note('The mapper handles seven number/character operations. Its insertion layer is chosen using development data.')
c.showPage()

# 5 dataset and operations
page(5,'What we have prepared so far','A repeatable data and experiment pipeline covers 12 operations, including examples that require no tool.')
box(40,233,276,150,'Count and measure','Character count, string length, word count and length of a selected word.',TEAL,'data')
box(342,233,276,150,'Find and retrieve','Find a character position; retrieve a character from text or a selected word.',BLUE,'tool')
box(644,233,276,150,'Edit and reverse','Insert text or words; reverse a string or a selected word.',TEAL,'chat')
rect(40,112,880,97,'#FFFFFF')
text('Original dataset',60,178,17,TEAL,True); text('50,000 examples',60,146,23,NAVY,True)
text('Expanded dataset',345,178,17,BLUE,True); text('150,000 examples',345,146,23,NAVY,True)
para('Includes longer sentence contexts;<br/>kept separate from the original protocol.',637,179,257,14)
note('Both releases: 70% train / 10% dev / 10% test / 10% new-wording test; 37% no-call controls.')
c.showPage()

# 6 current status + planned assessment
page(6,'Current progress and next steps','The implementation and experiment framework are in place; full research conclusions require completed runs.')
box(40,178,429,206,'Implemented','12 deterministic tools and dataset generation.<br/><br/>Selector and mapper training, full pipeline evaluation, recovery and reporting.',TEAL,'check')
box(491,178,429,206,'Prepared for validation','Separate 150k / two-GPU training protocol.<br/><br/>Matched comparisons: unaided model, direct-answer training, ordinary tool feedback and mapper pipeline.',BLUE,'data')
rect(40,108,880,50,NAVY)
for x,label in [(60,'Train'),(260,'Choose on dev'),(520,'Test + new wording'),(780,'Compare')]: text(label,x,127,17,'#FFFFFF',True)
for x1,x2 in [(153,245),(441,505),(721,765)]: arrow(x1,134,x2,134,'#67D5CE')
note('Next: verify server runs, repeat training across seeds, and compare accuracy, speed and GPU memory.',46)
c.showPage(); c.save()
r=PdfReader(OUT); assert len(r.pages)==6
for i,p in enumerate(r.pages):
    assert len(p.extract_text())>150, i
print(OUT)
print('Verified: exactly six pages, searchable text on every page.')
