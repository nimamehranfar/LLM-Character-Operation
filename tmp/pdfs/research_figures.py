from reportlab.lib.colors import HexColor
import math

INK='#273342'; BASE='#F1F3F5'; LORA='#087E8B'; MAP='#7653A6'; CODE='#9A6B21'

def label(c,s,x,y,size=12,color=INK,bold=False):
    c.setFont('Helvetica-Bold' if bold else 'Helvetica',size); c.setFillColor(HexColor(color)); c.drawCentredString(x,y,s)

def box(c,x,y,w,h,lines,edge=INK,fill='#FFFFFF',size=12,bold=False,dashed=False):
    c.setStrokeColor(HexColor(edge)); c.setFillColor(HexColor(fill)); c.setLineWidth(1.1)
    c.setDash(4,3) if dashed else c.setDash()
    c.rect(x,y,w,h,fill=1,stroke=1); c.setDash()
    if isinstance(lines,str): lines=[lines]
    for i,s in enumerate(lines):label(c,s,x+w/2,y+h/2+(len(lines)-1)*size*.58-i*size*1.16-size*.32,size,edge,bold)

def route(c,pts,color=INK,dashed=False,head=True):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(1.25); c.setDash(3,3) if dashed else c.setDash()
    p=c.beginPath(); p.moveTo(*pts[0])
    for pt in pts[1:]:p.lineTo(*pt)
    c.drawPath(p); c.setDash()
    if head:
        x,y=pts[-1]; a=math.atan2(y-pts[-2][1],x-pts[-2][0]); p=c.beginPath(); p.moveTo(x,y)
        p.lineTo(x-6*math.cos(a-.45),y-6*math.sin(a-.45)); p.lineTo(x-6*math.cos(a+.45),y-6*math.sin(a+.45)); p.close()
        c.setFillColor(HexColor(color));c.drawPath(p,fill=1,stroke=0)

def boundary(c,x,y,w,h,title):
    box(c,x,y,w,h,[],INK,'#FAFBFC',dashed=True)
    label(c,title,x+w/2,y+h-17,13,INK,True)

def legend(c):
    for x,col,s in [(43,INK,'Pretrained Qwen'),(260,LORA,'Our trained LoRA'),(480,MAP,'Our trained mapper'),(723,CODE,'Deterministic code')]:
        box(c,x,394,12,12,[],col,col); c.setFont('Helvetica',11);c.setFillColor(HexColor(col));c.drawString(x+18,395,s)

def pipeline(c):
    legend(c)
    # Phase 1: base model and explicit LoRA branches inside its boundary.
    box(c,40,279,114,49,['Request +','tool catalogue'],size=12)
    boundary(c,180,245,400,129,'Qwen3-8B: 36 blocks (L0-L35) | LoRA ON')
    for x,t in [(201,'L0'),(278,'L1'),(355,'...'),(432,'L35')]:
        box(c,x,274,59,40,t,INK,BASE,13,True)
        box(c,x,326,59,21,'LoRA',LORA,'#E7F5F2',11,True)
        route(c,[(x+29,326),(x+29,315)],LORA)
    box(c,509,274,53,40,['Output','head'],size=10)
    for a,b in [(154,201),(260,278),(337,355),(414,432),(491,509)]:route(c,[(a,294),(b,294)])
    box(c,606,272,108,44,['Generated','CALL + inputs'],size=11)
    route(c,[(562,294),(606,294)])
    box(c,742,272,75,44,'Parser',CODE,'#FFF8EB',12,True);route(c,[(714,294),(742,294)])
    box(c,844,265,76,59,['Exact','executor'],CODE,'#FFF8EB',12,True);route(c,[(817,294),(844,294)],CODE)
    # Explicit no-call branch and string-result bypass.
    route(c,[(660,272),(660,236),(755,236)],dashed=True)
    label(c,'NO_CALL: stop',818,232,11)
    route(c,[(882,265),(882,203),(736,203)],CODE)
    label(c,'Edited string: return directly',817,213,10,CODE)
    # Second model pass is the same backbone. Hook is downstream of Lk.
    box(c,40,115,114,49,['Same request','answer prompt'],size=11)
    box(c,180,86,582,103,[],INK,'#FAFBFC',dashed=True)
    label(c,'Same Qwen3-8B | LoRA disabled',295,173,11,INK,True)
    for x,t in [(202,'L0 ...'),(319,'Lk'),(503,'Lk+1 ... L35')]:box(c,x,115,88 if x!=503 else 124,43,t,INK,BASE,12,True)
    box(c,663,115,78,43,['Output','head'],size=11)
    route(c,[(154,137),(202,137)]);route(c,[(290,137),(319,137)])
    # State exits decoder k, goes into mapper, and is replaced before k+1.
    c.setFillColor(HexColor(MAP));c.circle(454,137,10,fill=0,stroke=1);label(c,'+',454,133,15,MAP,True)
    route(c,[(407,137),(444,137)]);route(c,[(464,137),(503,137)],MAP)
    route(c,[(627,137),(663,137)]);route(c,[(741,137),(787,137)])
    box(c,787,115,133,43,['Final answer','e.g. 3'],size=12,bold=True)
    # Outside-base trained result module; consumes hidden state and result.
    box(c,352,203,205,44,['Our result mapper','shared encoder + layer-k adapter / gate'],MAP,'#F2ECF8',10,True)
    route(c,[(882,203),(882,252),(575,252),(575,225),(557,225)],MAP)
    label(c,'Number / single character',700,259,10,MAP)
    route(c,[(414,137),(414,195),(429,195),(429,203)],MAP)
    label(c,'h',421,177,11,MAP)
    route(c,[(454,203),(454,147)],MAP)
    label(c,'gated update',507,178,10,MAP)
    label(c,'Hook after Lk: replace last prompt-token state during prefill',469,96,10,MAP)
    label(c,'LoRA = small adapters inside Qwen projections, not a separate stack of layers.',480,60,12)
    label(c,'Two generation passes. No ground-truth correction of the predicted call.',480,41,11)

def candidates(c):
    legend(c)
    box(c,40,314,117,49,['Exact result','number / character'],CODE,'#FFF8EB',11)
    boundary(c,196,279,724,105,'OUR TRAINED MAPPER (outside the pretrained LLM)')
    box(c,218,310,161,43,['Shared symbolic','result encoder'],MAP,'#F2ECF8',12,True)
    route(c,[(157,338),(218,338)],MAP)
    # Full mapper has shared encoder, distinct hidden-state adapters.
    for i,k in enumerate([1,3,5,7,9,11,13,15]):
        x=399+i*62
        box(c,x,310,52,43,'M'+str(k),MAP,'#F2ECF8',12,True)
    route(c,[(379,332),(388,332),(388,365),(894,365)],MAP,head=False)
    for i in range(8):route(c,[(425+i*62,365),(425+i*62,353)],MAP)
    box(c,40,110,880,138,[],INK,'#FAFBFC',dashed=True)
    # Omitted even-numbered blocks are stated explicitly beneath the chain.
    for i,k in enumerate([1,3,5,7,9,11,13,15]):
        x=70+i*105
        box(c,x,157,64,41,'L'+str(k),INK,BASE,13,True)
        c.setStrokeColor(HexColor(MAP));c.circle(x+77,177,4,fill=0,stroke=1)
        route(c,[(x+64,177),(x+73,177)])
        if i<7:route(c,[(x+81,177),(x+105,177)])
        route(c,[(425+i*62,310),(425+i*62,265),(x+77,265),(x+77,181)],MAP,dashed=True)
    label(c,'FROZEN QWEN3-8B | hooks after decoder block outputs',480,128,13,INK,True)
    label(c,'Even-numbered blocks and later blocks are omitted.',480,97,10)
    label(c,'Train across 8 candidates  /  select on development data  /  use one hook per answer',480,82,13,MAP,True)
    label(c,'Dashed paths are alternatives. Initial three-operation run chose zero-based L5.',480,51,11)

