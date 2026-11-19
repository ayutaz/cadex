import json,re,sys,itertools
import numpy as np
import xml.etree.ElementTree as ET
root='artifacts/tri3-r9-sw1-dr/evaluation-seeds/full-tri3-r9-sw1-dr-s0/policy/'
cut=ET.parse('artifacts/tri3-r9-sw1-dr/cut/cut.xml').getroot()
geoms=[]  # (body, name, type, size, pos, R)
def quat_mat(q):
    w,x,y,z=q;return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
for b in cut.iter('body'):
    for g in b.findall('geom'):
        t=g.get('type','sphere'); size=[float(v) for v in g.get('size').split()]
        pos=np.array([float(v) for v in g.get('pos','0 0 0').split()])*1000
        q=[float(v) for v in g.get('quat','1 0 0 0').split()]
        geoms.append((b.get('name'),g.get('name'),t,np.array(size)*1000,pos,quat_mat(q)))
def lowest_local_points(t,size,pos,R):
    if t=='box':
        pts=[pos+R@(np.array(s)*size) for s in itertools.product([-1,1],repeat=3)]
        return pts,0.0
    if t=='capsule':
        r,h=size[0],size[1]
        return [pos+R@np.array([0,0,h]),pos+R@np.array([0,0,-h])],r
    if t=='sphere': return [pos],size[0]
    raise SystemExit(t)
pre=[(b,n,*lowest_local_points(t,s,p,R)) for b,n,t,s,p,R in geoms]
def quat_xyzw(q):x,y,z,w=q;return quat_mat((w,x,y,z))
res={}
for seed in range(2101,2111):
    tr=json.load(open(root+f'seed-{seed}-trace.json'))
    fr=[f for f in tr['frames'] if f.get('component_placements')]
    groups={'shin/thigh/rollbr(非接触)':[], 'chassis':[]}
    per={}
    for f in fr:
        cp=f['component_placements']
        low={}
        for b,n,pts,r in pre:
            if b=='c_chassis' or b.startswith('c_foot'): pass
            if b not in cp: continue
            R=quat_xyzw(cp[b]['rotation_xyzw']);p=np.array(cp[b]['position_mm'])
            z=min((R@q+p)[2] for q in pts)-r
            low[n]=z
        for n,z in low.items(): per.setdefault(n,[]).append(z)
    out={}
    for key,sel in (('nc',lambda n:not n.startswith('c_foot') and not n.startswith('c_chassis')),('chassis',lambda n:n.startswith('c_chassis')),('foot',lambda n:n.startswith('c_foot'))):
        names=[n for n in per if sel(n)]
        m=np.min([per[n] for n in names],axis=0)
        out[key]=dict(frac2=float((m<-2).mean()),minz=float(m.min()),argmin=names[int(np.argmin([min(per[n]) for n in names]))],frames=len(m))
    res[seed]=out
    print(seed,json.dumps(out))
json.dump(res,open(root+'../floor_sink.json','w'),indent=1)
