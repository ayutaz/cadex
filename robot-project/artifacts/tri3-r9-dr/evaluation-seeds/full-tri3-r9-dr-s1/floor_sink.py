import json,sys,re,itertools
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
cut,pol=sys.argv[1],Path(sys.argv[2])
geoms=[]
def walk(el,body):
    for c in el:
        if c.tag=='body': walk(c,c.get('name'))
        elif c.tag=='geom' and body:
            geoms.append((body,c.get('name'),c.get('type','sphere'),np.array(list(map(float,c.get('size').split()))),np.array(list(map(float,c.get('pos','0 0 0').split())))*1000,np.array(list(map(float,c.get('quat','1 0 0 0').split())))))
walk(ET.parse(cut).getroot().find('worldbody'),None)
def rot(q):  # xyzw
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def lowest(g,P,R):
    body,name,typ,size,pos,quat=g
    Rg=rot((quat[1],quat[2],quat[3],quat[0]))
    s=size*1000
    if typ=='box':
        pts=np.array([[sx*s[0],sy*s[1],sz*s[2]] for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)]); r=0
    elif typ=='capsule':
        pts=np.array([[0,0,-s[1]],[0,0,s[1]]]); r=s[0]
    else:
        pts=np.zeros((1,3)); r=s[0]
    w=(P+(R@(pos+(Rg@pts.T).T).T).T)
    return w[:,2].min()-r
res={}
for sd in range(2101,2111):
    t=json.load(open(pol/f'seed-{sd}-trace.json'))
    fr=[f for f in t['frames'] if f['frame_kind']!='input']
    low={}
    for g in geoms:
        arr=[]
        for f in fr:
            p=f['component_placements'][g[0]]
            arr.append(lowest(g,np.array(p['position_mm']),rot(p['rotation_xyzw'])))
        low[g[1]]=np.array(arr)
    res[sd]=(low,len(fr))
out={}
for sd,(low,n) in res.items():
    nf={k:v for k,v in low.items() if 'foot' not in k}
    allnf=np.min(np.array(list(nf.values())),axis=0)
    nochas=np.min(np.array([v for k,v in nf.items() if 'chassis' not in k]),axis=0)
    foot=np.min(np.array([v for k,v in low.items() if 'foot' in k]),axis=0)
    deepest=min(nf,key=lambda k:nf[k].min())
    out[sd]=dict(frames=n,nonfoot_share_below_2mm=float((allnf<-2).mean()),nonfoot_max_depth_mm=float(max(0,-allnf.min())),
      nonfoot_excl_chassis_share=float((nochas<-2).mean()),nonfoot_excl_chassis_depth=float(max(0,-nochas.min())),
      deepest_geom=deepest,deepest_geom_depth=float(-nf[deepest].min()),foot_lowest_mm=float(foot.min()),
      per_geom_depth_mm={k:round(float(max(0,-v.min())),2) for k,v in nf.items() if v.min()<0},
      excluded=bool((allnf<-2).mean()>0 or False))
    out[sd]['excluded']=bool(out[sd]['nonfoot_max_depth_mm']>=2)
print(json.dumps(out,ensure_ascii=False,indent=1))
json.dump(out,open(sys.argv[3],'w'),ensure_ascii=False,indent=1)
