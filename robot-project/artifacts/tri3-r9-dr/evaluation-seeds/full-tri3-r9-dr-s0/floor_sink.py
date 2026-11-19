import json, sys, glob, numpy as np, mujoco
mj=mujoco.MjModel.from_xml_path('robot/mjcf/tri3-r9/model-model.xml')
d=mujoco.MjData(mj); mujoco.mj_forward(mj,d)
def quat_mat(q):
    x,y,z,w=q; n=np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
# per component: list of (type, center_ref(mm), R_ref, size mm)
comp={}
for g in range(mj.ngeom):
    if mj.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE: continue
    b=mj.body(mj.geom_bodyid[g]).name
    # only collision geoms (contype/conaffinity default); include all geoms of named collision
    nm=mj.geom(g).name
    if 'collision' not in nm: continue
    comp.setdefault(b,[]).append((int(mj.geom_type[g]),d.geom_xpos[g]*1000,d.geom_xmat[g].reshape(3,3),mj.geom_size[g]*1000))
def low(b,R,p):
    m=1e9
    for t,c,Rg,s in comp[b]:
        cw=R@c+p; Rw=R@Rg
        if t==mujoco.mjtGeom.mjGEOM_BOX:
            pts=[cw+Rw@(np.array([sx,sy,sz])*s) for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)]; z=min(q[2] for q in pts)
        elif t==mujoco.mjtGeom.mjGEOM_CAPSULE:
            ax=Rw[:,2]*s[1]; z=min(cw[2]+ax[2],cw[2]-ax[2])-s[0]
        elif t==mujoco.mjtGeom.mjGEOM_SPHERE: z=cw[2]-s[0]
        else: raise SystemExit(f'type {t}')
        m=min(m,z)
    return m
out={}
for f in sorted(glob.glob(sys.argv[1]+'/seed-*-trace.json')):
    t=json.load(open(f)); fr=[x for x in t['frames'] if x['frame_kind']!='input']
    series={}
    for b in comp:
        series[b]=np.array([low(b,quat_mat(x['component_placements'][b]['rotation_xyzw']),np.array(x['component_placements'][b]['position_mm'])) for x in fr])
    out[f.split('/')[-1][5:9]]=series
    print(f.split('/')[-1][:9],len(fr),{b:round(float(series[b].min()),2) for b in ('c_foot_rf','c_foot_hc')},file=sys.stderr)
res={}
for sd,ser in out.items():
    nonfoot={b:v for b,v in ser.items() if not b.startswith('c_foot')}
    m=np.min(np.stack(list(nonfoot.values())),0)
    worst=min(nonfoot,key=lambda b:nonfoot[b].min())
    res[sd]=dict(frames=len(m),frac_below_2mm=float((m<-2).mean()),max_depth_mm=float(max(0,-m.min())),worst=worst,
      per_body_depth={b:round(float(max(0,-v.min())),2) for b,v in nonfoot.items() if b.split('_',1)[0]=='c' and any(k in b for k in('thigh','shin','rollbr','chassis'))},
      frac_any_negative=float((m<0).mean()),min_foot=float(min(v.min() for b,v in ser.items() if b.startswith('c_foot'))))
json.dump(res,open(sys.argv[2],'w'),indent=1,ensure_ascii=False)
for sd,r in res.items(): print(sd,{k:v for k,v in r.items() if k!='per_body_depth'})
