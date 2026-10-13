import sys,pickle,numpy as np
from msclear.pixenv import PixelGame
from msclear.m11_env import restore
from msclear import ram as R
import msclear.search_teacher as S
g=PixelGame(sticky=0.25,max_steps=10**9); g.reset(1); restore(g,pickle.load(open(sys.argv[1],'rb')))
t=S.Teacher(H=40,R_=6,commit=2,eps=0.15)
hist=[]; prev=None
for i in range(3000):
    st=g.ale.cloneState(include_rng=True); r=g.ram.copy()
    a,q=t.act(g,prev); prev=a
    hist.append((st,a,q,(int(r[10]),int(r[16])),r[S.GX].astype(int).tolist(),r[S.GY].astype(int).tolist(),S.R.frightened(r).astype(int).tolist()))
    g.step(a)
    if S.caught(g.ram,6) and not S.R.frightened(g.ram).all():
        print('REAL caught at',i+1); break
    if g.ale.game_over(): print('GO',i); break
n=len(hist)
for j in range(max(0,n-22),n):
    st,a,q,p,gx,gy,fr=hist[j]
    print(j,'a',a,'p',p,'gx',gx,'gy',gy,'fr',fr,'qa mean %.0f max %.0f'%(q[a].mean(),q[a].max()),'means',np.round(q.mean(1)).tolist())
# fidelity: replay actual actions from n-20
k=max(0,n-20); g.ale.restoreState(hist[k][0])
for j in range(k,n):
    if j>k and j<n: pass
    for _ in range(4): g.ale.act(g.aset[hist[j][1]+1])
    rr=g.ale.getRAM()
    if j+1<n: print('replay',j+1,(int(rr[10]),int(rr[16])),'real',hist[j+1][3])
