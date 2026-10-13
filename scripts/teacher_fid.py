import sys,pickle,numpy as np
from msclear.pixenv import PixelGame
from msclear.m11_env import restore
g=PixelGame(sticky=0.25,max_steps=10**9); g.reset(1); restore(g,pickle.load(open(sys.argv[1],'rb')))
ale=g.ale; rng=np.random.default_rng(0); seq=rng.integers(4,size=60)
import inspect
print(ale.cloneState.__doc__)
for inc in (False,True):
    try: st=ale.cloneState(include_rng=inc)
    except TypeError as e: print('no kw',e); st=ale.cloneState()
    outs=[]
    for rep in range(3):
        ale.restoreState(st)
        if rep==2:  # pollute last action
            for _ in range(4): ale.act(g.aset[3])
            ale.restoreState(st)
        tr=[]
        for a in seq:
            for _ in range(4): ale.act(g.aset[a+1])
            r=ale.getRAM(); tr.append((int(r[10]),int(r[16])))
        outs.append(tr)
    print('include_rng',inc,'rep0==rep1',outs[0]==outs[1],'rep0==polluted',outs[0]==outs[2])
