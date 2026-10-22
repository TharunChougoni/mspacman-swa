"""Rebuild the report charts in docs/assets/ from the compact eval extracts in results/.
Run from the repo root: python tools/report/make_charts.py  (needs matplotlib, numpy)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.25,'figure.dpi':150})
C=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#6b7280','#0891b2']
R='results/'; A='docs/assets/'
E=json.load(open(R+'evals.json')); CF=json.load(open(R+'confirm.json')); MB=json.load(open(R+'multiboard.json'))
MS=list(json.load(open(R+'m22_multiseed.json')).values())[0]; RE=json.load(open(R+'m18_m19_reeval.json'))
def save(fig,n): fig.tight_layout(); fig.savefig(A+n); plt.close(fig)

# 1. board-1 clear rate across milestones (confirmation evals)
rows=[('M3','M3 HRA (seeds 0-99)',0),('M4','M4 HRA tuned (seeds 0-99)',0),('M4+','M4 HRA, bigger M1 model (300 held-out)',0),
      ('M5','M5 HRA + endgame/escape (300 held-out)',0),('M6','M6 pixel BC+DAgger',1),('M7','M7 PPO',1),('M8','M8 endgame PPO',1),
      ('M9','M9 PPO',1),('M10','M10 KV-cache distill',1)]
lab=[r[0] for r in rows]+['M11']; v=[CF[r[1]]['clear_rate']*100 for r in rows]+[MB['multiboard_m11']['reach_frac']['2']*100]
se=[CF[r[1]]['clear_se']*100 for r in rows]+[100*np.sqrt(.9*.1/100)]; col=[C[5] if r[2]==0 else C[0] for r in rows]+[C[0]]
fig,ax=plt.subplots(figsize=(9,4.2)); ax.bar(lab,v,yerr=se,color=col,capsize=3)
for i,x in enumerate(v): ax.text(i,x+se[i]+1.5,f'{x:.0f}%',ha='center',fontsize=9)
ax.set_ylim(0,105); ax.set_ylabel('board-1 clear rate (%)'); ax.set_title('Board-1 clears: RAM teacher (grey) → pixel-only student (blue)')
save(fig,'progression.png')

# 2. M11 multi-board PPO
m=E['m11']; st=[e['env_steps']/1e6 for e in m]
fig,ax=plt.subplots(figsize=(8,4)); ax.plot(st,[e['b1_clear']*100 for e in m],c=C[0],label='board-1 clear (%)')
ax.set_ylabel('board-1 clear (%)'); ax.set_xlabel('env steps (millions)')
a2=ax.twinx(); a2.plot(st,[e['mb_clears'] for e in m],c=C[2],label='boards cleared / game'); a2.set_ylabel('boards cleared per game')
a2.spines['right'].set_visible(True); a2.grid(False); ax.set_title('M11: multi-board PPO'); fig.legend(loc='lower right',bbox_to_anchor=(.88,.18),fontsize=9)
save(fig,'m11.png')

# 3. M12-M17 plateau: boards cleared per game and board-3 items
fig,(a,b)=plt.subplots(1,2,figsize=(10,4))
for i,r in enumerate(['m12','m13','m15','m16','m17']):
    m=[e for e in E[r] if 'mb_clears' in e]; a.plot([e['env_steps']/1e6 for e in m],[e['mb_clears'] for e in m],'o-',ms=3,c=C[i],label=r.upper())
    m=[e for e in E[r] if 'mb_b3_items' in e]
    if m: b.plot([e['env_steps']/1e6 for e in m],[e['mb_b3_items'] for e in m],'o-',ms=3,c=C[i],label=r.upper())
a.set_title('Boards cleared per game'); a.set_xlabel('env steps in run (M)'); a.legend(fontsize=8)
b.set_title('Items eaten on board 3 (of 154)'); b.set_xlabel('env steps in run (M)'); b.axhline(154,ls=':',c='k')
save(fig,'plateau.png')

# 4. M18 clean clears
m=E['m18']; st=[e['env_steps']/1e6 for e in m]
fig,ax=plt.subplots(figsize=(8,4)); ax.plot(st,[e['b1_clean']*100 for e in m],'o-',c=C[0],label='board 1 clean clear')
ax.plot(st,[e['b2_clean']*100 for e in m],'s-',c=C[1],label='board 2 clean clear')
ax.set_xlabel('env steps (millions)'); ax.set_ylabel('%'); ax.set_title('M18: survival + speed reward (Kaggle)'); ax.legend(); save(fig,'m18.png')

# 5. drift of fine-tunes resumed from M18 best
fig,ax=plt.subplots(figsize=(9,4.6))
def cm(e): return e['clean_mean']*100 if 'clean_mean' in e else (e['b1_clean']+e['b2_clean'])/2*100
for r,lbl,mk,c,lw in [('m18s2','M18 s2: board-2-heavy starts','o-',C[5],1.3),('m19','M19: endgame dot shaping','^-',C[3],1.3),
                      ('m20','M20: EMA of policy weights','v-',C[6],1.3),('m21','M21: full-life episodes, 256-step rollouts','s-',C[4],1.3),
                      ('m22','M22: separate value network','D-',C[0],2.4)]:
    m=E[r]; b0=m[0]['env_steps']; ax.plot([(e['env_steps']-b0)/1e6 for e in m],[cm(e) for e in m],mk,c=c,lw=lw,ms=4,label=lbl)
ax.axhline(39.5,ls='--',c='k',lw=1); ax.text(0.2,40.2,'start: M18 best (39.5%, single eval)',fontsize=9)
ax.set_xlabel('env steps since resume (millions)'); ax.set_ylabel('clean clear, mean of boards 1–2 (%)')
ax.set_title('Fine-tunes from the same checkpoint: only M22 holds its gains'); ax.legend(fontsize=8.5,loc='lower left'); save(fig,'drift.png')

# 6. multi-seed comparison (3 seed sets x 200 games per board)
items=[('M18 best',RE['runs/m18_s1/best.pt'],C[5]),('M18 last',RE['runs/m18_s1/last_pi.pt'],'#a3a3a3'),('M19 last',RE['runs/m19/last_pi.pt'],C[3]),('M22 best',MS,C[0])]
fig,ax=plt.subplots(figsize=(8.5,4.3)); xx=np.arange(2); w=.2
for j,(n,d,c) in enumerate(items):
    off=(j-1.5)*w; ax.bar(xx+off,[d['b1']['clean_mean']*100,d['b2']['clean_mean']*100],w,color=c,label=n)
    for i,k in enumerate(['b1','b2']):
        ax.scatter([i+off]*3,[x*100 for x in d[k]['clean']],c='k',s=8,zorder=3)
        ax.text(i+off,max(d[k]['clean'])*100+1.5,f"{d[k]['clean_mean']*100:.1f}",ha='center',fontsize=8)
ax.set_xticks(xx,['Board 1','Board 2']); ax.set_ylabel('clean clear rate (%)'); ax.set_ylim(0,68); ax.legend(fontsize=9,ncol=4,loc='upper right')
ax.set_title('Clean clears, 3 seed sets × 200 games per board (dots = seed sets)'); save(fig,'multiseed.png')

# 7. throughput
fig,ax=plt.subplots(figsize=(7,3.3)); lab=['Laptop\n(i7-12700H, 20 thr)','Kaggle\n(4 vCPU, 2×T4)','Kaggle M22\n(+value net)']
lo=[2000,1200,940]; hi=[2400,1320,1010]; ax.barh(lab,[h-l for l,h in zip(lo,hi)],left=lo,color=[C[2],C[4],C[1]],height=.5)
ax.set_xlim(0,2700); ax.set_xlabel('agent steps / s (range observed)'); ax.set_title('Throughput is CPU-bound (emulator)'); ax.invert_yaxis(); save(fig,'throughput.png')
print('charts written to', A)
