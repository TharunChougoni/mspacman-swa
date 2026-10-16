"""Greedy fp32 traces of a policy on the fixed board-2 test set (and board-1 seeds): per-death context."""
import glob, json, os, pickle, sys, collections
import numpy as np, torch
sys.path.insert(0, ".")
from msclear import ram as R
from msclear.pixenv import PixelGame
from msclear.m11_env import restore
from msclear.m10_fasteval import load_m10
dev = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
net = load_m10(sys.argv[1], dev); files = sorted(glob.glob("runs/m17/b2_test/*.pkl"))
G = len(files); games = []
for i, f in enumerate(files):
    g = PixelGame(sticky=0.25, max_steps=10**9); g.reset(i); restore(g, pickle.load(open(f, "rb"))); games.append(g)
b0 = [R.board(g.ram) for g in games]; e0 = [int(g.ram[R.DOTS_EATEN]) for g in games]
cache = net.new_cache(G, dev); new = torch.ones(G, dtype=torch.bool, device=dev)
alive = np.ones(G, bool); n = np.zeros(G, int); deaths = []; hist = [collections.deque(maxlen=12) for _ in range(G)]
posh = [collections.deque(maxlen=40) for _ in range(G)]; ringr = [collections.deque(maxlen=60) for _ in range(G)]; tleft = collections.Counter(); res = [None] * G; vals = []
while alive.any() :
    x = torch.from_numpy(np.stack([g.stack for g in games])).to(dev)
    with torch.no_grad():
        z = net.embed(x); net.prime(cache, new, z); h = net.step(cache, z); lg = net.adv(h)
    new[:] = False; a = lg.argmax(-1).cpu().numpy(); p = torch.softmax(lg / 0.25, -1).cpu().numpy()
    for i in np.flatnonzero(alive):
        g = games[i]; r0 = g.ram.copy()
        _, _, _, info = g.step(int(a[i])); n[i] += 1; hist[i].append(int(a[i]))
        px, py = int(r0[10]), int(r0[16]); posh[i].append((px, py)); ringr[i].append(r0); tleft[min(R.items_total(R.board(r0)) - int(r0[R.DOTS_EATEN]), 60) // 10] += 1
        if info["lost_life"]:
            rr = list(ringr[i]); k = len(rr) - 1
            while k > 0 and (rr[k][10], rr[k][16]) == (rr[k - 1][10], rr[k - 1][16]): k -= 1
            r0 = rr[max(k - 1, 0)]; px, py = int(r0[10]), int(r0[16]); lag = len(rr) - 1 - k
            gx = np.array([r0[k] for k in R.GHOST_X], int); gy = np.array([r0[k] for k in R.GHOST_Y], int)
            d = np.abs(gx - px) + np.abs(gy - py); fr = R.frightened(r0)
            stuck = len(set(list(posh[i])[-20:])) <= 3
            deaths.append(dict(game=i, step=int(n[i]), left=R.items_total(R.board(r0)) - int(r0[R.DOTS_EATEN]), px=px, py=py,
                               dmin=int(d.min()), nnear=int((d <= 12).sum()), fright_any=bool(fr.any()), fright_timer=int(r0[R.FRIGHT_TIMER]),
                               last_acts=list(hist[i]), pmax=float(p[i].max()), stuck=bool(stuck), lag=int(lag), lives_after=int(g.ram[R.LIVES]) & 7))
        b = R.board(g.ram)
        if b > b0[i] or g.ale.game_over() or n[i] >= 8000:
            alive[i] = False; res[i] = dict(cleared=b > b0[i], steps=int(n[i]))
cl = np.mean([r["cleared"] for r in res]); nd = collections.Counter(d["game"] for d in deaths)
clean = np.mean([r["cleared"] and nd[i] == 0 for i, r in enumerate(res)])
print(json.dumps(dict(games=G, clear=cl, clean=clean, deaths_per_game=len(deaths) / G)))
D = deaths
def q(k): return np.percentile([d[k] for d in D], [10, 25, 50, 75, 90]).tolist()
print("death step pct", q("step")); print("dots left at death pct", q("left")); print("min ghost dist pct", q("dmin"))
print("frac with ghost frightened", np.mean([d["fright_any"] for d in D]), "fright_timer>0", np.mean([d["fright_timer"] > 0 for d in D]))
print("frac >=2 ghosts within 12", np.mean([d["nnear"] >= 2 for d in D]), "stuck(<=3 pos in 20)", np.mean([d["stuck"] for d in D]))
print("policy max-prob at death pct", q("pmax"))
pos = collections.Counter((d["px"] // 8 * 8, d["py"] // 8 * 8) for d in D); print("top death cells", pos.most_common(10))
lastsame = np.mean([len(set(d["last_acts"][-6:])) == 1 for d in D]); print("last 6 actions identical", lastsame)
print("first-death step for games with deaths pct", np.percentile([min(d["step"] for d in D if d["game"] == g) for g in nd], [25, 50, 75]).tolist())
print("death lag (decisions frozen before lost_life) pct", q("lag")); print("decisions by dots-left decile (0-9,10-19,..,60+)", sorted(tleft.items()))
json.dump(D, open("runs/m18_s1/b2_deaths.json", "w"), default=int)
