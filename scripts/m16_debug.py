import json, pickle, sys, os, numpy as np, torch, cv2
from msclear.pixenv import PixelGame
from msclear.m13_env import restore
from msclear import ram as R, mazedist as MD
from msclear.m10_net import M10Net
from msclear.search_teacher import Teacher
B = json.load(open('runs/m16/buckets.json')); files = B['eval'][0]
pick = [files[i] for i in np.linspace(0, len(files) - 1, 5).astype(int)]
GX, GY = list(R.GHOST_X), list(R.GHOST_Y)
def gd(r):
    return (np.abs(r[GX].astype(int) - int(r[10])) + np.abs(r[GY].astype(int) - int(r[16]))).tolist()
ck = torch.load('runs/m16/ckpt_last.pt', map_location='cpu', weights_only=False)
net = M10Net(**ck['arch']); net.load_state_dict(ck['pi']); net.eval()
os.makedirs('/tmp/m16vid', exist_ok=True)
for fi, f in enumerate(pick):
    s = pickle.load(open(f, 'rb')); g = PixelGame(sticky=0.25, max_steps=10**9); g.reset(1000 + fi); restore(g, s)
    st = s['stack']; r = g.ram
    print(f"\n== {os.path.basename(f)} left {s['left']} dist {s['dist']} ram119 {int(r[119])} lives {int(r[123])&7} board {R.board(r)} "
          f"stack mean/frame {[int(x.mean()) for x in st]} frame-diffs {[int(np.abs(st[i].astype(int)-st[i-1]).sum()) for i in (1,2,3)]} "
          f"ghost dist {gd(r)} fright {R.frightened(r).astype(int).tolist()} player {(int(r[10]), int(r[16]))}")
    w, dots, _ = MD.masks(g.ale.getScreenRGB()); n, lab, sts, cen = cv2.connectedComponentsWithStats(dots.astype(np.uint8), connectivity=4)
    print('   screen dots', n - 1, 'sizes', sts[1:, cv2.CC_STAT_AREA].tolist(), 'centres', np.round(cen[1:]).astype(int).tolist())
    # (b) policy greedy + video
    root = g.ale.cloneState(include_rng=True); vw = cv2.VideoWriter(f'/tmp/m16vid/policy_L0_{fi}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (320, 420))
    cache = net.new_cache(1, 'cpu'); first = True; acts = []; mind = []; b0 = R.board(g.ram)
    with torch.no_grad():
        for t in range(600):
            z = net.embed(torch.from_numpy(g.stack[None].copy()))
            if first: net.prime(cache, torch.ones(1, dtype=torch.bool), z); first = False
            a = int(net.adv(net.step(cache, z)).argmax()); acts.append(a)
            _, _, _, info = g.step(a); mind.append(min(gd(g.ram)))
            if fi < 2: vw.write(cv2.resize(g.ale.getScreenRGB()[..., ::-1], (320, 420), interpolation=cv2.INTER_NEAREST))
            if info['lost_life'] or g.ale.game_over() or R.board(g.ram) != b0: break
    vw.release()
    print(f"   POLICY: steps {t+1} board {R.board(g.ram)} ram119 {int(g.ram[119])} lost_life {info['lost_life']} acts hist {np.bincount(acts, minlength=4).tolist()} "
          f"last acts {acts[-12:]} pos {(int(g.ram[10]), int(g.ram[16]))} min ghost dist last 5 {mind[-5:]}")
    # (a) teacher, real lives, no top-up
    g.ale.restoreState(root); g.ram = g.ale.getRAM().copy(); tch = Teacher(H=40, R_=6, commit=2, eps=0.15, seed=fi); n_ = 0; d = 0; prev = None
    while n_ < 1500:
        if n_ % 2 == 0: a, _ = tch.act(g, prev, 2); prev = a
        _, _, _, info = g.step(a); n_ += 1; tch.last = a; d += int(info['lost_life'])
        if R.board(g.ram) != b0 or g.ale.game_over(): break
    print(f"   TEACHER: steps {n_} board {b0}->{R.board(g.ram)} level byte {int(g.ram[R.LEVEL])} ram119 {int(g.ram[119])} deaths {d} gameover {g.ale.game_over()}")
