"""M16 value labels: search teacher (demo config: MIXMAX=1 CATCH=6 RANDDEF=0.15 COMMIT=2 H=40, R per action, lives top-up 2)
evaluated ONCE (no plan carry-over) at train curriculum states; saves per-action rollout values q[4,R] (+max/mean).
States: buckets 0-3 (<=15 left) all, plus FRAC_HI of buckets 4-7. Resumable: chunks of CH states -> runs/m16/qlabels/cNNNN.npz."""
import json, os, pickle, re, sys, time
import multiprocessing as mp
import numpy as np
E = os.environ.get
for k, v in dict(MIXMAX="1", CATCH="6", RANDDEF="0.15", HYST="0.5").items():
    os.environ.setdefault(k, v)
RR, CH, OUT = int(E("R", "8")), int(E("CH", "100")), "runs/m16/qlabels"


def label(job):
    import torch; torch.set_num_threads(1)
    from msclear import ram as R
    from msclear.pixenv import PixelGame
    from msclear.m11_env import restore
    from msclear.search_teacher import Teacher
    ci, files = job; t0 = time.time(); rows = []
    for f in files:
        s = pickle.load(open(f, "rb")); m = re.match(r".*/(.+)_(\d{5})\.pkl$", f); demo, i = m.group(1), int(m.group(2))
        acts = np.load(f"runs/teacher/demos/{demo}.npz")["act"]
        g = PixelGame(sticky=0.25, max_steps=10**9); g.reset(1234); restore(g, s)
        lv = int(g.ale.getRAM()[R.LIVES])
        if (lv & 7) < 2:
            g.ale.setRAM(R.LIVES, (lv & ~7) | 2); g.ram = g.ale.getRAM().copy(); g.prev_lives = int(g.ram[R.LIVES])
        t = Teacher(H=40, R_=RR, commit=2, eps=0.15, seed=(ci * 1000 + i) % 2**31)
        t.last = int(acts[i - 1]) if i > 0 else None; t.plan = None
        _, q = t.act(g)
        rows.append((demo, i, s["left"], s["dist"], int(acts[i]), q))
    np.savez(f"{OUT}/c{ci:04d}.npz", demo=np.array([r[0] for r in rows]), i=np.array([r[1] for r in rows]),
             left=np.array([r[2] for r in rows]), dist=np.array([r[3] for r in rows]), act=np.array([r[4] for r in rows]),
             q=np.stack([r[5] for r in rows]).astype(np.float32))
    return ci, len(rows), time.time() - t0


if __name__ == "__main__":
    b = json.load(open("runs/m16/buckets.json")); rng = np.random.default_rng(0); fs = []
    for k, lst in enumerate(b["train"]):
        lst = list(lst)
        if k >= 4:
            rng.shuffle(lst); lst = lst[:int(len(lst) * float(E("FRAC_HI", "1.0")))]
        fs += lst
    rng.shuffle(fs); chunks = [fs[j:j + CH] for j in range(0, len(fs), CH)]
    jobs = [(ci, c) for ci, c in enumerate(chunks) if not os.path.exists(f"{OUT}/c{ci:04d}.npz")]
    print(f"states {len(fs)} chunks {len(chunks)} todo {len(jobs)} R={RR}", flush=True); t0 = time.time(); n = 0
    with mp.get_context("spawn").Pool(int(E("WORKERS", "8"))) as p:
        for ci, k, sec in p.imap_unordered(label, jobs):
            n += k; print(f"[{time.strftime('%H:%M:%S')}] chunk {ci} n {k} {sec:.0f}s  total {n} rate {n / (time.time() - t0):.1f}/s", flush=True)
            if os.path.exists(f"{OUT}/STOP"): p.terminate(); break
    print("DONE", n, flush=True)
