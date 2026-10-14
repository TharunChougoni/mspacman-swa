"""DAgger labeller: the student (M10 net, CPU, per worker) plays maze-2 start states with lives top-up; the search
teacher labels a random subset (P_LABEL) of the visited decisions (teacher.act on the live state, then restore).
Output per game (resumable: skips existing): LABEL_DIR/<src>_r<rep>.npz with stack0, frames (last frame per decision),
act (teacher label, or student action where unlabelled), labeled (bool), student_act, deaths_before, cleared.
Usage: python scripts/dagger_label.py OUT.json @jobs.txt   env: CKPT LABEL_DIR P_LABEL STUDENT_TAU CAP WORKERS (+ teacher env)
"""
import json, os, sys, time
import multiprocessing as mp
import numpy as np

E = os.environ.get
CKPT, LDIR = E("CKPT", "runs/m14/dagger_init.pt"), E("LABEL_DIR", "runs/dagger1/labels")
P_LABEL, STAU, CAP, TOPUP = float(E("P_LABEL", "0.3")), float(E("STUDENT_TAU", "0.25")), int(E("CAP", "3000")), int(E("TOPUP", "2"))
H, RR, COMMIT = int(E("H", "40")), int(E("R", "6")), int(E("COMMIT", "2"))


def name(src, rep):
    return f"{LDIR}/{src.replace('/', '_').replace('runs_', '')}_r{rep}.npz"


def run(job):
    import pickle, torch
    torch.set_num_threads(1)
    from msclear import ram as R
    from msclear.pixenv import PixelGame
    from msclear.m11_env import restore
    from msclear.m10_net import M10Net
    from msclear.search_teacher import Teacher
    src, rep = job; t0 = time.time()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = M10Net(**ck["arch"]); net.load_state_dict(ck["model"] if "model" in ck else ck["pi"]); net.eval()
    g = PixelGame(sticky=0.25, max_steps=10**9)
    s = pickle.load(open(src, "rb")); g.reset(1234 + rep); restore(g, s)
    tch = Teacher(H=H, R_=RR, commit=COMMIT, eps=0.15, seed=(hash((src, rep)) % 2**31))
    rng = np.random.default_rng(hash((src, rep, 1)) % 2**31); trng = torch.Generator().manual_seed(int(rng.integers(2**31)))
    cache = net.new_cache(1, "cpu"); first = True
    b0 = R.board(g.ram); it0 = int(g.ram[R.DOTS_EATEN])
    frames, acts, sacts, lab, pdb = [], [], [], [], []; stack0 = g.stack.copy(); deaths = 0; n = 0; nl = 0
    with torch.no_grad():
        while n < CAP:
            z = net.embed(torch.from_numpy(g.stack[None].copy()))
            if first:
                net.prime(cache, torch.ones(1, dtype=torch.bool), z); first = False
            lg = net.adv(net.step(cache, z)) / STAU
            sa = int(torch.multinomial(torch.softmax(lg[0], -1), 1, generator=trng))
            frames.append(g.stack[-1].copy()); sacts.append(sa); pdb.append(deaths)
            if rng.random() < P_LABEL:
                tch.plan = None; ta, _ = tch.act(g, None, 1); acts.append(ta); lab.append(True); nl += 1
            else:
                acts.append(sa); lab.append(False)
            _, _, _, info = g.step(sa); n += 1; tch.last = sa
            deaths += int(info["lost_life"])
            if TOPUP and not g.ale.game_over():
                lv = int(g.ale.getRAM()[R.LIVES])
                if (lv & 7) < TOPUP:
                    g.ale.setRAM(R.LIVES, (lv & ~7) | TOPUP); g.ram = g.ale.getRAM().copy(); g.prev_lives = int(g.ram[R.LIVES])
            if R.board(g.ram) != b0 or g.ale.game_over():
                break
    cleared = R.board(g.ram) > b0
    os.makedirs(LDIR, exist_ok=True)
    np.savez_compressed(name(src, rep) + ".tmp.npz", stack0=stack0, frames=np.stack(frames).astype(np.uint8), act=np.array(acts, np.int8),
                        student_act=np.array(sacts, np.int8), labeled=np.array(lab), deaths_before=np.array(pdb, np.int16), cleared=cleared)
    os.replace(name(src, rep) + ".tmp.npz", name(src, rep))
    agree = float(np.mean([a == b for a, b, l in zip(acts, sacts, lab) if l])) if nl else None
    return dict(src=os.path.basename(src), rep=rep, cleared=bool(cleared), deaths=deaths, decisions=n, labels=nl,
                agree=agree, start_items=it0, sec=time.time() - t0)


if __name__ == "__main__":
    out = sys.argv[1]; srcs = []
    for gl in sys.argv[2:]:
        path, _, k = gl.lstrip("@").partition(":"); L = open(path).read().split()
        np.random.default_rng(1).shuffle(L); srcs += L[:int(k)] if k else L
    jobs = [(s, 0) for s in srcs]; n0 = len(jobs)
    jobs = [j for j in jobs if not os.path.exists(name(*j))]
    print(f"CKPT={CKPT} P_LABEL={P_LABEL} STUDENT_TAU={STAU} H={H} R={RR} jobs={len(jobs)} (skipped {n0 - len(jobs)} done)", flush=True)
    import glob
    prev = [np.load(f) for f in glob.glob(f"{LDIR}/*.npz") if not f.endswith(".tmp.npz")]
    t0 = time.time(); tot = dict(dec=sum(len(z["act"]) for z in prev), lab=sum(int(z["labeled"].sum()) for z in prev),
                                 g=len(prev), c=sum(bool(z["cleared"]) for z in prev))
    print(f"existing: {tot}", flush=True)
    with mp.get_context("spawn").Pool(int(E("WORKERS", "18"))) as p:
        for r in p.imap_unordered(run, jobs):
            tot["dec"] += r["decisions"]; tot["lab"] += r["labels"]; tot["g"] += 1; tot["c"] += r["cleared"]
            print(json.dumps(r), flush=True)
            if tot["g"] % 10 == 0:
                el = time.time() - t0
                print(f"PROGRESS games {tot['g']} cleared {tot['c']} decisions {tot['dec']} labels {tot['lab']} "
                      f"({tot['lab'] / el:.1f} labels/s, {tot['dec'] / el:.1f} dec/s)", flush=True)
                if tot["lab"] >= int(E("TARGET_LABELS", "100000")):
                    print("target labels reached", flush=True); p.terminate(); break
    json.dump(tot, open(out, "w"))
