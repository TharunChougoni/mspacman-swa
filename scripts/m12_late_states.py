"""M12 pre-pass: harvest LATE board-3 states (<= 60/45/30/20/15/8 items left) by playing CKPT (M11 best, greedy,
CPU fp32) from the board-3 start snapshots. Test seeds (m11_ppo.split_b3, same 7) -> runs/m12/late_test/,
train seeds -> runs/m12/late_train/. Each snapshot is replayed REPS times with different sticky seeds.
SAMPLE_TAU (0 = greedy), TAG (file suffix). Env: CKPT (runs/m11/best.pt), REPS_TEST (8), REPS_TRAIN (2), WORKERS (16)."""
import os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
THR = (60, 45, 30, 20, 15, 8)
TAU_S = float(os.environ.get("SAMPLE_TAU", "0"))   # >0: sample pi at this temperature (more diverse / deeper states)
TAG = os.environ.get("TAG", "")


def job(args):
    f, rep, out = args
    import torch
    torch.set_num_threads(1)
    from msclear import ram as R
    from msclear.m10_fasteval import load_m10
    from msclear.m11_env import restore, _book
    from msclear.pixenv import PixelGame
    global NET
    if "NET" not in globals():
        NET = load_m10(os.environ.get("CKPT", "runs/m11/best.pt"), "cpu")
    g = PixelGame(sticky=0.25, max_steps=10**9); seed = 7_000_000 + rep * 1000 + hash(f) % 997
    g.reset(seed); s = pickle.load(open(f, "rb")); restore(g, s); b0 = R.board(g.ram)
    cache = NET.new_cache(1, "cpu"); first = True; done_thr = set(); n = 0; stack = g.stack; tag = Path(f).stem
    torch.manual_seed(seed)
    with torch.no_grad():
        for t in range(6000):
            z = NET.embed(torch.from_numpy(stack[None].copy()))
            if first:
                NET.prime(cache, torch.ones(1, dtype=torch.bool), z); first = False
            lg = NET.adv(NET.step(cache, z))
            a = int(torch.distributions.Categorical(logits=lg / TAU_S).sample()) if TAU_S > 0 else int(lg.argmax(-1))
            stack, _, _, _ = g.step(a)
            if g.ale.game_over() or R.board(g.ram) != b0:
                break
            left = 154 - int(g.ram[R.DOTS_EATEN])
            for th in THR:
                if left <= th and th not in done_thr:
                    done_thr.update(x for x in THR if x >= th)
                    d = dict(state=g.ale.cloneState(), stack=g.stack.copy(), book=_book(g), board=b0, left=left,
                             lives=int(g.ram[R.LIVES]), src=tag, seed=s.get("seed"))
                    pickle.dump(d, open(f"{out}/{tag}{TAG}_r{rep}_l{left:03d}.pkl", "wb")); n += 1
                    break
    g.env.close()
    return n


def main():
    import multiprocessing as mp
    from scripts.m11_ppo import split_b3
    train, test, seeds = split_b3()
    jobs = []
    for files, reps, out in ((test, int(os.environ.get("REPS_TEST", "8")), "runs/m12/late_test"),
                             (train, int(os.environ.get("REPS_TRAIN", "2")), "runs/m12/late_train")):
        Path(out).mkdir(parents=True, exist_ok=True)
        jobs += [(f, r, out) for f in files for r in range(reps)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(int(os.environ.get("WORKERS", "16"))) as p:
        n = sum(p.imap_unordered(job, jobs))
    print(f"test seeds {seeds}; {len(jobs)} games, {n} late states in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
