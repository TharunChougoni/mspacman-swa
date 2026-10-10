"""Batched GPU eval: one vectorised PixelGame pool, greedy SWAv2 on device.

Workers (spawn) own the games and import only numpy + msclear.pixenv. The parent
keeps the 4-stacks and the (G, window, d) token window: at reset the window is
`window` copies of the first token, then each step shift-appends, matching
`demos.Student`. Q = net.head (last-token fast path when it is on). Finished games
are not reset. fp32 disables TF32 so argmax matches the CPU student except rare ties.
"""
from __future__ import annotations

import contextlib
import os
import sys
import multiprocessing as mp

# PixelGame.step stops at this many decisions. One more means a worker lost a done.
_MAX_STEPS = 6000


def _worker(remote):
    """One process, a round-robin slice of seeds. No torch."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import numpy as np
    from msclear import pixenv

    pixenv.cv2.setNumThreads(1)
    PixelGame = pixenv.PixelGame
    games, seeds = [], []

    def close_games():
        for g in games:
            try:
                g.env.close()
            except Exception:
                pass
        games.clear()
        seeds.clear()

    try:
        while True:
            try:
                cmd, data = remote.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
                break
            try:
                if cmd == "reset":
                    close_games()
                    seed_list, sticky = data
                    stacks = np.empty((len(seed_list), 4, 84, 84), np.uint8)
                    for i, seed in enumerate(seed_list):
                        g = PixelGame(sticky=float(sticky))
                        stacks[i] = g.reset(int(seed))
                        games.append(g)
                        seeds.append(int(seed))
                    remote.send(stacks)
                elif cmd == "step":
                    actions = data
                    if len(actions) != len(games):
                        raise RuntimeError(f"step got {len(actions)} actions for {len(games)} games")
                    n = len(games)
                    done = np.zeros(n, np.uint8)
                    frames = []
                    results = []
                    keep_g, keep_s = [], []
                    for i, a in enumerate(actions):
                        g = games[i]
                        g.step(int(a))
                        if g.done:
                            done[i] = 1
                            results.append(g.result(seeds[i]))
                            try:
                                g.env.close()
                            except Exception:
                                pass
                        else:
                            frames.append(g.frame.copy())
                            keep_g.append(g)
                            keep_s.append(seeds[i])
                    games, seeds = keep_g, keep_s
                    if frames:
                        fr = np.empty((len(frames), 84, 84), np.uint8)
                        for j, f in enumerate(frames):
                            fr[j] = f
                    else:
                        fr = np.zeros((0, 84, 84), np.uint8)
                    remote.send((done, fr, results))
                elif cmd == "close":
                    break
                else:
                    raise RuntimeError(f"unknown cmd {cmd!r}")
            except Exception:
                import traceback
                try:
                    remote.send(("__error__", traceback.format_exc()))
                except Exception:
                    pass
                break
    finally:
        close_games()


def _recv(remote):
    msg = remote.recv()
    if isinstance(msg, tuple) and len(msg) == 2 and isinstance(msg[0], str) and msg[0] == "__error__":
        raise RuntimeError(msg[1])
    return msg


def _spawn(n):
    """Start `n` daemon workers. Hide __main__ so spawn does not re-import the caller (torch)."""
    ctx = mp.get_context("spawn")
    main = sys.modules["__main__"]
    # Same trick as endgame_env.VecGames: a spawned child re-imports __main__ (the trainer
    # or the eval script, which imports torch). These workers only need pixenv.
    saved = {k: main.__dict__.pop(k) for k in ("__file__", "__spec__") if k in main.__dict__}
    main.__spec__ = None
    remotes, procs = [], []
    try:
        for w in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child,), daemon=True, name=f"fasteval-{w}")
            proc.start()
            child.close()
            remotes.append(parent)
            procs.append(proc)
    except Exception:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for remote in remotes:
            remote.close()
        raise
    finally:
        del main.__spec__
        main.__dict__.update(saved)
    return remotes, procs


@contextlib.contextmanager
def _strict_fp32():
    """IEEE fp32 matmul and convs. TF32 on the 3060 changes greedy actions versus the CPU student."""
    import torch
    mm = torch.backends.cuda.matmul.allow_tf32
    cd = torch.backends.cudnn.allow_tf32
    bench = torch.backends.cudnn.benchmark
    det = torch.backends.cudnn.deterministic
    prec = torch.get_float32_matmul_precision()
    conv = getattr(torch.backends.cudnn, "conv", None)
    conv_prec = getattr(conv, "fp32_precision", None) if conv is not None else None
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        if conv is not None and conv_prec is not None:
            conv.fp32_precision = "ieee"
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = mm
        torch.backends.cudnn.allow_tf32 = cd
        torch.backends.cudnn.benchmark = bench
        torch.backends.cudnn.deterministic = det
        torch.set_float32_matmul_precision(prec)
        if conv is not None and conv_prec is not None:
            conv.fp32_precision = conv_prec


def _load_net(src, device):
    import torch
    from msclear.swa_net import SWAv2
    if isinstance(src, (str, os.PathLike)):
        ck = torch.load(os.fspath(src), map_location="cpu", weights_only=False)
        net = SWAv2(**ck["arch"])
        net.load_state_dict(ck["model"])
    else:
        net = src
    net.eval()
    return net.to(device)


def _push(tokens, z, window):
    """(G, d) -> (G, window, d). First call repeats the token; later calls shift-append."""
    import torch
    if tokens is None:
        return z.unsqueeze(1).repeat(1, window, 1)
    return torch.cat([tokens[:, 1:], z.unsqueeze(1)], 1)


class FastEvaluator:
    """Persistent spawn pool. `evaluate` may be called again; `close` stops the workers."""

    def __init__(self, workers=14, device="cuda", bf16=False):
        if int(workers) < 1:
            raise ValueError("workers must be >= 1")
        self.workers = int(workers)
        self.device = device
        self.bf16 = bool(bf16)
        self._closed = False
        self.remotes, self.procs = _spawn(self.workers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        for remote in self.remotes:
            try:
                remote.close()
            except Exception:
                pass
        self.remotes, self.procs = [], []

    def evaluate(self, net_or_ckpt_path, seeds, sticky=0.25, progress=None):
        """Play `seeds` greedily. Returns (m6eval.summarize(results), results sorted by seed).

        `progress`, if given, is called as progress(dict(done, n, step, results)) a few times
        per second of play and whenever a game finishes.
        """
        import numpy as np
        import torch
        from msclear.m6eval import summarize

        if self._closed:
            raise RuntimeError("FastEvaluator is closed")
        seeds = [int(s) for s in seeds]
        G = len(seeds)
        if G == 0:
            raise ValueError("seeds is empty")

        groups = [[] for _ in range(self.workers)]
        gindex = [[] for _ in range(self.workers)]
        for i, seed in enumerate(seeds):
            w = i % self.workers
            groups[w].append(seed)
            gindex[w].append(i)

        # Workers build ALE envs while the checkpoint loads.
        for w, remote in enumerate(self.remotes):
            remote.send(("reset", (groups[w], float(sticky))))
        net = _load_net(net_or_ckpt_path, self.device)
        got = [_recv(remote) for remote in self.remotes]

        stacks = np.empty((G, 4, 84, 84), np.uint8)
        for w, st in enumerate(got):
            ids = gindex[w]
            if ids:
                stacks[np.asarray(ids, np.int64)] = st
        alive = np.ones(G, np.bool_)
        hist = np.zeros((G, 4), np.int64)
        worker_alive = [list(ids) for ids in gindex]
        results = [None] * G
        finished = []
        tokens = None
        step = 0
        reported = -1
        dev = self.device
        window = int(net.window)
        ctx = contextlib.nullcontext() if self.bf16 else _strict_fp32()
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else contextlib.nullcontext()

        with torch.no_grad(), ctx, autocast:
            while True:
                idx = np.flatnonzero(alive)
                if len(idx) == 0:
                    break
                if step >= _MAX_STEPS:
                    stuck = [seeds[i] for i in idx.tolist()]
                    raise RuntimeError(f"games still running after {_MAX_STEPS} steps: {stuck}")
                batch = stacks if alive.all() else stacks[idx]
                x = torch.from_numpy(np.ascontiguousarray(batch)).to(dev)
                z = net.embed(x)
                tokens = _push(tokens, z, window)
                q = net.head(tokens)
                actions = q.argmax(dim=-1).cpu().numpy()
                for j, gi in enumerate(idx.tolist()):
                    hist[gi, int(actions[j])] += 1
                act_at = {gi: int(a) for gi, a in zip(idx.tolist(), actions)}

                n_sent = 0
                for w, remote in enumerate(self.remotes):
                    ids = worker_alive[w]
                    if not ids:
                        continue
                    remote.send(("step", [act_at[i] for i in ids]))
                    n_sent += 1

                new_alive = []
                for w, remote in enumerate(self.remotes):
                    ids = worker_alive[w]
                    if not ids:
                        new_alive.append([])
                        continue
                    done, frames, done_results = _recv(remote)
                    ids_arr = np.asarray(ids, np.int64)
                    dead = np.flatnonzero(done)
                    live = np.flatnonzero(done == 0)
                    if len(done_results) != len(dead) or len(frames) != len(live):
                        raise RuntimeError(
                            f"worker {w}: done {int(done.sum())} results {len(done_results)} "
                            f"frames {len(frames)} live {len(live)}")
                    for li, res in zip(dead.tolist(), done_results):
                        gi = int(ids_arr[li])
                        res["hist"] = [int(hist[gi, a]) for a in range(4)]
                        results[gi] = res
                        finished.append(res)
                        alive[gi] = False
                    if len(live):
                        gis = ids_arr[live]
                        block = np.empty((len(gis), 4, 84, 84), np.uint8)
                        block[:, :-1] = stacks[gis, 1:]
                        block[:, -1] = frames
                        stacks[gis] = block
                        new_alive.append(gis.tolist())
                    else:
                        new_alive.append([])
                worker_alive = new_alive
                if n_sent == 0:
                    raise RuntimeError("alive games but no worker owned them")

                still = alive[idx]
                if not still.all():
                    tokens = tokens[torch.from_numpy(np.ascontiguousarray(still)).to(tokens.device)]
                step += 1
                if progress is not None and (step % 50 == 0 or len(finished) != reported):
                    reported = len(finished)
                    progress(dict(done=reported, n=G, step=step, results=finished))

        if any(r is None for r in results):
            raise RuntimeError("a game produced no result")
        if progress is not None and reported != G:
            progress(dict(done=G, n=G, step=step, results=finished))
        out = sorted(results, key=lambda r: r["seed"])
        return summarize(out), out
