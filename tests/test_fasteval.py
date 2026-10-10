"""fp32 FastEvaluator matches m6eval.play on the same seeds (rare float ties would still fail here)."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

M8_BEST = Path(__file__).resolve().parents[1] / "runs" / "m8" / "best.pt"
_KEYS = ("cleared", "steps", "eaten", "deaths", "score", "hist")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    pytest.mark.skipif(not M8_BEST.exists(), reason="runs/m8/best.pt not present"),
]


def _diff(tag, gpu, cpu):
    out = []
    for k in _KEYS:
        if gpu[k] != cpu[k]:
            out.append(f"{tag} seed {gpu['seed']} {k}: gpu={gpu[k]!r} cpu={cpu[k]!r}")
    return out


def test_fasteval_matches_cpu_play():
    import time

    from msclear.fasteval import FastEvaluator
    from msclear.m6eval import play, summarize

    seeds = [0, 1, 2, 3]
    sticky = 0.25
    # Two games per worker, so a finished game is dropped while its sibling keeps stepping.
    # The second evaluate reuses the pool. PixelGame's own max_steps=6000 is the same cap play() uses.
    t0 = time.time()
    last = [t0]

    def progress(info):
        now = time.time()
        if now - last[0] < 15 and info["done"] != info["n"]:
            return
        last[0] = now
        print(f"  gpu {info['done']}/{info['n']} step {info['step']} elapsed {now - t0:.0f}s", flush=True)

    ev = FastEvaluator(workers=2, device="cuda", bf16=False)
    try:
        _, first = ev.evaluate(str(M8_BEST), seeds, sticky=sticky, progress=progress)
        _, second = ev.evaluate(str(M8_BEST), seeds, sticky=sticky, progress=progress)
        for proc in ev.procs:
            maps = Path(f"/proc/{proc.pid}/maps").read_text(errors="replace")
            assert "libtorch" not in maps, f"worker pid {proc.pid} imported torch"
    finally:
        ev.close()

    by1 = {r["seed"]: r for r in first}
    by2 = {r["seed"]: r for r in second}
    mismatches = []
    for s in seeds:
        mismatches += _diff("reuse", by1[s], by2[s])
    cpu = []
    for s in seeds:
        t = time.time()
        r = play((str(M8_BEST), s, sticky))
        cpu.append(r)
        print(f"  cpu seed {s} steps {r['steps']} score {r['score']} cleared {r['cleared']} "
              f"({time.time() - t:.1f}s)", flush=True)
        mismatches += _diff("cpu", by1[s], r)
    if mismatches:
        raise AssertionError("fasteval diverged from m6eval.play:\n" + "\n".join(mismatches))
    summary = summarize(cpu)
    gpu_summary = summarize(first)
    assert gpu_summary == summary, (gpu_summary, summary)
    print(f"matched seeds {seeds} in {time.time() - t0:.0f}s", flush=True)
    print(summary, flush=True)
