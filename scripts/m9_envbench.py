"""M9: vector-env step latency (14x3, 18x3, and 14x3 while 6 CPU eval workers run). Analysis only."""
import sys, time, json, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from msclear.endgame_env import VecGames
from msclear import m6eval
def bench(v, n=300):
    rng = np.random.default_rng(0); v.step(rng.integers(0, 4, v.n))
    t = time.perf_counter()
    for _ in range(n): v.step(rng.integers(0, 4, v.n))
    return (time.perf_counter() - t) / n * 1e3
if __name__ == "__main__":
    out = {}
    for nw, per in ((14, 3), (18, 3)):
        v = VecGames(nw, per, p_pool=0.0); v.reset()
        out[f"vec_{nw}x{per}_ms_per_step"] = bench(v); v.close()
    pool = m6eval.make_pool(6)
    ar = pool.map_async(m6eval.play, [("runs/m8/best.pt", s, 0.25) for s in range(12)])
    time.sleep(8)
    v = VecGames(14, 3, p_pool=0.0); v.reset()
    out["vec_14x3_ms_per_step_with_6_eval_workers"] = bench(v); v.close()
    pool.terminate()
    print(json.dumps(out))
