"""M5: ablation (resumable: finished configs are read back from runs/m5/ablation.jsonl and skipped) of the three failure-analysis fixes on the 300 check seeds (20000-20299).

Groups (each a small grid over its key parameter, everything else = runs/m4/best.json):
  endgame : eg_n in {15, 25, 40} with pp_hold=15; eg_n=25 alone; pp_hold=15 alone
  escape  : esc_w in {0.3, 1, 2} x esc_slack in {0, 2}; larger gamma_g with smaller w_g (2 variants)
  blinky  : blinky_w in {1.5, 2, 3}; blinky_lead in {4, 8}
Then combos of each group's best (pairs + all three). Best overall (check-seed clear rate) is confirmed with
the baseline on held-out seeds 50000-50299 and on 0-99.  Writes runs/m5/{ablation.jsonl, ablation.md, best.json}.
Env: WORKERS (16), BUDGET_S (5400), N_CHECK (300).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msclear.hra import HRAConfig, _init, evaluate  # noqa: E402

W = int(os.environ.get("WORKERS", "16")); BUDGET = float(os.environ.get("BUDGET_S", "5400"))
NC = int(os.environ.get("N_CHECK", "300"))
OUT = Path("runs/m5"); OUT.mkdir(parents=True, exist_ok=True)
CHECK = list(range(20000, 20000 + NC)); HELD = list(range(50000, 50000 + NC)); FIXED = list(range(100))
t0 = time.time()
DONE = {}                                   # resume: results already on disk, keyed by (set, cfg)
if (OUT / "ablation.jsonl").exists():
    for line in open(OUT / "ablation.jsonl"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:          # half-written line from a power cut
            continue
        DONE[(r["set"], json.dumps(r["cfg"], sort_keys=True))] = r
jl = open(OUT / "ablation.jsonl", "a")


def base_cfg():
    kw = json.loads(Path("runs/m4/best.json").read_text()); kw["gammas_g"] = tuple(kw["gammas_g"])
    return HRAConfig(**kw)


def run(pool, group, label, cfg, seeds, setname, i=None, n=None):
    tag = f"[{i}/{n}] " if i else ""
    key = (setname, json.dumps(json.loads(json.dumps(asdict(cfg))), sort_keys=True))
    if key in DONE:
        r = dict(DONE[key], group=group, label=label)
        print(f"{tag}{group}:{label} [{setname}] already done (resumed): clear {r['clear_rate']:.3f}±{r['clear_se']:.3f}", flush=True)
        return r
    print(f"{tag}{group}:{label} on {setname} ({len(seeds)} games) ... elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    s, _ = evaluate(cfg, seeds, pool=pool, progress=lambda m: print(m, flush=True), every=30.0)
    row = dict(group=group, label=label, set=setname, cfg=asdict(cfg), **s)
    jl.write(json.dumps(row) + "\n"); jl.flush(); os.fsync(jl.fileno())     # survive power cuts
    print(f"  => {group}:{label} [{setname}] clear {s['clear_rate']:.3f}±{s['clear_se']:.3f}  items {s['eaten']:.1f}±{s['eaten_se']:.1f}  "
          f"score {s['score']:.0f}±{s['score_se']:.0f}  deaths {s['deaths']:.2f}", flush=True)
    return row


if __name__ == "__main__":
    B = base_cfg()
    grid = [("baseline", "m4 best", {})]
    grid += [("endgame", f"eg_n={n},pp_hold=15", dict(eg_n=n, pp_hold=15)) for n in (15, 25, 40)]
    grid += [("endgame", "eg_n=25", dict(eg_n=25)), ("endgame", "pp_hold=15", dict(pp_hold=15))]
    grid += [("escape", f"esc_w={w},slack={sl}", dict(esc_w=w, esc_slack=sl)) for w in (0.3, 1.0, 2.0) for sl in (0.0, 2.0)]
    grid += [("escape", "w_g=0.3,gammas=(0.7,0.95)", dict(w_g=0.3, gammas_g=(0.7, 0.95))),
             ("escape", "w_g=0.25,gammas=(0.8,0.95)", dict(w_g=0.25, gammas_g=(0.8, 0.95)))]
    grid += [("blinky", f"blinky_w={w}", dict(blinky_w=w)) for w in (1.5, 2.0, 3.0)]
    grid += [("blinky", f"blinky_lead={L}", dict(blinky_lead=L)) for L in (4, 8)]
    pool = mp.Pool(W, initializer=_init, initargs=("runs/m1/model.pkl",))
    rows = []; n = len(grid) + 4
    for i, (g, lab, kw) in enumerate(grid, 1):
        if time.time() - t0 > BUDGET:
            print("time box reached, skipping rest of grid", flush=True); break
        rows.append((g, lab, kw, run(pool, g, lab, replace(B, **kw), CHECK, "check", i, n)))
    base_cr = rows[0][3]["clear_rate"]
    best_kw = {}
    for g in ("endgame", "escape", "blinky"):
        gr = [r for r in rows if r[0] == g]
        if gr:
            b = max(gr, key=lambda r: r[3]["clear_rate"])
            if b[3]["clear_rate"] > base_cr:
                best_kw[g] = (b[1], b[2])
    print(f"group winners over baseline {base_cr:.3f}: { {g: v[0] for g, v in best_kw.items()} }", flush=True)
    combos = []
    gs = list(best_kw)
    for a in range(len(gs)):
        for b in range(a + 1, len(gs)):
            combos.append((gs[a], gs[b]))
    if len(gs) == 3:
        combos.append(tuple(gs))
    for j, cb in enumerate(combos, len(grid) + 1):
        if time.time() - t0 > BUDGET:
            print("time box reached, skipping combos", flush=True); break
        kw = {}
        for g in cb:
            kw.update(best_kw[g][1])
        lab = " + ".join(best_kw[g][0] for g in cb)
        rows.append(("combo", lab, kw, run(pool, "combo", lab, replace(B, **kw), CHECK, "check", j, n)))
    best = max(rows, key=lambda r: r[3]["clear_rate"])
    bcfg = replace(B, **best[2])
    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(bcfg).items()}, open(OUT / "best.json", "w"))
    print(f"BEST on check seeds: {best[0]}:{best[1]}  clear {best[3]['clear_rate']:.3f}", flush=True)
    conf = []
    for lab, cfg in (("baseline", B), (f"best: {best[1]}", bcfg)):
        for seeds, nm in ((HELD, "held-out 50000+"), (FIXED, "seeds 0-99")):
            conf.append((lab, nm, run(pool, "confirm", lab, cfg, seeds, nm)))
    pool.close()
    L = ["| group | config | clear ± SE | items | score ± SE | deaths |", "|---|---|---|---|---|---|"]
    for g, lab, kw, r in rows:
        L.append(f"| {g} | {lab} | {r['clear_rate']:.3f} ± {r['clear_se']:.3f} | {r['eaten']:.1f} | {r['score']:.0f} ± {r['score_se']:.0f} | {r['deaths']:.2f} |")
    L += ["", "| config | seed set | clear ± SE | items | score ± SE | deaths |", "|---|---|---|---|---|---|"]
    for lab, nm, r in conf:
        L.append(f"| {lab} | {nm} | {r['clear_rate']:.3f} ± {r['clear_se']:.3f} | {r['eaten']:.1f} | {r['score']:.0f} ± {r['score_se']:.0f} | {r['deaths']:.2f} |")
    (OUT / "ablation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
