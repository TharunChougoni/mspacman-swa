"""Teacher value labels (runs/m16/qlabels) -> soft targets for M16 aux soft-CE. Q = max over rollouts per action;
actions whose every rollout died (Q < DEAD) are masked to 0; target = softmax(Q/T) over the rest; states where all 4 die
are dropped. Rows index the decisions in runs/m16/ce_data (same train demos) -> runs/m16/qdata.npz (dec, p, left)."""
import glob, json, os
import numpy as np
T, DEAD = float(os.environ.get("QT", "1.0")), -250.0
md = np.load("runs/m16/ce_data/demo_meta.npz"); files = list(md["files"]); g0 = {f: int(md["g0"][md["gid"] == k][0]) for k, f in enumerate(files)}
D, P, LF, Q = [], [], [], []
for f in sorted(glob.glob("runs/m16/qlabels/c*.npz")):
    z = np.load(f); q = z["q"].max(2)
    for d, i, l, qq in zip(z["demo"], z["i"], z["left"], q):
        dead = qq < DEAD
        if dead.all() or str(d) not in g0:
            continue
        x = np.where(dead, -np.inf, (qq - qq[~dead].max()) / T); p = np.exp(x); p /= p.sum()
        D.append(g0[str(d)] + int(i)); P.append(p); LF.append(int(l)); Q.append(qq)
P = np.array(P, np.float32); D = np.array(D); Q = np.array(Q)
np.savez("runs/m16/qdata.npz", dec=D, p=P, left=np.array(LF), q=Q.astype(np.float32))
H = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1); s = np.sort(Q, 1)[:, ::-1]; nd = (Q < DEAD).sum(1)
print(json.dumps(dict(n=len(D), T=T, target_entropy_mean=float(H.mean()), uniform_ce=float(np.log(4)),
    uniform_ce_on_targets=float(-(P * np.log(0.25)).sum(1).mean()), max_p_mean=float(P.max(1).mean()),
    frac_near_tie_gap_lt_1=float(np.mean(s[:, 0] - s[:, 1] < 1)), frac_clear_best_gap_gt_3=float(np.mean(s[:, 0] - s[:, 1] > 3)),
    frac_some_action_dead=float(np.mean(nd > 0)), dead_count_hist=np.bincount(nd, minlength=5).tolist())))
