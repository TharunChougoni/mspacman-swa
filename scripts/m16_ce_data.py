"""M16 teacher-CE data: teacher demo decisions at the (replay-matched) TRAIN curriculum states (runs/m16/buckets.json
'train', all buckets) plus the next SPAN-1 decisions of the same demo. Output in the m14 aux format (frames.npy +
demo_meta.npz: fbase -> FRA[fbase:fbase+4] is the 4-frame stack at that decision; g0 = first decision of the demo)."""
import json, os, re, collections
import numpy as np
SPAN = int(os.environ.get("SPAN", "6")); OUT = "runs/m16/ce_data"; os.makedirs(OUT, exist_ok=True)
b = json.load(open("runs/m16/buckets.json"))
want = collections.defaultdict(set)
for lst in b["train"]:
    for f in lst:
        m = re.match(r".*/(.+)_(\d{5})\.pkl$", f); want[m.group(1)].update(range(int(m.group(2)), int(m.group(2)) + SPAN))
evd = {re.match(r".*/(.+)_(\d{5})\.pkl$", f).group(1) for lst in b["eval"] for f in lst}
assert not (evd & set(want)), "eval demo leaked"
fr, lab, act, fb, g0, gid, files = [], [], [], [], [], [], []; off = 0; dec = 0; bad = 0
for k, base in enumerate(sorted(want)):
    z = np.load(f"runs/teacher/demos/{base}.npz"); n = len(z["act"])
    bad += int(not np.array_equal(z["stack0"][-1], z["frames"][0]))
    seq = np.concatenate([z["stack0"][:3], z["frames"]]); fr.append(seq)
    l = np.zeros(n, bool); l[[i for i in want[base] if i < n]] = True
    lab.append(l); act.append(z["act"].astype(np.int8)); fb.append(off + np.arange(n)); g0.append(np.full(n, dec)); gid.append(np.full(n, k, np.int32))
    files.append(base); off += len(seq); dec += n
np.save(f"{OUT}/frames.npy", np.concatenate(fr))
L = np.concatenate(lab)
np.savez(f"{OUT}/demo_meta.npz", lab=L, act=np.concatenate(act), fbase=np.concatenate(fb), g0=np.concatenate(g0),
         pd=np.zeros(dec, bool), gid=np.concatenate(gid), val_games=np.zeros(len(files), bool), files=np.array(files))
A = np.concatenate(act)[L]
print(json.dumps(dict(demos=len(files), decisions=dec, labeled=int(L.sum()), stack0_mismatch=bad, act_hist=np.bincount(A, minlength=4).tolist())))
