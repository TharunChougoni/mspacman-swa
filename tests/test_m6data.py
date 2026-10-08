import numpy as np

from msclear.demos import write_shard
from msclear.m6data import Windows, pack


def _fake_rows(n, base):
    f = np.zeros((n, 84, 84), np.uint8)
    for t in range(n):
        f[t, 0, 0] = (base + t) % 256
    return dict(frames=list(f), label=[t % 4 for t in range(n)], act=[t % 4 for t in range(n)],
                reward=[0.0] * n, done=[0] * (n - 1) + [1], items=[154] * n, lives=[3] * n, lost=[0] * n,
                moved=[1] * n, q=[np.zeros(4, np.float32)] * n)


def test_pack_and_clamped_windows(tmp_path):
    src = tmp_path / "demos"
    write_shard(src / "shard_00000.npz", [dict(cleared=True), dict(cleared=False)], [_fake_rows(10, 0), _fake_rows(7, 100)])
    write_shard(src / "shard_00019.npz", [dict(cleared=True)], [_fake_rows(5, 200)])   # 19 % 20 == 19 -> val
    out = pack(src, log=lambda *_: None)
    assert pack(src, log=lambda *_: None) == out                                     # idempotent
    w = Windows([out], window=4, batch=8, split="train")
    assert set(w.rows[0]) == set(range(17))
    f, lab, q = w.gather(0, np.array([0, 3, 12]))
    assert f.shape == (3, 7, 84, 84)
    assert list(f[0, :, 0, 0]) == [0] * 7                          # episode start: all clamped to frame 0
    assert list(f[1, :, 0, 0]) == [0, 0, 0, 0, 1, 2, 3]
    assert list(f[2, :, 0, 0]) == [100, 100, 100, 100, 100, 101, 102]   # 2nd episode starts at row 10
    a, b = w.sample(5), w.sample(5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])    # batch k is deterministic
    v = Windows([out], window=4, batch=4, split="val")
    assert set(v.rows[0]) == {17, 18, 19, 20, 21}
