import collections
from pathlib import Path

import numpy as np
import pytest
import torch

import msclear.swa_net as swa_net
from msclear.demos import Student
from msclear.swa_net import SWAv2, load_stem

ARCHIVE = Path.home() / "HermesWorkspace/_archive/mspacman-rainbow-2026-10-08/ckpts/swa_rtc_attn_c3_300k.pt"
M8_BEST = Path(__file__).resolve().parents[1] / "runs" / "m8" / "best.pt"


def test_shapes_and_seq_equivalence():
    torch.manual_seed(0)
    net = SWAv2(window=6).eval()
    seq = torch.randint(0, 255, (2, 9, 84, 84), dtype=torch.uint8)
    q1 = net.forward_seq(seq)
    st = torch.stack([seq[:, k:k + 4] for k in range(6)], 1)
    assert q1.shape == (2, 4)
    assert torch.allclose(q1, net(st), atol=1e-5)


def test_causal_newest_token_only_sees_past():
    torch.manual_seed(0)
    net = SWAv2(window=5).eval()
    z = torch.randn(1, 5, 512)
    full = net.blocks[0](z)
    pre = net.blocks[0](z[:, :3])
    assert torch.allclose(full[:, :3], pre, atol=1e-5)


def test_student_cache_matches_clamped_window():
    torch.manual_seed(1)
    net = SWAv2(window=4).eval()
    frames = np.random.default_rng(0).integers(0, 255, (7, 84, 84), dtype=np.uint8)
    s = Student(net)
    s.reset()
    stack = np.repeat(frames[:1], 4, 0)
    for t in range(7):
        if t:
            stack = np.concatenate([stack[1:], frames[t:t + 1]])
        a = s.act(stack)
        idx = np.clip(np.arange(t - 6, t + 1), 0, t)            # W+3 frames, clamped at episode start
        q = net.forward_seq(torch.from_numpy(frames[idx])[None])
        assert a == int(q.argmax())


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archived checkpoint not on this machine")
def test_load_archived_stem():
    net = SWAv2(n_actions=4, window=16)
    keys = load_stem(net, str(ARCHIVE))
    assert any(k.startswith("conv.") for k in keys) and "fc.weight" in keys
    assert "adv.weight" not in keys


def _amax(a, b):
    return float((a - b).abs().max())


def _head_both(net, z):
    saved = swa_net.FAST_LAST
    try:
        swa_net.FAST_LAST = True
        fast = net.head(z)
        swa_net.FAST_LAST = False
        slow = net.head(z)
        return fast, slow
    finally:
        swa_net.FAST_LAST = saved


def test_fast_last_matches_full_trunk_and_head():
    """Last-token final block stays within 1e-5 of the full trunk/head (random init, several W and T)."""
    worst_trunk = worst_head = 0.0
    for window in (8, 1, 2, 16):
        torch.manual_seed(0)
        net = SWAv2(window=window).eval()
        for t in dict.fromkeys(t for t in (1, 3, 8, window) if t <= window):
            torch.manual_seed(t + 10 * window)
            z = torch.randn(37, t, net.d)
            with torch.no_grad():
                dt = _amax(net.trunk(z, last_only=True), net.trunk(z, last_only=False))
                qf, qs = _head_both(net, z)
                dh = _amax(qf, qs)
            assert dt < 1e-5, f"trunk W={window} T={t} diff {dt}"
            assert dh < 1e-5, f"head W={window} T={t} diff {dh}"
            worst_trunk, worst_head = max(worst_trunk, dt), max(worst_head, dh)
    print(f"fast-last max |diff| trunk {worst_trunk:.3e} head {worst_head:.3e}")


@pytest.mark.skipif(not M8_BEST.exists(), reason="runs/m8/best.pt not present")
def test_m8_best_head_last_token():
    ck = torch.load(M8_BEST, map_location="cpu", weights_only=False)
    net = SWAv2(**ck["arch"])
    net.load_state_dict(ck["model"])
    net.eval()
    torch.manual_seed(0)
    z = torch.relu(torch.randn(256, net.window, net.d))
    with torch.no_grad():
        qf, qs = _head_both(net, z)
        diff = _amax(qf, qs)
    assert diff < 1e-4, diff
    assert torch.equal(qf.argmax(-1), qs.argmax(-1))
    print(f"m8 best.pt head max |diff| {diff:.3e}")
