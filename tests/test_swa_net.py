import collections
from pathlib import Path

import numpy as np
import pytest
import torch

from msclear.demos import Student
from msclear.swa_net import SWAv2, load_stem

ARCHIVE = Path.home() / "HermesWorkspace/_archive/mspacman-rainbow-2026-10-08/ckpts/swa_rtc_attn_c3_300k.pt"


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
