"""
ext.lora_down / ext.lora_up (the decode graphs' LoRA stage, one adapter per row): o[r] += x[r] @ A[s] @ B[s] for
row r's adapter s among statics holding several side by side, each in as many columns as its rank, against the same
sum in fp32. Rows without an adapter (rank 0) are left bit for bit as they were.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

SHAPES = [
    # rows, K, N, ranks of the adapters in the statics
    (1, 896, 896, (16,)),
    (4, 896, 4864, (16, 16, 16, 16)),
    (8, 4864, 896, (64, 16, 8, 16)),
    (3, 1000, 130, (5, 1, 3)),
    (128, 896, 128, (16, 4, 64, 2)),
    (7, 33, 257, (1, 2)),
]


def reference(x, a_t, b, slots, o):
    ref = o.float().clone()
    for r in range(x.shape[0]):
        column, rank = (int(v) for v in slots[r])
        if rank:
            t = (x[r].float() @ a_t[column:column + rank].float().T).half().float()
            ref[r] += t @ b[column:column + rank].float()
    return ref


def make_slots(rows, ranks, device, extra = 3):
    """A table with more entries than rows (the kernels read only the first `rows`): adapters at random, with the
    last adapter and a row without one always there"""
    firsts = [sum(ranks[:i]) for i in range(len(ranks))]
    picks = torch.randint(-1, len(ranks), (rows + extra,)).tolist()
    picks[0] = len(ranks) - 1
    if rows > 1:
        picks[1] = -1
    table = [(0, 0) if p < 0 else (firsts[p], ranks[p]) for p in picks]
    return torch.tensor(table, device = device, dtype = torch.int32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows,K,N,ranks", SHAPES)
@torch.inference_mode()
def test_each_row_gets_its_own_adapter(device, dtype, rows, K, N, ranks):
    torch.manual_seed(rows * 1000 + K)
    width, highest = sum(ranks), max(ranks)
    x = torch.randn(rows, K, device = device).half()
    a_t = (torch.randn(width, K, device = device) * 0.1).half()
    b = (torch.randn(width, N, device = device) * 0.1).half()
    slots = make_slots(rows, ranks, device)

    t = torch.full((rows, highest), 7.0, device = device, dtype = torch.half)
    o = torch.randn(rows, N, device = device).to(dtype)
    before = o.clone()
    ext.lora_down(x, a_t, slots, t)
    ext.lora_up(t, b, slots, o)

    ref = reference(x, a_t, b, slots, before)
    assert (o.float() - ref).abs().max() <= 2e-3 * ref.abs().max()
    bits = torch.int16 if dtype == torch.float16 else torch.int32
    for r in range(rows):
        rank = int(slots[r, 1])
        assert t[r, rank:].eq(7).all(), f"row {r}: the intermediate was written beyond its rank {rank}"
        if rank == 0:
            assert torch.equal(o[r].view(bits), before[r].view(bits)), f"row {r} has no adapter but changed"
        else:
            assert not torch.equal(o[r], before[r])


@torch.inference_mode()
def test_strided_input_rows(device):
    """x may be the first columns of a wider buffer (a padded staging buffer)"""
    torch.manual_seed(5)
    rows, K, N, rank = 3, 100, 60, 4
    wide = torch.randn(rows, K + 28, device = device).half()
    x = wide[:, :K]
    a_t = (torch.randn(2 * rank, K, device = device) * 0.1).half()
    b = (torch.randn(2 * rank, N, device = device) * 0.1).half()
    slots = torch.tensor([[rank, rank], [0, rank], [0, 0]], device = device, dtype = torch.int32)
    t = torch.empty(rows, rank, device = device, dtype = torch.half)
    o = torch.zeros(rows, N, device = device)
    ext.lora_down(x, a_t, slots, t)
    ext.lora_up(t, b, slots, o)
    ref = reference(x.contiguous(), a_t, b, slots, torch.zeros(rows, N, device = device))
    assert (o - ref).abs().max() <= 2e-3 * ref.abs().max()


@torch.inference_mode()
def test_rejects_tensors_it_would_misread(device):
    half = dict(device = device, dtype = torch.half)
    x, a_t, b = torch.zeros(2, 8, **half), torch.zeros(4, 8, **half), torch.zeros(4, 6, **half)
    t, o = torch.zeros(2, 4, **half), torch.zeros(2, 6, **half)
    slots = torch.tensor([[0, 4], [0, 2]], device = device, dtype = torch.int32)
    ext.lora_down(x, a_t, slots, t)
    ext.lora_up(t, b, slots, o)
    with pytest.raises(RuntimeError):
        ext.lora_down(x, a_t, slots[:1], t)                             # a row without a slot entry
    with pytest.raises(RuntimeError):
        ext.lora_down(x, a_t, slots[:, 0].contiguous(), t)              # slots must be (rows, 2)
    with pytest.raises(RuntimeError):
        ext.lora_down(x, a_t, slots.long(), t)                          # slots must be int32
    with pytest.raises(RuntimeError):
        ext.lora_down(torch.zeros(2, 9, **half), a_t, slots, t)         # x and A differ in width
    with pytest.raises(RuntimeError):
        ext.lora_up(t, b, slots, torch.zeros(2, 7, **half))             # o and B differ in width
    with pytest.raises(RuntimeError):
        ext.lora_up(torch.zeros(2, 5, **half), b, slots, o)             # a scratch wider than the statics


@torch.inference_mode()
def test_a_rank_above_the_scratch_is_cut_to_it(device):
    """The table is trusted for the columns, but a rank can't make the kernels read past a row of t"""
    half = dict(device = device, dtype = torch.half)
    x, a_t, b = torch.ones(1, 8, **half), torch.ones(6, 8, **half), torch.ones(6, 5, **half)
    t, o = torch.zeros(1, 2, **half), torch.zeros(1, 5, device = device)
    slots = torch.tensor([[0, 6]], device = device, dtype = torch.int32)
    ext.lora_down(x, a_t, slots, t)
    ext.lora_up(t, b, slots, o)
    assert t.eq(8).all() and o.eq(16).all()
