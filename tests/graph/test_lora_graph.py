"""
A loaded LoRA must reach every decode step. The graphed paths don't go through Linear.forward, so with an adapter loaded
a module either takes the python path, or has a LoRA stage in its graphs (model.enable_graph_lora). Both must give the
same decode logits, across adapter swaps with no recapture, and with no adapter the stage must change nothing.

Run in child processes, one load per scenario. The adapters are random: only their effect matters here.
"""

import pytest
import torch

from testlib.env import get_test_device
from testlib.graph import assert_matches_eager, generate_with_logits, load_with_caches, unload_model
from testlib.isolated import device_env, run_isolated

NEW_TOKENS = 16
PROMPT = "The quick brown fox jumps over the lazy dog because"
RANK = 4
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class RandomAdapter:
    """What a loader leaves on the Linears: A (in_features, r), B (r, out_features)"""

    def __init__(self, model, seed: int, rank: int = RANK, scale: float = 0.05):
        from exllamav3.modules.linear import Linear
        gen = torch.Generator().manual_seed(seed)
        self.targets = [m for m in model if isinstance(m, Linear) and m.key.split(".")[-1] in TARGETS]
        assert self.targets, "model has no LoRA target projections"
        for m in self.targets:
            m.lora_a_tensors[self] = (torch.randn(m.in_features, rank, generator = gen) * scale).half().to(m.device)
            m.lora_b_tensors[self] = (torch.randn(rank, m.out_features, generator = gen) * scale).half().to(m.device)

    def unload(self):
        for m in self.targets:
            m.lora_a_tensors.pop(self, None)
            m.lora_b_tensors.pop(self, None)


def scenario(model_dir: str, stage: bool) -> list:
    """[(label, tokens, logits)] for: no adapter, adapter 1, adapter 2, adapter 1 again"""
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    if stage:
        counts = model.enable_graph_lora(RANK + 2)
        assert counts["stage"] > 0, f"no module took a LoRA stage: {counts}"
    runs = []

    def run(label, seed):
        adapter = RandomAdapter(model, seed) if seed is not None else None
        model.sync_graph_lora()
        ((tokens, logits),) = generate_with_logits(model, cache, tok, [PROMPT], NEW_TOKENS, max_batch_size = 1)
        runs.append((label, tokens, [l.cpu() for l in logits]))
        if adapter is not None:
            adapter.unload()
        model.sync_graph_lora()

    run("none", None)
    run("adapter 1", 1)
    run("adapter 2", 2)
    run("adapter 1 again", 1)
    unload_model(model)
    return runs


@pytest.mark.models("dense")
def test_lora_reaches_decode_with_and_without_a_graph_stage(model_id, model_dir, device):
    env = device_env(device)
    staged = run_isolated(scenario, model_dir, True, env = env, timeout = 900.0)
    python = run_isolated(scenario, model_dir, False, env = env, timeout = 900.0)

    # the rounding of the two paths differs, far below what an adapter does
    assert_matches_eager(staged, python, tol = 0.02, min_steps = 4)

    by_label = {label: logits for label, _, logits in python}
    with_adapter, without = by_label["adapter 1"][1], by_label["none"][1]
    finite = with_adapter.isfinite() & without.isfinite()   # some models mask tokens out with -inf
    change = (with_adapter - without)[finite].norm() / without[finite].norm()
    assert change > 0.05, f"{model_id}: the adapter barely changes decode logits ({change:.4f}); test is vacuous"

    # a swap copies weights into the statics: the same adapter gives the same logits again
    assert_matches_eager([r for r in staged if r[0] == "adapter 1 again"],
                         [("adapter 1 again", t, l) for label, t, l in staged if label == "adapter 1"], tol = 1e-3)


@pytest.mark.models("dense")
def test_rank_above_the_stage_takes_the_python_path(model_id, model_dir, device):
    env = device_env(device)
    assert run_isolated(rank_scenario, model_dir, env = env, timeout = 900.0)


def rank_scenario(model_dir: str) -> bool:
    """An adapter the stage can't hold declines the stage, so the python path runs it; a fitting one uses the stage"""
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.mlp import GatedMLP
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    model.enable_graph_lora(RANK)
    modules = [m for m in model if isinstance(m, (Attention, GatedMLP)) and m.lora_statics is not None]
    assert modules

    large = RandomAdapter(model, 3, rank = RANK + 1)
    model.sync_graph_lora()
    declined = not any(m.lora_stage_ok for m in modules)
    large.unload()

    fitting = RandomAdapter(model, 3, rank = RANK)
    model.sync_graph_lora()
    accepted = all(m.lora_stage_ok for m in modules)
    fitting.unload()
    unload_model(model)
    return declined and accepted
