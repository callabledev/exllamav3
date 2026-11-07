"""
A loaded LoRA must reach every decode step. The graphed paths don't go through Linear.forward, so with an adapter loaded
a module either takes the python path, or has a LoRA stage in its graphs (model.enable_graph_lora). Both must give the
same decode logits, single and batched, across adapter swaps with no recapture, and with no adapter the stage must
change nothing.

Run in child processes, one load per scenario. The adapters are random: only their effect matters here.
"""

import pytest
import torch

from testlib.env import get_test_device
from testlib.graph import assert_matches_eager, generate_with_logits, load_with_caches, unload_model
from testlib.isolated import device_env, run_isolated

NEW_TOKENS = 16
PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "Once upon a time, in a small town by the sea, there lived",
    "Three things matter most when you build a house:",
]
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


def staged_modules(model) -> list:
    from exllamav3.modules.attn import Attention
    from exllamav3.modules.mlp import GatedMLP
    return [m for m in model if isinstance(m, (Attention, GatedMLP)) and m.lora_statics is not None]


def scenario(model_dir: str, stage: bool) -> list:
    """
    [(label, tokens, logits)] per prompt, all prompts decoded as one batch, for: no adapter, adapter 1, adapter 2,
    adapter 1 again. Nothing calls sync_graph_lora: the stages must notice each swap themselves.
    """
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    if stage:
        counts = model.enable_graph_lora(RANK + 2)
        assert counts["stage"] > 0, f"no module took a LoRA stage: {counts}"
    staged = staged_modules(model)
    runs = []

    def run(label, seed):
        adapter = RandomAdapter(model, seed) if seed is not None else None
        results = generate_with_logits(model, cache, tok, PROMPTS, NEW_TOKENS, max_batch_size = len(PROMPTS))
        # the stages held what was loaded, so decode ran in the graphs, not on the python path
        assert all(m.lora_stage_ok for m in staged), f"{label}: a stage declined the adapter"
        for i, (tokens, logits) in enumerate(results):
            runs.append((f"{label} #{i}", tokens, [l.cpu() for l in logits]))
        if adapter is not None:
            adapter.unload()

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
    with_adapter, without = by_label["adapter 1 #0"][1], by_label["none #0"][1]
    finite = with_adapter.isfinite() & without.isfinite()   # some models mask tokens out with -inf
    change = (with_adapter - without)[finite].norm() / without[finite].norm()
    assert change > 0.05, f"{model_id}: the adapter barely changes decode logits ({change:.4f}); test is vacuous"

    # a swap copies weights into the statics: the same adapter gives the same logits again
    again = [(label.replace(" again", ""), t, l) for label, t, l in staged if label.startswith("adapter 1 again")]
    first = [r for r in staged if r[0].startswith("adapter 1 #")]
    assert again and len(again) == len(first)
    assert_matches_eager(again, first, tol = 1e-3)


@pytest.mark.models("dense")
def test_rank_above_the_stage_takes_the_python_path(model_id, model_dir, device):
    env = device_env(device)
    assert run_isolated(rank_scenario, model_dir, env = env, timeout = 900.0)


def rank_scenario(model_dir: str) -> bool:
    """An adapter the stage can't hold declines the stage, so the python path runs it; a fitting one uses the stage"""
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    model.enable_graph_lora(RANK)
    modules = staged_modules(model)
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


@pytest.mark.models("dense")
def test_stage_declined_once_graphs_are_captured(model_id, model_dir, device):
    env = device_env(device)
    counts = run_isolated(late_enable_scenario, model_dir, env = env, timeout = 900.0)
    assert counts["stage"] == 0, f"{model_id}: modules took a stage after capture: {counts}"


def late_enable_scenario(model_dir: str) -> dict:
    """enable_graph_lora after a decode captured the graphs: modules decline it, and an adapter still decodes"""
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    generate_with_logits(model, cache, tok, PROMPTS[:1], NEW_TOKENS, max_batch_size = 1)
    counts = model.enable_graph_lora(RANK)
    adapter = RandomAdapter(model, 1)
    generate_with_logits(model, cache, tok, PROMPTS[:1], NEW_TOKENS, max_batch_size = 1)
    adapter.unload()
    unload_model(model)
    return counts


def test_stage_follows_changes_to_the_lora_dicts():
    """No model or GPU: a stage resyncs after any change to its Linears' adapters, and an A without a B adds nothing"""
    from exllamav3.modules.linear import GraphLoraStage, LoraTensors

    class FakeLinear:
        def __init__(self):
            self.in_features, self.out_features = 6, 5
            self.lora_a_tensors, self.lora_b_tensors = LoraTensors(), LoraTensors()

    class FakeModule(GraphLoraStage):
        def __init__(self):
            self.device = "cpu"
            self.linear = FakeLinear()
            self.reset_graph_lora()

        def lora_targets(self):
            return ("x", self.linear),

    module, key = FakeModule(), object()
    assert not module.lora_stage_current()      # no stage
    module.lora_statics = module.make_lora_statics(3)
    a_static, b_static = module.lora_statics["x"]

    a, b = torch.ones(6, 2, dtype = torch.half), torch.full((2, 5), 2.0, dtype = torch.half)
    module.linear.lora_a_tensors[key] = a
    assert module.lora_stage_current() and a_static.sum() == 0     # A alone adds nothing
    module.linear.lora_b_tensors[key] = b
    assert module.lora_stage_current()
    assert torch.equal(a_static[:, :2], a) and torch.equal(b_static[:2], b) and a_static[:, 2:].abs().sum() == 0

    module.linear.lora_a_tensors.pop(key)
    module.linear.lora_b_tensors.clear()
    assert module.lora_stage_current() and a_static.abs().sum() == 0 and b_static.abs().sum() == 0

    module.linear.lora_a_tensors.update({key: torch.ones(6, 4, dtype = torch.half)})
    module.linear.lora_b_tensors[key] = torch.ones(4, 5, dtype = torch.half)
    assert not module.lora_stage_current()      # rank 4 doesn't fit a rank-3 stage
