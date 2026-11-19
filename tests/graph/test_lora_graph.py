"""
A loaded LoRA must reach every decode step. The graphed paths don't go through Linear.forward, so with an adapter loaded
a module either takes the python path, or has a LoRA stage in its graphs (model.enable_graph_lora). Both must give the
same decode logits, single and batched, across adapter swaps with no recapture, and with no adapter the stage must
change nothing.

The stage holds several adapters, and each job may name its own (Job(lora = ...)): jobs with different adapters, or
none, then decode in one batch, each as it would alone with only its adapter loaded, and the prefix cache keeps
each adapter's prompts apart.

Run in child processes, one load per scenario. The adapters are random: only their effect matters here.
"""

import pytest
import torch

from testlib.env import get_test_device
from testlib.graph import _rfn, assert_matches_eager, generate_with_logits, load_with_caches, unload_model
from testlib.isolated import device_env, run_isolated

NEW_TOKENS = 16
PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "Once upon a time, in a small town by the sea, there lived",
    "Three things matter most when you build a house:",
]
RANK = 4
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Each job's prompt and adapter seed (None: the base model). Two jobs share a prompt with different adapters, two
# share an adapter, and there are more jobs than adapters.
ROW_JOBS = [(0, 1), (1, 2), (2, None), (0, 3), (1, 1)]


class RandomAdapter:
    """What a loader leaves on the Linears: A (in_features, r), B (r, out_features)"""

    def __init__(self, model, seed: int, rank: int = RANK, scale: float = 0.05):
        from exllamav3.modules.linear import Linear
        gen = torch.Generator().manual_seed(seed)
        self.seed = seed
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
    runs = []

    def run(label, seed):
        adapter = RandomAdapter(model, seed) if seed is not None else None
        results = generate_with_logits(model, cache, tok, PROMPTS, NEW_TOKENS, max_batch_size = len(PROMPTS))
        if stage:
            # the stages held what was loaded, so decode ran in the graphs, not on the python path
            assert model.lora_slots.declined == 0, f"{label}: the stages declined {model.lora_slots.declined} forwards"
            assert list(model.lora_slots.placed) == ([adapter] if adapter else [])
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
    assert_same_until_parting(staged, python)

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


def assert_same_until_parting(got, expected, tol: float = 0.08, typical: float = 0.01, min_steps: int = 3):
    """
    Two runs of the same generations on paths that round differently (the stage accumulates its sums in fp32, the
    python path in fp16): per-step logits within `typical` (relative Frobenius) at the median step and within
    `tol` at every step, for as long as they picked the same tokens, the step where they part included. Greedy
    decoding parts at a near tie, after which the two aren't comparable; before that at least min_steps must have
    matched.

    A random adapter can leave the model sensitive enough that the rounding of one sum moves a step's logits by a
    few percent (the python path does that to itself when its sums are accumulated in fp32), which is why `tol`
    is what it is. A wrong or missing adapter moves them by tens of percent, at every step.
    """
    assert [r[0] for r in got] == [r[0] for r in expected]
    every = []
    for (label, tokens, logits), (_, tokens_e, logits_e) in zip(got, expected):
        same = next((i for i, (a, b) in enumerate(zip(tokens, tokens_e)) if a != b), min(len(tokens), len(tokens_e)))
        steps = min(same + 1, len(logits), len(logits_e))
        assert steps >= min_steps, f"{label}: only {steps} comparable decode steps"
        errs = [_rfn(a, b) for a, b in zip(logits[:steps], logits_e[:steps])]
        assert max(errs) <= tol, f"{label}: decode logits deviate (per-step rfn {[round(e, 5) for e in errs]}, tol {tol})"
        every += errs
    median = sorted(every)[len(every) // 2]
    assert median <= typical, f"decode logits deviate at the median step: rfn {median:.5f}, allowed {typical}"


def generate_rows(model, cache, tok, jobs: list, max_batch_size: int) -> list:
    """Greedy decode of [(prompt, lora)] as concurrent jobs, each naming its adapter: per job (tokens, [per-step
    logits], prompt tokens taken from the cache)"""
    from exllamav3 import Generator, Job, ArgmaxSampler
    with torch.inference_mode():
        gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = max_batch_size)
        return generate_rows_on(gen, tok, jobs)


def generate_rows_on(gen, tok, jobs: list) -> list:
    from exllamav3 import Job, ArgmaxSampler
    from exllamav3.constants import PAGE_SIZE
    js = [
        Job(input_ids = tok.encode(prompt, add_bos = True), max_new_tokens = NEW_TOKENS, sampler = ArgmaxSampler(),
            return_logits = True, lora = lora)
        for prompt, lora in jobs
    ]
    toks = {j: [] for j in js}
    logits = {j: [] for j in js}
    with torch.inference_mode():
        for j in js:
            gen.enqueue(j)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    toks[r["job"]] += r["token_ids"].view(-1).tolist()
                    logits[r["job"]].append(r["logits"].view(-1, r["logits"].shape[-1])[0].cpu().clone())
    return [(toks[j], logits[j], j.cached_pages * PAGE_SIZE + j.cached_tokens) for j in js]


def alone_scenario(model_dir: str) -> list:
    """Each job of ROW_JOBS by itself, with only its adapter loaded, no stage and nothing cached: the python path,
    as adapters ran before jobs could name theirs"""
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    runs = []
    for i, (prompt, seed) in enumerate(ROW_JOBS):
        adapter = RandomAdapter(model, seed, rank = ROW_RANKS[seed]) if seed is not None else None
        (tokens, logits), = generate_with_logits(model, cache, tok, [PROMPTS[prompt]], NEW_TOKENS, max_batch_size = 1)
        runs.append((f"job {i}", tokens, [l.cpu() for l in logits]))
        if adapter is not None:
            adapter.unload()
        # generate_with_logits makes a generator, and so an empty prefix cache, per call
    unload_model(model)
    return runs


# The rank of each adapter of ROW_JOBS, by seed: the stage holds them in rank columns each
ROW_RANKS = {1: RANK, 2: RANK // 2, 3: RANK}


def rows_scenario(model_dir: str, columns: int | None) -> dict:
    """
    ROW_JOBS as one batch with every adapter loaded, each job naming its own, with a stage of `columns` columns
    (None: no stage). Then the same batch again from the prefix cache, then every job with the next job's adapter.
    """
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    if columns is not None:
        counts = model.enable_graph_lora(RANK, columns)
        assert counts["stage"] > 0, f"no module took a LoRA stage: {counts}"
    adapters = {seed: RandomAdapter(model, seed, rank = ROW_RANKS[seed]) for seed in sorted(ROW_RANKS)}
    jobs = [(PROMPTS[prompt], adapters.get(seed)) for prompt, seed in ROW_JOBS]

    from exllamav3 import Generator
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = len(jobs))
    first = generate_rows_on(gen, tok, jobs)
    again = generate_rows_on(gen, tok, jobs)
    # Each job with the adapter of the job after it, one at a time: what the cache holds of its prompt was
    # computed with another adapter, unless that pair already ran
    seen = {(prompt, seed) for prompt, seed in ROW_JOBS}
    shifted = []
    for i, (prompt, _) in enumerate(ROW_JOBS):
        seed = ROW_JOBS[(i + 1) % len(ROW_JOBS)][1]
        (tokens, logits, cached), = generate_rows_on(gen, tok, [(PROMPTS[prompt], adapters.get(seed))])
        shifted.append((prompt, seed, tokens, [l for l in logits], cached, (prompt, seed) in seen))
        seen.add((prompt, seed))

    result = {
        "first": [(f"job {i}", t, l) for i, (t, l, _) in enumerate(first)],
        "again": [(f"job {i}", t, l) for i, (t, l, _) in enumerate(again)],
        "first_cached": [c for _, _, c in first],
        "again_cached": [c for _, _, c in again],
        "shifted": shifted,
        "declined": model.lora_slots.declined if columns is not None else None,
        "unplaced": len(model.lora_slots.unplaced) if columns is not None else None,
    }
    unload_model(model)
    return result


def shifted_alone_scenario(model_dir: str) -> list:
    """rows_scenario's shifted jobs, each by itself with only its adapter loaded and nothing cached"""
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    runs = []
    for i, (prompt, _) in enumerate(ROW_JOBS):
        seed = ROW_JOBS[(i + 1) % len(ROW_JOBS)][1]
        adapter = RandomAdapter(model, seed, rank = ROW_RANKS[seed]) if seed is not None else None
        (tokens, logits), = generate_with_logits(model, cache, tok, [PROMPTS[prompt]], NEW_TOKENS, max_batch_size = 1)
        runs.append((f"shifted {i}", tokens, [l.cpu() for l in logits]))
        if adapter is not None:
            adapter.unload()
    unload_model(model)
    return runs


@pytest.mark.models("dense")
@pytest.mark.parametrize("columns", [
    sum(ROW_RANKS.values()), 4 * RANK, 2 * RANK, None,
], ids = ["columns for all", "spare columns", "too few columns", "no stage"])
def test_jobs_with_different_adapters_decode_in_one_batch(model_id, model_dir, device, columns):
    """
    Each row of a batch runs with its job's adapter as it would alone, in the stage when it holds every adapter and
    on the python path otherwise, and a prompt cached with one adapter is no prefix for another.
    """
    env = device_env(device)
    rows = run_isolated(rows_scenario, model_dir, columns, env = env, timeout = 900.0)
    alone = run_isolated(alone_scenario, model_dir, env = env, timeout = 900.0)
    shifted_alone = run_isolated(shifted_alone_scenario, model_dir, env = env, timeout = 900.0)

    assert_same_until_parting(rows["first"], alone)
    assert_same_until_parting(rows["again"], alone)

    # the adapters do tell the rows apart: two jobs of one prompt with different adapters
    a, b = alone[0][2][1], alone[3][2][1]
    assert _rfn(a, b) > 0.05, f"{model_id}: adapters barely change decode logits; test is vacuous"

    if columns is not None:
        if columns >= sum(ROW_RANKS.values()):
            assert rows["declined"] == 0 and rows["unplaced"] == 0, "the stage holds every adapter, yet forwards left it"
        else:
            assert rows["declined"] > 0 and rows["unplaced"] == 1, "an adapter without columns must take the python path"

    # nothing was cached at first; the second time each job found its own prompt, under its own adapter
    assert rows["first_cached"] == [0] * len(ROW_JOBS)
    assert all(c > 1 for c in rows["again_cached"]), rows["again_cached"]

    shifted = [(f"shifted {i}", t, l) for i, (_, _, t, l, _, _) in enumerate(rows["shifted"])]
    assert_same_until_parting(shifted, shifted_alone)
    for prompt, seed, _, _, cached, seen in rows["shifted"]:
        if seen:
            assert cached > 1, f"prompt {prompt} ran with adapter {seed} before, yet nothing was cached"
        else:
            assert cached == 0, f"prompt {prompt} with adapter {seed} took {cached} tokens cached with another adapter"


@pytest.mark.models("dense")
def test_rank_above_the_stage_takes_the_python_path(model_id, model_dir, device):
    env = device_env(device)
    assert run_isolated(rank_scenario, model_dir, env = env, timeout = 900.0)


def rank_scenario(model_dir: str) -> bool:
    """An adapter the stage can't hold gets no columns, so the python path runs it; a fitting one uses the stage"""
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device())
    model.enable_graph_lora(RANK)
    slots = model.lora_slots
    assert staged_modules(model)

    large = RandomAdapter(model, 3, rank = RANK + 1)
    model.sync_graph_lora()
    declined = large in slots.unplaced and not slots.placed and not slots.ready({}, 1, 1, get_test_device())
    large.unload()

    fitting = RandomAdapter(model, 3, rank = RANK)
    model.sync_graph_lora()
    accepted = not slots.unplaced and list(slots.placed) == [fitting] and slots.ready({}, 1, 1, get_test_device())
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
