"""
LoraSlots and Linear.apply_lora, on the CPU with no model: which columns of the graph LoRA stages each adapter holds
as adapters come and go on the Linears, which of them each row of a forward reads, when the stages can't hold a
forward (the python path then runs it), and the python path's own adapter per row.
"""

import pytest
import torch

from exllamav3.modules.linear import ALL_LORAS, GraphLoraStage, Linear, LoraSlots, LoraTensors, lora_rows_for_tokens

pytestmark = pytest.mark.nogpu

IN, OUT, RANK = 6, 5, 3


class FakeLinear:
    def __init__(self):
        self.in_features, self.out_features = IN, OUT
        self.lora_a_tensors, self.lora_b_tensors = LoraTensors(), LoraTensors()

    apply_lora = Linear.apply_lora


class FakeModule(GraphLoraStage):
    def __init__(self):
        self.device = "cpu"
        self.linear = FakeLinear()
        self.other = FakeLinear()
        self.reset_graph_lora()

    def lora_targets(self):
        return ("x", self.linear), ("y", self.other)


class Adapter:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def load(module, adapter, rank = 2, value = 1.0, linears = ("linear", "other")):
    for name in linears:
        linear = getattr(module, name)
        linear.lora_a_tensors[adapter] = torch.full((IN, rank), value, dtype = torch.half)
        linear.lora_b_tensors[adapter] = torch.full((rank, OUT), value * 2, dtype = torch.half)


def unload(module, adapter):
    for linear in (module.linear, module.other):
        linear.lora_a_tensors.pop(adapter, None)
        linear.lora_b_tensors.pop(adapter, None)


def staged(columns = 2 * RANK):
    module = FakeModule()
    slots = LoraSlots(RANK, columns, max_rows = 8)
    module.make_lora_statics(slots)
    return module, slots


def table(slots, rows):
    """[(first column, rank)] per row"""
    return [tuple(row) for row in slots.tables["cpu"][:rows].tolist()]


NONE = (0, 0)


def test_without_a_stage_the_graph_runs_only_with_no_adapter():
    module = FakeModule()
    assert module.lora_stage_ready({}, 1, 1)
    load(module, Adapter("a"))
    assert not module.lora_stage_ready({}, 1, 1)


def test_adapters_take_columns_as_they_load_and_leave_them_when_unloaded():
    module, slots = staged()
    a, b, c = Adapter("a"), Adapter("b"), Adapter("c")
    assert module.lora_stage_ready({}, 2, 1) and table(slots, 2) == [NONE, NONE]     # nothing loaded: no row adds anything

    load(module, a, rank = 2, value = 1.0)
    assert module.lora_stage_ready({}, 2, 1) and slots.placed == {a: (0, 2)}
    a_static, b_static = module.lora_statics["x"]
    assert a_static.shape == (2 * RANK, IN) and b_static.shape == (2 * RANK, OUT)
    # A transposed in the adapter's rows, nothing anywhere else
    assert torch.equal(a_static[:2], torch.ones(2, IN, dtype = torch.half)) and not a_static[2:].any()
    assert torch.equal(b_static[:2], torch.full((2, OUT), 2.0, dtype = torch.half)) and not b_static[2:].any()
    assert table(slots, 2) == [(0, 2), (0, 2)]                                      # every row runs the one adapter

    load(module, b, rank = 3, value = 3.0)
    assert module.lora_stage_ready({"lora_rows": [b, a]}, 2, 1) and slots.placed == {a: (0, 2), b: (2, 3)}
    assert torch.equal(a_static[2:5], torch.full((3, IN), 3.0, dtype = torch.half))
    assert torch.equal(a_static[:2], torch.ones(2, IN, dtype = torch.half))          # a's columns are untouched
    assert table(slots, 2) == [(2, 3), (0, 2)]

    load(module, c, rank = 2)
    assert slots.declined == 0
    assert not module.lora_stage_ready({"lora_rows": [c, a]}, 2, 1)                  # one column left, c needs two
    assert slots.unplaced == {c} and slots.declined == 1
    assert module.lora_stage_ready({"lora_rows": [a, b]}, 2, 1)                      # rows that have columns still run

    unload(module, a)
    assert module.lora_stage_ready({"lora_rows": [c, b]}, 2, 1) and slots.placed == {b: (2, 3), c: (0, 2)}
    assert not slots.unplaced and table(slots, 2) == [(0, 2), (2, 3)]


def test_adapters_are_packed_together_when_the_free_columns_are_in_pieces():
    module, slots = staged(columns = 6)
    a, b, c, d = Adapter("a"), Adapter("b"), Adapter("c"), Adapter("d")
    load(module, a, rank = 2, value = 1.0)
    load(module, b, rank = 2, value = 2.0)
    load(module, c, rank = 2, value = 3.0)
    assert module.lora_stage_ready({"lora_rows": [a, b, c]}, 3, 1)
    assert slots.placed == {a: (0, 2), b: (2, 2), c: (4, 2)}
    unload(module, a)
    unload(module, c)
    load(module, d, rank = 3, value = 4.0)      # four columns are free, two on each side of b
    assert module.lora_stage_ready({"lora_rows": [d, b, None]}, 3, 1)
    assert slots.placed == {b: (0, 2), d: (2, 3)} and not slots.unplaced
    a_static, b_static = module.lora_statics["y"]
    # (the free column keeps what was there: no row reads it)
    assert a_static[:2].eq(2).all() and a_static[2:5].eq(4).all()
    assert b_static[:2].eq(4).all() and b_static[2:5].eq(8).all()
    assert table(slots, 3) == [(2, 3), (0, 2), NONE]


def test_columns_are_cleared_for_the_adapter_that_takes_them():
    module, slots = staged(columns = RANK)
    big, small = Adapter("big"), Adapter("small")
    load(module, big, rank = 3, value = 5.0)
    assert module.lora_stage_ready({}, 1, 1)
    unload(module, big)
    load(module, small, rank = 1, value = 1.0, linears = ("linear",))     # a lower rank, and on one projection only
    assert module.lora_stage_ready({}, 1, 1) and slots.placed == {small: (0, 1)}
    a_x, b_x = module.lora_statics["x"]
    a_y, b_y = module.lora_statics["y"]
    assert a_x[0].eq(1).all() and b_x[0].eq(2).all()
    assert not a_y[0].any() and not b_y[0].any()
    assert table(slots, 1) == [(0, 1)]      # what big left in the other columns is past small's rank


def test_an_adapter_whose_projections_differ_in_rank_takes_the_highest():
    module, slots = staged()
    a = Adapter("a")
    load(module, a, rank = 1, value = 1.0, linears = ("linear",))
    load(module, a, rank = 3, value = 2.0, linears = ("other",))
    assert module.lora_stage_ready({}, 1, 1) and slots.placed == {a: (0, 3)}
    a_x, _ = module.lora_statics["x"]
    a_y, _ = module.lora_statics["y"]
    assert a_x[0].eq(1).all() and not a_x[1:3].any() and a_y[:3].eq(2).all()


def test_a_rank_above_the_stage_gets_no_columns():
    module, slots = staged()
    large = Adapter("large")
    load(module, large, rank = RANK + 1)
    assert not module.lora_stage_ready({}, 1, 1) and slots.unplaced == {large} and not slots.placed
    # ...and rows that don't use it run in the stage all the same
    assert module.lora_stage_ready({"lora_rows": [None]}, 1, 1) and table(slots, 1) == [NONE]


def test_an_a_without_its_b_adds_nothing():
    module, slots = staged()
    a = Adapter("a")
    module.linear.lora_a_tensors[a] = torch.ones(IN, 2, dtype = torch.half)
    assert module.lora_stage_ready({}, 1, 1) and not slots.placed and table(slots, 1) == [NONE]
    assert not module.lora_statics["x"][0].any()
    module.linear.lora_b_tensors[a] = torch.ones(2, OUT, dtype = torch.half)
    assert module.lora_stage_ready({}, 1, 1) and module.lora_statics["x"][0][:2].eq(1).all()
    assert table(slots, 1) == [(0, 2)]


def test_replaced_weights_are_copied_again():
    module, slots = staged()
    a = Adapter("a")
    load(module, a, value = 1.0)
    assert module.lora_stage_ready({}, 1, 1)
    load(module, a, value = 4.0)        # the same adapter, new tensors
    assert module.lora_stage_ready({}, 1, 1) and slots.placed == {a: (0, 2)}
    assert module.lora_statics["x"][0][:2].eq(4).all() and module.lora_statics["y"][1][:2].eq(8).all()
    load(module, a, rank = 3, value = 5.0)      # and of another rank
    assert module.lora_stage_ready({}, 1, 1) and slots.placed == {a: (0, 3)}
    assert module.lora_statics["x"][0][:3].eq(5).all()


def test_rows_without_names_run_everything_loaded():
    module, slots = staged()
    a, b = Adapter("a"), Adapter("b")
    load(module, a)
    assert module.lora_stage_ready(None, 3, 1) and table(slots, 3) == [(0, 2)] * 3
    assert module.lora_stage_ready({"lora_rows": [ALL_LORAS, None, a]}, 3, 1) and table(slots, 3) == [(0, 2), NONE, (0, 2)]
    load(module, b)
    # both on one row: 2 + 2 columns, more than the stage's scratch of RANK is wide
    assert not module.lora_stage_ready({}, 3, 1)
    assert not module.lora_stage_ready({"lora_rows": [ALL_LORAS, None, a]}, 3, 1)
    assert module.lora_stage_ready({"lora_rows": [b, None, a]}, 3, 1) and table(slots, 3) == [(2, 2), NONE, (0, 2)]


def test_a_row_without_a_name_spans_every_loaded_adapter_when_their_ranks_fit():
    module, slots = staged()
    a, b, c = Adapter("a"), Adapter("b"), Adapter("c")
    load(module, a, rank = 1, value = 1.0)
    load(module, b, rank = 1, value = 2.0, linears = ("linear",))     # on one projection only: zeros on the other
    load(module, c, rank = 1, value = 3.0)
    assert module.lora_stage_ready({}, 2, 1) and table(slots, 2) == [(0, 3), (0, 3)]     # 1 + 1 + 1 columns, as one run
    a_y, _ = module.lora_statics["y"]
    assert a_y[0].eq(1).all() and not a_y[1].any() and a_y[2].eq(3).all()
    assert module.lora_stage_ready({"lora_rows": [ALL_LORAS, c]}, 2, 1) and table(slots, 2) == [(0, 3), (2, 1)]

    # b goes: a and c are apart, and are moved together for a row that runs both
    unload(module, b)
    layout = slots.layout
    params = {"lora_rows": [c, ALL_LORAS, None]}
    assert module.lora_stage_ready(params, 3, 1) and slots.placed == {a: (0, 1), c: (1, 1)} and slots.layout == layout + 1
    assert table(slots, 3) == [(1, 1), (0, 2), NONE]
    assert a_y[0].eq(1).all() and a_y[1].eq(3).all()
    assert module.lora_stage_ready(params, 3, 1) and slots.layout == layout + 1 and table(slots, 3) == [(1, 1), (0, 2), NONE]


def test_weights_that_failed_to_follow_their_adapters_are_copied_again():
    """Moving the adapters together and then failing to copy them must not leave rows on columns of other weights"""
    module, slots = staged()
    a, b, c = Adapter("a"), Adapter("b"), Adapter("c")
    load(module, a, rank = 1, value = 1.0)
    load(module, b, rank = 1, value = 2.0)
    load(module, c, rank = 1, value = 3.0)
    assert module.lora_stage_ready({}, 1, 1)
    unload(module, b)
    assert module.lora_stage_ready({"lora_rows": [c]}, 1, 1)      # c stays in the third column while rows are named

    copy = slots.copy
    def failing(*args):
        raise RuntimeError("device error")
    slots.copy = failing
    with pytest.raises(RuntimeError, match = "device error"):
        module.lora_stage_ready({}, 1, 1)                           # moves c beside a, and the copy fails
    slots.copy = copy
    assert module.lora_stage_ready({}, 1, 1) and table(slots, 1) == [(0, 2)]
    a_x, b_x = module.lora_statics["x"]
    assert a_x[0].eq(1).all() and a_x[1].eq(3).all() and b_x[1].eq(6).all()


def test_a_plan_made_before_the_adapters_moved_is_made_again():
    module, slots = staged()
    a, b, c = Adapter("a"), Adapter("b"), Adapter("c")
    for adapter in (a, b, c):
        load(module, adapter, rank = 1)
    named = {"lora_rows": [c]}
    assert module.lora_stage_ready(named, 1, 1) and table(slots, 1) == [(2, 1)]
    unload(module, b)
    assert module.lora_stage_ready(named, 1, 1) and table(slots, 1) == [(2, 1)]
    assert module.lora_stage_ready({}, 1, 1) and table(slots, 1) == [(0, 2)]     # moves c beside a
    assert module.lora_stage_ready(named, 1, 1) and table(slots, 1) == [(1, 1)]   # not the columns its plan had


def test_an_adapter_of_other_widths_than_the_projections_gets_no_columns():
    """It can't be copied in: left to the python path, which fails for the rows that run it, and for nobody else"""
    module, slots = staged()
    good, wide = Adapter("good"), Adapter("wide")
    load(module, good)
    module.linear.lora_a_tensors[wide] = torch.ones(IN + 2, 2, dtype = torch.half)
    module.linear.lora_b_tensors[wide] = torch.ones(2, OUT, dtype = torch.half)
    assert module.lora_stage_ready({"lora_rows": [good]}, 1, 1) and slots.unplaced == {wide}
    assert not module.lora_stage_ready({"lora_rows": [wide]}, 1, 1) and not module.lora_stage_ready({}, 1, 1)
    # made for another model's width, where there was room before
    load(module, good, value = 2.0)
    module.linear.lora_b_tensors[good] = torch.ones(2, OUT + 1, dtype = torch.half)
    assert not module.lora_stage_ready({"lora_rows": [good]}, 1, 1) and slots.unplaced == {good, wide} and not slots.placed


def test_every_token_of_a_row_gets_the_rows_adapter():
    module, slots = staged()
    a, b = Adapter("a"), Adapter("b")
    load(module, a)
    load(module, b, rank = 3)
    assert module.lora_stage_ready({"lora_rows": [b, None, a]}, 3, 2)
    assert table(slots, 6) == [(2, 3), (2, 3), NONE, NONE, (0, 2), (0, 2)]
    assert not module.lora_stage_ready({"lora_rows": [b, a, a]}, 3, 3)      # more rows than the table has: 9 of 8


def test_an_adapter_that_is_not_on_the_stages_projections_adds_nothing_there():
    module, slots = staged()
    a, elsewhere = Adapter("a"), Adapter("elsewhere")
    load(module, a)
    assert module.lora_stage_ready({"lora_rows": [elsewhere, a]}, 2, 1) and table(slots, 2) == [NONE, (0, 2)]


def test_the_plan_is_made_once_per_forward_and_again_when_adapters_change():
    module, slots = staged()
    a, b = Adapter("a"), Adapter("b")
    load(module, a)
    params = {"lora_rows": [a, None]}
    calls = []
    row_places = slots.row_places
    slots.row_places = lambda *args: calls.append(args) or row_places(*args)
    for _ in range(3):      # the modules of one forward
        assert module.lora_stage_ready(params, 2, 1)
    assert len(calls) == 1
    load(module, b)
    assert module.lora_stage_ready(params, 2, 1) and len(calls) == 2
    assert module.lora_stage_ready(params, 2, 2) and len(calls) == 3    # the same params with another shape
    for _ in range(2):
        assert module.lora_stage_ready(params, 2, 2)
    assert len(calls) == 3
    params["lora_rows"] = [b, a]        # the same dict again, for other rows
    assert module.lora_stage_ready(params, 2, 2) and len(calls) == 4
    assert table(slots, 4) == [(2, 2), (2, 2), (0, 2), (0, 2)]


def python_path(lora_rows, bsz, q_len, loaded):
    linear = FakeLinear()
    torch.manual_seed(1)
    weights = {}
    for lora in loaded:
        a, b = torch.randn(IN, 2), torch.randn(2, OUT)
        linear.lora_a_tensors[lora], linear.lora_b_tensors[lora] = a, b
        weights[lora] = (a, b)
    x_in = torch.randn(bsz, q_len, IN)
    out = torch.zeros(bsz, q_len, OUT)
    params = None if lora_rows is None else {"lora_rows": lora_rows}
    linear.apply_lora(x_in, out, params)
    return x_in, out, weights


def test_python_path_gives_each_row_its_adapter():
    a, b = Adapter("a"), Adapter("b")
    x, out, w = python_path([b, None, a, ALL_LORAS], 4, 3, [a, b])
    delta = {lora: x @ w[lora][0] @ w[lora][1] for lora in (a, b)}
    assert torch.allclose(out[0], delta[b][0], atol = 1e-5)
    assert not out[1].any()
    assert torch.allclose(out[2], delta[a][2], atol = 1e-5)
    assert torch.allclose(out[3], (delta[a] + delta[b])[3], atol = 1e-5)


def test_python_path_without_names_adds_every_adapter_to_every_row():
    a, b = Adapter("a"), Adapter("b")
    x, out, w = python_path(None, 2, 3, [a, b])
    assert torch.allclose(out, x @ w[a][0] @ w[a][1] + x @ w[b][0] @ w[b][1], atol = 1e-5)
    x, out_all, _ = python_path([ALL_LORAS, ALL_LORAS], 2, 3, [a, b])
    assert torch.equal(out, out_all)


def test_python_path_takes_flattened_rows():
    """Callers pass (1, bsz * q_len, K) views too: token rows divide among the batch rows in order"""
    a, b = Adapter("a"), Adapter("b")
    x, out, w = python_path([b, a], 2, 3, [a, b])
    linear = FakeLinear()
    for lora in (a, b):
        linear.lora_a_tensors[lora], linear.lora_b_tensors[lora] = w[lora]
    flat_out = torch.zeros(1, 6, OUT)
    linear.apply_lora(x.view(1, 6, IN), flat_out, {"lora_rows": [b, a]})
    assert torch.allclose(flat_out.view(2, 3, OUT), out, atol = 1e-6)
    with pytest.raises(AssertionError):
        linear.apply_lora(x.view(1, 6, IN), flat_out, {"lora_rows": [b, a, a, a]})     # 6 rows don't divide by 4


def test_adapters_named_by_value_match_by_equality_on_both_paths():
    """Tensors registered under a name: a row naming it by an equal string gets it, in the stage and on the python path"""
    module, slots = staged()
    key, name = "voice-1", "".join(["voice", "-1"])     # equal, not the same object
    load(module, key)
    assert name is not key and module.lora_stage_ready({"lora_rows": [name, None]}, 2, 1)
    assert table(slots, 2) == [(0, 2), NONE]
    out = torch.zeros(2, 1, OUT)
    module.linear.apply_lora(torch.ones(2, 1, IN), out, {"lora_rows": [name, None]})
    assert out[0].any() and not out[1].any()


def test_tokens_picked_from_a_forward_keep_their_rows_adapters():
    """A block-sparse MLP hands each expert some of the forward's token rows: params for those name each one's"""
    a, b = Adapter("a"), Adapter("b")
    params = {"lora_rows": [a, None, b], "other": 1}
    picked = lora_rows_for_tokens(params, [5, 0, 3, 4], 6)      # 3 batch rows of 2 tokens
    assert picked["lora_rows"] == [b, a, None, b] and picked["other"] == 1 and params["lora_rows"] == [a, None, b]
    assert lora_rows_for_tokens({"other": 1}, [1], 6) == {"other": 1}      # no names: params as they are
    assert lora_rows_for_tokens(None, [1], 6) is None

    x, out, w = python_path([b, a, None, b], 4, 1, [a, b])     # what the expert's Linear then does with them
    assert torch.allclose(out[0], x[0] @ w[b][0] @ w[b][1], atol = 1e-5) and not out[2].any()


def test_a_module_that_dropped_its_stage_is_left_out():
    """A module loaded again or unloaded by itself resets its stage: the slots go on with the others"""
    module, slots = staged()
    other = FakeModule()
    other.make_lora_statics(slots)
    a = Adapter("a")
    load(module, a)
    load(other, a)
    assert module.lora_stage_ready({}, 1, 1) and slots.stages == [module, other]
    other.reset_graph_lora()
    load(module, Adapter("b"))
    assert module.lora_stage_ready({}, 1, 1) is False and slots.stages == [module]     # two adapters, no names
    assert module.lora_stage_ready({"lora_rows": [a]}, 1, 1)


def test_tokens_may_be_a_tensor_read_only_when_rows_are_named():
    a = Adapter("a")
    picked = lora_rows_for_tokens({"lora_rows": [a, None]}, torch.tensor([3, 0]), 4)
    assert picked["lora_rows"] == [None, a]

    class NeverRead:
        def tolist(self):
            raise AssertionError("no rows are named: the tokens need not be fetched")

    assert lora_rows_for_tokens({}, NeverRead(), 4) == {}
