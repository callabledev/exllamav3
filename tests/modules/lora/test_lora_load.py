"""
LoRA (PEFT adapter files onto a model's Linears), on the CPU with a stand-in model: tensors land transposed and
scaled on the projections they name, and a load that fails partway, in the graph stages' sync too, leaves nothing
behind, since the caller gets no adapter to unload.
"""

import json

import pytest
import torch
from safetensors.torch import save_file

from exllamav3.model.lora import LoRA
from exllamav3.modules.linear import Linear, LoraTensors

pytestmark = pytest.mark.nogpu

IN, OUT, RANK = 6, 5, 2
KEYS = ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj")


def linear(key):
    module = object.__new__(Linear)
    module.key, module.in_features, module.out_features = key, IN, OUT
    module.device = torch.device("cpu")
    module.is_sliced = False
    module.lora_a_tensors, module.lora_b_tensors = LoraTensors(), LoraTensors()
    return module


class Model:
    def __init__(self, sync = None):
        self.linears = [linear(key) for key in KEYS]
        self.modules_dict = {m.key: m for m in self.linears}
        self.sync = sync
        self.synced = 0

    def sync_graph_lora(self):
        self.synced += 1
        if self.sync is not None:
            self.sync()


def adapter_dir(tmp_path, alpha = 4):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": RANK, "lora_alpha": alpha}))
    tensors = {}
    for i, key in enumerate(KEYS):
        tensors[f"base_model.model.{key}.lora_A.weight"] = torch.full((RANK, IN), float(i + 1))
        tensors[f"base_model.model.{key}.lora_B.weight"] = torch.full((OUT, RANK), 0.5)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    return str(tmp_path)


def test_tensors_land_on_the_projections_they_name(tmp_path):
    model = Model()
    lora = LoRA.from_directory(model, adapter_dir(tmp_path))
    for i, module in enumerate(model.linears):
        a, b = module.lora_a_tensors[lora], module.lora_b_tensors[lora]
        assert a.shape == (IN, RANK) and b.shape == (RANK, OUT) and a.dtype == b.dtype == torch.half
        assert a.eq(i + 1).all() and b.eq(0.5 * 4 / RANK).all()     # B carries alpha / r
    assert model.synced == 1
    lora.unload()
    assert all(not m.lora_a_tensors and not m.lora_b_tensors for m in model.linears) and model.synced == 2


def test_a_load_that_fails_in_the_stages_sync_leaves_nothing_on_the_model(tmp_path):
    calls = []

    def sync():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("the stages can't take it")

    model = Model(sync)
    with pytest.raises(RuntimeError, match = "can't take it"):
        LoRA.from_directory(model, adapter_dir(tmp_path))
    assert all(not m.lora_a_tensors and not m.lora_b_tensors for m in model.linears)
    assert len(calls) == 2      # and the stages were told it is gone


def test_a_load_that_fails_on_a_tensor_leaves_nothing_on_the_model(tmp_path):
    model = Model()
    # The second projection can't be reached: its tensors fail to move there, after the first one's are on
    model.linears[1].device = "no such device"
    with pytest.raises(RuntimeError):
        LoRA.from_directory(model, adapter_dir(tmp_path))
    assert all(not m.lora_a_tensors and not m.lora_b_tensors for m in model.linears)
