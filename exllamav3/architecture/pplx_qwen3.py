from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP
from ..modules.attn import prepare_for_attn
from ..util.rope import RopeStyle, RoPE


class PPLXQwen3Config(Config):
    arch_string = "PPLXQwen3Model"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": PPLXQwen3Model},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)

        # PPLX embedding checkpoints use Qwen3 blocks with bidirectional
        # self-attention and no language-modeling head.
        self.use_bidirectional_attention = self.read_cfg(bool, "use_bidirectional_attention", True)
        self.assert_cfg(bool, "use_cache", False, True)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)


class PPLXQwen3Model(Model):
    config_class = PPLXQwen3Config

    def __init__(
        self,
        config: PPLXQwen3Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        head_dim = config.head_dim,
                        num_q_heads = config.num_q_heads,
                        num_kv_heads = config.num_kv_heads,
                        rope_settings = config.rope_settings,
                        sm_scale = None,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "o_proj",
                        qmap = "block.attn",
                        q_norm = RMSNorm(
                            config = config,
                            key = f"layers.{idx}.self_attn.q_norm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                        k_norm = RMSNorm(
                            config = config,
                            key = f"layers.{idx}.self_attn.k_norm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                        out_dtype = torch.float,
                        use_cu_seqlens = True,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            RMSNorm(
                config = config,
                key = "norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            )
        ]

        # No logit layer: forward() returns final token hidden states.
        self.caps.update({"mrope": True})
        self.g_rope = RoPE("cpu", config.rope_settings)


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params.setdefault("attn_mode", "flash_attn_nc")
        params.setdefault("causal", not self.config.use_bidirectional_attention)
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def per_layer_quant_preamble(self, params: dict):
        params.setdefault("causal", not self.config.use_bidirectional_attention)
