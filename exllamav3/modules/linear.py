from __future__ import annotations
from functools import cached_property
import threading
from typing_extensions import override
import torch
import numpy as np
from ..model.config import Config
from . import Module
from .quant import LinearFP16, LinearFP16Packed, PackedSegment, packed_fmt, LinearEXL3
from .quant.exl3_lib.hessian_packed import accumulate_, packed_numel
from .quant.exl3_lib import quantize_exl3, quantize_exl3_batch
from .quant.exl3_lib.quantize import codebook_mcg_mult, codebook_mul1_mult
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
from ..util.pinned_arena import PinnedArena

# MXFP4 (e2m1 + e8m0 block scale) as stored by gpt-oss: each 16-byte block packs 32 fp4 values
# (low nibble first), one power-of-two scale byte per block
_MXFP4_LUT = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def _mxfp4_dequant(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """(N, G, 16) uint8 blocks + (N, G) uint8 scales -> (N, G * 32) fp16"""
    lut = torch.tensor(_MXFP4_LUT, dtype = torch.float, device = blocks.device)
    b = blocks.int()
    w = torch.stack((lut[b & 0x0F], lut[b >> 4]), dim = -1).view(blocks.shape[0], blocks.shape[1], 32)
    # exp2 multiply rather than torch.ldexp: exact for these integer exponents, and the CUDA
    # ldexp kernel in some torch builds reads out of bounds on broadcast integer exponents
    # (silent on a warm allocator, illegal-memory-access on a freshly initialized device)
    w = w * torch.exp2(scales.float() - 127.0).unsqueeze(-1)
    return w.view(blocks.shape[0], -1).half()


_lora_generation = 0


def _counting(name: str):
    method = getattr(dict, name)
    def counted(self, *args, **kwargs):
        global _lora_generation
        _lora_generation += 1
        return method(self, *args, **kwargs)
    return counted


class LoraTensors(dict):
    """
    A Linear's adapter tensors, {lora: A} or {lora: B}. Every change counts in _lora_generation, so a graph's LoRA stage
    notices an adapter that came or went since it last copied them in, whoever changed the dict
    """
    __setitem__ = _counting("__setitem__")
    __delitem__ = _counting("__delitem__")
    __ior__ = _counting("__ior__")
    pop = _counting("pop")
    popitem = _counting("popitem")
    clear = _counting("clear")
    update = _counting("update")
    setdefault = _counting("setdefault")


class _AllLoras:
    def __repr__(self):
        return "ALL_LORAS"

    def __reduce__(self):
        return "ALL_LORAS"      # pickles as this module's own, so it stays the one object


# A row, or a job, that runs with every adapter loaded on the model, which is what a forward does when nothing names
# its rows' adapters (params["lora_rows"]). Its own object: no value an adapter could be registered under
ALL_LORAS = _AllLoras()


def lora_rows_of(params: dict | None, rows: int) -> list | None:
    """
    The adapter of each batch row of a forward over `rows` token rows, or None when every row runs all loaded
    adapters. params["lora_rows"] has one entry per batch row: an adapter (the key it is loaded under on the Linears),
    None for the base model, or ALL_LORAS.
    """
    lora_rows = params.get("lora_rows") if params is not None else None
    if lora_rows is None or all(r is ALL_LORAS for r in lora_rows):
        return None
    assert len(lora_rows) > 0 and rows % len(lora_rows) == 0, \
        f"lora_rows names {len(lora_rows)} batch rows, the input has {rows} token rows"
    return lora_rows


def lora_rows_for_tokens(params: dict | None, tokens, rows: int) -> dict | None:
    """
    params for a forward over some of the token rows of a forward over `rows` of them, picked by index (a list or
    a tensor; what a block-sparse MLP hands each expert): each picked token becomes a batch row with its own row's
    adapter, since they no longer divide among the batch rows in order.
    """
    lora_rows = lora_rows_of(params, rows)
    if lora_rows is None:
        return params
    per_row = rows // len(lora_rows)
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()    # only now: it waits for the device
    return {**params, "lora_rows": [lora_rows[int(t) // per_row] for t in tokens]}


def is_lora_row(row, lora) -> bool:
    """Whether a batch row named `row` runs the adapter loaded under the key `lora`: by the key's own equality, as
    the Linears' dicts and LoraSlots find it"""
    return row is lora or row is ALL_LORAS or (row is not None and row == lora)


class LoraSlots:
    """
    The adapters in a model's graph LoRA stages (model.enable_graph_lora), and which of them each row of a decode
    step runs with.

    Every stage's statics are `columns` wide and hold the loaded adapters side by side, each in as many columns
    as its rank, the same adapter in the same columns of every projection's A and B. One int32 table per device,
    read by all the stages on it, names each row's first column and rank (rank 0: no adapter). So a row costs what
    its own adapter's rank does, whatever else is loaded, and a change of rows only rewrites the table.

    Adapters reach the columns by copy whenever the Linears' adapter dicts changed, whoever changed them: one that
    is loaded takes free columns and keeps them until it is unloaded. If the free columns would do but are in
    pieces, the adapters are packed together first, between two forwards. A row that names no adapter runs all
    that are loaded: their columns as one run, which adds up every adapter's part.

    A step the stages can't hold makes ready() return False and runs on the python path, which gives the same
    result: one with a row whose adapter found no room, is above the stages' `rank` or doesn't have the
    projections' widths, or with a row running all the loaded adapters when their ranks come to more than `rank`.
    """

    def __init__(self, rank: int, columns: int, max_rows: int):
        assert rank > 0 and columns > 0
        self.rank = rank                    # the highest rank an adapter may have: the width of the stages' scratch
        self.columns = columns              # the width of the statics, shared by the adapters loaded
        self.max_rows = max_rows
        self.stages = []                    # modules with a stage (GraphLoraStage)
        self.placed = {}                    # adapter -> (first column, rank)
        self.copied = {}                    # adapter -> (first column, [(stage, name, a, b)]) as last copied
        self.unplaced = set()               # loaded adapters with no columns
        self.synced = -1                    # _lora_generation at the last refresh
        self.layout = 0                     # counts the times adapters were moved (pack)
        self.lock = threading.RLock()       # ready(): a prefill pipeline runs each device's modules on a thread
        self.tables = {}                    # device -> int32 (max_rows, 2): each row's first column and rank
        self.table_rows = {}                # device -> what its table holds, as a tuple
        self.declined = 0                   # forwards the stages couldn't hold, per staged module (ready)


    def table(self, device) -> torch.Tensor:
        table = self.tables.get(device)
        if table is None:
            table = torch.zeros((self.max_rows, 2), dtype = torch.int32, device = device)
            self.tables[device] = table
            self.table_rows[device] = ()
        return table


    def loaded(self) -> dict:
        """{adapter: [(stage, name, a, b)]} for everything on the stages' projections. An A without its B is left
        out: it adds nothing, as in Linear.apply_lora"""
        loaded = {}
        for stage in self.stages:
            for name, linear in stage.lora_targets():
                for lora, a in linear.lora_a_tensors.items():
                    parts = loaded.setdefault(lora, [])
                    b = linear.lora_b_tensors.get(lora)
                    if b is not None:
                        parts.append((stage, name, a, b))
        return loaded


    def free(self) -> int:
        return self.columns - sum(rank for _, rank in self.placed.values())


    def find(self, rank: int) -> int | None:
        """The first run of `rank` free columns"""
        column = 0
        for first, taken in sorted(self.placed.values()):
            if first - column >= rank:
                return column
            column = first + taken
        return column if self.columns - column >= rank else None


    def pack(self):
        """Move the adapters together, leaving the free columns in one piece. Their weights follow (copy_placed)"""
        column = 0
        for lora, (_, rank) in sorted(self.placed.items(), key = lambda item: item[1]):
            self.placed[lora] = (column, rank)
            column += rank
        self.layout += 1


    def refresh(self):
        """Bring the columns up to date with the adapters on the Linears"""
        generation = _lora_generation
        # A module that was loaded again or unloaded by itself has dropped its stage
        self.stages = [stage for stage in self.stages if stage.lora_slots is self and stage.lora_statics is not None]
        loaded = self.loaded()
        for lora in list(self.placed):
            if lora not in loaded:
                self.release(lora)
        self.unplaced = set()
        for lora, parts in loaded.items():
            rank = max((max(a.shape[1], b.shape[0]) for _, _, a, b in parts), default = 0)
            # One made for other widths than the projections' can't be copied in. The python path says so, to
            # the rows that run it
            fits = rank <= self.rank and all(
                a.shape[0] == stage.lora_statics[name][0].shape[1] and b.shape[1] == stage.lora_statics[name][1].shape[1]
                for stage, name, a, b in parts
            )
            place = self.placed.get(lora)
            if place is not None and (place[1] != rank or not fits):
                self.release(lora)
                place = None
            if place is None and rank > 0:
                column = self.find(rank) if fits else None
                if column is None and fits and rank <= self.free():
                    self.pack()
                    column = self.find(rank)
                if column is None:
                    self.unplaced.add(lora)
                else:
                    self.placed[lora] = (column, rank)
        # After every adapter has its place: packing moves the ones placed before
        self.copy_placed(loaded)
        self.synced = generation


    def copy_placed(self, loaded: dict):
        """Copy in the adapters whose weights or columns aren't the ones last copied"""
        for lora, (column, rank) in self.placed.items():
            parts = loaded[lora]
            copied = self.copied.get(lora)
            if copied is None or copied[0] != column or len(copied[1]) != len(parts) or any(
                p[0] is not c[0] or p[1] != c[1] or p[2] is not c[2] or p[3] is not c[3] for p, c in zip(parts, copied[1])
            ):
                self.copy(column, rank, parts)
                self.copied[lora] = (column, parts)


    def everything(self) -> tuple | None:
        """
        (first column, rank) of a row that runs every loaded adapter, or None if the stages can't hold that: the
        adapters' columns as one run, each projection adding up the parts of all of them (an adapter has zeros
        where it has no weights). They are packed together for it if they aren't, which is between two forwards.
        """
        if self.unplaced:
            return None
        if not self.placed:
            return (0, 0)
        total = sum(rank for _, rank in self.placed.values())
        if total > self.rank:
            return None
        first = min(column for column, _ in self.placed.values())
        if max(column + rank for column, rank in self.placed.values()) - first != total:
            self.pack()
            try:
                self.copy_placed(self.loaded())
            except BaseException:
                self.synced = -1    # the weights didn't all follow their adapters: refresh() copies them again
                raise
            first = 0
        return (first, total)


    def release(self, lora):
        self.placed.pop(lora, None)
        self.copied.pop(lora, None)


    def copy(self, column: int, rank: int, parts: list):
        """Put an adapter's weights into its columns of every stage: zeros where it has none"""
        for stage in self.stages:
            for a_static, b_static in stage.lora_statics.values():
                a_static[column:column + rank].zero_()
                b_static[column:column + rank].zero_()
        for stage, name, a, b in parts:
            a_static, b_static = stage.lora_statics[name]
            a_static[column:column + a.shape[1]].copy_(a.T)
            b_static[column:column + b.shape[0]].copy_(b)


    def row_places(self, params: dict | None, bsz: int, q_len: int) -> tuple | None:
        """(first column, rank) of every token row of a forward, flattened, or None if the stages can't hold it"""
        lora_rows = lora_rows_of(params, bsz * q_len)
        if lora_rows is None:
            everything = self.everything()
            return None if everything is None else everything * (bsz * q_len)
        if len(lora_rows) != bsz:
            return None
        if any(lora is ALL_LORAS for lora in lora_rows):
            everything = self.everything()      # first: it may move the adapters
        places = ()
        for lora in lora_rows:
            if lora is None:
                place = (0, 0)
            elif lora is ALL_LORAS:
                place = everything
            elif lora in self.unplaced:
                place = None
            else:
                # One that isn't on the stages' projections at all adds nothing there
                place = self.placed.get(lora, (0, 0))
            if place is None:
                return None
            places += place * q_len
        return places


    def ready(self, params: dict | None, bsz: int, q_len: int, device) -> bool:
        """
        Whether the stages on `device` can run this forward, having set the row table for it if so. Called by each
        staged module before its graph: the plan is made once per forward and kept in params, with the rows it
        was made for.
        """
        with self.lock:
            return self.ready_locked(params, bsz, q_len, device)


    def ready_locked(self, params: dict | None, bsz: int, q_len: int, device) -> bool:
        if self.synced != _lora_generation:
            self.refresh()
        if bsz * q_len > self.max_rows:
            self.declined += 1
            return False
        plan = params.get("lora_stage_plan") if params is not None else None
        lora_rows = params.get("lora_rows") if params is not None else None
        rows_key = None if lora_rows is None else tuple(lora_rows)
        if plan is None or plan[0] != (self, self.synced, self.layout, bsz, q_len, rows_key):
            places = self.row_places(params, bsz, q_len)
            # The key after the places: making them may have moved the adapters
            plan = ((self, self.synced, self.layout, bsz, q_len, rows_key), places)
            if params is not None:
                params["lora_stage_plan"] = plan
        places = plan[1]
        if places is None:
            self.declined += 1
            return False
        table = self.table(device)
        if self.table_rows[device] != places:
            # A blocking copy, on the stream the graphs launch on: the steps already queued have read the table
            table[:bsz * q_len].copy_(torch.tensor(places, dtype = torch.int32).view(-1, 2))
            self.table_rows[device] = places
        return True


class GraphLoraStage:
    """
    For modules whose decode graphs can carry a LoRA stage (Attention, GatedMLP): fp16 statics the captured graphs
    read, holding the adapters of the model's LoraSlots for lora_targets().
    """

    def lora_targets(self) -> tuple:
        """((name, Linear), ...) the stage applies LoRA to"""
        raise NotImplementedError()


    def has_lora(self) -> bool:
        return any(p.lora_a_tensors for _, p in self.lora_targets())


    def reset_graph_lora(self):
        self.lora_slots = None      # enable_graph_lora: the model's LoraSlots
        self.lora_statics = None    # {name: (A transposed, B)} the captured graphs read


    def make_lora_statics(self, slots: LoraSlots):
        """Join the model's slots: statics of (columns, in_features) and (columns, out_features)"""
        def z(*shape):
            return torch.zeros(shape, dtype = torch.half, device = self.device)
        width = slots.columns
        self.lora_statics = {name: (z(width, p.in_features), z(width, p.out_features)) for name, p in self.lora_targets()}
        self.lora_slots = slots
        slots.stages.append(self)
        slots.synced = -1


    def lora_stage_ready(self, params: dict | None, bsz: int, q_len: int) -> bool:
        """
        Whether the graphed path gives this forward its adapters: with a stage, when the stage holds every row's
        (the row table is then set for it); without one, only when no adapter is loaded on the projections.
        """
        if self.lora_slots is None:
            return not self.has_lora()
        return self.lora_slots.ready(params, bsz, q_len, self.device)


class Linear(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        in_features: int,
        out_features: int,
        qmap: str | None = None,
        alt_key: str | list | None = None,
        qbits_key: str = "bits",
        fkey : str | None = None,
        frange: tuple[int, int] | None = None,
        frange_dim: int = 0,
        fidx: int | None = None,
        finterleaved: bool = False,
        fdequant = None,
        caps: dict = None,
        softcap: float = 0.0,
        pad_to: int = 128,
        trim_padded_out: bool = False,
        full_in_features: int | None = None,
        full_out_features: int | None = None,
        first_in_feature: int | None = None,
        first_out_feature: int | None = None,
        out_dtype: torch.dtype | None = None,
        allow_input_padding: bool = False,
        pre_scale: float = 1.0,
        post_scale: float = 1.0,
        weight_scale: float = 1.0,
        transposed_load: bool = True,
        transpose_fused_weights: bool = True,
        ftranspose_after_load: bool = True,
        select_hq_bits: int = 0,
        qgroup: str = None
    ):
        super().__init__(config, key, qmap)

        self.alt_key = alt_key
        self.in_features_unpadded = in_features
        self.in_features = (in_features + pad_to - 1) // pad_to * pad_to
        self.out_features_unpadded = out_features
        self.out_features = (out_features + pad_to - 1) // pad_to * pad_to
        self.trim_padded_out = trim_padded_out
        self.full_in_features = full_in_features if full_in_features is not None else self.in_features
        self.full_out_features = full_out_features if full_out_features is not None else self.out_features
        self.first_in_feature = first_in_feature if first_in_feature is not None else 0
        self.first_out_feature = first_out_feature if first_out_feature is not None else 0
        self.inner = None
        self.qbits_key = qbits_key
        self.fkey = fkey
        self.frange = frange
        self.frange_dim = frange_dim
        self.fidx = fidx
        self.finterleaved = finterleaved
        # Optional architecture-supplied reader for the fused checkpoint tensor named by fkey.
        # Signature: (stc, fkey, device) -> fp16 (out_features, in_features) tensor
        self.fdequant = fdequant
        self.quant_type = None
        self.softcap = softcap
        self.is_sliced = self.in_features < self.full_in_features or self.out_features < self.full_out_features
        self.out_dtype = out_dtype
        self.pre_scale = pre_scale
        self.post_scale = post_scale
        self.weight_scale = weight_scale
        self.transposed_load = transposed_load
        self.transpose_fused_weights = transpose_fused_weights
        self.ftranspose_after_load = ftranspose_after_load
        self.select_hq_bits = select_hq_bits
        self.qgroup = qgroup or key
        self.lora_a_tensors = LoraTensors()
        self.lora_b_tensors = LoraTensors()
        # Optional frequency-selected column subset for drafting (see modules/sparse_head.py); the
        # regular forward never consults it
        self.sparse_head = None

        # in_features_unpadded < in_features is handled at runtime: inputs narrower than the
        # (padded) weight are zero-extended in forward(). allow_input_padding marks layers whose
        # inputs already arrive padded (e.g. MoE down projections fed from padded gate/up)
        self.allow_input_padding = allow_input_padding

        if caps is not None:
            self.caps.update(caps)


    @override
    def optimizer_targets(self):
        return [self.key]


    def pad_out(self, w: torch.Tensor | None) -> torch.Tensor | None:
        """Zero-pad a weight in (in_features, out_features) orientation (or a bias) to the padded
        dims. Padding either or both dims with zeros is inert: pad output columns produce zero
        (trimmed or consumed by a matching pad on the consumer's input dim), pad input rows only
        ever see zero activations."""
        if w is None or self.out_features == self.out_features_unpadded and self.in_features == self.in_features_unpadded:
            return w
        if w.dim() == 2:
            padded = torch.zeros((self.in_features, self.out_features), dtype = w.dtype, device = w.device)
            padded[:w.shape[0], :w.shape[1]] = w
        else:
            assert w.dim() == 1
            padded = torch.zeros((self.out_features,), dtype = w.dtype, device = w.device)
            padded[:w.shape[0]] = w
        return padded.contiguous()


    def _packed_source(self) -> bool:
        """Conversion with packed_fp16_source: natively quantized tensors stay packed (LinearFP16Packed)"""
        return bool(getattr(self.config.infer_params, "packed_fp16_source", False)) and not self.is_sliced

    def _make_packed(self, segments: list, bias: torch.Tensor | None) -> bool:
        self.inner = LinearFP16Packed(
            self.in_features, self.out_features, segments, bias, self.pad_out,
            global_scale = self.weight_scale, out_dtype = self.out_dtype, key = self.key,
        )
        self.quant_type = "fp16"
        return True

    # Row scales (Transformers-specific?)
    def apply_fp8_scales_(self, weight: torch.Tensor, scale: torch.Tensor):
        ws = weight.shape
        ss = scale.shape
        assert len(ss) == 0 or len(ws) == len(ss) == 2
        if len(ss) == 2:
            if 1 < ss[0] < ws[0]:
                scale = torch.nn.functional.pad(scale, (0, 0, 0, ws[0] - ss[0]), "constant", 1)
            if 1 < ss[1] < ws[1]:
                scale = torch.nn.functional.pad(scale, (0, ws[1] - ss[1]), "constant", 1)
        weight = weight.float() * scale
        return weight.view(ws).half()


    def apply_fp8_scales_inv_(self, weight: torch.Tensor, scale_inv: torch.Tensor):
        ws = weight.shape
        ss = scale_inv.shape
        if len(ss) > 0:
            assert len(ws) == len(ss) == 2
            assert all(s == (w + 127) // 128 for w, s in zip(ws, ss)), \
                f"fp8 scale grid {tuple(ss)} does not cover a {tuple(ws)} weight in 128x128 blocks"
            if all(w == s * 128 for w, s in zip(ws, ss)):
                weight = weight.view(ss[0], 128, ss[1], 128)
                scale_inv = scale_inv.view(ss[0], 1, ss[1], 1)
            else:
                # A dimension that is not a multiple of 128 ends in a partial block (Ling's MTP
                # head gate: 64 rows); expand the grid over whole blocks and crop
                scale_inv = scale_inv.repeat_interleave(128, 0)[:ws[0]].repeat_interleave(128, 1)[:, :ws[1]]
        weight = weight.float() * scale_inv
        return weight.view(ws).half()


    # DeepSeek-V4-style quantized checkpoint tensors, dequantized to FP16 at load (they get
    # requantized to EXL3 anyway; no runtime paths for exotic dtypes). One generic routine:
    # the block shape is derived from the scale grid, covering FP8 E4M3 with a [128, 128]
    # grid and packed FP4 (two e2m1 nibbles per int8 byte, low nibble first) with a [1, 32]
    # grid. E8M0 scales arrive as uint8 exponent bytes, value 2^(byte - 127). Mirrors the
    # transformers finegrained_fp8 reference dequant.
    def dequant_e8m0_blocks_(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # MiMo-V2 stores the same packed-FP4 expert weights as U8 rather than I8; the bytes are
        # identical (SGLang's loader only bitcasts to match its int8 parameter, then masks as
        # unsigned again), so both dtypes take the unpack path
        if weight.dtype in (torch.int8, torch.uint8):
            u8 = weight.contiguous().view(torch.uint8)
            lut = torch.tensor(_MXFP4_LUT, dtype = torch.float, device = weight.device)
            w = torch.stack((lut[(u8 & 0x0F).long()], lut[(u8 >> 4).long()]), dim = -1)
            w = w.view(weight.shape[0], -1)
        else:
            w = weight.float()
        rows, cols = w.shape
        sr, sc = scale.shape
        assert rows % sr == 0 and cols % sc == 0, \
            f"{self.key}: weight {rows}x{cols} not divisible by scale grid {sr}x{sc}"
        s = (scale.float() - 127.0).exp2() if scale.dtype == torch.uint8 else scale.float()
        w = (w.view(sr, rows // sr, sc, cols // sc) * s.view(sr, 1, sc, 1)).view(rows, cols)
        return w.half()


    def load_fp16(self, key: str | list[str]) -> bool:

        if self.config.stc.has_tensor_group(key, ["weight"]):

            self.used_alt_key = key == self.alt_key
            if self.used_alt_key and isinstance(key, list):
                dev = self.device
                if self._packed_source() and self.transposed_load:
                    segs, col0, ok = [], 0, True
                    for k in key:
                        w = self.config.stc.get_tensor(k + ".weight", dev, no_defer = True, arena = False)
                        fmt = packed_fmt(w)
                        if fmt is None or fmt == 1:
                            ok = False
                            break
                        si = self.config.stc.get_tensor(k + ".weight_scale_inv", dev, optional = True, no_defer = True)
                        block = (128, 128) if si is not None and si.dim() == 2 else None
                        segs.append(PackedSegment(w, si, fmt, col0 = col0, block = block))
                        col0 += w.shape[0]
                    if ok:
                        bias = [self.config.stc.get_tensor(k + ".bias", dev, float2half = True, optional = True, no_defer = True) for k in key]
                        bias = torch.cat(bias, dim = -1) if bias[0] is not None else None
                        return self._make_packed(segs, self.pad_out(bias))
                weight = []
                for k in key:
                    # fp8 parts (Ling 3.x q/k/v) carry their own block scales: dequantize each before the concat
                    w = self.config.stc.get_tensor(k + ".weight", dev, float2half = True, transpose = self.transposed_load, no_defer = True)
                    scale_inv = self.config.stc.get_tensor(k + ".weight_scale_inv", dev, transpose = self.transposed_load, optional = True, no_defer = True)
                    weight.append(self.apply_fp8_scales_inv_(w, scale_inv) if scale_inv is not None else w)
                bias = [self.config.stc.get_tensor(k + ".bias", dev, float2half = True, optional = True, no_defer = True) for k in key]
                weight = torch.cat(weight, dim = -1)
                bias = torch.cat(bias, dim = -1) if bias[0] is not None else None
            else:
                dev = "cpu" if self.is_sliced else self.device
                pad1 = (self.out_features,) if not self.is_sliced else None
                pad2 = (self.in_features, self.out_features) if not self.is_sliced else None
                scale_q = self.config.stc.get_tensor(key + ".scale", dev, optional = True, no_defer = True)
                if scale_q is None and self.config.stc.has_tensor(key + ".weight_scale"):
                    # MiMo-V2 routed experts: same E8M0 block grid, named "weight_scale" and
                    # stored as U8 exponent bytes. Only 2D uint8 scales route here; float
                    # "weight_scale" tensors keep the per-row/per-tensor fp8 path below. Fetched
                    # untransposed because the dequant runs in checkpoint (out, in) orientation
                    ws_q = self.config.stc.get_tensor(key + ".weight_scale", dev, optional = True, no_defer = True)
                    if ws_q is not None and ws_q.dim() == 2 and ws_q.dtype == torch.uint8:
                        scale_q = ws_q
                if scale_q is not None and scale_q.dim() == 2:
                    # DeepSeek-V4 style: fp8/fp4 blocks + E8M0 scale grid, checkpoint
                    # orientation (out, in); dequant first, then orient/pad like a plain load
                    w_raw = self.config.stc.get_tensor(key + ".weight", dev, no_defer = True, arena = False)
                    if self._packed_source() and self.transposed_load and scale_q.dtype == torch.uint8 \
                            and packed_fmt(w_raw) is not None:
                        # Conversion: keep the bytes packed, dequantize per use (LinearFP16Packed)
                        return self._make_packed([PackedSegment(w_raw, scale_q, packed_fmt(w_raw))], None)
                    weight = self.dequant_e8m0_blocks_(w_raw, scale_q)
                    if self.transposed_load:
                        weight = weight.T
                    weight = weight.contiguous() if self.is_sliced else self.pad_out(weight.contiguous())
                    bias = None
                    scale = scale_inv = None
                else:
                    if self._packed_source() and self.transposed_load:
                        # fp8 with a per-tensor / per-row weight_scale or a block weight_scale_inv (Transformers
                        # finegrained fp8, DeepSeek-V3 style): the fp8 bytes stay packed
                        w_raw = self.config.stc.get_tensor(key + ".weight", dev, no_defer = True, arena = False)
                        fmt = packed_fmt(w_raw)
                        if fmt is not None and fmt != 1:
                            sc_ = self.config.stc.get_tensor(key + ".weight_scale", dev, optional = True, no_defer = True)
                            si_ = self.config.stc.get_tensor(key + ".weight_scale_inv", dev, optional = True, no_defer = True)
                            assert sc_ is None or si_ is None
                            block = (128, 128) if si_ is not None and si_.dim() == 2 else None
                            sc_ = sc_ if sc_ is not None else si_
                            if sc_ is None or sc_.dim() <= 2:
                                bias = self.config.stc.get_tensor(key + ".bias", dev, float2half = True, optional = True, pad_to = pad1, no_defer = True)
                                return self._make_packed([PackedSegment(w_raw, sc_, fmt, block = block)], bias)
                        del w_raw
                    scale = self.config.stc.get_tensor(key + ".weight_scale", dev, transpose = self.transposed_load, optional = True, no_defer = True)
                    scale_inv = self.config.stc.get_tensor(key + ".weight_scale_inv", dev, transpose = self.transposed_load, optional = True, no_defer = True)
                    assert scale is None or scale_inv is None
                    # Scaled loads are rewritten into a new tensor below: keep the raw load out of
                    # the loader's slab blocks, where the dead copy would otherwise stay resident
                    rescaled = scale is not None or scale_inv is not None or self.weight_scale != 1.0
                    no_defer = rescaled
                    weight = self.config.stc.get_tensor(key + ".weight", dev, float2half = True, transpose = self.transposed_load, pad_to = pad2, no_defer = no_defer, arena = not rescaled)
                    bias = self.config.stc.get_tensor(key + ".bias", dev, float2half = True, optional = True, pad_to = pad1, no_defer = no_defer, arena = not rescaled)
                if scale is not None:
                    weight = self.apply_fp8_scales_(weight, scale)
                elif scale_inv is not None:
                    weight = self.apply_fp8_scales_inv_(weight, scale_inv)
                if self.weight_scale != 1.0:
                    weight = (weight.float() * self.weight_scale).to(weight.dtype)
                    if bias is not None:
                        bias = (bias.float() * self.weight_scale).to(bias.dtype)

            self.inner = LinearFP16(
                self.in_features,
                self.out_features,
                weight,
                bias,
                self.full_in_features,
                self.full_out_features,
                self.first_in_feature,
                self.first_out_feature,
                self.out_dtype,
                key = self.key
            )
            if self.is_sliced:
                self.inner.swap_device = self.device
                self.inner.unswap_cpu()
            self.quant_type = "fp16"
            return True

        # Special dumb loading mode for Qwen3VLMoE
        elif self.fkey and self.config.stc.has_tensor(self.fkey) and self.fidx is not None:

            # Load and split fused weight along first dim
            weight = self.config.stc.get_tensor(
                self.fkey,
                self.device,
                transpose = self.transpose_fused_weights,
                no_defer = True,
                fidx = self.fidx
            )
            if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # fp8 batch tensor (Mistral-Small-4): per-expert scalar scale stored as
                # {fkey}_scale_inv with the expert on the first dim
                scale_inv = self.config.stc.get_tensor(
                    self.fkey + "_scale_inv", self.device, optional = True, no_defer = True, fidx = self.fidx
                )
                weight = weight.float()
                if scale_inv is not None:
                    weight = weight * scale_inv.float().reshape(-1)[0]
                weight = weight.half()
            if self.frange is not None:
                if self.frange_dim == 0:
                    weight = weight[self.frange[0] : self.frange[1]].contiguous()
                elif self.frange_dim == 1:
                    weight = weight[:, self.frange[0] : self.frange[1]].contiguous()
                else:
                    raise ValueError(f"Unsupported frange_dim={self.frange_dim} for {self.key}")
            if self.ftranspose_after_load:
                weight = weight.T.contiguous()
            # pad_out expects (in_features, out_features) orientation
            weight = self.pad_out(weight)
            if self.weight_scale != 1.0:
                weight = (weight.float() * self.weight_scale).to(weight.dtype)
            self.inner = LinearFP16(
                self.in_features,
                self.out_features,
                weight,
                None,
                self.full_in_features,
                self.full_out_features,
                self.first_in_feature,
                self.first_out_feature,
                out_dtype = self.out_dtype,
                key = self.key
            )
            self.quant_type = "fp16"
            return True

        # MXFP4 batch tensor (gpt-oss): experts packed as {fkey}_blocks/_scales/_bias, one 3D/4D
        # tensor per projection with the expert on the first dim. Dequantized to fp16 per expert
        # slice here; the interleaved flag handles fused tensors that alternate gate/up rows
        elif self.fkey and self.fidx is not None and self.config.stc.has_tensor(self.fkey + "_blocks"):

            assert self.ftranspose_after_load, \
                f"MXFP4 batch tensor for {self.key} requires ftranspose_after_load"
            blocks = self.config.stc.get_tensor(
                self.fkey + "_blocks", self.device, no_defer = True, fidx = self.fidx
            )
            scales = self.config.stc.get_tensor(
                self.fkey + "_scales", self.device, no_defer = True, fidx = self.fidx
            )
            bias = self.config.stc.get_tensor(
                self.fkey + "_bias", self.device, optional = True, no_defer = True, fidx = self.fidx
            )
            # Slice out rows (output features) before dequantizing
            if self.frange is not None:
                n = self.frange[1] - self.frange[0]
                if self.finterleaved:
                    step = blocks.shape[0] // n
                    off = self.frange[0] // n
                    sl = slice(off, None, step)
                else:
                    sl = slice(self.frange[0], self.frange[1])
                blocks = blocks[sl].contiguous()
                scales = scales[sl].contiguous()
                if bias is not None:
                    bias = bias[sl].contiguous()
            if self._packed_source():
                # (out, G, 16) bytes are (out, in / 2) nibble pairs in row order, scales (out, G) E8M0
                seg = PackedSegment(blocks.reshape(blocks.shape[0], -1), scales, 1, block = (1, 32))
                return self._make_packed([seg], self.pad_out(bias.half() if bias is not None else None))
            weight = _mxfp4_dequant(blocks, scales)
            weight = weight.T.contiguous()
            weight = self.pad_out(weight)
            bias = self.pad_out(bias)
            if self.weight_scale != 1.0:
                weight = (weight.float() * self.weight_scale).to(weight.dtype)
                if bias is not None:
                    bias = (bias.float() * self.weight_scale).to(bias.dtype)
            self.inner = LinearFP16(
                self.in_features,
                self.out_features,
                weight,
                bias,
                self.full_in_features,
                self.full_out_features,
                self.first_in_feature,
                self.first_out_feature,
                out_dtype = self.out_dtype,
                key = self.key
            )
            self.quant_type = "fp16"
            return True

        elif self.fkey and self.config.stc.has_tensor_group(self.fkey, ["weight"]) and self.frange is not None:

            if self.fdequant is not None:
                # Architecture-supplied reader for a fused tensor whose checkpoint layout the
                # generic paths cannot describe (MiMo-V2: TP-shard-interleaved fused QKV with a
                # per-shard, ceiling-padded FP8 scale grid). Returns the canonical, contiguous
                # (out_features, in_features) fp16 matrix that frange indexes into
                weight = self.fdequant(self.config.stc, self.fkey, self.device)
                bias = None
            else:
                weight = self.config.stc.get_tensor(
                    self.fkey + ".weight",
                    self.device,
                    no_defer = True
                )
                # Fused checkpoint tensor in FP8/FP4-block form (DeepSeek-V4 wo_a): dequantize the
                # whole tensor before slicing out this group's rows
                scale_q = self.config.stc.get_tensor(self.fkey + ".scale", self.device, optional = True, no_defer = True)
                if scale_q is not None and scale_q.dim() == 2 and self._packed_source() and self.ftranspose_after_load \
                        and packed_fmt(weight) is not None and scale_q.dtype == torch.uint8:
                    r0, r1 = self.frange
                    br = weight.shape[0] // scale_q.shape[0]
                    if r0 % br == 0 and r1 % br == 0:
                        bias = self.config.stc.get_tensor(self.fkey + ".bias", self.device, optional = True, no_defer = True)
                        if bias is not None:
                            bias = self.pad_out(bias[r0:r1].contiguous())
                        seg = PackedSegment(weight[r0:r1].contiguous(), scale_q[r0 // br:r1 // br].contiguous(), packed_fmt(weight))
                        return self._make_packed([seg], bias)
                if scale_q is not None and scale_q.dim() == 2:
                    weight = self.dequant_e8m0_blocks_(weight, scale_q)
            weight = weight[self.frange[0] : self.frange[1]].contiguous()
            weight = self.pad_out(weight)
            if self.fdequant is None:
                bias = self.config.stc.get_tensor(
                    self.fkey + ".bias",
                    self.device,
                    optional = True,
                    no_defer = True
                )
            if bias is not None:
                bias = bias[self.frange[0] : self.frange[1]].contiguous()
                bias = self.pad_out(bias)
            if self.ftranspose_after_load:
                weight = weight.T.contiguous()
            self.inner = LinearFP16(
                self.in_features,
                self.out_features,
                weight,
                bias,
                self.full_in_features,
                self.full_out_features,
                self.first_in_feature,
                self.first_out_feature,
                out_dtype = self.out_dtype,
                key = self.key
            )
            self.quant_type = "fp16"
            return True

        return False


    def is_exl3_storage(self, key: str):
        return self.config.stc.has_tensor_group(
            key,
            [["sv", "svh"], ["su", "suh"], "trellis"]
        )

    def load_exl3(self, key: str) -> bool:
        if not self.is_exl3_storage(key):
            return False
        self.used_alt_key = key == self.alt_key
        # Probe the header before fetching: most formats ship only a subset of these tensors,
        # and the misses otherwise add up to a significant share of bulk-load Python overhead
        stc = self.config.stc
        def opt(subkey, device, **kwargs):
            k = key + subkey
            return stc.get_tensor(k, device, optional = True, **kwargs) if stc.has_tensor(k) else None
        scale = None  # opt(".scale", self.device)
        su = opt(".su", self.device, no_defer = True)
        suh = opt(".suh", self.device)
        sv = opt(".sv", self.device, no_defer = True)
        svh = opt(".svh", self.device)
        trellis = stc.get_tensor(key + ".trellis", self.device)
        mcg = stc.get_codebook_marker(key + ".mcg", codebook_mcg_mult)
        mul1 = stc.get_codebook_marker(key + ".mul1", codebook_mul1_mult)
        bias = opt(".bias", self.device)
        self.inner = LinearEXL3(
            self.config,
            self.in_features,
            self.out_features,
            scale,
            su,
            sv,
            suh,
            svh,
            trellis,
            mcg,
            mul1,
            bias,
            self.out_dtype,
            key = self.key
        )
        self.quant_type = "exl3"
        return True


    @override
    def can_defer_load(self):
        if self.is_sliced: return False
        return super().can_defer_load()


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        keys = [self.key]
        if self.alt_key:
            keys += [self.alt_key]
        if any(self.load_exl3(k) for k in keys): return
        if any(self.load_fp16(k) for k in keys): return
        raise ValueError(f"No tensors found for {self.key} matching supported quantization format.")


    @override
    def pin_linears(self):
        """Move this layer's bulk weight storage (fp16 weight or EXL3 trellis) to pinned host
        memory and swap in a zero-copy CUDA alias, freeing the VRAM. Compute paths read the
        weights over PCIe: fine for occasional, compute-bound vision-tower batches. Small
        tensors (bias, suh/svh) stay in VRAM. Runs post-load, so it covers every load flavor
        (fused/sliced/fp8-dequant) and deferred fills have already landed."""
        inner = self.inner
        if inner is None or self.device is None:
            return
        device = torch.device(self.device)
        if device.type != "cuda":
            return

        def pin_alias(t):
            # An arena of the tensor's own size rather than a tensor from torch's pinned allocator, which
            # rounds each request up to a power of two and keeps the block after the tower is unloaded
            if not t.is_contiguous():
                t = t.contiguous()
            nbytes = t.numel() * t.element_size()
            # Registered memory belongs to the device current at registration, and the alias must be
            # on that device. The current device is otherwise whatever the loading thread last used
            with torch.cuda.device(device):
                inner._pinned_arena = PinnedArena(nbytes, f"Pinned weights of {self.key}", mapped = True, required = True)
            p = inner._pinned_arena.tensor[:nbytes].view(t.dtype).view(t.shape)
            p.copy_(t)
            return p, ext.pinned_cuda_view(p, device.index if device.index is not None else 0)

        if isinstance(inner, LinearFP16) and inner.swap_device is None and not isinstance(inner, LinearFP16Packed):
            inner._pinned_store, inner.weight = pin_alias(inner.weight)
            inner.bc = ext.BC_LinearFP16(inner.weight, inner.bias)
        elif isinstance(inner, LinearEXL3):
            from ..util.tensor import g_tensor_cache
            inner._pinned_store, inner.trellis = pin_alias(inner.trellis)
            inner.bc = ext.BC_LinearEXL3(
                inner.trellis, inner.suh, inner.svh, inner.K, inner.bias,
                inner.mcg, inner.mul1, g_tensor_cache.get(*inner.bsz1_xh_args))

    @override
    def unload(self):
        if self.inner is not None:
            self.inner.unload()
        self.device = None
        self.inner = None
        self.sparse_head = None
        self.lora_a_tensors.clear()
        self.lora_b_tensors.clear()


    @override
    def make_sparse_head(self, capacity: int, vocab_size: int | None = None, **kwargs):
        """Attach a SparseHead mirroring this layer's columns (requires a loaded inner with column extraction)"""
        from .sparse_head import SparseHead
        assert self.inner is not None and hasattr(self.inner, "reconstruct_columns"), \
            f"{self.key}: sparse head needs a loaded EXL3 or fp16 layer"
        self.sparse_head = SparseHead(self, capacity, vocab_size or self.out_features_unpadded, **kwargs)
        return self.sparse_head


    def get_tensors(self):
        if self.device:
            return self.inner.get_tensors(self.key)
        else:
            return {}


    def convert_exl3(
        self,
        H_data: dict,
        quant_args: dict,
        progress_str: str | None = None,
        return_weight_q: bool = False,
        verbose: bool = False,
        save_reg: str = None,
        override_swap_device: torch.device | None = None
    ):
        assert isinstance(self.inner, LinearFP16), \
            "Inner layer is already quant type"

        # Destroy original layer here to save VRAM, we only need weights
        swap_to_device = self.inner.swap_device  # in case weights are swapped to CPU
        if swap_to_device is not None and override_swap_device is not None:
            swap_to_device = override_swap_device

        if isinstance(self.inner, LinearFP16Packed):
            # Dequantize straight onto the quantizing device (never the load device of a swapped layer)
            dev = torch.device(swap_to_device) if swap_to_device is not None else self.inner.device
            orig_weight = self.inner.dequant_weight(dev).float()
        else:
            orig_weight = self.inner.get_weight_tensor().float()
        orig_bias = self.inner.get_bias_tensor()
        self.inner = None

        weight_q, proxy_err, out_tensors = quantize_exl3(
            orig_weight,
            H_data,
            quant_args,
            return_weight_q,
            progress_str,
            verbose,
            swap_to_device,
            save_reg = save_reg
        )

        self.inner = LinearEXL3(
            self.config,
            self.in_features,
            self.out_features,
            out_tensors.get("scale"),
            out_tensors.get("su"),
            out_tensors.get("sv"),
            out_tensors.get("suh"),
            out_tensors.get("svh"),
            out_tensors.get("trellis"),
            out_tensors.get("mcg"),
            out_tensors.get("mul1"),
            orig_bias,
            self.out_dtype,
            key = self.key
        )

        if return_weight_q:
            return proxy_err, weight_q
        else:
            return proxy_err


    def init_H_data(self, init_tensor: bool):
        # The capture accumulates into the block-packed upper triangle (hessian_packed); the full matrix is
        # reconstructed when the tensor is finalized. The uncalibrated placeholder stays a meta (in, in) matrix
        return {
            "H": torch.zeros(packed_numel(self.in_features), dtype = torch.float32, device = self.device) if init_tensor else
                 torch.zeros(self.in_features, self.in_features, dtype = torch.float32, device = "meta"),
            "packed": init_tensor,
            "n": self.in_features,
            "first_key": self.key,
            "count": 0,
            "finalized": False,
            "num_total": 0,
            "inf_nan": torch.zeros(2, dtype = torch.long, device = self.device),
            "device": self.device,
        }


    def capture_H(self, x: torch.Tensor, params: dict):
        if self.qmap not in params["capture"]:
            params["capture"][self.qmap] = self.init_H_data(True)

        params["capture"][self.qmap]["num_total"] += x.numel()
        ext.count_inf_nan(x, params["capture"][self.qmap]["inf_nan"])

        if params["capture"][self.qmap]["first_key"] == self.key:
            rows = np.prod(x.shape[:-1])
            dim = x.shape[-1]
            x = x.reshape((rows, dim))

            # fp16 activation overflow (+-inf) would poison entire rows/columns of H through the
            # accumulation, and no diagonal damping can repair non-finite entries afterwards.
            # Drop the affected rows; the inf_nan counter above still reports them
            finite = torch.isfinite(x).all(dim = 1)
            row_w = params.get("capture_row_w")
            if not finite.all():
                x = x[finite]
                rows = x.shape[0]
                if row_w is not None:
                    row_w = row_w[finite]

            accumulate_(params["capture"][self.qmap]["H"], x)
            params["capture"][self.qmap]["count"] += rows

            # Row-weighted Hessian alongside the plain one (affinity-weighted expert Hessians, see
            # conversion/moe_affinity.py), combined with it before quantization. The weights are an
            # expert's routing weights, so most rows are zero and only the routed ones accumulate
            if row_w is not None:
                hd = params["capture"][self.qmap]
                if "H_aff" not in hd:
                    hd["H_aff"] = torch.zeros_like(hd["H"])
                    hd["aff_sum"] = torch.zeros((), dtype = torch.float, device = hd["H"].device)
                row_w = row_w.to(torch.float).clamp_min(0)
                routed = row_w.nonzero().flatten()
                if routed.numel():
                    xw = x[routed] * row_w[routed].sqrt().unsqueeze(1).to(x.dtype)
                    accumulate_(hd["H_aff"], xw)
                    hd["aff_sum"] += row_w.sum()


    @override
    def weights_numel(self):
        return self.in_features * self.out_features


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        if self.out_features == 0:
            dtype = out_dtype or self.out_dtype or torch.half
            return x.new_empty((*x.shape[:-1], 0), dtype = dtype)

        # When in_features is padded past the incoming activation width (dims not a multiple of
        # pad_to, e.g. gpt-oss hidden_size 2880), zero-extend the input. The padded weight rows
        # are zeros, so this is exact; H capture below sees the padded width, keeping the
        # quantizer's Hadamard blocks aligned
        if x.shape[-1] < self.in_features:
            x = torch.nn.functional.pad(x, (0, self.in_features - x.shape[-1]))

        if self.qmap and "capture" in params:
            # A capture split into passes (convert_model.capture_passes) names the qmaps of the
            # current pass; everything else runs without accumulating
            pass_qmaps = params.get("capture_qmaps")
            if pass_qmaps is None or self.qmap in pass_qmaps:
                self.capture_H(x, params)

        if self.lora_a_tensors:
            lora_input = x
        else:
            lora_input = None

        x = self.inner.forward(x, params, out_dtype)

        # Padded output columns carry only zeros (fp16) or quantization noise (EXL3); trim when
        # the consumer expects the exact width (e.g. projections back into the residual stream).
        # Contiguous copy: downstream kernels index flat memory
        if self.trim_padded_out and self.out_features != self.out_features_unpadded:
            x = x[..., :self.out_features_unpadded].contiguous()

        if lora_input is not None:
            self.apply_lora(lora_input, x, params)

        if self.pre_scale != 1.0:
            x *= self.pre_scale
        if self.softcap != 0.0:
            ext.softcap(x, x, self.softcap)
        if self.post_scale != 1.0:
            x *= self.post_scale
        return x


    def apply_lora(self, lora_input: torch.Tensor, x: torch.Tensor, params: dict | None = None):
        """
        x += lora_input @ A @ B for the loaded adapters: all of them on every row, or each batch row's own when
        params["lora_rows"] names them (see lora_rows_of). Rows are batch-major: the input's token rows divide
        evenly among the batch rows, in order.
        """
        orig_shape = lora_input.shape
        flat = lora_input.view(-1, orig_shape[-1])
        lora_rows = lora_rows_of(params, flat.shape[0])
        for lora, a in self.lora_a_tensors.items():
            b = self.lora_b_tensors.get(lora)
            if b is None:
                continue
            lora_in = flat if flat.dtype == a.dtype else flat.to(a.dtype)
            if lora_rows is None:
                delta = lora_in @ a @ b
                x += delta.view(*orig_shape[:-1], -1).to(x.dtype)
                continue
            picked = [i for i, r in enumerate(lora_rows) if is_lora_row(r, lora)]
            if not picked:
                continue
            if len(picked) == len(lora_rows):
                delta = lora_in @ a @ b
                x += delta.view(*orig_shape[:-1], -1).to(x.dtype)
                continue
            bsz = len(lora_rows)
            index = torch.tensor(picked, dtype = torch.long, device = x.device)
            delta = lora_in.view(bsz, -1, orig_shape[-1])[index] @ a @ b
            x.view(bsz, -1, x.shape[-1]).index_add_(0, index, delta.to(x.dtype))


    def quant_format_id(self):
        # alt_key is only used when loading unquantized model
        if self.is_exl3_storage(self.key):
            return "exl3"
        else:
            return None


    @cached_property
    def _storage_size(self):
        # alt_key is only used when loading unquantized model
        if self.is_exl3_storage(self.key):
            return sum(self.config.stc.get_tensor_sizes(prefix = self.key))
        else:
            return 2 * self.in_features * self.out_features
    def storage_size(self):
        # A packed conversion-time inner holds the checkpoint bytes, not the fp16 form the size was planned for
        if self.inner is not None and hasattr(self.inner, "storage_bytes"):
            return self.inner.storage_bytes()
        return self._storage_size


    def recons_size(self):
        return 2 * self.in_features * self.out_features


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.storage_size()
        overhead_d = self.out_features * (self.out_dtype or torch.half).itemsize
        overhead_s = self.out_features * (self.out_dtype or torch.half).itemsize
        recons = self.recons_size()
        tpa = TPAllocation(
            key = self.key,
            channel_width = 128,
            channel_unit = "channels",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.out_features // 128,
            limit_key = "linear"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None and self.inner is not None, "Cannot export module for TP before loading."
        return {
            "cls": Linear,
            "kwargs": {
                "key": self.key,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "out_dtype": self.out_dtype,
                "alt_key": self.alt_key,
                "fkey": self.fkey,
                "frange": self.frange,
                "frange_dim": self.frange_dim,
                "fidx": self.fidx,
                "finterleaved": self.finterleaved,
                "trim_padded_out": self.trim_padded_out,
                "ftranspose_after_load": self.ftranspose_after_load,
                "caps": self.caps,
                "softcap": self.softcap,
                "full_in_features": self.full_in_features,
                "full_out_features": self.full_out_features,
                "first_in_feature": self.first_in_feature,
                "first_out_feature": self.first_out_feature,
                "pre_scale": self.pre_scale,
                "post_scale": self.post_scale,
            },
            # Not constructor args: restored post-construction by _adopt_inner_dims for dims the
            # split leaves whole (padded models pad the hidden dims, which are never split)
            "unpadded": (self.in_features_unpadded, self.out_features_unpadded),
            "inner": self.inner.tp_export(plan, producer),
            "device": self.device
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        device = local_context["device"]
        module = Linear(
            config = None,
            **exported["kwargs"],
        )
        module.device = device
        module.inner = exported["inner"]["cls"].tp_import_split(local_context, exported["inner"], plan, split)
        module.quant_type = module.inner.quant_type
        if split is not None:
            split_out, first, _ = split
            module._adopt_inner_dims(exported, out_first = first if split_out else None,
                                     in_first = first if not split_out else None)
        else:
            module._adopt_inner_dims(exported)
        return module

    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2):
        device = local_context["device"]
        module = Linear(
            config = None,
            **exported["kwargs"],
        )
        module.device = device
        module.inner = exported["inner"]["cls"].tp_import_split_3(local_context, exported["inner"], plan, split_0, split_1, split_2)
        module.quant_type = module.inner.quant_type
        module._adopt_inner_dims(exported)
        return module

    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits):
        # N-range output split (Mamba2 in_proj: [z | x | B | C | dt] sections split by state
        # group, with the dt section replicated). The ranges select only real (unpadded) columns,
        # so the local slice is exact by construction: no output trim, no padded-out bookkeeping
        device = local_context["device"]
        module = Linear(
            config = None,
            **exported["kwargs"],
        )
        module.device = device
        module.inner = exported["inner"]["cls"].tp_import_split_n(local_context, exported["inner"], plan, splits)
        module.quant_type = module.inner.quant_type
        module.in_features = module.inner.in_features
        module.out_features = module.inner.out_features
        exp_in = exported["kwargs"]["in_features"]
        unp_in, _ = exported.get("unpadded", (exp_in, None))
        module.in_features_unpadded = unp_in if module.in_features == exp_in else module.in_features
        module.out_features_unpadded = module.out_features
        module.trim_padded_out = False
        module.is_sliced = True
        return module

    def _adopt_inner_dims(self, exported: dict, in_first: int | None = None, out_first: int | None = None):
        # The exported kwargs carry the full unsplit (padded) dims; forward() sizes its input padding and
        # output trim against the wrapper dims, so they must track the local slice. A dim the split left
        # whole keeps its true unpadded value (padded models pad the hidden dims, which are never split).
        # Splitting a padded dim is fine when the slice offset is known: slices below the pad boundary
        # are exact, and the slice containing it has (unpadded - first) real columns (e.g. an lm_head
        # with a padded vocab split across ranks -- the pad columns ride in the tail slice like they do
        # on a single device)
        exp_in, exp_out = exported["kwargs"]["in_features"], exported["kwargs"]["out_features"]
        unp_in, unp_out = exported.get("unpadded", (exp_in, exp_out))
        self.in_features = self.inner.in_features
        self.out_features = self.inner.out_features
        if self.in_features == exp_in:
            self.in_features_unpadded = unp_in
        elif exp_in == unp_in:
            self.in_features_unpadded = self.in_features
        else:
            assert in_first is not None, f"{self.key}: cannot split the padded input dim without the slice offset"
            self.in_features_unpadded = max(0, min(unp_in - in_first, self.in_features))
        if self.out_features == exp_out:
            self.out_features_unpadded = unp_out
        elif exp_out == unp_out:
            self.out_features_unpadded = self.out_features
        else:
            assert out_first is not None, f"{self.key}: cannot split the padded output dim without the slice offset"
            self.out_features_unpadded = max(0, min(unp_out - out_first, self.out_features))
            # A load-bearing output trim (projections back into the residual stream) has no
            # defined per-slice gather semantics; no current model splits such a dim
            assert not self.trim_padded_out or self.out_features_unpadded == self.out_features, \
                f"{self.key}: cannot split a padded output dim that requires trimming"
        self.is_sliced = self.in_features < self.full_in_features or self.out_features < self.full_out_features


    @staticmethod
    def tp_import(local_context, exported, plan):
        key = exported["kwargs"]["key"]
        first, last, unit = plan[key] if key in plan else (None, None, "channels")
        assert unit == "channels"
        split = (True, first, last) if first is not None else None
        return Linear.tp_import_split(local_context, exported, plan, split)


def convert_exl3_group(
    linears: list,
    H_datas: list[dict],
    quant_args_list: list[dict],
    progress_str: str | None = None,
):
    """
    Convert a group of Linear modules to EXL3 in one batched quantization pass (see quantize_exl3_batch for
    the grouping requirements: shared Hessian with equal in_features, or same-shape tensors with individual
    Hessians, all at the same bitrate).

    :return:
        list of proxy errors, one per module; each module's quant_args dict in quant_args_list is updated
        the same way Linear.convert_exl3 updates its quant_args
    """
    weights = []
    biases = []
    for linear in linears:
        assert isinstance(linear.inner, LinearFP16), \
            "Inner layer is already quant type"
        # Checkpoint precision, usually still swapped to CPU: quantize_exl3_batch stages the
        # upload through pinned memory and casts to fp32 on the device. Packed inners are handed
        # over as they are and dequantized by the stager one at a time on the quantizing device
        weights.append(linear.inner if isinstance(linear.inner, LinearFP16Packed) else linear.inner.get_weight_tensor())
        biases.append(linear.inner.get_bias_tensor())
        linear.inner = None

    results = quantize_exl3_batch(weights, H_datas, quant_args_list, progress_str)

    proxy_errs = []
    for linear, bias, (proxy_err, out_tensors) in zip(linears, biases, results):
        linear.inner = LinearEXL3(
            linear.config,
            linear.in_features,
            linear.out_features,
            out_tensors.get("scale"),
            out_tensors.get("su"),
            out_tensors.get("sv"),
            out_tensors.get("suh"),
            out_tensors.get("svh"),
            out_tensors.get("trellis"),
            out_tensors.get("mcg"),
            out_tensors.get("mul1"),
            bias,
            linear.out_dtype,
            key = linear.key
        )
        proxy_errs.append(proxy_err)
    return proxy_errs
