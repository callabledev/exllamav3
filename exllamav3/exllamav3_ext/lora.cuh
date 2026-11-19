#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

// LoRA with a different adapter per row, for the decode graphs' LoRA stage: o[r] += x[r] @ A[s] @ B[s], s being
// row r's adapter. The stage's statics hold several adapters side by side, each in as many columns as its rank, and
// `slots` names each row's: (first column, rank) per row, rank 0 for no adapter (the row is left as it is). A row
// costs what its own adapter's rank does, whatever else is loaded.
//
// lora_down: t[r, j] = sum_k x[r, k] * a_t[col_r + j, k] for j < rank_r; t beyond a row's rank is not written
//   x: (rows, K) fp16, a_t: (width, K) fp16 (A transposed, so an adapter's columns are contiguous rows),
//   slots: int32 (at least rows, 2), t: (rows, highest rank) fp16, contiguous
//
// lora_up: o[r, n] += sum_{j < rank_r} t[r, j] * b[col_r + j, n]
//   t: (rows, highest rank) fp16, b: (width, N) fp16, o: rows * N elements, fp16 or fp32, contiguous. With patch_o
//   its address is a patchable site, GP_lora_up_o
//
// Sums are accumulated in fp32. A rank above t's width is cut to it.

void lora_down_gr
(
    const at::Tensor& x,
    const at::Tensor& a_t,
    const at::Tensor& slots,
    at::Tensor& t,
    Graph* graph
);

void lora_up_gr
(
    const at::Tensor& t,
    const at::Tensor& b,
    const at::Tensor& slots,
    at::Tensor& o,
    bool patch_o,
    Graph* graph
);

void lora_down
(
    const at::Tensor& x,
    const at::Tensor& a_t,
    const at::Tensor& slots,
    at::Tensor& t
);

void lora_up
(
    const at::Tensor& t,
    const at::Tensor& b,
    const at::Tensor& slots,
    at::Tensor& o
);
