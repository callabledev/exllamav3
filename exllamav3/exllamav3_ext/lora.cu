#include <cuda_fp16.h>
#include "lora.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

#define LORA_THREADS 256
#define LORA_WARPS (LORA_THREADS / 32)

// One block per (row, rank column): the dot product of the row with one row of A transposed. Blocks beyond the row's
// own rank leave at once

__launch_bounds__(LORA_THREADS)
__global__ void lora_down_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ a_t,
    const int* __restrict__ slots,
    half* __restrict__ t,
    const int x_stride,
    const int size_k,
    const int rank
)
{
    const int j = blockIdx.x;
    const int row = blockIdx.y;
    if (j >= slots[2 * row + 1]) return;

    const half* xr = x + (int64_t) row * x_stride;
    const half* ar = a_t + (int64_t) (slots[2 * row] + j) * size_k;

    float acc = 0.0f;
    for (int k = threadIdx.x; k < size_k; k += LORA_THREADS)
        acc += __half2float(xr[k]) * __half2float(ar[k]);

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ float partial[LORA_WARPS];
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = acc;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float sum = 0.0f;
        for (int w = 0; w < LORA_WARPS; ++w) sum += partial[w];
        t[(int64_t) row * rank + j] = __float2half_rn(sum);
    }
}

// One thread per output element

__device__ inline void lora_accumulate(half* o, float delta) { *o = __float2half_rn(__half2float(*o) + delta); }
__device__ inline void lora_accumulate(float* o, float delta) { *o += delta; }

template <typename T>
__launch_bounds__(LORA_THREADS)
__global__ void lora_up_kernel
(
    const half* __restrict__ t,
    const half* __restrict__ b,
    const int* __restrict__ slots,
    T* __restrict__ o,
    const int size_n,
    const int rank
)
{
    const int n = blockIdx.x * LORA_THREADS + threadIdx.x;
    const int row = blockIdx.y;
    const int row_rank = min(slots[2 * row + 1], rank);
    if (n >= size_n || row_rank <= 0) return;

    const half* tr = t + (int64_t) row * rank;
    const half* bn = b + (int64_t) slots[2 * row] * size_n + n;

    float acc = 0.0f;
    for (int j = 0; j < row_rank; ++j)
        acc += __half2float(tr[j]) * __half2float(bn[(int64_t) j * size_n]);

    lora_accumulate(o + (int64_t) row * size_n + n, acc);
}

static int lora_check_slots(const at::Tensor& slots, const at::Tensor& t, const at::Tensor& weights)
{
    TORCH_CHECK_DTYPE(slots, kInt);
    TORCH_CHECK_DTYPE(t, kHalf);
    TORCH_CHECK_DTYPE(weights, kHalf);
    TORCH_CHECK(t.dim() == 2 && weights.dim() == 2, "lora: t and the weights must be 2D");
    TORCH_CHECK(t.is_contiguous() && weights.is_contiguous() && slots.is_contiguous(), "lora: tensors must be contiguous");
    TORCH_CHECK(slots.dim() == 2 && slots.size(1) == 2 && slots.size(0) >= t.size(0), "lora: slots must be (rows, 2)");
    TORCH_CHECK(t.size(1) <= weights.size(0), "lora: the weights are narrower than the rank");
    return (int) t.size(0);
}

void lora_down_gr
(
    const at::Tensor& x,
    const at::Tensor& a_t,
    const at::Tensor& slots,
    at::Tensor& t,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    int rows = lora_check_slots(slots, t, a_t);
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == rows, "lora_down: x must be (rows, K)");
    TORCH_CHECK(x.size(1) == a_t.size(1), "lora_down: x and A differ in width");
    TORCH_CHECK(x.stride(1) == 1, "lora_down: x must have contiguous columns");
    int rank = (int) t.size(1);
    if (!rows || !rank) return;

    dim3 grid(rank, rows);
    lora_down_kernel<<<grid, LORA_THREADS, 0, stream>>>
    (
        (const half*) x.data_ptr(),
        (const half*) a_t.data_ptr(),
        (const int*) slots.data_ptr(),
        (half*) t.data_ptr(),
        (int) x.stride(0),
        (int) x.size(1),
        rank
    );
    // Every launch records at least its end: capture matches recorded sites to nodes by kernel, in order
    if (graph) graph->record_param((void*) &lora_down_kernel, GP_end, 0);
    cuda_check(cudaPeekAtLastError());
}

void lora_up_gr
(
    const at::Tensor& t,
    const at::Tensor& b,
    const at::Tensor& slots,
    at::Tensor& o,
    bool patch_o,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(t.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    int rows = lora_check_slots(slots, t, b);
    int rank = (int) t.size(1);
    int size_n = (int) b.size(1);
    TORCH_CHECK_FLOAT_HALF(o);
    TORCH_CHECK(o.is_contiguous() && o.numel() == (int64_t) rows * size_n, "lora_up: o must be rows * N contiguous elements");
    if (!rows || !rank || !size_n) return;

    dim3 grid(CEIL_DIVIDE(size_n, LORA_THREADS), rows);

    #define INSTANCE(dt, T) \
    if (o.dtype() == dt) \
    { \
        lora_up_kernel<T><<<grid, LORA_THREADS, 0, stream>>> \
        ( \
            (const half*) t.data_ptr(), \
            (const half*) b.data_ptr(), \
            (const int*) slots.data_ptr(), \
            (T*) o.data_ptr(), \
            size_n, \
            rank \
        ); \
        if (graph && patch_o) graph->record_param((void*) &lora_up_kernel<T>, GP_lora_up_o, 3); \
        if (graph) graph->record_param((void*) &lora_up_kernel<T>, GP_end, 0); \
    }

    INSTANCE(at::kHalf, half)
    INSTANCE(at::kFloat, float)

    #undef INSTANCE

    cuda_check(cudaPeekAtLastError());
}

void lora_down
(
    const at::Tensor& x,
    const at::Tensor& a_t,
    const at::Tensor& slots,
    at::Tensor& t
)
{
    lora_down_gr(x, a_t, slots, t, nullptr);
}

void lora_up
(
    const at::Tensor& t,
    const at::Tensor& b,
    const at::Tensor& slots,
    at::Tensor& o
)
{
    lora_up_gr(t, b, slots, o, false, nullptr);
}
