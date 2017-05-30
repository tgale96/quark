#include "quark/util/math_functions.h"

namespace quark {

template <typename T>
__global__ void EltwiseProduct(int64 n, const T* alpha, const T* a, const T* b, T* c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = alpha * b[tid] * c[tid];
  }
}

template <typename T>
void quark_gpu_eltwise_prod(cudaStream_t stream, const T* alpha, const GpuTensor<T> &a,
    const GpuTensor<T>& b, GpuTensor<T>* c) {
  int64 n = Prod(a.shape());

  int num_thread = QUARK_CUDA_BLOCK_SIZE;
  int64 num_block = QUARK_GET_NUM_BLOCK(num_thread);
  EltwiseProduct<<<num_block, num_thread, 0, stream>>>(n, alpha, a.data(),
      b.data(), c->mutable_data());
}

} // namespace quark
