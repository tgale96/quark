#ifndef QUARK_UTIL_MATH_FUNCTIONS_H_
#define QUARK_UTIL_MATH_FUNCTIONS_H_

#include "quark/tensor/tensor.h"
#include "quark/util/cuda_util.h"

namespace quark {

template <typename T>
void quark_gpu_geam(cublasHandle_t handle, const T* alpha, bool trans_a,
    const GpuTensor<T>& a, const T* beta, bool trans_b, const GpuTensor<T>& b,
    GpuTensor<T>* c);

template <typename T>
void quark_gpu_gemm(cublasHandle_t handle, const T* alpha, bool trans_a,
    const GpuTensor<T>& a, const T* beta, bool trans_b, const GpuTensor<T>& b,
    GpuTensor<T>* c);

template <typename T>
void quark_gpu_eltwise_prod(cudaStream_t stream, const T* alpha, const GpuTensor<T> &a,
    const GpuTensor<T>& b, GpuTensor<T>* c);

} // namespace quark

#endif // QUARK_UTIL_MATH_FUNCTIONS_H_
