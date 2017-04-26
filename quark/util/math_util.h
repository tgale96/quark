#ifndef QUARK_UTIL_MATH_UTIL_H_
#define QUARK_UTIL_MATH_UTIL_H_

#include "quark/tensor.h"
#include "quark/util/cuda_util.h"

namespace quark {

template <typename T>
void quark_gpu_geam(cublasHandle_t handle, const T* alpha, bool trans_a,
    const GpuTensor<T>&a, const T* beta, bool trans_b, const GpuTensor<T>& b,
    GpuTensor<T>* c);

} // namespace quark

#endif // QUARK_UTIL_MATH_UTIL_H_
