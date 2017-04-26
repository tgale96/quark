#include "quark/util/math_util.h"

namespace quark {

// Note: In Quark, we use the same dimension naming and data layout conventions as TensorFlow
// (see https://www.tensorflow.org/performance/xla/shapes). With a shape of {A, B, C}, A is refered
// to as dimensions 0, B is dimension 1, and C is dimension 3. For data layout, we store Tensors
// in a major-to-minor ordering. With the shape of {A, B, C}, this means A is the most major
// dimension (It would have the slowest changing index if you were to iterate over the Tensor
// as it is sequentially layed out in memory) and C is the most minor dimension. With a Tensor
// of 2 dimensions, this corresponds to a row-major layout. BLAS routines assume data is stored
// in column-major format, so we calculate the transpose of the output by default (A tranposed
// row-major matrix is the same as its (not transposed) column-major equivalent)

template <>
void quark_gpu_geam(cublasHandle_t handle, const float* alpha, bool trans_a, const GpuTensor<float>&a,
    const float* beta, bool trans_b, const GpuTensor<float>& b, GpuTensor<float>* c) {
  int m = trans_a ? a.shape()[0] : a.shape()[1];
  int n = trans_a ? a.shape()[1] : a.shape()[0];

  int lda = trans_a ? n : m;
  int ldb = lda;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  CUBLAS_CALL(cublasSgeam(handle, transpose_a , transpose_b, m, n,
          alpha, a.data(), lda, beta, b.data(), ldb, c->mutable_data(), c->shape()[1]));
}

template <>
void quark_gpu_geam(cublasHandle_t handle, const double* alpha, bool trans_a, const GpuTensor<double>&a,
    const double* beta, bool trans_b, const GpuTensor<double>& b, GpuTensor<double>* c) {
  int m = trans_a ? a.shape()[0] : a.shape()[1];
  int n = trans_a ? a.shape()[1] : a.shape()[0];

  int lda = trans_a ? n : m;
  int ldb = lda;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  CUBLAS_CALL(cublasDgeam(handle, transpose_a , transpose_b, m, n,
          alpha, a.data(), lda, beta, b.data(), ldb, c->mutable_data(), c->shape()[1]));
}

template <>
void quark_gpu_gemm(cublasHandle_t handle, const float* alpha, bool trans_a,
    const GpuTensor<float>&a, const float* beta, bool trans_b, const GpuTensor<float>& b,
    GpuTensor<float>* c) {
  int m = trans_b ? b.shape()[0] : b.shape()[1];
  int k = trans_b ? b.shape()[1] : b.shape()[0];
  int n = trans_a ? a.shape()[1] : a.shape()[0];

  int ldb = trans_b ? k : m;
  int lda = trans_a ? n : k;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CALL(cublasSgemm(handle, transpose_a, transpose_b, m, n, k,
          alpha, b.data(), ldb, a.data(), lda, beta, c->mutable_data(), c->shape()[1]));
  
  // Good with trans
  // int lda = trans_a ? a.shape()[1] : a.shape()[1];
  // int ldb = trans_b ? b.shape()[1] : b.shape()[1];
  // cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  // cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  // CUBLAS_CALL(cublasSgemm(handle, transpose_a, transpose_b, b.shape()[0], a.shape()[1], b.shape()[1],
  //         alpha, b.data(), ldb, a.data(), lda, beta, c->mutable_data(), c->shape()[1]));
  
  // Good with no_trans
  // int lda = trans_a ? a.shape()[0] : a.shape()[1];
  // int ldb = trans_b ? b.shape()[0] : b.shape()[1];
  // cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  // cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  // CUBLAS_CALL(cublasSgemm(handle, transpose_a, transpose_b, b.shape()[1], a.shape()[0], b.shape()[0],
  //         alpha, b.data(), ldb, a.data(), lda, beta, c->mutable_data(), c->shape()[1]));
}

template <>
void quark_gpu_gemm(cublasHandle_t handle, const double* alpha, bool trans_a,
    const GpuTensor<double>&a, const double* beta, bool trans_b, const GpuTensor<double>& b,
    GpuTensor<double>* c) {
  int m = trans_b ? b.shape()[0] : b.shape()[1];
  int k = trans_b ? b.shape()[1] : b.shape()[0];
  int n = trans_a ? a.shape()[1] : a.shape()[0];

  int ldb = trans_b ? k : m;
  int lda = trans_a ? n : k;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CALL(cublasDgemm(handle, transpose_a, transpose_b, m, n, k,
          alpha, b.data(), ldb, a.data(), lda, beta, c->mutable_data(), c->shape()[1]));
}

} // namespace quark
