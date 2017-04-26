#ifndef QUARK_OPS_MATMUL_OP_H_
#define QUARK_OPS_MATMUL_OP_H_

#include "quark/ops/op_base.h"
#include "quark/util/cuda_util.h"
#include "quark/util/math_util.h"

namespace quark {

/**
 * Computes:
 * c = \alpha * Op(a) * op(b)
 */
template <typename T>
class MatmulOp final : public OpBase<T> {
public:

  /**
   * Stores and validates inputs.
   *
   * @throws runtime_error if output Tenosor ptr is nullptr
   * @throws runtime_error if input Tensors are not 2 dimensional
   * @throws runtime_error if inner Tensor dimensions do not match
   */
  MatmulOp(T alpha, bool trans_a, const GpuTensor<T>& a,
      bool trans_b, const GpuTensor<T>& b, GpuTensor<T>* c) :
    trans_a_(trans_a), trans_b_(trans_b), d_alpha_(nullptr),
    d_beta_(nullptr), a_(a), b_(b), c_(c) {
    QUARK_CHECK(c, "c Tensor cannot be nullptr");
    QUARK_CHECK(a.shape().size() == 2, "Input Tensor a must be two dimensional");
    QUARK_CHECK(b.shape().size() == 2, "Input Tensor b must be two dimensional");

    if (trans_a == trans_b) {
      QUARK_CHECK(a.shape()[1] == b.shape()[0] || a.shape()[0] == b.shape()[1],
          "Inner Tensor dimensions must match");
    } else {
      QUARK_CHECK(a.shape()[1] == b.shape()[1] || a.shape()[0] == b.shape()[0],
          "Inner Tensor dimensions must match");
    }

    int64 out_rows = trans_a ? a.shape()[1] : a.shape()[0];
    int64 out_cols = trans_b ? b.shape()[0] : b.shape()[1];
    std::cout << out_rows << " " << out_cols << std::endl;
    c_->Resize({out_rows, out_cols});

    // allocate cublas handle, set ptr mode, and copy args to device
    CUBLAS_CALL(cublasCreate(&handle_));
    CUBLAS_CALL(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE));
    d_alpha_ = (T*)CudaBackend::New(2 * sizeof(T));
    d_beta_ = d_alpha_ + 1;

    vector<T> packed_coeffs = {alpha, 0.0};
    CUDA_CALL(cudaMemcpy(d_alpha_, packed_coeffs.data(), 2 * sizeof(T), cudaMemcpyHostToDevice));
  }

  DISABLE_COPY_ASSIGN_MOVE(MatmulOp);
  ~MatmulOp() {
    CudaBackend::Delete(d_alpha_);
  }

  /**
   * Returns the input Tensors
   */
  vector<const GpuTensor<T>*> inputs() const override {
    return {&a_, &b_};
  }

  /**
   * Returns the output Tensor
   */
  vector<const GpuTensor<T>*> outputs() const override {
    return {c_};
  }
  
private:
  bool trans_a_, trans_b_;
  T* d_alpha_;
  T* d_beta_;
  const GpuTensor<T>& a_;
  const GpuTensor<T>& b_;
  GpuTensor<T>* c_;

  cublasHandle_t handle_;
  
  void LaunchKernel(cudaStream_t stream) override {
    CUBLAS_CALL(cublasSetStream(handle_, stream));
    quark_gpu_gemm(handle_, d_alpha_, trans_a_, a_, d_beta_, trans_b_, b_, c_);
  }
};

} // namespace quark

#endif // QUARK_OPS_MATMUL_OP_H_
