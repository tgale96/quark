#ifndef QUARK_OPS_ELTWISE_PRODUCT_OP_H_
#define QUARK_OPS_ELTWISE_PRODUCT_OP_H_

#include "quark/ops/op_base.h"
#include "quark/util/math_functions.h"

namespace quark {

template <typename T>
class EltwiseProductOp final : public OpBase<T> {
public:
  /**
   * Stores and validates inputs. Tensors a and b must be of matching shape. 
   * Output Tensor c is resized to match Tensors a and b.
   */
  EltwiseProductOp(T alpha, const GpuTensor<T>& a, const GpuTensor<T>& b,
      GpuTensor<T>* c) : a_(a), b_(b), c_(c), user_alpha_(nullptr) {
    QUARK_CHECK(a.shape() == b.shape(), "Input Tensors a & b must have the same shape");
    QUARK_CHECK(c, "Input Tensor c cannot be nullptr");
    c_->Resize(a.shape());

    // load the coefficient into the alpha Tensor
    vector<T> tmp = {alpha};
    alpha_.Resize({1});
    alpha_.Copy(tmp);
  }
  
  /**
   * Stores and validates inputs. Tensor alpha must be a scalar, and Tensors a and b must
   * be of matching shape. Output Tensor c is resized to match Tensors a and b.
   */
  EltwiseProductOp(const GpuTensor<T>& alpha, const GpuTensor<T>& a,
      const GpuTensor<T>& b, GpuTensor<T>* c) : a_(a), b_(b), c_(c), user_alpha_(&alpha) {
    QUARK_CHECK(a.shape() == b.shape(), "Input Tensors a & b must have hte same shape");
    QUARK_CHECK(alpha.size() == 1, "Input Tensor alpha must be a scalar");
    QUARK_CHECK(c, "Input Tensor c cannot be nullptr");
    c_->Resize(a.shape());
  }

  DISABLE_COPY_ASSIGN_MOVE(EltwiseProductOp);
  ~EltwiseProductOp() = default;

  /**
   * Returns the input Tensors
   */
  vector<const GpuTensor<T>*> inputs() const override {
    if (user_alpha_) return {&a_, &b_, user_alpha_};
    return {&a_, &b_};
  }

  /**
   * Returns the output Tensors
   */
  vector<const GpuTensor<T>*> outputs() const override {
    return {c_};
  }
  
private:
  const GpuTensor<T>& a_;
  const GpuTensor<T>& b_;
  GpuTensor<T>* c_;

  const GpuTensor<T>* user_alpha_;
  GpuTensor<T> alpha_;
  
  void LaunchKernel(cudaStream_t stream) override {
    if (user_alpha_) {
      quark_gpu_eltwise_prod(stream, user_alpha_->data(), a_, b_, c_); 
    } else {
      quark_gpu_eltwise_prod(stream, alpha_.data(), a_, b_, c_);
    }
  }
};

} // namespace quark

#endif // QUARK_OPS_ELTWISE_PRODUCT_OP_H_
