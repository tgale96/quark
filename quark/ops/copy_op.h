#ifndef QUARK_OPS_COPY_OP_H_
#define QUARK_OPS_COPY_OP_H_

#include "quark/ops/op_base.h"

namespace quark {

/**
 * @brief Copies the input Tensor's data into the output Tensor
 */
template <typename T>
class CopyOp final : public OpBase<T> {
public:
  /**
   * Stores the Tensors and resizes output tensor to match input tensor
   *
   * @throws runtime_error if output Tensor is nullptr
   */
  CopyOp(const GpuTensor<T>& input_tensor, GpuTensor<T>* output_tensor) :
    input_tensor_(input_tensor), output_tensor_(output_tensor) {
    QUARK_CHECK(output_tensor, "Output Tensor cannot be nullptr");
    
    output_tensor_->Resize(input_tensor.shape());
  }
  
  /**
   * Returns the input Tensor
   */
  vector<const GpuTensor<T>*> inputs() const override {
    return {&input_tensor_};
  }

  /**
   * Returns the output Tensor
   */
  vector<const GpuTensor<T>*> outputs() const override {
    return {output_tensor_};
  }

private:
  const GpuTensor<T>& input_tensor_;
  GpuTensor<T>* output_tensor_;
  
  void LaunchKernel(cudaStream_t stream) override {
    QUARK_ASSERT(input_tensor_.size() == output_tensor_->size(), "Input and output tensors are not the same size");
    QUARK_ASSERT(input_tensor_.shape() == output_tensor_->shape(), "Input and output tensors are not the same shape");
        
    size_t bytes = input_tensor_.size() * sizeof(T);
    CUDA_CALL(cudaMemcpyAsync(output_tensor_->mutable_data(), input_tensor_.data(), bytes, cudaMemcpyDeviceToDevice, stream));
  }

  
};

} // namespace quark

#endif // QUARK_OPS_COPY_OP_H_
