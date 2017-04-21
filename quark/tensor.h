#ifndef QUARK_TENSOR_H_
#define QUARK_TENSOR_H_

#include "quark/common.h"

namespace quark {
// TODO(Trevor):
// 1. Backends for CPU and CUDA. Handle allocation, de-allocation
// 2. Utility methods for Tensor
//    - resize
//    - reshape
//    - printing for different backends
//    - copying between different backends
// 3. Pinned memory for CPU
// 4. I/O methods for tensors
// Question
// 1. How do we want to manage datatypes? We want ops to be able to be performed
//    in different precision


/**    
 * @brief Tensor is the basic unit of data storage in quark. All data storage
 * is contiguous in memory to avoid issues with strides when performing 
 * distributed computation.
 *
 * This Tensor class is heavily influenced by Caffe2. In particular the idea of 
 * templating on a class that absracts the device specific memory management
 * functions. In Caffe2 these are refered to as a "Context". Here we have used 
 * the name "Backend" to avoid confusion with the term Context in the CUDA
 * programming language. Template parameter T specifies the type of data stored 
 * in the Tensor.
 */
template <typename T, typename Backend>
class Tensor {
public:
  /**
   * Creates empty Tensor
   */
  Tensor() {}

  /**
   * Creates a tensor with the input shape
   */
  Tensor(vector<int64> shape);
  
  // Disable copy, assign, move-copy, move-assign
  Tensor(const Tensor &other) = delete;
  Tensor(Tensor &&other) = delete;
  Tensor& operator=(const Tensor &other) = delete;
  Tensor& operator=(Tensor &&other) = delete;

  ~Tensor();

  /**
   * Alters the shape of the tensor without changing the underlying memory. 
   * Requires that the size of the tensor stays the same.
   *
   * @throws runtime_error if the new tensor shape does not have the same 
   * number of elements as the current tensor
   */
  void Reshape(vector<int64> new_shape);

  /**
   * Resizes a tensor. The underlying memory is only reallocated if the new
   * tensor size is larger than the current capacity of the tensor
   */
  void Resize(vector<int64> new_shape);

  /**
   * Returns a pointer to the underlying data store in a tensor
   *
   * @throws runtime_error if tensor stores no data
   */
  T* mutable_data() {
    QUARK_CHECK(shape_.size() != 0, "Tensor stores no data (shape == {})");
    return data_;
  }

  /**
   * Returns a const pointer to the underlying data store in a tensor
   *
   * @throws runtime_error if tensor stores no data
   */
  const T* data() {
    QUARK_CHECK(shape_.size() != 0, "Tensor stores no data (shape == {})");
    return data_;
  }

  /**
   * Returns the shape of a tensor
   */
  vector<int64> shape() { return shape_; }
  
  /**
   * Returns the size of a tensor
   */
  int64 size() { return size_; }
  
protected:
  T* data_ = nullptr;
  vector<int64> shape_;
  int64 size_ = 0;
  size_t capacity_ = 0;
};

} // namespace quark

#endif // QUARK_TENSOR_H_
