#ifndef QUARK_TENSOR_H_
#define QUARK_TENSOR_H_

#include <memory>

#include "quark/util/common.h"

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
 * templating on an object that absracts the device specific memory management
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

  // Disable copy, assign, move-copy, move-assign
  Tensor(const Tensor &other) = delete;
  Tensor(Tensor &&other) = delete;
  Tensor& operator=(const Tensor &other) = delete;
  Tensor& operator=(Tensor &&other) = delete;

  // TODO(Trevor): handle de-allocation of memory
  ~Tensor() = default;

  /**
   * Getter for data pointer
   */
  
protected:
  shared_ptr<T> data_;
  vector<int64> shape_;
  int64 size_;
  size_t capacity_ = 0;
};

} // namespace quark

#endif // QUARK_TENSOR_H_
