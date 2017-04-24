#ifndef QUARK_TENSOR_H_
#define QUARK_TENSOR_H_

#include <functional>

#include "quark/backend_util.h"
#include "quark/common.h"
#include "quark/cpu_backend.h"
#include "quark/cuda_backend.h"

namespace quark {

template <typename Type>
class TensorTest;

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
   * Creates empty Tensor.
   */
  Tensor() {}

  /**
   * Creates a tensor with the input shape.
   */
  explicit Tensor(vector<int64> shape) {
    size_ = Prod(shape);
    capacity_ = size_ * sizeof(T);
    shape_ = shape;
    data_ = (T*)Backend::New(capacity_);
  }

  /**
   * Creates a tensor with the input shape that stores the input data.
   *
   * @throws runtime_error if the size of the tensor does not match the
   * number of elements in the data vector.
   */
  Tensor(vector<int64> shape, const vector<T>& data) {
    QUARK_CHECK(Prod(shape) == data.size(),
        "New tensor shape does not match the number of elements in the data vector");

    Resize(shape);
    CopyData(data.size(), data.data(), data_);
  }
  
  /**
   * Creates a tensor by copying a different tensor with an arbitrary backend.
   */
  template <typename SrcBackend>
  explicit Tensor(const Tensor<T, SrcBackend>& other) {
    Resize(other.shape());
    Copy(other);
  }

  // TODO(Trevor): is there a way to have these copy constructors consolidated into
  // a single function? The first one should be able to handle both, but the
  // compiler seems to look for this function when the backend match.
  Tensor(const Tensor<T, Backend>& other) {
    Resize(other.shape());
    Copy(other);
  }
  
  Tensor(Tensor<T, Backend>&& other) = delete;
  Tensor<T, Backend>& operator=(const Tensor<T, Backend>& rhs) = delete;
  Tensor<T, Backend>& operator=(Tensor<T, Backend>&& rhs) = delete;

  ~Tensor() { Backend::Delete(data_); }

  /**
   * Alters the shape of the tensor without changing the underlying memory. 
   * Requires that the size of the tensor stays the same.
   *
   * @throws runtime_error if the new tensor shape does not have the same 
   * number of elements as the current tensor.
   */
  void Reshape(vector<int64> new_shape) {
    QUARK_CHECK(Prod(new_shape) == size_, "Input tensor shape has different number of elements than current tensor");
    shape_ = new_shape;
  }

  /**
   * Resizes a tensor. The underlying memory is only reallocated if the new
   * tensor size is larger than the current capacity of the tensor. Thus, 
   * Resize is not guaranteed to maintain the data currently stored in 
   * the Tensor
   */
  void Resize(vector<int64> new_shape) {
    int64 new_size = Prod(new_shape);
    size_t new_capacity = new_size * sizeof(T);
    
    if (new_capacity <= capacity_) {
      // no need to reallocate
      size_ = new_size;
      shape_ = new_shape;
    } else {
      // clean up current memory & allocate new memory
      Backend::Delete(data_);
      data_ = (T*)Backend::New(new_capacity);
      
      shape_ = new_shape;
      size_ = new_size;
      capacity_ = new_capacity;
    }
  }

  /**
   * Copies data from the input tensor into this tensor
   *
   * @throws runtime_error if the shape of calling tensor and input tensor
   * don't match
   */
  template <typename SrcBackend>
  void Copy(const Tensor<T, SrcBackend>& src) {
    QUARK_CHECK(src.shape() == shape(), "Input tensor shape must match current tensor shape to perform copy");
    CopyData(src.size(), src.data(), data_);
  }
  
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
  const T* data() const {
    QUARK_CHECK(shape_.size() != 0, "Tensor stores no data (shape == {})");
    return data_;
  }

  /**
   * Returns the shape of a tensor
   */
  vector<int64> shape() const { return shape_; }
  
  /**
   * Returns the size of a tensor
   */
  int64 size() const { return size_; }

  // TODO(Trevor): This should evaluate the data at the pointer not the pointer.
  // TensorHash has same problem. Rather than having this and the hash function,
  // we should just ID every tensor with an int64. This is hashable, and then
  // we don't need to deal with these hacky comparison and hash functions.
  bool operator==(const Tensor<T, Backend>& other) const {
    if (data_ == other.data_ && shape_ == other.shape_ &&
        size_ == other.size_ && capacity_ == other.capacity_) {
      return true;
    }
    return false;
  }

  friend struct TensorHash;

  template <typename Type>
  friend class TensorTest;
  
protected:
  T* data_ = nullptr;
  vector<int64> shape_;
  int64 size_ = 0;
  size_t capacity_ = 0;
};

/**
 * Overloaded output operator for Tensor class w/ CpuBackend.
 */
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T, CpuBackend>& t);

/**
 * Overloaded output operator for Tensor class w/ CudaBackend. This is a potentially
 * expensive operation, as the data is copied from GPU to a temporary CpuBackend
 * Tensor.
 */
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T, CudaBackend>& t);

// Hash function for Tensor class. Needed for unordered_map w/ Tensor as key. 
struct TensorHash {
  template <typename T, typename Backend>
  size_t operator()(const Tensor<T, Backend>& t);
};

} // namespace quark

#endif // QUARK_TENSOR_H_
