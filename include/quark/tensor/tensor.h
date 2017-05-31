#ifndef QUARK_TENSOR_TENSOR_H_
#define QUARK_TENSOR_TENSOR_H_

#include "quark/common.h"
#include "quark/tensor/cpu_backend.h"
#include "quark/tensor/cuda_backend.h"
#include "quark/util/backend_util.h"

namespace quark {

template <typename Type>
class TensorTest;

// Used to maintain a globaly consistent id system for all Tensor objects 
typedef int64 TensorId;
class TensorTag {
public:
  TensorTag() : id_(++count_) {}
  DISABLE_COPY_ASSIGN_MOVE(TensorTag);
  ~TensorTag() = default;

  TensorId id() const { return id_; }

private:
  static TensorId count_;
  const TensorId id_;
};

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
   * Creates a Tensor with the input shape.
   */
  explicit Tensor(vector<int64> shape) {
    size_ = Prod(shape);
    capacity_ = size_ * sizeof(T);
    shape_ = shape;
    data_ = (T*)Backend::New(capacity_);
  }

  /**
   * Creates a Tensor with the input shape that stores the input data.
   *
   * @throws runtime_error if the size of the Tensor does not match the
   * number of elements in the data vector.
   */
  Tensor(vector<int64> shape, const vector<T>& data) {
    QUARK_CHECK(Prod(shape) == data.size(),
        "New Tensor shape does not match the number of elements in the data vector");

    Resize(shape);
    CopyData(data.size(), data.data(), data_);
  }

  // TODO(Trevor): Having these copy constructors is very dangerous with the global Tensor id
  // system. Allowing copying can make issues with the graph compilation process. Consider
  // removing these and requiring the user to do some more explicit form of copying to do this.
  /**
   * Creates a Tensor by copying a different Tensor with an arbitrary backend.
   */
  template <typename SrcBackend>
  explicit Tensor(const Tensor<T, SrcBackend>& other) {
    Resize(other.shape());
    Copy(other);
  }

  // TODO(Trevor): Is there a way to have these copy constructors consolidated into
  // a single function? The first one should be able to handle both, but the
  // compiler seems to look for this function when the backend match.
  explicit Tensor(const Tensor<T, Backend>& other) {
    Resize(other.shape());
    Copy(other);
  }
  
  Tensor(Tensor<T, Backend>&& other) = delete;
  Tensor<T, Backend>& operator=(const Tensor<T, Backend>& rhs) = delete;
  Tensor<T, Backend>& operator=(Tensor<T, Backend>&& rhs) = delete;

  ~Tensor() { Backend::Delete(data_); }

  /**
   * Alters the shape of the tensor without changing the underlying memory. 
   * Requires that the size of the Tensor stays the same.
   *
   * @throws runtime_error if the new Tensor shape does not have the same 
   * number of elements as the current Tensor.
   */
  void Reshape(vector<int64> new_shape) {
    QUARK_CHECK(Prod(new_shape) == size_, "Input Tensor shape has different number of elements than current Tensor");
    shape_ = new_shape;
  }

  /**
   * Resizes a Tensor. The underlying memory is only reallocated if the new
   * Tensor size is larger than the current capacity of the Tensor. Thus, 
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
   * Copies data from the input tensor into this Tensor
   *
   * @throws runtime_error if the shape of calling Tensor and input Tensor
   * don't match
   */
  template <typename SrcBackend>
  void Copy(const Tensor<T, SrcBackend>& src) {
    QUARK_CHECK(src.shape() == shape(), "Input Tensor shape must match current Tensor shape to perform copy");
    CopyData(src.size(), src.data(), data_);
  }

  /**
   * Copies data from the input vector into this Tensor. Prod(this->shape()) must be
   * equal to the lenght of the input vector
   */
  void Copy(const vector<T>& src) {
    QUARK_CHECK(src.size() == Prod(shape()),
        "Input vector must have same number of elements as the calling Tensor");
    CopyData(size(), src.data(), data_);
  }
  
  /**
   * Returns a pointer to the underlying data store in a Tensor
   *
   * @throws runtime_error if Tensor stores no data
   */
  T* mutable_data() {
    QUARK_CHECK(shape_.size() != 0, "Tensor stores no data (shape == {})");
    return data_;
  }

  /**
   * Returns a const pointer to the underlying data store in a Tensor
   *
   * @throws runtime_error if Tensor stores no data
   */
  const T* data() const {
    QUARK_CHECK(shape_.size() != 0, "Tensor stores no data (shape == {})");
    return data_;
  }

  /**
   * Returns the shape of a Tensor
   */
  vector<int64> shape() const { return shape_; }
  
  /**
   * Returns the size of a Tensor
   */
  int64 size() const { return size_; }

  /**
   * Returns the id of the Tensor
   */
  TensorId id() const { return tid_.id(); }
  
  template <typename Type>
  friend class TensorTest;
  
protected:
  T* data_ = nullptr;
  vector<int64> shape_;
  int64 size_ = 0;
  size_t capacity_ = 0;
  
  const TensorTag tid_;
};

// Aliases for convenience
template <typename T>
using GpuTensor = Tensor<T, CudaBackend>;

template <typename T>
using CpuTensor = Tensor<T, CpuBackend>;

/**
 * Overloaded output operator for Tensor class w/ CpuBackend.
 */
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T, CpuBackend>& t) {
  const T* data = t.data();
  for (int i = 0 ; i < t.size(); ++i) {
    stream << data[i] << " ";
  }
  stream << std::endl;
  return stream;
}

/**
 * Overloaded output operator for Tensor class w/ CudaBackend. This is a potentially
 * expensive operation, as the data is copied from GPU to a temporary CpuBackend
 * Tensor.
 */
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T, CudaBackend>& t) {
  Tensor<T, CpuBackend> cpu_t(t);
  stream << cpu_t;
  return stream;
}

} // namespace quark

#endif // QUARK_TENSOR_TENSOR_H_
