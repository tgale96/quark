#include "quark/backend_util.h"
#include "quark/tensor.h"

namespace quark {

template <typename T, typename Backend>
Tensor<T, Backend>::Tensor(vector<int64> shape) {
  size_ = Prod(shape);
  capacity_ = size_ * sizeof(T);
  shape_ = shape;
  data_ = Backend::New(capacity_);
}

// TODO(Trevor): make sure this works when Backend == SrcBackend
template <typename T, typename Backend>
template <typename SrcBackend>
Tensor<T, Backend>::Tensor(const Tensor<T, SrcBackend>& src) {
  Resize(src.shape());
  Copy(src);
}

template <typename T, typename Backend>
void Tensor<T, Backend>::Reshape(vector<int64> new_shape) {
  QUARK_CHECK(Prod(new_shape) == size_, "Input tensor shape has different number of elements than current tensor");
  shape_ = new_shape;
}

template <typename T, typename Backend>
void Tensor<T, Backend>::Resize(vector<int64> new_shape) {
  int64 new_size = Prod(new_shape);
  
  if (new_size == size_) {
    shape_ = new_shape;
  } else if (new_size > size_) {
    // clean up current memory & allocate new memory
    Backend::Delete(data_);
    data_ = Backend::New(new_size * sizeof(T));

    shape_ = new_shape;
    size_ = new_size;
    capacity_ = new_size * sizeof(T);
  } else {
    shape_ = new_shape;
    size_ = new_size;
  }
}

template <typename T, typename Backend>
template <typename SrcBackend>
void Tensor<T, Backend>::Copy(const Tensor<T, SrcBackend>& src) {
  QUARK_CHECK(src.shape() == shape(), "Input tensor shape must match current tensor shape to perform copy");
  CopyData(src.size(), src.data(), data_);
}

template <typename T, typename Backend>
bool Tensor<T, Backend>::operator==(const Tensor<T, Backend>& other) const {
  if (data_ == other.data_ && shape_ == other.shape_ && size_ == other.size_ && capacity_ == other.capacity_) {
    return true;
  }
  return false;
}

// TODO(Trevor): make these print in a more organized manner that shows the shape of the tensor
template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T, CpuBackend>& t) {
  const T* data = t.data();
  for (int i = 0 ; i < t.size(); ++i) {
    stream << data[i] << " ";
  }
  stream << std::endl;
  return stream;
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Tensor<T, CudaBackend>& t) {
  Tensor<T, CpuBackend> cpu_t(t);
  stream << cpu_t;
  return stream;
}

} // namespace quark
