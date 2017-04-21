#include "quark/tensor.h"

namespace quark {

template <typename T, typename Backend>
Tensor<T, Backend>::Tensor(vector<int64> shape) {
  size_ = Prod(shape);
  capacity_ = size_ * sizeof(T);
  shape_ = shape;
  data_ = Backend::New(capacity_);
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

} // namespace quark
