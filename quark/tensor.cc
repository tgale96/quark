#include "quark/tensor.h"

namespace quark {

int64 TensorTag::count_ = 0;

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
