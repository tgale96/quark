#include "quark/tensor.h"

namespace quark {

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

// This will need to be benchmarked to see how it actually performs. Based off the simple
// hash function suggested at http://stackoverflow.com/a/1646913/126995. Ideally we would
// use something like boost's hash_combine, but I wanted to avoid adding dependencies.
template <typename T, typename Backend>
size_t TensorHash::operator()(const Tensor<T, Backend>& t) {
  size_t res = 17;
  res = res * 31 + std::hash<T*>()(t.data_);
  for (const auto& val : t.shape_) {
    res = res * 31 + std::hash<int64>()(val);
  }
  res = res * 31 + std::hash<int64>()(t.size_);
  res = res * 31 + std::hash<size_t>()(t.capacity_);
  return res;
}

} // namespace quark
