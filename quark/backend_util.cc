#include "quark/backend_util.h"
#include "quark/cuda_util.h"

namespace quark {

template <typename T>
void CopyData(int64 num, const T* src, const T* dst) {
  QUARK_ASSERT(src != nullptr, "Src pointer must not be nullptr");
  QUARK_ASSERT(dst != nullptr, "Dst pointer must not be nullptr");
  QUARK_ASSERT(num >= 0, "Cannot copy negative number of elements");
  
  if (num == 0) return;

  CUDA_CALL(cudaMemcpy(dst, src, num * sizeof(T), cudaMemcpyDefault));
}

} // namespace quark
