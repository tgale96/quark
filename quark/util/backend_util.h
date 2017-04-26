#ifndef QUARK_UTIL_BACKEND_UTIL_H_
#define QUARK_UTIL_BACKEND_UTIL_H_

#include "quark/common.h"
#include "quark/util/cuda_util.h"

namespace quark {

/**
 * Handles cross-backend copying of data. Only works as long as
 * the only backends we support are cuda and cpu.
 */
template <typename T>
void CopyData(int64 num, const T* src, T* dst) {
  QUARK_ASSERT(src != nullptr, "Src pointer must not be nullptr");
  QUARK_ASSERT(dst != nullptr, "Dst pointer must not be nullptr");
  QUARK_ASSERT(num >= 0, "Cannot copy negative number of elements");
  
  if (num == 0) return;

  CUDA_CALL(cudaMemcpy(dst, src, num * sizeof(T), cudaMemcpyDefault));
}

} // namespace quark

#endif // QUARK_UTIL_BACKEND_UTIL_H_
