#ifndef QUARK_CUDA_UTIL_H_
#define QUARK_CUDA_UTIL_H_

#include <cuda_runtime.h>

#include "quark/common.h"

namespace quark {

// Macro for cuda error checking. All calls that return cudaError_t should be wrapped with this
#define CUDA_CALL(code) { GpuAssert(code, __FILE__, __LINE__); }

// TODO(Trevor): update output to use logging functions
inline void GpuAssert(cudaError_t code, const string file, int line) {
  if (code != cudaSuccess) {
    std::cout << file << "(" << to_string(line) << "): "
              << cudaGetErrorString(code) << std::endl;
  }
}

} // namespace quark

#endif // QUARK_CUDA_UTIL_H_
