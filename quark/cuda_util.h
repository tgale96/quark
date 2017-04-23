#ifndef QUARK_CUDA_UTIL_H_
#define QUARK_CUDA_UTIL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "quark/common.h"

namespace quark {

// Macro for cuda error checking. All calls that return cudaError_t should be wrapped with this
#define CUDA_CALL(code) {                                               \
  if (code != cudaSuccess) {                                            \
    string file = __FILE__;                                             \
    string line = to_string(__LINE__);                                  \
    string err_str = cudaGetErrorString(code);                          \
    std::cout << file << "(" << line << "): " << err_str << std::endl;  \
  }                                                                     \
}

} // namespace quark

#endif // QUARK_CUDA_UTIL_H_
