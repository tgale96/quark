#ifndef QUARK_CUDA_UTIL_H_
#define QUARK_CUDA_UTIL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "quark/common.h"

namespace quark {

// Macro for cuda error checking. All calls that
// return cudaError_t should be wrapped with this
#define CUDA_CALL(code) {                                               \
  if (code != cudaSuccess) {                                            \
    string file = __FILE__;                                             \
    string line = to_string(__LINE__);                                  \
    string err_str = cudaGetErrorString(code);                          \
    std::cout << file << "(" << line << "): " << err_str << std::endl;  \
  }                                                                     \
}

// Allocates cudaStreams and keeps track of them
// so that they can be cleaned up appropriately
class StreamManager final {
public:
  StreamManager() {}

  DISABLE_COPY_ASSIGN_MOVE(StreamManager);

  ~StreamManager() {
    for (auto& stream : streams_) {
      if (stream) {
        CUDA_CALL(cudaStreamDestroy(stream));
      }
    }
  }

  // Allocate a new stream
  cudaStream_t GetStream() {
    cudaStream_t new_stream;
    CUDA_CALL(cudaStreamCreate(&new_stream));
    streams_.push_back(new_stream);
    return new_stream;
  }
  
private:
  vector<cudaStream_t> streams_;
};

} // namespace quark

#endif // QUARK_CUDA_UTIL_H_
