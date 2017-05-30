#ifndef QUARK_UTIL_CUDA_UTIL_H_
#define QUARK_UTIL_CUDA_UTIL_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "quark/common.h"

namespace quark {

// Macro for cuda error checking. All calls that return cudaError_t
// should be wrapped with this. Note: do {...{ while(0) wrapping is
// to avoid extra semi-colon errors from nvcc
#define CUDA_CALL(code)                                                 \
  do {                                                                  \
    cudaError_t status = code;                                          \
    QUARK_CHECK(status == cudaSuccess, cudaGetErrorString(status));     \
  } while(0)                                                             

// Handler function for cublas errors
inline const string cublasGetErrorString(cublasStatus_t status) {
  switch(status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE"; 
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH"; 
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED"; 
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown error";
}

// Macro for cublas error checking. All calls that
// return cublasStatus_t should be wrapped with this
#define CUBLAS_CALL(code)                                               \
  do {                                                                  \
    cublasStatus_t status = code;                                          \
    QUARK_CHECK(status == CUBLAS_STATUS_SUCCESS, cublasGetErrorString(status)); \
  } while(0)

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

// Block size used for custom kernels
const int QUARK_CUDA_BLOCK_SIZE = 512;

inline int64 QUARK_GET_NUM_BLOCK(const int n) {
  return ceil(float(n) / QUARK_CUDA_BLOCK_SIZE);
}

} // namespace quark

#endif // QUARK_UTIL_CUDA_UTIL_H_
