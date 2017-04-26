#include "quark/cuda_backend.h"
#include "quark/util/cuda_util.h"

namespace quark {

void* CudaBackend::New(size_t nbytes) {
  void* ptr = nullptr;
  CUDA_CALL(cudaMalloc(&ptr, nbytes));
  return ptr;
}

void CudaBackend::Delete(void* ptr) {
  CUDA_CALL(cudaFree(ptr));
}

} // namespace quark
