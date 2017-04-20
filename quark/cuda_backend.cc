#include "quark/cuda_backend.h"
#include "quark/cuda_util.h"

namespace quark {

void CudaBackend::New(void* ptr, size_t nbytes) {
  CUDA_CALL(cudaMalloc(&ptr, nbytes));
}

void CudaBackend::Delete(void* ptr) {
  CUDA_CALL(cudaFree(ptr));
}

} // namespace quark
