#ifndef QUARK_CUDA_BACKEND_H_
#define QUARK_CUDA_BACKEND_H_

#include "quark/common.h"

namespace quark {

/**
 * @brief Backend object for cuda
 *
 * Provides functions for allocation and deallocation 
 * of cuda memory
 */
class CudaBackend final {
public:

  CudaBackend(const CudaBackend &other) = delete;
  CudaBackend(CudaBackend &&other) = delete;
  CudaBackend& operator=(const CudaBackend &rhs) = delete;
  CudaBackend& operator=(CudaBackend &&rhs) = delete;
  ~CudaBackend() = delete;
  
  /**
   * Allocates `nbytes` of cuda memory
   */
  static void* New(size_t nbytes);

  /**
   * De-allocates pointer
   */
  static void Delete(void* ptr);

private:
  // CudaBackend shoud not be instantiated
  CudaBackend() {}
};

} // namespace quark

#endif // QUARK_CUDA_BACKEND_H_
