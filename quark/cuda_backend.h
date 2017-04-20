#ifndef QUARK_CUDA_BACKEND_H_
#define QUARK_CUDA_BACKEND_H_

#include "quark/backend_base.h"

namespace quark {

/**
 * @brief backend object for cuda
 *
 * Provides functions for allocation and deallocation 
 * of cuda memory
 */
class CudaBackend final : public BackendBase {
public:
  
  /**
   * Allocates `nbytes` of cuda memory
   */
  static void New(void* ptr, int nbytes) override;

  /**
   * De-allocates `data` pointer
   */
  static void Delete(void* ptr) override;

private:
  // CudaBackend shoud not be instantiated
  CudaBackend() {}
};

} // namespace quark

#endif // QUARK_CUDA_BACKEND_H_
