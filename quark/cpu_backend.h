#ifndef QUARK_CPU_BACKEND_H_
#define QUARK_CPU_BACKEND_H_

#include "quark/common.h"

namespace quark {

/**
 * @brief Backend object for cpu data
 *
 * Provides functions for allocatin and deallocation
 * of cpu memory
 */
class CpuBackend final {
public:
  CpuBackend(const CpuBackend &other) = delete;
  CpuBackend(CpuBackend &&other) = delete;
  CpuBackend& operator=(const CpuBackend &rhs) = delete;
  CpuBackend& operator=(CpuBackend &&rhs) = delete;
  ~CpuBackend() = delete;
  
  /**
   * Allocates `nbytes` of memory
   */
  static void* New(size_t nbytes);

  /**
   * De-allocates pointer
   */
  static void Delete(void* ptr);

private:
  // CpuBackend shoud not be instantiated
  CpuBackend() {}
};

} // namespace quark

#endif // QUARK_CPU_BACKEND_H_
