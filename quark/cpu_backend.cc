#include "quark/cpu_backend.h"

namespace quark {

void* CpuBackend::New(size_t nbytes) {
  void* ptr = ::operator new(nbytes);
  return ptr;
}

void CpuBackend::Delete(void* ptr) {
  ::operator delete(ptr);
}

} // namespace quark
