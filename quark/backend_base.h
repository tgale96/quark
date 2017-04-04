#ifndef QUARK_BACKEND_BASE_H_
#define QUARK_BACKEND_BASE_H_

namespace quark {

/**
 * @brief abstract base class for backend objects
 *
 * Defines the interface that all backend objects must implement, such 
 * as allocation, and de-allocation methods
 */
class BackendBase {
public:

  /**
   * Allocation method for the backend
   */
  virtual static void* New(int nbytes) = 0;

  /**
   * De-allocation method for the backend
   */
  virtual static void* Delete(void *data) = 0;

  BackendBase(const BackendBase &other) = delete;
  BackendBase(BackendBase &&other) = delete;
  BackendBase& operator=(const BackendBase &rhs) = delete;
  BackendBase& operator=(BackendBase &&rhs) = delete;
  ~BackendBase() = default;
  
protected:
  // Backend objects should not be instantiated
  BackendBase() {}
}
  
} // namespace quark

#endif // QUARK_BACKEND_BASE_H_
