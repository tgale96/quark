#ifndef QUARK_OPS_OP_BASE_H_
#define QUARK_OPS_OP_BASE_H_

#include "quark/cuda_util.h"
#include "quark/tensor.h"

namespace quark {

template <typename T>
using GpuTensor = Tensor<T, CudaBackend>;

/**
 * @brief Defines the interface for all operations.
 *
 * Quark leverages cudaStreams and cudaEvents to executed ComputeGraphs
 * completely asynchronously. Ideally we would be able to run graphs
 * with both CPU and GPU operations, but this is a bit complicated and
 * I haven't figured out a good way to do this while still doing async
 * execution. To keep it simple, ops in quark only run on GPU.
 */
template <typename T>
class OpBase {
public:
  /**
   * Default constructor. Does nothing.
   */
  OpBase() {}

  DISABLE_COPY_ASSIGN_MOVE(OpBase);
  
  ~OpBase() = default;

  /**
   * @brief Executes the operation on GPU.
   *
   * Before launching this ops kernel, the op calls cudaStreamWaitEvent() 
   * on the input cudaEvents to ensure that all of the ops parent 
   * operations have completed before its work begins.
   */
  void Run(cudaStream_t stream, vector<cudaEvent_t> events) {
    for (const auto &event : events) {
      CUDA_CALL(cudaStreamWaitEvent(stream, event, 0));
    }
    LaunchKernel(stream);
  }

  /**
   * Returns vector of the input Tensors to the op
   */
  virtual vector<const GpuTensor<T>&> inputs() const = 0;

  /**
   * Returns vector of the output Tensors of the op
   */
  virtual vector<const GpuTensor<T>&> outputs() const = 0;
  
protected:
  // Enqueues this ops kernel in the input stream
  virtual void LaunchKernel(cudaStream_t stream) = 0;
};

} // namespace quark

#endif // QUARK_OPS_OP_BASE_H_
