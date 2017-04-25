#ifndef QUARK_OPS_OP_BASE_H_
#define QUARK_OPS_OP_BASE_H_

#include "quark/cuda_util.h"
#include "quark/tensor.h"

namespace quark {

template <typename T>
using GpuTensor = Tensor<T, CudaBackend>;

// Used to maintain a globally consisten id system for all Ops
typedef int64 OpId;
class OpTag {
public:
  OpTag() : id_(++count_) {}
  DISABLE_COPY_ASSIGN_MOVE(OpTag);
  ~OpTag() = default;

  OpId id() const { return id_; }
  
private:
  static OpId count_;
  const OpId id_;
};

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
   * operations have completed before its work begins. After launching the
   * operation, it enqueues its event so that future ops can check for its
   * completion.
   */
  void Run(cudaStream_t stream, cudaEvent_t op_event, vector<cudaEvent_t> parent_op_events) {
    for (const auto &event : parent_op_events) {
      CUDA_CALL(cudaStreamWaitEvent(stream, event, 0));
    }
    LaunchKernel(stream);
    CUDA_CALL(cudaEventRecord(op_event, stream));
  }

  /**
   * Returns vector of the input Tensors to the op
   */
  virtual const vector<GpuTensor<T>>& inputs() const = 0;

  /**
   * Returns vector of the output Tensors of the op
   */
  virtual const vector<GpuTensor<T>>& outputs() const = 0;

  /**
   * Returns the id of the operation
   */
  OpId id() const { return oid_.id(); }
  
protected:
  // Enqueues this ops kernel in the input stream
  virtual void LaunchKernel(cudaStream_t stream) = 0;

  const OpTag oid_;
};

} // namespace quark

#endif // QUARK_OPS_OP_BASE_H_
