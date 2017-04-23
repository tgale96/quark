#ifndef QUARK_OPS_MATMUL_OP_H_
#define QUARK_OPS_MATMUL_OP_H_

#include "quark/ops/compute_graph.h"
#include "quark/ops/op_base.h"

namespace quark {

// TODO(Trevor): Move this function into a "math_ops.h" file so that users can include
// a single file to get all mathematical operations.
void Matmul(ComputeGraph *cg, const Tensor& a, const Tensor& b, Tensor* c);

template <typename T>
class MatmulOp final : OpBase {
public:
  MatmulOp();

  DISABLE_COPY_ASSIGN_MOVE(MatmulOp);
  ~MatmulOp() = default;
  
private:
  void LaunchKernel(cudaStream_t stream) override;
};

} // namespace quark

#endif // QUARK_OPS_MATMUL_OP_H_
