#ifndef QUARK_OPS_UTILITY_OPS_H_
#define QUARK_OPS_UTILITY_OPS_H_

#include "quark/compute_graph.h"
#include "quark/ops/copy_op.h"
#include "quark/ops/op_base.h"

namespace quark {

// Note: These could all be done with a macro, but this is the
// interface exposed to the user so it is important that they
// are able to understand what operations they can call and what
// each operation is actually doing.

/**
 * Copies the data from the "src" tensor to the "dst" tensor. "dst" is automatically 
 * resized to match "src" in shape and size.
 */
template <typename T>
void Copy(ComputeGraph<T> *cg, const GpuTensor<T>& src, GpuTensor<T>* dst) {
  shared_ptr<CopyOp<T>> op = make_shared<CopyOp<T>>(src, dst);
  cg->AddOp(op);
}

} // namespace quark

#endif // QUARK_OPS_UTILITY_OPS_H_

