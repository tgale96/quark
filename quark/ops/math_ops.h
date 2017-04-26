#ifndef QUARK_OPS_MATH_OPS_H_
#define QUARK_OPS_MATH_OPS_H_

#include "quark/compute_graph.h"
#include "quark/common.h"
#include "quark/ops/add_op.h"

namespace quark {

/**
 * Computes the element-wise addition of "a" and "b"
 */
template <typename T>
void Add(ComputeGraph<T> *cg, T alpha, bool trans_a, const GpuTensor<T>& a,
    T beta, bool trans_b, const GpuTensor<T>& b, GpuTensor<T>* c) {
  shared_ptr<AddOp<T>> op =
    make_shared<AddOp<T>>(alpha, trans_a, a, beta, trans_b, b, c);

  cg->AddOp(op);
}

} // namespace quark

#endif // QUARK_OPS_MATH_OPS_H_
