#ifndef QUARK_OPS_MATH_OPS_H_
#define QUARK_OPS_MATH_OPS_H_

#include "quark/compute_graph.h"
#include "quark/common.h"
#include "quark/ops/add_op.h"
#include "quark/ops/eltwise_product_op.h"
#include "quark/ops/matmul_op.h"

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

/**
 * Computes the matrix prod of "a" and "b"
 */
template <typename T>
void Matmul(ComputeGraph<T> *cg, T alpha, bool trans_a, const GpuTensor<T>& a,
    bool trans_b, const GpuTensor<T>& b, GpuTensor<T>* c) {
  shared_ptr<MatmulOp<T>> op =
    make_shared<MatmulOp<T>>(alpha, trans_a, a, trans_b, b, c);

  cg->AddOp(op);
}

/**
 * Computes the element-wise product of "a" and "b" and scales the output by the
 * constant "alpha".
 */
template <typename T>
void EltwiseProduct(ComputeGraph<T> *cg, T alpha, const GpuTensor<T>& a,
    const GpuTensor<T>& b, GpuTensor<T>* c) {
  shared_ptr<EltwiseProductOp<T>> op =
    make_shared<EltwiseProductOp<T>>(alpha, a, b, c);

  cg->AddOp(op);
}

/**
 * Computes the element-wise product of "a" and "b" and scales the output by "alpha".
 * "alpha" must be a scalar.
 */
template <typename T>
void EltwiseProduct(ComputeGraph<T> *cg, const GpuTensor<T>& alpha,
    const GpuTensor<T>& a, const GpuTensor<T>& b, GpuTensor<T>* c) {
  shared_ptr<EltwiseProductOp<T>> op =
    make_shared<EltwiseProductOp<T>>(alpha, a, b, c);

  cg->AddOp(op);
}

} // namespace quark

#endif // QUARK_OPS_MATH_OPS_H_
