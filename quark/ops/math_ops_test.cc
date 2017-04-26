#include <gtest/gtest.h>

#include "quark/ops/add_op.h"
#include "quark/ops/matmul_op.h"
#include "quark/test/quark_test.h"
#include "quark/util/cuda_util.h"

namespace quark {

template <typename T>
class MathOpsTest : public QuarkTest<T> {
public:

protected:
};

typedef ::testing::Types <float, double> Implementations;
TYPED_TEST_CASE(MathOpsTest, Implementations);

// TODO(Trevor): Test combinations of transposes and larger matrices
TYPED_TEST(MathOpsTest, TestAddOpNoTrans) {
  vector<TypeParam> data = {1, 2, 3, 4, 5, 6};
  vector<TypeParam> other_data = {1, 2, 3, 4, 5, 6};
  Tensor<TypeParam, CudaBackend> a({3, 2}, data);
  Tensor<TypeParam, CudaBackend> b({3, 2}, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  AddOp<TypeParam> op(1.0, false, a, 1.0, false, b, &c);
  
  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());
  
  // check the output
  vector<TypeParam> result = {2, 4, 6, 8, 10, 12};
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestMatmulOpNoTrans) {
  vector<TypeParam> data = {1, 2};
  vector<TypeParam> other_data = {1, 3, 5, 2, 4, 6};
  Tensor<TypeParam, CudaBackend> a({1, 2}, data);
  Tensor<TypeParam, CudaBackend> b({2, 3}, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  MatmulOp<TypeParam> op(1.0, false, a, false, b, &c);
  
  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  
  // check the output
  vector<TypeParam> result = {5, 11, 17};
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestMatmulOpTrans) {
  vector<TypeParam> data = {1, 2};
  vector<TypeParam> other_data = {1, 2};
  Tensor<TypeParam, CudaBackend> a({1, 2}, data);
  Tensor<TypeParam, CudaBackend> b({2, 1}, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  MatmulOp<TypeParam> op(1.0, true, a, true, b, &c);

  std::cout << c << std::endl;
  
  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  
  // check the output
  vector<TypeParam> result = {1, 2, 2, 4};
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

} // namepace quark
