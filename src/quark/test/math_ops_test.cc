#include <gtest/gtest.h>

#include "quark/ops/add_op.h"
#include "quark/ops/matmul_op.h"
#include "quark/test/quark_test.h"
#include "quark/util/cuda_util.h"

namespace quark {

template <typename T>
class MathOpsTest : public QuarkTest<T> {
public:

  vector<int64> GetRandMatrixDims() {
    int rows = std::rand() % 100 + 1;
    int cols = std::rand() % 100 + 1;
    return {rows, cols};
  }

  vector<T> ComputeSum(const T* d1, const T* d2, int64 num) {
    QUARK_ASSERT(d1, "d1 ptr must not be nullptr");
    QUARK_ASSERT(d2, "d2 ptr must not be nullptr");

    T* h_d1 = new T[num];
    T* h_d2 = new T[num];
    CUDA_CALL(cudaMemcpy(h_d1, d1, num * sizeof(T), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(h_d2, d2, num * sizeof(T), cudaMemcpyDefault));

    vector<T> res(num, 0);

    for (int i = 0; i < num; ++i) {
      res[i] = h_d1[i] + h_d2[i];
    }

    delete[] h_d1;
    delete[] h_d2;
    return res;
  }

  vector<T> TransposeMatrix(const T* d, int rows, int cols) {    
    QUARK_ASSERT(d, "d must not be nullptr");
    QUARK_ASSERT(rows > 0 && cols > 0, "Dims must be > 0");

    T* h_d = new T[rows*cols];
    CUDA_CALL(cudaMemcpy(h_d, d, rows * cols * sizeof(T), cudaMemcpyDefault));
    
    vector<T> res(rows*cols, 0);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        res[j * rows + i] = h_d[i * cols + j];
      }
    }
    delete[] h_d;
    return res;
  }

  vector<T> ComputeProduct(const T* d1, const T* d2, int m, int n, int k) {
    QUARK_ASSERT(d1, "d1 ptr must not be nullptr");
    QUARK_ASSERT(d2, "d2 ptr must not be nullptr");

    T* h_d1 = new T[m * k];
    T* h_d2 = new T[k * n];
    CUDA_CALL(cudaMemcpy(h_d1, d1, m * k * sizeof(T), cudaMemcpyDefault));
    CUDA_CALL(cudaMemcpy(h_d2, d2, k * n * sizeof(T), cudaMemcpyDefault));

    vector<T> res(m * n, 0);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int l = 0; l < k; ++l) {
          res[i * n + j] += h_d1[i * k + l] * h_d2[l * n + j];
        }
      }
    }
    delete[] h_d1;
    delete[] h_d2;
    return res;
  }
  
protected:
};

typedef ::testing::Types <float, double> Implementations;
TYPED_TEST_CASE(MathOpsTest, Implementations);

TYPED_TEST(MathOpsTest, TestSmallAddOpNoTrans) {
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

TYPED_TEST(MathOpsTest, TestAddOpNoTrans) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(dims, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  AddOp<TypeParam> op(1.0, false, a, 1.0, false, b, &c);
  
  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());
  
  // check the output
  vector<TypeParam> result = this->ComputeSum(a.data(), b.data(), a.size());
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestSmallAddOpTrans) {
  vector<TypeParam> data = {1, 2, 3, 4, 5, 6};
  vector<TypeParam> other_data = {1, 2, 3, 4, 5, 6};
  Tensor<TypeParam, CudaBackend> a({3, 2}, data);
  Tensor<TypeParam, CudaBackend> b({3, 2}, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  AddOp<TypeParam> op(1.0, true, a, 1.0, true, b, &c);
  
  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());
  
  // check the output
  vector<TypeParam> result = {2, 6, 10, 4, 8, 12};
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestAddOpTransB) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<int64> trans_dims = {dims[1], dims[0]};
  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(trans_dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(trans_dims, other_data);
  Tensor<TypeParam, CudaBackend> c;
  
  // Get the operator
  AddOp<TypeParam> op(1.0, false, a, 1.0, true, b, &c);

  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());
  
  // check the output
  vector<TypeParam> trans_b = this->TransposeMatrix(b.data(), b.shape()[0], b.shape()[1]);
  vector<TypeParam> result = this->ComputeSum(a.data(), trans_b.data(), a.size());
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestAddOpTransA) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<int64> trans_dims = {dims[1], dims[0]};
  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(trans_dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(trans_dims, other_data);
  Tensor<TypeParam, CudaBackend> c;
  
  // Get the operator
  AddOp<TypeParam> op(1.0, true, a, 1.0, false, b, &c);

  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());
  
  // check the output
  vector<TypeParam> trans_a = this->TransposeMatrix(a.data(), a.shape()[0], a.shape()[1]);
  vector<TypeParam> result = this->ComputeSum(trans_a.data(), b.data(), a.size());
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestAddOpTransAB) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(dims, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  AddOp<TypeParam> op(1.0, true, a, 1.0, true, b, &c);
  
  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());
  
  // check the output
  vector<TypeParam> trans_a = this->TransposeMatrix(a.data(), a.shape()[0], a.shape()[1]);
  vector<TypeParam> trans_b = this->TransposeMatrix(b.data(), b.shape()[0], b.shape()[1]);
  vector<TypeParam> result = this->ComputeSum(trans_a.data(), trans_b.data(), a.size());
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestSmallMatmulOpNoTrans) {
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

  // check the output
  vector<TypeParam> result = {5, 11, 17};
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestMatmulOpNoTrans) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<int64> other_dims = this->GetRandMatrixDims();
  other_dims[0] = dims[1];

  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(other_dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(other_dims, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  MatmulOp<TypeParam> op(1.0, false, a, false, b, &c);
  
  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());
  
  // check the output
  vector<TypeParam> result = this->ComputeProduct(a.data(), b.data(), a.shape()[0],
      b.shape()[1], a.shape()[1]);
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size(), .00001));
}

TYPED_TEST(MathOpsTest, TestSmallMatmulOpTrans) {
  vector<TypeParam> data = {1, 2};
  vector<TypeParam> other_data = {1, 2};
  Tensor<TypeParam, CudaBackend> a({1, 2}, data);
  Tensor<TypeParam, CudaBackend> b({2, 1}, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  MatmulOp<TypeParam> op(1.0, true, a, true, b, &c);

  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());

  // check the output
  vector<TypeParam> result = {1, 2, 2, 4};
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size()));
}

TYPED_TEST(MathOpsTest, TestMatmulOpTransA) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<int64> other_dims = this->GetRandMatrixDims();  
  dims[0] = other_dims[0];

  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(other_dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(other_dims, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  MatmulOp<TypeParam> op(1.0, true, a, false, b, &c);

  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());

  // check the output
  vector<TypeParam> trans_a = this->TransposeMatrix(a.data(), a.shape()[0], a.shape()[1]);
  vector<TypeParam> result = this->ComputeProduct(trans_a.data(), b.data(), a.shape()[1],
      b.shape()[1], a.shape()[0]);
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size(), .00001));
}

TYPED_TEST(MathOpsTest, TestMatmulOpTransB) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<int64> other_dims = this->GetRandMatrixDims();  
  other_dims[1] = dims[1];

  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(other_dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(other_dims, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  MatmulOp<TypeParam> op(1.0, false, a, true, b, &c);

  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());

  // check the output
  vector<TypeParam> trans_b = this->TransposeMatrix(b.data(), b.shape()[0], b.shape()[1]);
  vector<TypeParam> result = this->ComputeProduct(a.data(), trans_b.data(), a.shape()[0],
      b.shape()[0], a.shape()[1]);
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size(), .00001));
}

TYPED_TEST(MathOpsTest, TestMatmulOpTransAB) {
  vector<int64> dims = this->GetRandMatrixDims();
  vector<int64> other_dims = this->GetRandMatrixDims();  
  other_dims[1] = dims[0];

  vector<TypeParam> data, other_data;
  this->GetRandData(dims, &data);
  this->GetRandData(other_dims, &other_data);

  Tensor<TypeParam, CudaBackend> a(dims, data);
  Tensor<TypeParam, CudaBackend> b(other_dims, other_data);
  Tensor<TypeParam, CudaBackend> c;

  // Get the operator
  MatmulOp<TypeParam> op(1.0, true, a, true, b, &c);

  // run the op
  cudaEvent_t event;
  CUDA_CALL(cudaEventCreate(&event));
  op.Run(0, event, {});
  CUDA_CALL(cudaDeviceSynchronize());

  // check the output
  vector<TypeParam> trans_a = this->TransposeMatrix(a.data(), a.shape()[0], a.shape()[1]);
  vector<TypeParam> trans_b = this->TransposeMatrix(b.data(), b.shape()[0], b.shape()[1]);
  vector<TypeParam> result = this->ComputeProduct(trans_a.data(), trans_b.data(), a.shape()[1],
      b.shape()[0], a.shape()[0]);
  ASSERT_TRUE(this->CompareData(result.data(), c.data(), c.size(), .00001));
}

} // namepace quark
