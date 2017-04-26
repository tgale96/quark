#include <cstdlib>
#include <ctime>
#include <gtest/gtest.h>
#include <random>

#include "quark/common.h"
#include "quark/tensor.h"
#include "quark/test/quark_test.h"

namespace quark {

// Test fixture to run typed tests
template <typename Type>
class TensorTest : public QuarkTest<typename Type::T> {
public:

  template <typename T, typename Backend>
  size_t GetCapacity(const Tensor<T, Backend>& t) {
    return t.capacity_;
  }

protected:
};

// Hack to get gtest to run with multiple test types
template <typename A, typename B>
struct TypePair {
  typedef A T;
  typedef B Backend;
};

// Macro to define the template params for the Tensors in the regular way
#define DEFINE_TYPES()                          \
  typedef typename TypeParam::T T;              \
  typedef typename TypeParam::Backend Backend;

typedef ::testing::Types <TypePair<float, CpuBackend>,
                          TypePair<double, CpuBackend>,
                          TypePair<float, CudaBackend>,
                          TypePair<double, CudaBackend>> Implementations;

TYPED_TEST_CASE(TensorTest, Implementations);

TYPED_TEST(TensorTest, ConstructWithShape) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  Tensor<T, Backend> tensor(shape);

  EXPECT_EQ(tensor.shape(), shape);
  EXPECT_EQ(tensor.size(), Prod(shape));
  EXPECT_EQ(this->GetCapacity(tensor), Prod(shape) * sizeof(T));
}

TYPED_TEST(TensorTest, ConstructFromVector) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  vector<T> data(Prod(shape), 0);
  this->GetRandData(shape, &data);  
  Tensor<T, Backend> tensor(shape, data);

  ASSERT_EQ(tensor.shape(), shape);
  ASSERT_EQ(tensor.size(), data.size());
  ASSERT_EQ(this->GetCapacity(tensor), Prod(shape) * sizeof(T));
  ASSERT_TRUE(this->CompareData(tensor.data(), data.data(), tensor.size()));
}

TYPED_TEST(TensorTest, ConstructFromCpuTensor) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  vector<T> data(Prod(shape), 0);
  this->GetRandData(shape, &data);  
  Tensor<T, CpuBackend> src_tensor(shape, data);

  Tensor<T, Backend> new_tensor(src_tensor);
  ASSERT_EQ(src_tensor.shape(), new_tensor.shape());
  ASSERT_EQ(src_tensor.size(), new_tensor.size());
  ASSERT_EQ(this->GetCapacity(src_tensor), this->GetCapacity(new_tensor));
  ASSERT_TRUE(this->CompareData(src_tensor.data(), new_tensor.data(), src_tensor.size()));
}

TYPED_TEST(TensorTest, ConstructFromCudaTensor) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  vector<T> data(Prod(shape), 0);
  this->GetRandData(shape, &data);  
  Tensor<T, CudaBackend> src_tensor(shape, data);

  Tensor<T, Backend> new_tensor(src_tensor);
  ASSERT_EQ(src_tensor.shape(), new_tensor.shape());
  ASSERT_EQ(src_tensor.size(), new_tensor.size());
  ASSERT_EQ(this->GetCapacity(src_tensor), this->GetCapacity(new_tensor));
  ASSERT_TRUE(this->CompareData(src_tensor.data(), new_tensor.data(), src_tensor.size()));
}


TYPED_TEST(TensorTest, ReshapeTensor) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  Tensor<T, Backend> tensor(shape);
  size_t start_capacity = this->GetCapacity(tensor);
  
  vector<int64> new_shape = {Prod(shape)};
  tensor.Reshape(new_shape);
  
  EXPECT_EQ(tensor.shape(), new_shape);
  EXPECT_EQ(tensor.size(), Prod(shape));
  EXPECT_EQ(this->GetCapacity(tensor), start_capacity);
}

TYPED_TEST(TensorTest, ResizeTensor) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  size_t start_capacity = Prod(shape) * sizeof(T);
  Tensor<T, Backend> tensor(shape);

  vector<int64> new_shape = shape;
  for (auto &val : new_shape) {
    if (val != 1) val -= 1;
  }

  // should not change capacity
  tensor.Resize(new_shape);
  
  EXPECT_EQ(tensor.shape(), new_shape);
  EXPECT_EQ(tensor.size(), Prod(new_shape));
  EXPECT_EQ(this->GetCapacity(tensor), start_capacity);

  // should change capcity
  for (auto &val : new_shape) val += 2;
  tensor.Resize(new_shape);

  EXPECT_EQ(tensor.shape(), new_shape);
  EXPECT_EQ(tensor.size(), Prod(new_shape));
  EXPECT_EQ(this->GetCapacity(tensor), Prod(new_shape) * sizeof(T));
}

TYPED_TEST(TensorTest, CopyFromCpuTensor) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  vector<T> data(Prod(shape), 0);
  this->GetRandData(shape, &data);  
  Tensor<T, CpuBackend> src_tensor(shape, data);

  // copy data into new tensor
  Tensor<T, Backend> new_tensor(shape);
  new_tensor.Copy(src_tensor);

  // make sure the data is the same
  ASSERT_TRUE(this->CompareData(src_tensor.data(), new_tensor.data(), src_tensor.size()));
}

TYPED_TEST(TensorTest, CopyFromCudaTensor) {
  DEFINE_TYPES();
  vector<int64> shape = this->GetRandDims();
  vector<T> data(Prod(shape), 0);
  this->GetRandData(shape, &data);  
  Tensor<T, CudaBackend> src_tensor(shape, data);

  // copy data into new tensor
  Tensor<T, Backend> new_tensor(shape);
  new_tensor.Copy(src_tensor);
  
  // make sure the data is the same
  ASSERT_TRUE(this->CompareData(src_tensor.data(), new_tensor.data(), src_tensor.size()));
}

TYPED_TEST(TensorTest, TestTensorIds) {
  DEFINE_TYPES();
  
  for (int i = 0; i < 100; ++i) {
    Tensor<T, Backend> tensor;
    Tensor<T, CudaBackend> d_tensor;
    Tensor<T, CpuBackend> h_tensor;

    // make sure ids are unique and incremental
    ASSERT_EQ(tensor.id(), d_tensor.id() - 1);
    ASSERT_EQ(tensor.id(), h_tensor.id() - 2);
  }
}

} // namespace quark
