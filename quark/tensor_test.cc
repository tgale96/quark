#include <gtest/gtest.h>

#include "quark/tensor.h"

namespace quark {

// Test fixture to run typed tests
template <typename Type>
class TensorTest : public ::testing::Test {};

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

typedef ::testing::Types <TypePair<float, CpuBackend>, TypePair<double, CpuBackend>,
                          TypePair<float, CudaBackend>, TypePair<double, CudaBackend>> Implementations;
TYPED_TEST_CASE(TensorTest, Implementations);

/* To Test:
 * for both backends:
 * 1. allocation
 * 2. reshape
 * 3. resize
 * 4. copy between all pairs of backends
 * 5. tensor hash
 */
TYPED_TEST(TensorTest, Reshape) {
  DEFINE_TYPES();

  Tensor<T, Backend> tensor;
}

} // namespace quark
