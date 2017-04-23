#ifndef QUARK_COMMON_H_
#define QUARK_COMMON_H_

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace quark {

// commonly used types
typedef int64_t int64;
typedef int32_t int32;

// commonly used std lib objects
using std::shared_ptr;
using std::size_t;
using std::string;
using std::to_string;
using std::unordered_map;
using std::vector;

// Used for internal error checking
#define QUARK_ASSERT(result, message)             \
  if (!result) {                                  \
    string file = __FILE__;                       \
    string line = to_string(__LINE__);            \
    string err_str = file + "(" + line + "): ";   \
    std::cout << err_str << message;              \
    std::terminate;                               \
  }

// used for checking user input
#define QUARK_CHECK(result, message)                \
  if (!result) {                                    \
    string file = __FILE__;                         \
    string line = to_string(__LINE__);              \
    string err_str = file + "(" + line + "): ";     \
    throw std::runtime_error(err_str + message);    \
  }  

// disables copy, move-copy, copy-assignment, and move-assignment operators
#define DISABLE_COPY_ASSIGN_MOVE(class_name)                 \
  class_name(const class_name& other) = delete;              \
  class_name(class_name&& other) = delete;                   \
  class_name& operator=(const class_name& rhs) = delete;     \
  class_name& operator=(class_name&& rhs) = delete;          \
 

// TODO(Trevor): Should probably move this into a utility file
// Calculates the product of all elements in a vector
template <typename T>
inline T Prod(vector<T> v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

} // namespace quark

#endif // QUARK_COMMON_H_
