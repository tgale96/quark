#ifndef QUARK_BACKEND_UTIL_H_
#define QUARK_BACKEND_UTIL_H_

#include "quark/common.h"

namespace quark {

/**
 * Handles cross-backend copying of data. Only works as long as
 * the only backends we support are cuda and cpu.
 */
template <typename T>
void CopyData(int64 num, const T* src, const T* dst);

} // namespace quark

#endif // QUARK_BACKEND_UTIL_H_
