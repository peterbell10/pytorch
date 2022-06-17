#pragma once

// The PtrTraits argument to the TensorAccessor/GenericPackedTensorAccessor
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.

namespace at { namespace native {

template <typename T>
struct DefaultPtrTraits {
  using PtrType = T*;
};

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

template <typename T>
struct RestrictPtrTraits {
  using PtrType = T* RESTRICT;
};

}}  // namespace at::native
