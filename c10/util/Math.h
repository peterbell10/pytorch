#pragma once

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_HCC__)
#include <c10/cuda/CUDAMathCompat.h>
#else
#include <c10/util/math_compat.h>
#endif


namespace c10 {
namespace math {

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_HCC__)

using namespace c10::cuda::compat;

#else

using std::abs;
using std::exp;
using std::ceil;
using std::copysign;
using std::floor;
using std::log;
using std::log1p;
using std::max;
using std::min;
using std::pow;
using std::sincos;
using std::sqrt;
using std::rsqrt;
using std::tan;
using std::tanh;
using std::normcdf;
using std::lgamma;

#endif

}}  // namespace c10::math
