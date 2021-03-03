#include <ATen/NativeFunctions.h>
#include <ATen/NumericUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>


namespace at { namespace native {

namespace {

void where_kernel_impl(TensorIterator &iter, ScalarType condition_type) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, iter.dtype(), "where_cuda", [&] {
    if (condition_type == at::ScalarType::Byte) {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (uint8_t cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
    } else {
      gpu_kernel(
        iter,
        [=] GPU_LAMBDA (bool cond_val, scalar_t self_val, scalar_t other_val) -> scalar_t {
          return cond_val ? self_val : other_val;
        });
    }
  });
}

void isposinf_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

void isneginf_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cuda", [&]() {
    gpu_kernel(
      iter,
      [] GPU_LAMBDA (scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); }
    );
  });
}

void clamp_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_cuda", [&] {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t v, scalar_t lower, scalar_t upper) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (at::_isnan(v)) {
        return v;
      } else {
        return ::min(::max(v, lower), upper);
      }
    });
  });
}

void clamp_scalar_kernel_impl(TensorIterator& iter, Scalar min, Scalar max) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_scalar_cuda", [&] {
    const auto lower = min.to<scalar_t>();
    const auto upper = max.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (at::_isnan(v)) {
        return v;
      } else {
        return ::min(::max(v, lower), upper);
      }
    });
  });
}

void clamp_min_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_min_cuda", [&] {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t v, scalar_t lower) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(v)) {
        return v;
      } else {
        return ::max(v, lower);
      }
    });
  });
}

void clamp_min_scalar_kernel_impl(TensorIterator& iter, Scalar min) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_min_scalar_cuda", [&] {
    auto lower = min.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(v)) {
        return v;
      } else {
        return ::max(v, lower);
      }
    });
  });
}

void clamp_max_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_max_cuda", [&] {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t v, scalar_t upper) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(v)) {
        return v;
      } else {
        return ::min(v, upper);
      }
    });
  });
}

void clamp_max_scalar_kernel_impl(TensorIterator& iter, Scalar max) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "clamp_max_scalar_cuda", [&] {
    const auto upper = max.to<scalar_t>();
    gpu_kernel(iter, [=]GPU_LAMBDA(scalar_t v) -> scalar_t {
      // Propagate nan, which doesn't propagate automatically for ROCm
      if (_isnan(v)) {
        return v;
      } else {
        return ::min(v, upper);
      }
    });
  });
}

} // anonymous namespace


REGISTER_DISPATCH(where_kernel, &where_kernel_impl);
REGISTER_DISPATCH(isposinf_stub, &isposinf_kernel_impl);
REGISTER_DISPATCH(isneginf_stub, &isneginf_kernel_impl);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl);
REGISTER_DISPATCH(clamp_min_stub, &clamp_min_kernel_impl);
REGISTER_DISPATCH(clamp_max_stub, &clamp_max_kernel_impl);
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl);

}} // namespace at::native
