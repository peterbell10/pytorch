#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/cpu/Reduce.h>
#include <c10/util/llvmMathExtras.h>

#include <algorithm>


namespace at {
namespace native {
namespace {

int64_t ceil_log2(int64_t x) {
  if (x <= 2) {
    return 1;
  }

  auto ux = static_cast<uint64_t>(x);
  // Last set bit is floor(log2(x)), floor + 1 is ceil
  // except when x is an exact powers of 2, so subtract 1 first
  return static_cast<int64_t>(llvm::findLastSet(ux - 1)) + 1;
}

/** Simultaneously sum over n rows at once

This algorithm calculates the sum without loss of precision over large axes. It
does this by chunking the sum into groups of 16 or more elements. The sums of
these chunks are also summed in chunks and so on until there is just a single sum
value remaining. This means only numbers of a similar order of magnitude are
added together, thus minimising rounding errors.

This is done in a single linear pass over the data and with O(1) extra storage.
A simplified recursive implementation would look like this:

  scalar_t row_sum(const scalar_t * data, int64_t n) {
    // Note, in practice the chunk size can increase with n
    // This allows the recursion depth to be limited to O(1).
    constexpr int64_t min_chunk_size = 16;

    scalar_t sum = 0;
    if (n <= min_chunk_size) {
      // Recursive base case, calculate a simple running sum
      for (int64_t i = 0; i < n; ++i) {
        sum += data[i];
      }
      return sum;
    }

    // Recursively sum larger chunks of elements
    const int64_t chunk_size = std::max(divup(n, min_chunk_size), min_chunk_size);
    for (int64_t i = 0; i < n; i += chunk_size) {
      sum += row_sum(data + i, std::min(chunk_size, n - i));
    }
    return sum;
  }
*/
template <typename scalar_t, int64_t nrows>
std::array<scalar_t, nrows> multi_row_sum(
    const char * C10_RESTRICT in_data,
    const int64_t row_stride,
    const int64_t col_stride,
    const int64_t size) {
  constexpr int64_t num_levels = 4;

  const int64_t level_power =
      std::max(int64_t(4), ceil_log2(size) / num_levels);
  const int64_t level_step = (1 << level_power);
  const int64_t level_mask = level_step - 1;

  scalar_t acc[num_levels][nrows];
  std::fill_n(&acc[0][0], num_levels * nrows, scalar_t(0));

  int64_t i = 0;
  for (; i + level_step <= size;) {
    for (int64_t j = 0; j < level_step; ++j, ++i) {
      const char * sum_base = in_data + i * row_stride;
      #pragma unroll
      for (int64_t k = 0; k < nrows; ++k) {
        acc[0][k] += load<scalar_t>(sum_base, col_stride, k);
      }
    }

    for (int64_t j = 1; j < num_levels; ++j) {
      #pragma unroll
      for (int64_t k = 0; k < nrows; ++k) {
        acc[j][k] += acc[j-1][k];
        acc[j-1][k] = scalar_t(0);
      }

      const auto mask = (level_mask << (j * level_power));
      if ((i & mask) != 0) {
        break;
      }
    }
  }

  for (; i < size; ++i) {
    const char * sum_base = in_data + i * row_stride;
    #pragma unroll
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += load<scalar_t>(sum_base, col_stride, k);
    }
  }

  for (int64_t j = 1; j < num_levels; ++j) {
    #pragma unroll
    for (int64_t k = 0; k < nrows; ++k) {
      acc[0][k] += acc[j][k];
    }
  }

  std::array<scalar_t, nrows> ret;
  for (int64_t k = 0; k < nrows; ++k) {
    ret[k] = acc[0][k];
  }
  return ret;
}

template <typename T>
struct CascadeSumReduction {
  using scalar_t = T;
  static constexpr int ilp_factor = 4;

  static constexpr scalar_t identity() { return scalar_t(0); }

  template <typename U>
  static constexpr U reduce(U a, U b) {
    return a + b;
  }

  template <typename U>
  static std::array<U, ilp_factor> multi_row_reduce(
      const char * C10_RESTRICT in_data,
      const int64_t row_stride,
      const int64_t col_stride,
      const int64_t size) {
    return multi_row_sum<U, ilp_factor>(in_data, row_stride, col_stride, size);
  }
};

void sum_kernel_impl(TensorIterator &iter) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(ScalarType::Bool, iter.dtype(), "sum_cpu",
      [&] {
        binary_kernel_reduce_vec(
            iter, simple_vec_reduce(
                [=](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
                [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a + b; }));
      });
    return;
  }

  // Custom floating point sum for better accuracy
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "sum_cpu",
    [&] {
      binary_kernel_reduce_vec(iter, CascadeSumReduction<scalar_t>{});
    });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(sum_stub, &sum_kernel_impl);

}}  // namespace at::native
