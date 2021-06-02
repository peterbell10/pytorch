#include <ATen/TensorIterator.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <memory>
#include <ATen/Functions.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorIteratorInternal.h>

#include <c10/util/irange.h>

/// Contains the implementation of parallel reductions in TensorIterator.

namespace at {

using loop2d_t = TensorIteratorBase::loop2d_t;

static bool use_two_pass_reduction(TensorIteratorBase& iter);
static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop);
static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop);

void TensorIteratorBase::parallel_reduce(loop2d_t loop) {
  TORCH_CHECK(ntensors() == 2, "parallel_reduce only supports one input and one output");
  int64_t numel = this->numel();
  if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
      at::in_parallel_region()) {
    serial_for_each(loop, {0, numel});
  } else if (use_two_pass_reduction(*this)) {
    two_pass_reduction(*this, loop);
  } else {
    parallel_dim_reduction(*this, loop);
  }
}

static bool use_two_pass_reduction(TensorIteratorBase& iter) {
  return iter.output(0).numel() == 1;
}

static void two_pass_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  const int max_threads = at::get_num_threads();

  auto dst = iter.output(0);
  auto unsqueezed = dst.unsqueeze(0);
  auto buffer_shape = DimVector(unsqueezed.sizes());
  buffer_shape[0] = max_threads;
  auto buffer = at::empty(buffer_shape, dst.options());
  // Fill with the identity
  buffer.copy_(unsqueezed);

  auto buffer_stride = buffer.strides()[0] * buffer.element_size();
  auto buffer_0 = buffer[0];
  auto first_reduce = TensorIterator::reduce_op(buffer_0, iter.input(0));
  TORCH_INTERNAL_ASSERT(first_reduce.output(0).is_alias_of(buffer_0));

  at::parallel_for(0, iter.numel(), internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    const auto thread_num = at::get_thread_num();
    auto shape = first_reduce.shape();
    auto strides = first_reduce.get_strides();

    // Bump output ptr so each thread has its own ouput slice
    auto base_ptrs = first_reduce.get_base_ptrs();
    base_ptrs[0] += buffer_stride * thread_num;

    at::internal::serial_for_each(shape, strides, base_ptrs.data(),
                                  base_ptrs.size(), loop, {begin, end});
  });

  auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
  final_reduce.for_each(loop);
}

/// Chooses a dimension over which to parallelize. Prefers the outer-most
/// dimension thats larger than the number of available threads.
static int find_split_dim(TensorIteratorBase& iter) {
  int num_threads = at::get_num_threads();
  auto shape = iter.shape();

  // start with the outer-most dimension
  int best_dim = iter.ndim() - 1;
  for (int dim = best_dim; dim >= 0 && !iter.is_dim_reduced(dim); dim--) {
    if (shape[dim] >= num_threads) {
      return dim;
    } else if (shape[dim] > shape[best_dim]) {
      best_dim = dim;
    }
  }

  AT_ASSERT(!iter.is_dim_reduced(best_dim));
  return best_dim;
}

static std::tuple<int64_t, int64_t>
round_columns(TensorIteratorBase& iter, int dim, int multiple, int64_t begin, int64_t end) {
  begin = begin - (begin % multiple);
  if (end != iter.shape()[dim]) {
    // only round the 'end' column down if it's not the final column
    end = end - (end % multiple);
  }
  return std::make_tuple(begin, end);
}

static void parallel_dim_reduction(TensorIteratorBase& iter, loop2d_t loop) {
  AT_ASSERT(iter.ndim() >= 1);
  int dim = find_split_dim(iter);
  int64_t cols = iter.shape()[dim];
  int element_size = iter.element_size(/*arg=*/1);

  bool should_round_columns = iter.strides(1)[dim] == element_size;
  at::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
    if (should_round_columns) {
      // round columns to multiples of 128 bytes if adjacent columns are
      // contiguous in memory.
      int64_t cols_per_128_bytes = 128 / element_size;
      std::tie(begin, end) = round_columns(iter, dim, cols_per_128_bytes, begin, end);
    }
    if (begin == end) {
      return;
    }
    auto sub_iter = TensorIterator(iter);
    sub_iter.narrow(dim, begin, end - begin);
    sub_iter.for_each(loop);
  });
}

void TensorIteratorBase::serial_foreach_reduced_elt(loop_subiter_t loop, Range range) const {
  TORCH_INTERNAL_ASSERT(ninputs() == 1);
  TORCH_INTERNAL_ASSERT(noutputs() >= 1);
  if (range.size() == 0) {
    return;
  }

  const auto shape = this->shape();
  const auto ndim = this->ndim();
  const auto reduce_dims = num_reduce_dims();

  const size_t noperands = operands_.size();
  const auto base_ptrs = this->get_base_ptrs();
  auto all_strides = this->get_strides();
  auto non_reduced_strides = IntArrayRef{all_strides}.slice(
      noperands * reduce_dims, all_strides.size() - noperands * reduce_dims);
  auto non_reduced_shape = shape.slice(reduce_dims, ndim - reduce_dims);

  // Duplicate the TensorIterator, and narrow it to point to one output element.
  // Then update the pointers & view offsets appropriately for each call.
  TensorIterator reduce_iter = *this;
  for (int i = reduce_dims; i < ndim; ++i) {
    reduce_iter.shape_[i] = 1;
  }

  c10::SmallBuffer<char*, 8> data_ptrs(noperands);
  DimCounter dims {non_reduced_shape, range};
  while (!dims.is_done()) {
    for (int i = reduce_dims; i < ndim; ++i) {
      reduce_iter.view_offsets_[i] = view_offsets_[i] + dims.values[i - reduce_dims];
    }
    at::internal::get_data_ptrs(data_ptrs.data(), base_ptrs,
                                non_reduced_strides, dims.values);
    for (size_t iop = 0; iop < noperands; ++iop) {
      reduce_iter.operands_[iop].data = data_ptrs[iop];
    }

    loop(reduce_iter);
    dims.increment({1, 1});
  }
}

void TensorIteratorBase::foreach_reduced_elt(loop_subiter_t loop) const {
  const auto output_numel = num_output_elements();
  if (output_numel == 0) {
    return;
  }
  if (output_numel == 1) {
    TORCH_INTERNAL_ASSERT(ninputs() == 1);
    TORCH_INTERNAL_ASSERT(noutputs() >= 1);
    return loop(*this);
  }

  const auto input_numel = numel();
  const auto grain_size = at::internal::GRAIN_SIZE / (input_numel / output_numel);
  at::parallel_for(0, output_numel, grain_size, [&](int64_t begin, int64_t end) {
    serial_foreach_reduced_elt(loop, {begin, end});
  });
}

}  // namespace at
