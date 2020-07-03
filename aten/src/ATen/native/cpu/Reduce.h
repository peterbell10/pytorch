#pragma once

#include <ATen/native/cpu/Loops.h>
#include <ATen/Parallel.h>
#include <c10/util/TypeList.h>

#include <array>
#include <sstream>

namespace at { namespace native { namespace {

using namespace vec256;

template <typename scalar_t>
struct LoadImpl {
  static scalar_t load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = reinterpret_cast<const scalar_t*>(data + index * stride);
    return *ptr;
  }
};

template <typename scalar_t>
struct LoadImpl<Vec256<scalar_t>> {
  static Vec256<scalar_t> load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
    auto *ptr = data + index * stride;
    return Vec256<scalar_t>::loadu(ptr);
  }
};

template <typename T>
T load(const char * C10_RESTRICT data, int64_t stride, int64_t index) {
  return LoadImpl<T>::load(data, stride, index);
}

template <typename op_t>
void accumulate_result(char * C10_RESTRICT data, int64_t stride, int64_t index,
                       typename op_t::scalar_t value, const op_t &op) {
  auto * ptr = reinterpret_cast<typename op_t::scalar_t*>(data + index * stride);
  *ptr = op.reduce(*ptr, value);
}

template <typename op_t, size_t numel, typename scalar_t>
void accumulate_result(char * C10_RESTRICT data, int64_t stride, int64_t index,
                       const std::array<scalar_t, numel> &values, const op_t &op) {
  auto *base_ptr = data + stride * index;
  for (int64_t k = 0; k < numel; ++k) {
    accumulate_result(base_ptr, stride, k, values[k], op);
  }
}

template <typename op_t, typename scalar_t>
void accumulate_result(char * C10_RESTRICT data, int64_t stride, int64_t index,
                       const Vec256<scalar_t> &values, const op_t &op) {
  if (C10_UNLIKELY(stride != sizeof(scalar_t))) {
    // Slow path, if out stride not contiguous split vector into an array
    std::array<scalar_t, values.size()> arr;
    values.store(arr.data());
    accumulate_result(data, stride, index, arr, op);
    return;
  }

  auto *base_ptr = data + stride * index;
  auto v = Vec256<scalar_t>::loadu(base_ptr);
  op.reduce(v, values).store(base_ptr);
}

template <typename F>
static inline void UNARY_OUTER_LOOP(char* data[2], const int64_t strides[2], int64_t n, F f) {
  for (int j = 0; j < n; j++) {
    f();
    data[0] += strides[0];
    data[1] += strides[1];
  }
}

template <typename T, typename op_t>
T row_reduce(const char * C10_RESTRICT in_data, const int64_t in_stride, const int64_t size, const op_t &op) {
  constexpr int ilp_factor = op_t::ilp_factor;
  // Interpret row as a (-1, ilp_factor) shaped array and find partial reductions
  const int64_t size_ilp = size / ilp_factor;
  auto partial_results = op.template multi_row_reduce<T>(
      in_data, in_stride * ilp_factor, in_stride, size_ilp);

  for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
    partial_results[0] = op.reduce(partial_results[0], load<T>(in_data, in_stride, i));
  }

  for (int64_t k = 1; k < ilp_factor; ++k) {
    partial_results[0] = op.reduce(partial_results[0], partial_results[k]);
  }

  return partial_results[0];
}



// computes the reduction out = op(out, in)
template <typename op_t>
static inline void vectorized_inner_reduction(
    char * C10_RESTRICT data[2], int64_t outer_stride, int64_t out_stride,
    int64_t size0, int64_t size1, const op_t &op) {
  using scalar_t = typename op_t::scalar_t;
  using vec_t = Vec256<scalar_t>;
  constexpr int64_t vec_stride = vec_t::size() * sizeof(scalar_t);
  const int64_t vec_size = size0 / vec_t::size();

  // Input is contiguous over the first (reduced) dimension
  for (int64_t j = 0; j < size1; ++j) {
    const auto *row_in = data[1] + j * outer_stride;
    auto vec_acc = row_reduce<vec_t>(row_in, vec_stride, vec_size, op);

    scalar_t final_acc = op.identity();
    for (int64_t k = vec_size * vec_t::size(); k < size0; ++k) {
      final_acc = op.reduce(final_acc, load<scalar_t>(row_in, sizeof(scalar_t), k));
    }

    scalar_t partials[vec_t::size()];
    vec_acc.store(partials);
    for (int64_t k = 0; k < vec_t::size(); ++k) {
      final_acc = op.reduce(final_acc, partials[k]);
    }
    accumulate_result(data[0], out_stride, j, final_acc, op);
  }
}

template <typename op_t>
void scalar_inner_reduction(
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1, const op_t &op) {
  using scalar_t = typename op_t::scalar_t;
  for (int64_t j = 0; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    scalar_t ans = row_reduce<scalar_t>(row_in, in_strides[0], size0, op);
    accumulate_result(data[0], out_stride, j, ans, op);
  }
}

// computes the reduction out = op(out, in)
template <typename op_t>
inline void vectorized_outer_reduction(
    char * C10_RESTRICT data[2], int64_t inner_stride, int64_t out_stride,
    int64_t size0, int64_t size1, const op_t &op) {
  using scalar_t = typename op_t::scalar_t;
  using vec_t = Vec256<scalar_t>;
  constexpr int64_t vec_stride = vec_t::size() * sizeof(scalar_t);
  constexpr int nrows = op_t::ilp_factor;

  // Input is contiguous over the second (non-reduced) dimension
  int64_t j = 0;
  for (; j + nrows * vec_t::size() <= size1; j += nrows * vec_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    auto res = op.template multi_row_reduce<vec_t>(
        row_in, inner_stride, vec_stride, size0);

    for (int64_t i = 0; i < nrows; ++i) {
      const int64_t base_idx = j + i * vec_t::size();
      accumulate_result(data[0], out_stride, base_idx, res[i], op);
    }
  }

  for (; j + vec_t::size() <= size1; j += vec_t::size()) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    const vec_t res = row_reduce<vec_t>(row_in, inner_stride, size0, op);
    accumulate_result(data[0], out_stride, j, res, op);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * sizeof(scalar_t);
    scalar_t ans = row_reduce<scalar_t>(row_in, inner_stride, size0, op);
    accumulate_result(data[0], out_stride, j, ans, op);
  }
}

template <typename op_t>
void scalar_outer_reduction(
    char * C10_RESTRICT data[2], int64_t in_strides[2], int64_t out_stride,
    int64_t size0, int64_t size1, const op_t &op) {
  using scalar_t = typename op_t::scalar_t;
  constexpr int nrows = op_t::ilp_factor;
  int64_t j = 0;
  for (; j + (nrows - 1) < size1; j += nrows) {
    const auto *row_in = data[1] + j * in_strides[1];
    auto results = op.template multi_row_reduce<scalar_t>(
        row_in, in_strides[0], in_strides[1], size0);
    accumulate_result(data[0], out_stride, j, results, op);
  }

  for (; j < size1; ++j) {
    const auto *row_in = data[1] + j * in_strides[1];
    scalar_t ans = row_reduce<scalar_t>(row_in, in_strides[0], size0, op);
    accumulate_result(data[0], out_stride, j, ans, op);
  }
}

template<typename traits, typename res_t>
static void set_result(const int index, const res_t result, const TensorIterator &iter, const int num_outputs) {
  // static_assert(std::is_same<res_t, typename traits::arg2_t>::value, "data types must match");
  if (index < num_outputs) {
    char *out = (char *) iter.data_ptr(index);
    *(res_t *) out = result;
  }
}

template<typename traits, typename res_t>
static void set_results(const res_t result, const TensorIterator &iter, const int num_outputs) {
  AT_ASSERT(num_outputs == 1);
  set_result<traits>(0, result, iter, num_outputs);
}

template<typename traits, std::size_t i = 0, typename... tuple_t>
static inline typename std::enable_if<i == sizeof...(tuple_t), std::size_t>::type
for_each_in_tuple(const std::tuple<tuple_t...>& t, const TensorIterator &iter, const int num_outputs) {
  return i;
}

template<typename traits, std::size_t i = 0, typename... tuple_t>
static inline typename std::enable_if<i < sizeof...(tuple_t), std::size_t>::type
for_each_in_tuple(const std::tuple<tuple_t...>& t, const TensorIterator &iter, const int num_outputs) {
  if (i < (size_t)num_outputs) {
    set_result<traits>(i, std::get<i>(t), iter, num_outputs);
    return for_each_in_tuple<traits, i + 1, tuple_t...>(t, iter, num_outputs);
  }
  return i;
}

template<typename traits, typename... res_t>
static void set_results(const std::tuple<res_t...>& result, const TensorIterator &iter, const int num_outputs) {
  AT_ASSERT(num_outputs >= 1);
  std::size_t result_size = for_each_in_tuple<traits>(result, iter, num_outputs);
  AT_ASSERT((size_t)num_outputs == result_size);
}

template <typename T, typename... Args>
struct all_same : guts::conjunction<
  std::is_same<T, Args>...
> {};

// data_t is the input/output data type.
// acc_t is a type that contains all the necessary data
// to continue reducing.
// index_t is a one-dimensional index
//
// ops_t is such that &ops_t::reduce, &ops_t::combine, and &ops_t::project exist and satisfy
// the following.
// reduce: (acc_t, data_t, index_t) -> acc_t adds one data point to the accumulated value.
// combine: (acc_t, acc_t) -> acc_t combines two accumulated values into one.
// project: acc_t -> out_t finishes the reduction, getting the required output.
//
// Additionally, acc_t must be default-constructible:
// acc_t {} is an identity for combine,
// and project(acc_t {}) is the value of the operation on zero elements.
//
// The point of `combine` is to support parallelization -
// the idea is to one sequence of `reduce` calls per thread of execution,
// and then to combine them at the end with `combine`.
//
// If there is more than one output element,
// our parallelization strategy is to use one thread for each of them,
// which means that `combine` will never be called.
//
// If, on the other hand, there is only one, then we split the input into
// into several pieces, reduce each separately, and then combine them.

template <typename ops_t, typename init_t>
void binary_kernel_reduce(TensorIterator& iter, ops_t ops, init_t init) {
  using rf_t = decltype(&ops_t::reduce);
  using cf_t = decltype(&ops_t::combine);
  using pf_t = decltype(&ops_t::project);
  using r_traits = binary_function_traits<rf_t>;
  using c_traits = binary_function_traits<cf_t>;
  using p_traits = unary_function_traits<pf_t>;
  using acc_t = typename p_traits::arg1_t;
  using data_t = typename r_traits::arg2_t;
  static_assert(
    all_same<
      acc_t,
      init_t,
      typename r_traits::arg1_t,
      typename r_traits::result_type,
      typename c_traits::arg1_t,
      typename c_traits::arg2_t,
      typename c_traits::result_type>::value,
    "all accumulate types must match");
  static_assert(
    std::is_default_constructible<acc_t>::value,
    "the accumulate type must be default-constructible"
  );
  const int num_outputs = iter.noutputs();
  iter.foreach_reduced_elt([&ops, &init, num_outputs](TensorIterator &sub_iter) {
    auto reduction_body = [&ops, &sub_iter, num_outputs](acc_t acc, int64_t begin, int64_t end) -> acc_t {
      int ntensors = sub_iter.ntensors();
      sub_iter.serial_for_each([&acc, &ops, num_outputs, ntensors, begin](char** data, const int64_t* strides, int64_t size) {
        AT_ASSERT(ntensors - num_outputs == 1);
        char *in = data[ntensors - 1];
        int64_t stride = strides[ntensors - 1];
        for (int64_t i = 0; i < size; ++i) {
          acc = ops.reduce(acc, *(data_t*)in, begin + i);
          in += stride;
        }
      }, {begin, end});
      return ops.translate_idx(acc, sub_iter.view_offsets()[0]);
    };
    acc_t total_acc = init;
    auto numel = sub_iter.numel();
    if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
        at::in_parallel_region()) {
      total_acc = reduction_body(total_acc, 0, numel);
    } else {
      int max_threads = at::get_num_threads();
      AT_ASSERT(max_threads > 0);
      static_assert(
        !std::is_same<acc_t, bool>::value,
        "Concurrently modifying different references into std::vector<bool> is UB."
      );
      std::vector<acc_t> buffer((unsigned)max_threads, init);
      at::parallel_for(0, numel, internal::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
          auto& acc = buffer[at::get_thread_num()];
          acc = reduction_body(acc, begin, end);
        }
      );
      for (int i = 0; i < max_threads; ++i) {
        total_acc = ops.combine(total_acc, buffer[i]);
      }
    }
    set_results<r_traits>(ops.project(total_acc), sub_iter, num_outputs);
  });
}


template <typename func_t, typename vec_func_t>
class SimpleVecReduce {
public:

  using scalar_t = typename binary_function_traits<func_t>::result_type;

  /** Determines the number of rows reduced by multi_row_reduce

   Since each row has a separate accumulator, each row reduction can be calculated
   in parallel on the CPU. i.e instruction level parallelism (ILP).
   Can be tuned to achieve best results with a given op.
  */
  static constexpr int ilp_factor = 4;

  SimpleVecReduce(func_t scalar_op, vec_func_t vector_op, scalar_t identity) :
    scalar_op_(scalar_op),
    vector_op_(vector_op),
    identity_(identity) {}

  scalar_t reduce(scalar_t a, scalar_t b) const {
    return scalar_op_(a, b);
  }

  Vec256<scalar_t> reduce(Vec256<scalar_t> a, Vec256<scalar_t> b) const {
    return vector_op_(a, b);
  }

  scalar_t identity() const {
    return identity_;
  }

  /** Simultaneously reduce ilp_factor rows of a 2d strided memory view

   \tparam T The element type, may be scalar_t or Vec256<scalar_t> for vector reductions
  */
  template <typename T>
  std::array<T, ilp_factor> multi_row_reduce(
      const char * C10_RESTRICT in_data,
      const int64_t row_stride,
      const int64_t col_stride,
      const int64_t size) const {
    std::array<T, ilp_factor> acc;
    acc.fill(T(identity()));

    for (int64_t i = 0; i < size; ++i) {
      const char * column_base = in_data + i * row_stride;
      #pragma unroll
      for (int64_t k = 0; k < ilp_factor; ++k) {
        acc[k] = reduce(acc[k], load<T>(column_base, col_stride, k));
      }
    }

    return acc;
  }

private:
  func_t scalar_op_;
  vec_func_t vector_op_;
  scalar_t identity_;
};

template <typename func_t, typename vec_func_t,
          typename scalar_t = typename binary_function_traits<func_t>::result_type>
SimpleVecReduce<func_t, vec_func_t> simple_vec_reduce(
    func_t op, vec_func_t vop, scalar_t identity=0) {
  return {op, vop, identity};
}


template <typename op_t>
void binary_kernel_reduce_vec(TensorIterator& iter, const op_t &op) {
  using scalar_t = typename op_t::scalar_t;

  iter.output().fill_(op.identity());
  iter.parallel_reduce([&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
    int64_t in_strides[] = { strides[1], strides[3] };
    int64_t out_strides[] = { strides[0], strides[2] };

    // Move reduction to be the 1st dim
    if (out_strides[0] != 0 && out_strides[1] == 0) {
      std::swap(in_strides[0], in_strides[1]);
      std::swap(out_strides[0], out_strides[1]);
      std::swap(size0, size1);
    }

    // Special case? - not a true reduction
    if (out_strides[0] != 0 && out_strides[1] != 0) {
      int64_t outer_strides[] = { strides[2], strides[3] };
      UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
        char* ptrs[3] = { data[0], data[0], data[1] };
        int64_t inner_strides[3] = { strides[0], strides[0], strides[1] };
        basic_loop(ptrs, inner_strides, 0, size0,
                   [&](scalar_t a, scalar_t b) { return op.reduce(a, b); });
      });
      return;
    }

    const int64_t out_stride = out_strides[1];
    TORCH_INTERNAL_ASSERT(out_strides[0] == 0);

    if (in_strides[0] == sizeof(scalar_t) && size0 >= Vec256<scalar_t>::size()) {
      // Contiguous inner reduction
      vectorized_inner_reduction(data, in_strides[1], out_stride, size0, size1, op);
    } else if (in_strides[1] == sizeof(scalar_t) && size1 >= Vec256<scalar_t>::size()) {
      // Contiguous outer reduction
      vectorized_outer_reduction(data, in_strides[0], out_stride, size0, size1, op);
    } else if (in_strides[0] < in_strides[1]) {
      scalar_inner_reduction(data, in_strides, out_stride, size0, size1, op);
    } else {
      scalar_outer_reduction(data, in_strides, out_stride, size0, size1, op);
    }
  });
}

}}}  // namespace at::native::<anonymous>
