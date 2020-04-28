#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_sigmoid(const Tensor& self) {
  AT_ERROR("mkldnn_sigmoid: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  AT_ERROR("mkldnn_sigmoid_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_sigmoid(const Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  at::internal::lazy_init_num_threads();
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  at::internal::lazy_init_num_threads();
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
