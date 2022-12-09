import torch.library
from torch.autograd import Function

_test_lib_def = torch.library.Library("_inductor_test", "DEF")
_test_lib_def.define("realize(Tensor self) -> Tensor")

for device in ("CPU", "CUDA", "Meta"):
    _test_lib_impl = torch.library.Library("_inductor_test", device)
    _test_lib_impl.impl("realize", lambda x: x.clone())

class Realize(Function):
    @staticmethod()
    def forward(ctx, x):
        return torch.ops._inductor_test.realize(x)

    @staticmethod()
    def backward(ctx, grad_output):
        return grad_output

realize = Realize.apply
