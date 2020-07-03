import operator_benchmark as op_bench
import torch

"""Microbenchmarks for reduction operators."""

# Configs for PT add operator
reduce_configs = op_bench.cross_product_configs(
    R=[64, 256],  # Length of reduced dimension
    V=[32, 512],  # Length of other dimension
    dim=[0, 1],
    contiguous=[True, False],
    device=['cpu', 'cuda'],
    tags=['short']
) + op_bench.cross_product_configs(
    R=[1024, 8192],
    V=[512, 1024],
    dim=[0, 1],
    contiguous=[True, False],
    device=['cpu', 'cuda'],
    tags=['long']
)


class ReduceBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, R, V, dim, contiguous, device):
        shape = (R, V) if dim == 0 else (V, R)
        tensor = torch.rand(shape, device=device).to(self.dtype)

        if not contiguous:
            storage = torch.empty(size=[s * 2 for s in shape],
                                  device=tensor.device, dtype=self.dtype)
            storage[::2, ::2] = tensor
            self.input_tensor = storage[::2, ::2]
        else:
            self.input_tensor = tensor

        self.dim = dim
        self.set_module_name(self.operator.__name__)

    def forward(self):
        return self.operator(self.input_tensor, dim=self.dim)

class SumBenchmark(ReduceBenchmark):
    def __init__(self, *args, **kwargs):
        self.operator = torch.sum
        self.dtype = torch.float
        super().__init__(*args, **kwargs)

class ProdBenchmark(ReduceBenchmark):
    def __init__(self, *args, **kwargs):
        self.operator = torch.prod
        self.dtype = torch.float
        super().__init__(*args, **kwargs)

class AllBenchmark(ReduceBenchmark):
    def __init__(self, *args, **kwargs):
        self.operator = torch.all
        self.dtype = torch.bool
        super().__init__(*args, **kwargs)

class AnyBenchmark(ReduceBenchmark):
    def __init__(self, *args, **kwargs):
        self.operator = torch.any
        self.dtype = torch.bool
        super().__init__(*args, **kwargs)

class MinBenchmark(ReduceBenchmark):
    def __init__(self, *args, **kwargs):
        self.operator = torch.min
        self.dtype = torch.float
        super().__init__(*args, **kwargs)

class MaxBenchmark(ReduceBenchmark):
    def __init__(self, *args, **kwargs):
        self.operator = torch.max
        self.dtype = torch.float
        super().__init__(*args, **kwargs)

op_bench.generate_pt_test(reduce_configs, SumBenchmark)
op_bench.generate_pt_test(reduce_configs, ProdBenchmark)
op_bench.generate_pt_test(reduce_configs, AllBenchmark)
op_bench.generate_pt_test(reduce_configs, AnyBenchmark)
op_bench.generate_pt_test(reduce_configs, MinBenchmark)
op_bench.generate_pt_test(reduce_configs, MaxBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
