import functools
import logging
from typing import cast, List, Tuple

import sympy

import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2

log = logging.getLogger(__name__)

from ..mm_common import triton_config, acc_type, addmm_epilogue


def filtered_configs(
    m: int,
    n: int,
    k: int,
    configs: List[Tuple[int, int, int]],
    has_int8_tensor=False,
):
    """Heuristic to shrink configs when they are bigger than the input size"""

    min_block_size = 16
    m = max(next_power_of_2(V.graph.sizevars.size_hint(m)), min_block_size)
    k = max(next_power_of_2(V.graph.sizevars.size_hint(k)), min_block_size)
    used = set()
    for block_m, block_k, num_warps in configs:
        # shrink configs for small sizes
        block_m = max(min(block_m, m), min_block_size)
        block_k = max(min(block_k, k), min_block_size)
        # each warp computes 16x16 tile = 256
        num_warps = min(num_warps, block_m * block_n // 256)
        if (block_m, block_n, block_k, num_stages, num_warps) not in used:
            used.add((block_m, block_n, block_k, num_stages, num_warps))
            yield triton_config(
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                num_stages=1,
                num_warps=num_warps,
            )


# List of dictionaries to store the kernel configs. Configs that evaluate to true
# will be utilised on the target platform
mv_kernel_configs = [
    # "BLOCK_M", "BLOCK_K", "num_warps"
    {"config": (1, 512, 4), "cond": True},
    {"config": (1, 1024, 8), "cond": True},
    {"config": (2, 256, 4), "cond": True},
    {"config": (4, 128, 4), "cond": True},
    {"config": (8, 64, 4), "cond": True},
    {"config": (16, 32, 4), "cond": True},
    {"config": (16, 64, 4), "cond": True},
    {"config": (32, 32, 4), "cond": True},
    {"config": (32, 64, 4), "cond": True},
    {"config": (64, 16, 4), "cond": True},
    {"config": (64, 32, 4), "cond": True},
    {"config": (128, 32, 4), "cond": True},
    {"config": (64, 32, 8), "cond": True},
    {"config": (128, 32, 8), "cond": True},
]

# Create filtered list of configs based on cond evaluation
mv_platform_configs = tuple(
    cast(Tuple[int, int, int], config["config"])
    for config in mv_kernel_configs
    if config["cond"]
)

mv_configs = functools.partial(
    filtered_configs,
    configs=mm_platform_configs,
)


def mv_grid(m, meta):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta["BLOCK_M"]), 1, 1)


def mm_options(config, sym_k, layout, b_prologue_cast_type=None):
    """
    Common options to matmul triton templates.
    """
    even_k_symbolic = (
        # it isn't worth guarding on this
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"])
        == config.kwargs["BLOCK_K"]
    )
    return dict(
        GROUP_M=8,
        EVEN_K=even_k_symbolic,
        ACC_TYPE=acc_type(layout.dtype),
        B_PROLOGUE_CAST_TYPE=b_prologue_cast_type,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        **config.kwargs,
    )


def mv_args(mat, vec, *others, layout=None, out_dtype=None, use_4x2_dim=False):
    """
    Common arg processing for mv,addmv,etc
    """
    mat, vec = realize_inputs(mat, vec)
    m, k1 = mat.get_size()
    k2 = vec.get_size()
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()
        layout = FixedLayout(
            mat.get_device(),
            out_dtype,
            [m],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    from ..lowering import expand

    others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, k, layout, mat, vec, *others]


def addmv_epilogue(dtype, alpha, beta):
    return addmm_epilogue(dtype, alpha, beta)
