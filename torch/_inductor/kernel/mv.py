import logging

import torch

from .. import config as inductor_config
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
    realize_inputs
)
from ..utils import use_aten_gemm_kernels, use_triton_template
from .mv_common import (
    addmv_epilogue,
    mv_args,
    mv_configs,
    mv_grid,
    mv_options,
)

log = logging.getLogger(__name__)
aten = torch.ops.aten

mv_template = TritonTemplate(
    name="mv",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    K = {{size("A", 1)}}
    if M == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = 1

    # re-order program ID for better L2 performance
    width = GROUP_M
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + rk[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += a * b
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    sum = tl.sum(acc, 1)

    # rematerialize rm to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_m = rm[:, None]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m",), "acc", "mask")}}
""",
)

aten_mv = ExternKernelChoice(torch.mv, "at::mv_out")
aten_addmv = ExternKernelChoice(torch.addmv, "at::addmv_out")


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


@register_lowering(aten.mv)
def tuned_mv(mat, vec, *, layout=None):
    m, k, layout, mat1, vec = mv_args(mat, vec, layout=layout)

    # options to tune from
    choices = [aten_mv.bind((mat, vec), layout)]
    if m != 0 and use_triton_template(layout):
        for config in mv_configs(m, k):
            mm_template.maybe_append_choice(
                choices,
                (mat1, mat2),
                layout,
                **mm_options(config, k, layout),
            )

    return autotune_select_algorithm("mm", choices, [mat1, mat2], layout)


@register_lowering(aten.addmv)
def tuned_addmv(inp, mat, vec, *, alpha=1, beta=1, layout=None):
    ordered_kwargs_for_cpp_kernel = ("beta", "alpha")

    m, k, layout, mat, vec, inp_expanded = mv_args(mat, vec, inp, layout=layout)
    choices = [
        aten_addmv.bind(
            (inp, mat, vec),
            layout,
            ordered_kwargs_for_cpp_kernel,
            alpha=alpha,
            beta=beta,
        )
    ]
    if m == 0 or not use_triton_template(layout):
        return autotune_select_algorithm("addmm", choices, [inp, mat, vec], layout)

    for config in mv_configs(m, k):
        mv_template.maybe_append_choice(
            choices,
            (inp_expanded, mat, vec),
            layout,
            **mv_options(config, k, layout),
            prefix_args=1,
            epilogue_fn=addmv_epilogue(layout.dtype, alpha, beta),
        )

    return autotune_select_algorithm(
        "addmv", choices, [inp_expanded, mat, vec], layout
    )
