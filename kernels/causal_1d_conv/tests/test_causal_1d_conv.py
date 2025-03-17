# Copyright (C) 2025, IBM Research.
# python -m pytest tests/test_causal_conv1d.py

import sys
from einops import rearrange
import pytest
import torch.nn.functional as F
import torch
import math

import os
from pathlib import Path

base_path = Path(os.path.abspath(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, str(base_path / "../causal_1d_conv"))

try:
    from causal_1d_conv import causal_conv1d_fn
except ImportError:
    raise


def _undecorated_test_causal_conv1d(
    batch,
    dim,
    seqlen,
    width,
    has_bias,
    silu_activation,
    itype,
    channel_last,
    has_initial_states,
    return_final_states,
    check_backward,
):
    if not channel_last and (has_initial_states or return_final_states):
        pytest.skip("Only channel_last support initial_states or return_final_states")
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[
            :, 4096: 4096 + dim, :
        ].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096: 4096 + dim],
            "b s d -> b d s",
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    if has_initial_states:
        initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
    else:
        initial_states = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    initial_states_ref = initial_states.detach().clone().requires_grad_() if initial_states is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(
        x, weight, bias, initial_states=initial_states, return_final_states=return_final_states, activation=activation
    )
    out_ref = causal_conv1d_ref(
        x_ref,
        weight_ref,
        bias_ref,
        initial_states=initial_states_ref,
        return_final_states=return_final_states,
        activation=activation,
    )
    if return_final_states:
        out, final_states = out
        out_ref, final_states_ref = out_ref
        print(f"Final states max diff: {(final_states - final_states_ref).abs().max().item()}")
        print(f"Final states mean diff: {(final_states - final_states_ref).abs().mean().item()}")
        assert torch.allclose(final_states, final_states_ref, rtol=rtol, atol=atol)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    if return_final_states:
        out += F.sigmoid(final_states).sum(dim=-1, keepdim=True)
        out_ref += F.sigmoid(final_states_ref).sum(dim=-1, keepdim=True)

    if check_backward:
        g = torch.randn_like(out)
        out.backward(g)
        out_ref.backward(g)

        print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
        print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
        if has_bias:
            print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")
        if has_initial_states:
            print(f"dinitial_states max diff: {(initial_states.grad - initial_states_ref.grad).abs().max().item()}")

        assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
        assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
        if has_bias:
            assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)
        if has_initial_states:
            assert torch.allclose(initial_states.grad, initial_states_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    torch.cuda.empty_cache()
    del x_ref, x, weight, weight_ref, bias, bias_ref, out, out_ref


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """[copied from causal_conv1d/causal_conv1d_interface.py]
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


@pytest.mark.parametrize("batch", [1, 2, 3, 8, 16, 32, 64])  # END-GOAL
# @pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("dim", [64, 4096 + 32])  # END-GOAL
# @pytest.mark.parametrize('dim', [64])
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 512, 784, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize(
    "seqlen", [1, 2, 8, 16, 32, 64, 128, 129, 130, 151, 256, 372, 512, 784, 1024, 1134, 2048, 4096]
)  # END-GOAL
@pytest.mark.parametrize("width", [2, 3, 4, 5])  # END-GOAL
# @pytest.mark.parametrize('width', [3])
@pytest.mark.parametrize("has_bias", [False, True])  # END-GOAL
# @pytest.mark.parametrize('has_bias', [True])
# @pytest.mark.parametrize('has_bias', [False])
@pytest.mark.parametrize("silu_activation", [False, True])  # END-GOAL
# @pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
# @pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("channel_last", [True])  # END-GOAL
@pytest.mark.parametrize("has_initial_states", [False, True])  # END-GOAL
# @pytest.mark.parametrize("has_initial_states", [False])
# @pytest.mark.parametrize("return_final_states", [False, True]) # END-GOAL
@pytest.mark.parametrize("return_final_states", [False])
# @pytest.mark.parametrize('check_backward', [True]) # END-GOAL
@pytest.mark.parametrize("check_backward", [False])
def test_causal_conv1d(
    batch,
    dim,
    seqlen,
    width,
    has_bias,
    silu_activation,
    itype,
    channel_last,
    has_initial_states,
    return_final_states,
    check_backward,
):
    return _undecorated_test_causal_conv1d(
        batch,
        dim,
        seqlen,
        width,
        has_bias,
        silu_activation,
        itype,
        channel_last,
        has_initial_states,
        return_final_states,
        check_backward,
    )
