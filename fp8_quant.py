import functools
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from loguru import logger

# from .quantizer import QuantizerMixin

E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max

FP16_MAX_POS = torch.finfo(torch.float16).max

EPS = 1e-12


from abc import ABC, abstractmethod

import torch


class QuantizerMixin(ABC):
    @abstractmethod
    def find_scale_zero(self, w: torch.Tensor):
        """
        Determines scale and zero-point for quantization.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes input tensor.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def pseudo_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies pseudo quantization (forward pass simulation).
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def n_bit(self):
        """
        Returns the number of bits used for quantization.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def maxq(self):
        """
        Returns the maximum quantization level.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def scales(self):
        """
        Returns the scale factors used for quantization.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def zeros(self):
        """
        Returns the zero points used for quantization.
        Must be implemented by subclasses.
        """
        pass

    @property
    @abstractmethod
    def symm_q(self):
        """
        Indicates whether symmetric quantization is used.
        Must be implemented by subclasses.
        """
        pass


class Quantizer(QuantizerMixin):
    """
    RTN(round-to-nearest) Quantizer for weight quantization.
    This is a simple baseline quantizer.
    """

    def __init__(
        self,
        n_bit: int,
        *,
        per_tensor: bool = False,
        symm_q: bool = False,
        group_size: int = 128,
        zeropoint: bool = True,
        mse: bool = False,
        norm=2.4,
        grid=100,
        max_shrink=0.8,
    ):
        self._n_bit = n_bit
        self._group_size = group_size
        self._symm_q = symm_q
        self._per_tensor = per_tensor
        self._zeropoint = zeropoint

        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.max_shrink = max_shrink

        if not symm_q:
            assert zeropoint, "asymm quantization must have zeropoint"
            self._maxq = 2**n_bit - 1
        else:
            self._maxq = 2 ** (n_bit - 1) - 1

    def _find_scale_zero_per_tensor(self, w: torch.Tensor):
        assert (
            self._symm_q
        ), "per-tensor quantization only support symmetric quantization right now!"
        scales = w.abs().max().float() / self._maxq

        self._scales = scales

    def _find_scale_zero_per_channel(self, w: torch.Tensor):
        # w: c_out, c_in
        org_shape = w.shape
        device = w.device

        if self._group_size > 0:
            assert w.shape[1] % self._group_size == 0
            w = w.view(-1, self._group_size)

        tmp = torch.zeros(w.shape[0], dtype=w.dtype, device=device)
        xmin = torch.minimum(w.amin(1), tmp)
        xmax = torch.maximum(w.amax(1), tmp)

        if self._symm_q:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        zero_range = (xmin == 0) & (xmax == 0)
        xmin[zero_range] = -1
        xmax[zero_range] = +1

        if self._symm_q:
            scales = xmax.float() / self._maxq
            zeros = torch.full_like(scales, self._maxq + 1)
        else:
            scales = (xmax - xmin).clamp(1e-5).float() / self._maxq
            zeros = torch.round(-xmin / scales).clamp(0, self._maxq)

        shape = [org_shape[0]] + [-1]
        self._scales = scales.reshape(shape)
        self._zeros = zeros.reshape(shape)

    def find_scale_zero(self, w):
        if self._per_tensor:
            self._find_scale_zero_per_tensor(w)
        else:
            self._find_scale_zero_per_channel(w)

    @property
    def n_bit(self):
        return self._n_bit

    @property
    def maxq(self):
        return self._maxq

    @property
    def group_size(self):
        return self._group_size

    @property
    def scales(self):
        return self._scales

    @property
    def zeros(self):
        return self._zeros

    @property
    def symm_q(self):
        return self._symm_q

    def quantize(self, x: torch.Tensor):
        if not hasattr(self, "_scales"):
            self.find_scale_zero(x)

        if self._per_tensor:
            return self._quantize_per_tensor(x)
        else:
            return self._quantize_per_channel(x)

    def _quantize_per_tensor(self, x: torch.Tensor):
        min_int = -self._maxq - 1
        max_int = self._maxq
        return torch.clamp(x.div(self._scales).round(), min_int, max_int)

    def _pseudo_quantize_per_tensor(self, x: torch.Tensor):
        q = self._quantize_per_tensor(x)
        return q * self._scales

    def _quantize_per_channel(self, x: torch.Tensor):
        org_shape = x.shape
        if self._group_size > 0:
            assert x.shape[1] % self._group_size == 0
            assert self._scales.shape[1] == x.shape[1] // self._group_size
            x = x.view(-1, self._group_size)
            scales = self._scales.view(-1, 1)
            zeros = self._zeros.view(-1, 1)
        else:
            scales = self._scales
            zeros = self._zeros
        if self._zeropoint:
            max_int = 2 * self._maxq + 1 if self._symm_q else self._maxq
            q = torch.clamp(torch.round(x / scales) + zeros, 0, max_int)
        else:
            q = torch.clamp(torch.round(x / scales), -self._maxq - 1, self._maxq)
        return q.view(org_shape)

    def _pseudo_quantize_per_channel(self, x: torch.Tensor):
        q = self._quantize_per_channel(x)
        org_shape = q.shape
        if self._group_size > 0:
            q = q.view(-1, self._group_size)
            scales = self._scales.view(-1, 1)
            zeros = self._zeros.view(-1, 1)
        else:
            scales = self._scales
            zeros = self._zeros
        dq = scales * (q - zeros) if self._zeropoint else scales * q
        return dq.view(org_shape)

    def pseudo_quantize(self, x: torch.Tensor):
        if not hasattr(self, "_scales"):
            self.find_scale_zero(x)

        if self._per_tensor:
            return self._pseudo_quantize_per_tensor(x)
        else:
            return self._pseudo_quantize_per_channel(x)

# NOTE(xingyu): This function is essential for obtaining fp8 quantized linear values.
# Since the fp8 quantization scale is a scalar,
# when combining two linear values, it's crucial to ensure they share the same scale.
# This function facilitates the creation of a structure that can determine
# if the linear values have been combined, and exclude the not quantized linears.
def get_linears_for_fp8_scale(layer: nn.Module):
    layer_name = layer.__class__.__name__

    if layer_name == "LlamaDecoderLayer":
        layer_to_scale = _get_llama_linears_for_fp8_scale(layer)
    elif layer_name == "MixtralDecoderLayer":
        layer_to_scale = _get_mixtral_linears_for_fp8_scale(layer)
    elif layer_name == "Qwen2DecoderLayer":
        layer_to_scale = _get_qwen2_linears_for_fp8_scale(layer)
    elif layer_name == "Qwen2MoeDecoderLayer":
        layer_to_scale = _get_qwen2moe_linears_for_fp8_scale(layer)
    elif layer_name == "TelechatBlock":
        layer_to_scale = _get_telechat_linears_for_fp8_scale(layer)
    elif layer_name in ["BaichuanLayer", "DecoderLayer"]:
        layer_to_scale = _get_baichuan1_linears_for_fp8_scale(layer)
    elif layer_name == "GLMBlock":
        layer_to_scale = _get_glm3_linears_for_fp8_scale(layer)
    elif layer_name == "QWenBlock":  # qwen1
        layer_to_scale = _get_qwen1_linears_for_fp8_scale(layer)
    elif layer_name == "DeepseekV2DecoderLayer":
        layer_to_scale = _get_deepseekv2_linears_for_fp8_scale(layer)
    elif layer_name in ["GemmaDecoderLayer", "Gemma2DecoderLayer"]:
        layer_to_scale = _get_gemma_linears_for_fp8_scale(layer)
    else:
        raise NotImplementedError(f"Unsupported layer for FP8 quantization right now: {layer_name}")

    return layer_to_scale


@torch.inference_mode()
def amax_to_scale(amax, float8_dtype, orig_dtype):
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == torch.float8_e4m3fn:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=FP16_MAX_POS)
    scale.copy_(res)
    return scale


class FP8Quantizer(QuantizerMixin):
    def __init__(self, *, fp8_dtype: torch.device = torch.float8_e4m3fn, per_tensor: bool = True):
        assert fp8_dtype in {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }, f"Unsupported fp8_dtype: {fp8_dtype}"
        self._fp8_dtype = fp8_dtype
        self._per_tensor = per_tensor

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_scales"):
            self.find_scale_zero(x)

        # clamp the tensor to bring it to the representative range of float8 data type
        # (as default cast is unsaturated)
        x_fp8_sat = (x * self._scales).clamp(min=-self.maxq, max=self.maxq)
        return x_fp8_sat.to(self._fp8_dtype)

    def pseudo_quantize(self, x: torch.Tensor) -> torch.Tensor:
        x_fp8_sat = self.quantize(x)
        return x_fp8_sat * (1 / self._scales)

    def find_scale_zero(self, x: torch.Tensor):
        # w: c_out, c_in
        amax = x.abs().max() if self._per_tensor else x.abs().amax(dim=1)
        self._scales = amax_to_scale(amax, self._fp8_dtype, x.dtype)

    @property
    def n_bit(self):
        return 8

    @property
    def maxq(self):
        return E4M3_MAX_POS if self._fp8_dtype == torch.float8_e4m3fn else E5M2_MAX_POS

    @property
    def scales(self):
        return self._scales

    @property
    def zeros(self):
        return 0

    @property
    def symm_q(self):
        return True


@torch.inference_mode()
def get_activation_scales_for_fp8(
    fp8_dtype: torch.dtype,
    layer: nn.Module,
    linears_to_scale: List[Dict],
    all_inps: List[torch.Tensor],
    kwargs: Optional[Dict] = None,
):
    if kwargs is None:
        kwargs = {}
    act_amax_dict = defaultdict(dict)

    def stat_io_hook(_, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if x.numel() == 0:
            return
        if name not in act_amax_dict or "input" not in act_amax_dict[name]:
            act_amax_dict[name]["input"] = x.abs().max()
        else:
            act_amax_dict[name]["input"] = max(act_amax_dict[name]["input"], x.abs().max())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_amax_dict or "output" not in act_amax_dict[name]:
            act_amax_dict[name]["output"] = y.abs().max()
        else:
            act_amax_dict[name]["output"] = max(act_amax_dict[name]["output"], y.abs().max())

    hooks = []
    for layer_info in linears_to_scale:
        linears = layer_info["linears"]
        if not isinstance(linears, list):
            linears = [linears]
        names = layer_info["names"]
        if not isinstance(names, list):
            names = [names]
        for n, m in zip(names, linears):
            hooks.append(m.register_forward_hook(functools.partial(stat_io_hook, name=n)))

    # mean input scales for all linears in this layer
    mean_scale = []
    for i in range(len(all_inps)):
        layer(all_inps[i : i + 1], **kwargs)
        mean_scale.append(np.mean([v["input"].cpu().item() for v in act_amax_dict.values()]))
    mean_scale = np.mean(mean_scale)

    for hook in hooks:
        hook.remove()

    scale_dict = {
        f"{key}_input": amax_to_scale(amax["input"], fp8_dtype, amax["input"].dtype)
        for key, amax in act_amax_dict.items()
    }

    # NOTE(xingyu): for packed linears, set the input scales to the minimum one
    for linear2scale in linears_to_scale:
        linear_names = linear2scale["names"]
        if isinstance(linear_names, list) and len(linear_names) > 1:
            scale_list = [
                scale_dict[f"{name}_input"]
                for name in linear_names
                if f"{name}_input" in scale_dict  # in case expert is not assigned tokens
            ]
            if len(scale_list) == 0:
                logger.warning(
                    f"The linear {linear_names[0]} was not found in scale_dict. This may occur in "
                    f"MoE models if the expert containing this linear did not assign "
                    f"any tokens, resulting in the setting the mean scale for this "
                    f"linear. To avoid this, consider adjusting the calibration set to "
                    f"ensure all experts are assigned tokens."
                )
                scale = torch.mean(torch.stack([v["input"] for v in act_amax_dict.values()]))
            else:
                scale = torch.min(torch.stack(scale_list))
            for name in linear_names:
                scale_dict[f"{name}_input"] = scale

    return mean_scale, scale_dict


@torch.inference_mode()
def get_weight_scales_for_fp8(
    linears_to_scale: Dict[str, nn.Module],
    fp8_dtype: torch.device = torch.float8_e4m3fn,
    weight_per_tensor: bool = True,
):
    for layer_info in linears_to_scale:
        linears = layer_info["linears"]
        if not isinstance(linears, list):
            linears = [linears]

        weight = torch.cat([_m.weight for _m in linears], dim=0)  # c_out, c_in

        quantizer = FP8Quantizer(fp8_dtype=fp8_dtype, per_tensor=weight_per_tensor)
        quantizer.find_scale_zero(weight)

        for linear in linears:
            linear.quantizer = quantizer


def _get_llama_linears_for_fp8_scale(layer):
    return [
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            names=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ),
        dict(
            prev_op=layer.self_attn.v_proj,
            linears=[layer.self_attn.o_proj],
            names=["self_attn.o_proj"],
        ),
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
            names=["mlp.gate_proj", "mlp.up_proj"],
        ),
        dict(prev_op=layer.mlp.up_proj, linears=[layer.mlp.down_proj], names=["mlp.down_proj"]),
    ]


def _get_mixtral_linears_for_fp8_scale(layer):
    # self-attn
    linears = [
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            names=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ),
        dict(
            prev_op=layer.self_attn.v_proj,
            linears=[layer.self_attn.o_proj],
            names=["self_attn.o_proj"],
        ),
    ]

    num_experts = layer.block_sparse_moe.num_experts
    for i in range(num_experts):
        expert = layer.block_sparse_moe.experts[i]

        linears.append(
            dict(
                prev_op=layer.post_attention_layernorm,
                linears=[expert.w3, expert.w1],
                names=[f"block_sparse_moe.experts.{i}.w3", f"block_sparse_moe.experts.{i}.w1"],
            )
        )

        linears.append(
            dict(prev_op=expert.w3, linears=expert.w2, names=f"block_sparse_moe.experts.{i}.w2")
        )
    return linears


def _get_qwen1_linears_for_fp8_scale(layer):
    return [
        dict(
            prev_op=layer.ln_1,
            linears=[layer.attn.c_attn],
            names=["attn.c_attn"],
        ),
        dict(
            prev_op=layer.attn.c_attn,
            linears=[layer.attn.c_proj],
            names=["attn.c_proj"],
        ),
        dict(
            prev_op=layer.ln_2,
            linears=[layer.mlp.w2, layer.mlp.w1],
            names=["mlp.w2", "mlp.w1"],
        ),
        dict(
            prev_op=layer.mlp.w1,
            linears=[layer.mlp.c_proj],
            names=["mlp.c_proj"],
        ),
    ]


def _get_qwen2_linears_for_fp8_scale(layer):
    return [
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            names=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ),
        dict(
            prev_op=layer.self_attn.v_proj,
            linears=[layer.self_attn.o_proj],
            names=["self_attn.o_proj"],
        ),
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
            names=["mlp.gate_proj", "mlp.up_proj"],
        ),
        dict(prev_op=layer.mlp.up_proj, linears=[layer.mlp.down_proj], names=["mlp.down_proj"]),
    ]


def _get_telechat_linears_for_fp8_scale(layer):
    return [
        dict(
            prev_op=layer.input_layernorm,
            linears=[layer.self_attention.query, layer.self_attention.key_value],
            names=["self_attention.query", "self_attention.key_value"],
        ),
        dict(
            prev_op=layer.self_attention.key_value,
            linears=[layer.self_attention.dense],
            names=["self_attention.dense"],
        ),
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
            names=["mlp.gate_proj", "mlp.up_proj"],
        ),
        dict(
            prev_op=layer.mlp.up_proj,
            linears=[layer.mlp.down_proj],
            names=["mlp.down_proj"],
        ),
    ]


def _get_baichuan1_linears_for_fp8_scale(layer):
    return [
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.W_pack,
            ],
            names=["self_attn.W_pack"],
        ),
        dict(
            prev_op=layer.self_attn.W_pack,
            linears=[layer.self_attn.o_proj],
            names=["self_attn.o_proj"],
        ),
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
            names=["mlp.gate_proj", "mlp.up_proj"],
        ),
        dict(prev_op=layer.mlp.up_proj, linears=[layer.mlp.down_proj], names=["mlp.down_proj"]),
    ]


def _get_glm3_linears_for_fp8_scale(layer):
    return [
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attention.query_key_value,
            ],
            names=["self_attention.query_key_value"],
        ),
        dict(
            prev_op=layer.self_attention.query_key_value,
            linears=[layer.self_attention.dense],
            names=["self_attention.dense"],
        ),
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[layer.mlp.dense_h_to_4h],
            names=["mlp.dense_h_to_4h"],
        ),
        dict(
            prev_op=layer.mlp.dense_h_to_4h,
            linears=[layer.mlp.dense_4h_to_h],
            names=["mlp.dense_4h_to_h"],
        ),
    ]


def _get_deepseekv2_linears_for_fp8_scale(layer):
    # self-attn
    linears = [
        # bmm, split kv_b_proj -> k_b_proj, v_b_proj
        dict(
            prev_op=layer.self_attn.kv_a_layernorm,
            linears=[layer.self_attn.k_b_proj],
            names=["self_attn.k_b_proj"],
        ),
        dict(
            linears=[layer.self_attn.v_b_proj],
            names=["self_attn.v_b_proj"],
        ),
        dict(
            prev_op=layer.self_attn.v_b_proj,
            linears=[layer.self_attn.o_proj],
            names=["self_attn.o_proj"],
        ),
    ]

    if hasattr(layer.self_attn, "q_proj"):
        # deepseek-v2-lite
        linears.extend([
            dict(
                prev_op=layer.input_layernorm,
                linears=[layer.self_attn.q_proj],
                names=["self_attn.q_proj"],
            ),
            dict(
                prev_op=layer.input_layernorm,
                linears=[layer.self_attn.kv_a_proj_with_mqa],
                names=["self_attn.kv_a_proj_with_mqa"],
            ),
        ])
    else:
        linears.extend([
            dict(
                prev_op=layer.input_layernorm,
                linears=[
                    layer.self_attn.q_a_proj,
                    layer.self_attn.kv_a_proj_with_mqa,
                ],
                names=["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"],
            ),
            dict(
                prev_op=layer.self_attn.q_a_layernorm,
                linears=[layer.self_attn.q_b_proj],
                names=["self_attn.q_b_proj"],
            ),
        ])

    # dense-mlp for the first layer
    if hasattr(layer.mlp, "gate_proj"):
        linears.append(
            dict(
                prev_op=layer.post_attention_layernorm,
                linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
                names=["mlp.gate_proj", "mlp.up_proj"],
            )
        )
        linears.append(
            dict(prev_op=layer.mlp.up_proj, linears=[layer.mlp.down_proj], names=["mlp.down_proj"]),
        )

    # experts
    if hasattr(layer.mlp, "experts"):
        for i in range(len(layer.mlp.experts)):
            linears.append(
                dict(
                    prev_op=layer.post_attention_layernorm,
                    linears=[
                        layer.mlp.experts[i].up_proj,
                        layer.mlp.experts[i].gate_proj,
                    ],
                    names=[f"mlp.experts.{i}.up_proj", f"mlp.experts.{i}.gate_proj"],
                )
            )

            linears.append(
                dict(
                    prev_op=[layer.mlp.experts[i].up_proj],
                    linears=[layer.mlp.experts[i].down_proj],
                    names=[f"mlp.experts.{i}.down_proj"],
                )
            )

    # shared experts
    if hasattr(layer.mlp, "shared_experts"):
        linears.append(
            dict(
                linears=[layer.mlp.shared_experts.gate_proj, layer.mlp.shared_experts.up_proj],
                names=["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
            )
        )
        linears.append(
            dict(
                prev_op=layer.mlp.shared_experts.up_proj,
                linears=[layer.mlp.shared_experts.down_proj],
                names=["mlp.shared_experts.down_proj"],
            ),
        )

    return linears


def _get_qwen2moe_linears_for_fp8_scale(layer):
    # self-attn
    linears = [
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            names=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ),
        dict(
            prev_op=layer.self_attn.v_proj,
            linears=[layer.self_attn.o_proj],
            names=["self_attn.o_proj"],
        ),
    ]

    num_experts = layer.mlp.num_experts
    for i in range(num_experts):
        expert = layer.mlp.experts[i]

        linears.append(
            dict(
                prev_op=layer.post_attention_layernorm,
                linears=[expert.up_proj, expert.gate_proj],
                names=[f"mlp.experts.{i}.up_proj", f"mlp.experts.{i}.gate_proj"],
            )
        )

        linears.append(
            dict(
                prev_op=expert.up_proj, linears=expert.down_proj, names=f"mlp.experts.{i}.down_proj"
            )
        )

    # shared experts
    shared_expert = layer.mlp.shared_expert
    linears.append(
        dict(
            prev_op=layer.post_attention_layernorm,
            linears=[shared_expert.up_proj, shared_expert.gate_proj],
            names=["mlp.shared_expert.up_proj", "mlp.shared_expert.gate_proj"],
        )
    )
    linears.append(
        dict(
            prev_op=shared_expert.up_proj,
            linears=shared_expert.down_proj,
            names="mlp.shared_expert.down_proj",
        )
    )

    return linears


def _get_gemma_linears_for_fp8_scale(layer):
    linears = [
        dict(
            prev_op=layer.input_layernorm,
            linears=[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
            names=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ),
        dict(
            prev_op=layer.self_attn.v_proj,
            linears=[layer.self_attn.o_proj],
            names=["self_attn.o_proj"],
        ),
    ]
    if hasattr(layer, "pre_feedforward_layernorm"):
        linears.append(
            dict(
                prev_op=layer.pre_feedforward_layernorm,
                linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
                names=["mlp.gate_proj", "mlp.up_proj"],
            )
        )
    else:
        linears.append(
            dict(
                prev_op=layer.post_attention_layernorm,
                linears=[layer.mlp.gate_proj, layer.mlp.up_proj],
                names=["mlp.gate_proj", "mlp.up_proj"],
            )
        )
    linears.append(
        dict(prev_op=layer.mlp.up_proj, linears=[layer.mlp.down_proj], names=["mlp.down_proj"])
    )
    return linears
