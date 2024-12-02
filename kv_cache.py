from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from .utils.helper import DEBUG, can_accept_argument

from .quantizer import QuantizerMixin
from .utils_helper import get_layers_to_quantize, get_model_inputs, get_num_heads


@dataclass
class KVQuantizeArguments:
    kv_quant_pass: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to quantize kv cache"},
    )
    kv_quant_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to quantize kv cache only"},
    )
    kv_bit: Optional[int] = field(
        default=8,
        metadata={"help": "Number of kv cache bits for quantization"},
    )
    kv_dtype: Optional[str] = field(
        default="int8",
        metadata={"help": "Data type for kv cache quantization, choose from int8 and fp8_e4m3"},
    )
    kv_symm_q: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use symmetric quantization for kv cache"},
    )

    def __post_init__(self):
        assert self.kv_dtype in ["int8", "fp8_e4m3"], "kv_dtype must be int8 or float8_e4m3"
        self.kv_dtype = torch.int8 if self.kv_dtype == "int8" else torch.float8_e4m3fn


E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
INT8_MAX_POS = torch.iinfo(torch.int8).max
INT8_MIN_NEG = torch.iinfo(torch.int8).min
UINT8_MAX_POS = torch.iinfo(torch.uint8).max

FP16_MAX_POS = torch.finfo(torch.float16).max


def _adjust_tensor_layout(x: torch.Tensor, num_head: int, head_dim: int):
    """Change the tensor layout to (..., heads, dims)"""
    if len(x.shape) == 3:
        x = x.unsqueeze(0)  # add `batch` dim
    assert len(x.shape) == 4
    assert x.size(1) != x.size(2)

    if x.size(2) == num_head and x.size(3) == head_dim:
        # layout: (bs, seq_len, heads, dims)
        x = x
    elif x.size(1) == num_head and x.size(3) == head_dim:
        # layout: (bs, heads, seq_len, dims) => (bs, seq_len, heads, dims)
        x = x.transpose(1, 2)
    else:
        raise RuntimeError(f"Invalid input shape: {x.shape}")

    return x


class KVCacheQuantizer(QuantizerMixin):
    def __init__(
        self,
        *,
        n_bit: int = 8,
        kv_dtype: torch.device = torch.int8,
        symm_q: bool = True,
        num_head: int,
        head_dim: int,
    ) -> None:
        self._n_bit = n_bit
        self._kv_dtype = kv_dtype
        self._symm_q = symm_q
        self.num_head = num_head
        self.head_dim = head_dim

        self.max_val = torch.full((num_head, head_dim), float("-inf"), dtype=torch.float32)
        self.min_val = torch.full((num_head, head_dim), float("inf"), dtype=torch.float32)
        self.absmax_val = torch.full((num_head, head_dim), 0, dtype=torch.float32)

        assert symm_q, "Only symmetric quantization is supported for kv cache right now!"

    @torch.inference_mode()
    def observe(self, x: torch.Tensor) -> None:
        """Function to observe the input tensor and update the max, min, and absolute max values"""

        x = _adjust_tensor_layout(x, self.num_head, self.head_dim)
        x_flat = x.flatten(0, 1)

        assert not torch.isnan(x_flat).any(), "KVCache contains NaN values!"
        cur_max = x_flat.max(dim=0).values.cpu()
        cur_min = x_flat.min(dim=0).values.cpu()
        cur_absmax = x_flat.abs().max(dim=0).values.cpu()

        self.max_val = torch.maximum(self.max_val, cur_max)
        self.min_val = torch.minimum(self.min_val, cur_min)
        self.absmax_val = torch.maximum(self.absmax_val, cur_absmax)

    def find_scale_zero(self):
        if self._symm_q:
            self._scales = self.absmax_val.max() / self.maxq
            self._zeros = torch.zeros_like(self._scales)
        else:
            assert (
                self._kv_dtype == torch.int8
            ), "Only int8 kv cache supports asymmetric quantization"
            # NOTE(xingyu): zero-point integer constrain may be dispensed since
            # activation is seldom zero.
            # zp = (min+max) / 2
            # scale = (max-min) / 255
            # quant: q = round( (f-zp) / scale)
            # dequant: f = q * scale + zp
            max_val = self.max_val.max()
            min_val = self.min_val.min()
            self._scales = (max_val - min_val) / self.maxq
            self._zeros = (max_val + min_val) / 2

    def quantize(self, x):
        x = _adjust_tensor_layout(x, self.num_head, self.head_dim)
        if not hasattr(self, "_scales"):
            self.find_scale_zero()

        if self._kv_dtype == torch.int8:
            return (
                torch.clamp(torch.round(x / self._scales), INT8_MIN_NEG, INT8_MAX_POS)
                if self._symm_q
                else torch.clamp(torch.round((x - self._zeros) / self._scales), 0, UINT8_MAX_POS)
            )
        else:
            return torch.clamp(x / self._scales, -E4M3_MAX_POS, E4M3_MAX_POS)

    def pseudo_quantize(self, x):
        x = _adjust_tensor_layout(x, self.num_head, self.head_dim)
        q = self.quantize(x)
        if self._symm_q:
            return x, q * self._scales
        else:
            return x, q * self._scales + self._zeros

    @property
    def n_bit(self):
        return self._n_bit

    @property
    def maxq(self):
        if self._kv_dtype == torch.int8:
            return INT8_MAX_POS if self._symm_q else UINT8_MAX_POS
        # fp8_e4m3
        assert self._symm_q, "Only symmetric quantization is supported for fp8 kv cache."
        return E4M3_MAX_POS

    @property
    def scales(self):
        return self._scales

    @property
    def zeros(self):
        return self._zeros

    @property
    def symm_q(self):
        return self._symm_q


@torch.inference_mode()
def kv_cache_quantize(
    model: nn.Module,
    calib_dataloader: List[torch.Tensor],
    *,
    kv_bit: int,
    kv_dtype: torch.dtype,
    kv_symm_q: bool,
    seq_len: int,
    device="cuda",
):
    layers = get_layers_to_quantize(model)
    dtype = next(iter(model.parameters())).dtype
    if model.__class__.__name__ == "DeepseekV2ForCausalLM":
        # For Deepseek-V2, k_head_dim represents c_latent and v_head_dim represents k_rope
        num_kv_heads = 1
        k_head_dim = model.config.kv_lora_rank
        v_head_dim = model.config.qk_rope_head_dim
    elif model.__class__.__name__ in ["GemmaForCausalLM", "Gemma2ForCausalLM"]:
        num_kv_heads, num_attn_heads = get_num_heads(model)
        k_head_dim = model.config.head_dim
        v_head_dim = k_head_dim
    else:
        num_kv_heads, num_attn_heads = get_num_heads(model)
        k_head_dim = model.config.hidden_size // num_attn_heads
        v_head_dim = k_head_dim

    n_samples = len(calib_dataloader)

    all_inps = torch.zeros(
        (n_samples, seq_len, model.config.hidden_size), dtype=dtype, device=device
    )
    all_outs = torch.zeros_like(all_inps)

    layer_kwargs = get_model_inputs(model, calib_dataloader, all_inps=all_inps, device=device)

    k_scales = []
    v_scales = []
    if not kv_symm_q:
        k_zero = []
        v_zero = []
    for layer_idx, layer in enumerate((pbar := tqdm(layers, disable=(DEBUG > 0)))):
        layer = layer.to(device)

        # set layer name
        layer_name = f"{layer.__class__.__name__}_{layer_idx}"

        k_quantizer = KVCacheQuantizer(
            n_bit=kv_bit,
            kv_dtype=kv_dtype,
            symm_q=kv_symm_q,
            num_head=num_kv_heads,
            head_dim=k_head_dim,
        )
        v_quantizer = KVCacheQuantizer(
            n_bit=kv_bit,
            kv_dtype=kv_dtype,
            symm_q=kv_symm_q,
            num_head=num_kv_heads,
            head_dim=v_head_dim,
        )

        def _build_empty_cache():
            # NOTE(xingyu): transformers has refactored the kv cache to DynamicCache
            nonlocal layer_idx
            past_key_value = DynamicCache.from_legacy_cache(None)
            for _ in range(layer_idx):
                past_key_value.key_cache.append([])
                past_key_value.value_cache.append([])
            return past_key_value

        # kv_cache will be returned appending to the output
        layer_kwargs["use_cache"] = True
        layer_kwargs["past_key_value"] = _build_empty_cache()

        for i in range(len(calib_dataloader)):
            # Check if the layer forward method can accept the "past_key_value" argument
            # this is for compatibility with models using older versions of transformers.
            if (
                not can_accept_argument(layer.forward, "past_key_value")
                # NOTE(xingyu): baichuan use older transformer inference code whose past_key_value
                # is not compatible with newer transformer.
                or model.__class__.__name__ in ["BaichuanForCausalLM", "BaiChuanForCausalLM"]
            ):
                # If not, remove the "past_key_value" argument from layer_kwargs
                layer_kwargs.pop("past_key_value")
            outs = layer(all_inps[i : i + 1], **layer_kwargs)
            # outs: (hidden_states, present_key_value) when use_cache=True
            outs = list(outs)
            kv_cache = outs.pop(-1)
            if hasattr(kv_cache, "key_cache"):
                key = kv_cache.key_cache[-1]
                value = kv_cache.value_cache[-1]
            else:
                # If 'key_cache' attribute is not present, this might be an older version
                # of the model. In that case, assume 'kv_cache' is a list-like structure
                # and fetch the first two elements as key and value respectively.
                if isinstance(kv_cache, (list, tuple)):
                    key = kv_cache[0]
                    value = kv_cache[1]
                else:
                    # chat-glm4 case
                    assert isinstance(kv_cache, torch.Tensor)
                    key = kv_cache[0, 0]
                    value = kv_cache[0, 1]
            all_outs[i] = outs.pop(0)
            k_quantizer.observe(key)
            v_quantizer.observe(value)

            layer_kwargs["past_key_value"] = _build_empty_cache()
            del key, value, kv_cache
            torch.cuda.empty_cache()

        k_quantizer.find_scale_zero()
        v_quantizer.find_scale_zero()

        # Update the progress bar with quantized error
        k_quant_err = []
        v_quant_err = []
        layer_kwargs["past_key_value"] = _build_empty_cache()
        for i in range(len(calib_dataloader)):
            # outs: (hidden_states, present_key_value) when use_cache=True
            if (
                not can_accept_argument(layer.forward, "past_key_value")
                # NOTE(xingyu): baichuan use older transformer inference code whose past_key_value
                # is not compatible with newer transformer.
                or model.__class__.__name__ in ["BaichuanForCausalLM", "BaiChuanForCausalLM"]
            ):
                # If not, remove the "past_key_value" argument from layer_kwargs
                layer_kwargs.pop("past_key_value")
            outs = layer(all_inps[i : i + 1], **layer_kwargs)
            outs = list(outs)
            kv_cache = outs.pop(-1)
            if hasattr(kv_cache, "key_cache"):
                key = kv_cache.key_cache[-1]
                value = kv_cache.value_cache[-1]
            else:
                if isinstance(kv_cache, (list, tuple)):
                    key = kv_cache[0]
                    value = kv_cache[1]
                else:
                    assert isinstance(kv_cache, torch.Tensor)
                    key = kv_cache[0, 0]
                    value = kv_cache[0, 1]

            key, q_key = k_quantizer.pseudo_quantize(key)
            value, q_value = v_quantizer.pseudo_quantize(value)
            k_quant_err.append((key - q_key).abs().float().max().item())
            v_quant_err.append((value - q_value).abs().float().max().item())

            layer_kwargs["past_key_value"] = _build_empty_cache()
            torch.cuda.empty_cache()

        if DEBUG >= 1:
            logger.debug(
                f"{layer_name}: key quant err: {np.mean(k_quant_err):.4f}, "
                f"value quant err: {np.mean(v_quant_err):.4f}"
            )
        else:
            pbar.set_description(
                f"{layer_name}: key quant err: {np.mean(k_quant_err):.4f}, "
                f"value quant err: {np.mean(v_quant_err):.4f}"
            )

        k_scales.append(k_quantizer.scales)
        v_scales.append(v_quantizer.scales)
        if not kv_symm_q:
            k_zero.append(k_quantizer.zeros)
            v_zero.append(v_quantizer.zeros)

        layer = layer.cpu()
        # Swap for next layer inputs
        all_inps.data.copy_(all_outs.data)

    kv_scales = {"k_scale": torch.stack(k_scales), "v_scale": torch.stack(v_scales)}
    if not kv_symm_q:
        kv_scales["k_zero"] = torch.stack(k_zero)
        kv_scales["v_zero"] = torch.stack(v_zero)

    return kv_scales, np.mean(k_quant_err), np.mean(v_quant_err)
