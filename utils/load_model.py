import json
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
from accelerate import load_checkpoint_and_dispatch
from loguru import logger
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from crossing.core.causal_lm_text_generation_engine_base import KernelDispatcher
from crossing.core.nn import W8A8QuantLinear
from crossing.core.operators_plugin_registry import OperatorsPluginRegistry
from crossingbits.qmodules import FP8QuantLinear, QuantLinear
from crossingbits.quantize import Quantizer, QuantType
from crossingbits.quantize.fp8_quant import FP8Quantizer
from crossingbits.quantize.utils import get_layers_to_quantize
from utils.helper import DEBUG

from .skip_init import skip_init


def load_hf_model(model_name_or_path: Union[str, Path], dtype: torch.dtype):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config.architectures is not None and "Deepseek" in config.architectures:
        config._attn_implementation = "eager"
    if DEBUG > 1:
        # using 2 layers for debug
        config.num_hidden_layers = 2
    # for saving full parameters: if `tie_word_embeddings` is set to True,
    # model will skip lm_head when saving.
    # FIXME(xingyu): maybe there is no lm_head in state_dict for some models,
    # and this may cause model cannot use the correct lm_head weights.
    # We can add assert to check if lm_head weights are the same as embedding weights.
    tie_embed = config.tie_word_embeddings
    config.tie_word_embeddings = False
    if "GemmaForCausalLM" in config.architectures or "Gemma2ForCausalLM" in config.architectures:
        config.tie_word_embeddings = True
    # no kv_cache for weight quantizing
    config.use_cache = False
    with skip_init():
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    model.eval()
    if tie_embed:
        assert torch.equal(
            model.lm_head.weight, model.get_input_embeddings().weight
        ), "lm_head weights are not the same as embedding weights when tie_embedding"
    return model


def load_model(
    model_path: str,
    *,
    is_quantized: Optional[bool] = False,
    quant_weight_format: str = "crossingbits",
    warmup_autotune: Optional[bool] = False,
    device: Union[str, torch.device] = "cuda",
):
    if is_quantized:
        QuantLinear.weight_format = quant_weight_format
        model = load_quantized_model(model_path, device)
    else:
        with skip_init(dtype=torch.float16):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=device, trust_remote_code=True
            )
    model.eval()

    return model


def _get_no_split_module_name(model: nn.Module) -> List[str]:
    try:
        if model.__class__.__name__ in {"QWenLMHeadModel", "TelechatForCausalLM"}:
            return [model.transformer.h[0].__class__.__name__]
        else:
            return [model.model.layers[0].__class__.__name__]
    except AttributeError:
        return []


def _swap_weight_only_quant_linear(
    model: nn.Module,
    packing_dtype: torch.dtype,
    exclude_layers: List[str],
    bits: int,
    group_size: int,
    symm_q: bool,
):
    """Swap out the linear layers for quantized ones"""
    logger.info("Swapping QuantLinear...")
    operators_name: str = os.environ.get("CROSSING_OPERATOR", "cuda")
    ops = OperatorsPluginRegistry.get(operators_name).create()
    if operators_name == "cuda":
        kernel_dispatcher = KernelDispatcher("interleaved")
    else:
        kernel_dispatcher = KernelDispatcher(operators_name)

    layers = get_layers_to_quantize(model)
    for i, layer in enumerate(layers):
        # Add quantizer for all linears
        for name, sub_layer in layer.named_modules():
            if isinstance(sub_layer, nn.Linear) and name not in exclude_layers:
                sub_layer.quantizer = Quantizer(bits, group_size=group_size, symm_q=symm_q)
        q_layer = QuantLinear.from_linear(
            layer, packing_dtype=packing_dtype, operators=ops, kernel_dispatcher=kernel_dispatcher
        )
        layers[i] = q_layer


def _swap_w8a8_quant_linear(model: nn.Module, exclude_layers: List[str]):
    logger.info("Swapping W8A8QuantLinear...")

    layers = get_layers_to_quantize(model)
    for i, layer in enumerate(layers):
        for name, sub_layer in layer.named_modules():
            if isinstance(sub_layer, nn.Linear) and name not in exclude_layers:
                sub_layer.quantizer = Quantizer(
                    n_bit=8, per_tensor=False, group_size=-1, symm_q=True, zeropoint=False
                )

        q_layer = W8A8QuantLinear.from_linear(
            layer, operators=OperatorsPluginRegistry.get("torch").create()
        )
        layers[i] = q_layer


@torch.inference_mode()
def _swap_fp8_quant_linear(fp8_dtype: torch.device, model: nn.Module, exclude_layers: List[str]):
    logger.info("Swapping FP8QuantLinear...")

    layers = get_layers_to_quantize(model)
    for i, layer in enumerate(layers):
        for name, sub_layer in layer.named_modules():
            if isinstance(sub_layer, nn.Linear) and name not in exclude_layers:
                sub_layer.quantizer = FP8Quantizer(fp8_dtype=fp8_dtype, per_tensor=True)

        q_layer = FP8QuantLinear.from_linear(
            layer, fp8_dtype=fp8_dtype, operators=OperatorsPluginRegistry.get("cuda").create()
        )
        layers[i] = q_layer


def load_quantized_model(model_path, device, dtype=torch.float16):
    quant_config = json.load(open(Path(model_path) / "quantize_config.json"))

    # Load model config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False

    # Build model
    with skip_init(dtype):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

    quant_type = quant_config["quant_type"]
    exclude_layers = quant_config["exclude_layers"]
    load_format = quant_config.get("load_format", "safetensors")

    if QuantType(quant_type) == QuantType.WEIGHT_ONLY:
        bits = quant_config["bits"]
        group_size = quant_config["group_size"]
        packing_dtype = quant_config["packing_dtype"]
        symm_q = quant_config["symm_q"]
        if isinstance(packing_dtype, str):
            packing_dtype = getattr(torch, packing_dtype)
        _swap_weight_only_quant_linear(
            model, packing_dtype, exclude_layers, bits, group_size, symm_q
        )
    elif QuantType(quant_type) == QuantType.WEIGHT_ACTIVATION_INT8:
        _swap_w8a8_quant_linear(model, exclude_layers)
    elif QuantType(quant_type) == QuantType.WEIGHT_ACTIVATION_FP8:
        fp8_dtype = quant_config["fp8_dtype"]
        if fp8_dtype == "e4m3":
            fp8_dtype = torch.float8_e4m3fn
        else:
            assert fp8_dtype == "e5m2", f"Unknown fp8_dtype {fp8_dtype}"
            fp8_dtype = torch.float8_e5m2
        _swap_fp8_quant_linear(fp8_dtype, model, exclude_layers)
    else:
        raise ValueError(f"Unknown quant_type {quant_type}")

    # Load the quantized model checkpoint
    logger.info("Loading quantized model...")

    if load_format == "safetensors":
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map=device,
            no_split_module_classes=_get_no_split_module_name(model),
        )
    elif load_format == "pt":
        model.load_state_dict(
            torch.load(Path(model_path) / "model.pt"),
        )
    else:
        raise FileNotFoundError(
            f"Could not find model checkpoint at {model_path}; please ensure that the path is "
            "correct and contains a `model.pt` or `model.safetensors` file."
        )

    return model
