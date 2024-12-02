import functools
from enum import Enum
from functools import lru_cache
from typing import Optional, Union

import torch
from loguru import logger
from torch import device, dtype, nn

from crossing.core.causal_lm_text_generation_engine_base import KernelDispatcher
from crossing.core.nn import FP8QuantLinear, QuantLinear
from crossing.core.operators import Operators
from crossing.core.operators_plugin_registry import OperatorsPluginRegistry
from crossing.quant.quant_context import ActQuantType
from crossing.quant.quant_kernel_dispatcher import QuantKernelDispatcherPluginRegistry
from crossing.quant.quant_weight_converter import QuantWeightConverterPluginRegistry
from crossingbits.utils.helper import DEBUG

E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max


@lru_cache(maxsize=None)
def get_cuda_arch():
    property = torch.cuda.get_device_properties(torch.cuda.current_device())
    return property.major * 10 + property.minor


class DispatchState(str, Enum):
    READY = "ready"
    NOT_READY = "not_ready"


class QuantLinear(QuantLinear):
    weight_format: str = "crossingbits"

    def __init__(
        self,
        *,
        w_bit: int,
        group_size: int,
        symm_q: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: str | device | None = None,
        dtype: dtype | None = None,
        operators: Operators,
        kernel_dispatcher: KernelDispatcher,
    ) -> None:
        super().__init__(
            w_bit=w_bit,
            group_size=group_size,
            symm_q=symm_q,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            operators=operators,
            kernel_dispatcher=kernel_dispatcher,
        )
        self._dispatch_state: DispatchState = DispatchState.NOT_READY

        self._dispatch_kernel_name = kernel_dispatcher.value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dispatch to certain quant kernel at the first forward
        if self._dispatch_state == DispatchState.NOT_READY:
            self.dispatch_to_kernel()
        return super().forward(x)

    def dispatch_to_kernel(
        self,
    ):
        qweight, scales, qzeros = (
            QuantWeightConverterPluginRegistry.get(self.weight_format)
            .convert_to_intermediate(
                bit_width=self.w_bit,
                weight=self.qweight,
                scales=self.scales,
                zeros=None if self._symm_q else self.qzeros,
            )
            .set_dispatcher(QuantKernelDispatcherPluginRegistry.get(self._dispatch_kernel_name))
            .dispatch_to_kernel()
        )
        self.qweight.data = qweight
        self.scales.data = scales
        if not self._symm_q:
            self.qzeros.data = qzeros
        torch.cuda.empty_cache()
        self._dispatch_state = DispatchState.READY


class FP8QuantLinear(FP8QuantLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()

        out_shape = x.shape[:-1] + (self.out_features,)
        if get_cuda_arch() >= 89:
            out = self.fp8_qdq_per_tensor_gemm(
                x.reshape(-1, x.shape[-1]),
                self.qweight.view(self.fp8_dtype),
                self.fp8_scale_x,
                self.fp8_scale_w,
                self.bias,
            )
        else:
            if DEBUG > 1:
                logger.debug("FP8 is not supported on this device, fallback to FP32 gemm.")
            dtype = x.dtype
            out = torch.empty(out_shape, device=x.device, dtype=dtype)

            fp8_x = x * self.fp8_scale_x_reciprocal
            if self.fp8_dtype == torch.float8_e4m3fn:
                fp8_x = torch.clamp(fp8_x, min=-E4M3_MAX_POS, max=E4M3_MAX_POS)
            else:
                assert self.fp8_dtype == torch.float8_e5m2
                fp8_x = torch.clamp(fp8_x, min=-E5M2_MAX_POS, max=E5M2_MAX_POS)
            fp8_x = fp8_x.reshape(-1, x.shape[-1])

            out = (
                torch.matmul(
                    fp8_x.float(),
                    self.qweight.view(self.fp8_dtype).float().transpose(0, 1).contiguous(),
                )
                * self.fp8_scale_x
                * self.fp8_scale_w
            ).to(dtype)
            if self.bias is not None:
                out += self.bias

        return out.view(out_shape)  # (..., N)


class FP8BmmQuantLinear(FP8QuantLinear):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_heads: int,
        bias: bool = False,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        act_quant: Union[str, ActQuantType] = "per_tensor_static",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        operators: Operators,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            fp8_dtype=fp8_dtype,
            act_quant=act_quant,
            device=device,
            dtype=dtype,
            operators=operators,
        )
        self.num_heads = num_heads

        self.fp8_scaled_bmm = operators.fp8_scaled_bmm
        self.fp8_per_token_quantize = operators.fp8_per_token_quantize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()

        qweight = self.qweight.view(self.fp8_dtype).view(self.num_heads, -1, self.in_features)
        if self.in_features == x.size(-1):
            qweight = qweight.transpose(1, 2)
        else:
            qweight = qweight

        out_shape = x.shape[:-1] + (qweight.size(-1),)
        out = torch.empty(out_shape, device=x.device, dtype=torch.float32)

        x_shape = x.shape
        x = x.transpose(0, 1).reshape(x_shape[1], -1)
        fp8_x, fp8_scale_x = self.fp8_per_token_quantize(x)
        fp8_x = fp8_x.reshape(x_shape[1], x_shape[0], x_shape[2]).transpose(0, 1).contiguous()

        torch.matmul(fp8_x.float(), qweight.float(), out=out)
        out = (out * fp8_scale_x * self.fp8_scale_w).to(x.dtype)
        return out

    @classmethod
    def from_linear(
        cls,
        module: nn.Module,
        *,
        name: Optional[str] = None,
        fp8_dtype: torch.device = torch.float8_e4m3fn,
        input_scale: torch.tensor = torch.tensor(1.0, dtype=torch.float32),
        operators: Operators,
    ):
        """Helper function to convert all `nn.Linear` modules
        in a model to `FP8QuantLinear` modules"""
        module_output = module
        if name is None:
            name = module_output.__class__.__name__
        if module.__class__.__name__ == "BMMLinear":
            if hasattr(module, "quantizer"):
                quantizer = module.quantizer
                del module.quantizer

                module_output = cls(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    num_heads=module.num_heads,
                    bias=module.bias is not None,
                    fp8_dtype=fp8_dtype,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                    operators=operators,
                )

                fp8_weight = quantizer.quantize(module.weight)
                weight_scale = quantizer.scales
                # NOTE(xingyu): float8 dtype is not supported in pytorch when serializing.
                # Use view to bitcast the tensor to int8 and then back to float8 when loading.
                module_output.qweight.data.copy_(fp8_weight.view(torch.int8))
                if module.bias is not None:
                    module_output.bias.data = module.bias.data
                module_output.fp8_scale_x.data.copy_(torch.reciprocal(input_scale))
                module_output.fp8_scale_x_reciprocal.data.copy_(input_scale)
                module_output.fp8_scale_w.data.copy_(torch.reciprocal(weight_scale))

            else:
                logger.info(
                    f"Linear module {name} has no quantizer attribute, "
                    "using the un-quantized one temporarily."
                )

        for name, child in module.named_children():
            module_output.add_module(
                name,
                cls.from_linear(
                    child,
                    name=name,
                    fp8_dtype=fp8_dtype,
                    input_scale=input_scale,
                    operators=operators,
                ),
            )
        del module
        return module_output


def autotune_warmup(model):
    F = OperatorsPluginRegistry.get("triton").create()
    # Find all the QuantLinear layers
    modules = (m for m in model.modules() if isinstance(m, QuantLinear))
    kn_values = {
        (m.in_features, m.out_features, m.w_bit): (
            m.qweight,
            m.scales,
            m.qzeros,
            m.group_size,
        )
        for m in modules
    }

    logger.info(f"QuantLinear Warmup: Found {len(kn_values)} unique KN values.")

    def func(m, k, bits, q_weight, scales, q_zeros, group_size):
        a = torch.randn(m, k, dtype=torch.float16, device=q_weight.device)
        F.quantized_matmul(bits, group_size, a, q_weight, scales, q_zeros)

    return (
        functools.partial(
            func,
            k=k,
            bits=bits,
            q_weight=q_weight,
            scales=scales,
            q_zeros=q_zeros,
            group_size=group_size,
        )
        for (k, n, bits), (q_weight, scales, q_zeros, group_size) in kn_values.items()
    )
