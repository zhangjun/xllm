import torch
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from loguru import logger
from transformers import AutoTokenizer, HfArgumentParser

from .kv_cache import KVQuantizeArguments, kv_cache_quantize
from .utils import CalibDataLoader, copy_hf_config_files, save_model
from .utils.load_model import load_hf_model
from .qlinear import FP8BmmQuantLinear, FP8QuantLinear


from .utils_helper import (
    clear_memory,
    get_layers_to_quantize,
    get_lm_head_to_quantize,
    get_model_inputs,
    get_module_outputs,
    get_named_linears,
    set_op_by_name,
    compute_quant_errors,
    QuantType,
)

from .fp8_quant import (
    get_activation_scales_for_fp8,
    get_linears_for_fp8_scale,
    get_weight_scales_for_fp8,
)

from .utils.helper import DEBUG
from .utils.helper import HF_MODEL_BASE_DIR

@torch.inference_mode()
def run_fp8_quantize(
    model: nn.Module,
    calib_dataloader: List[torch.Tensor],
    *,
    fp8_dtype: torch.device = torch.float8_e4m3fn,
    weight_per_tensor: bool,
    seq_len: int,
    exclude_layers: Optional[List[str]] = None,
    quant_lm_head: Optional[bool] = False,
    lm_head_name: Optional[str] = "lm_head",
    device: Union[str, torch.device] = "cuda",
):
    if exclude_layers is None:
        exclude_layers = []
    if quant_lm_head:
        lm_head_layer, lm_head_name_full = get_lm_head_to_quantize(model, lm_head_name)
    layers = get_layers_to_quantize(model)
    dtype = next(iter(model.parameters())).dtype
    n_samples = len(calib_dataloader)

    all_inps = torch.zeros(
        (n_samples, seq_len, model.config.hidden_size), dtype=dtype, device=device
    )
    is_seq_first = False

    layer_kwargs = get_model_inputs(model, calib_dataloader, all_inps=all_inps, device=device)

    error_list = []
    mean_scale_list = []
    for layer_idx, layer in enumerate(
        (pbar := tqdm(layers, desc="Running FP8 Quantizing...", disable=(DEBUG > 0)))
    ):
        layer.to(device)

        all_outs = get_module_outputs(
            layer, all_inps, kwargs=layer_kwargs, is_seq_first=is_seq_first
        )
        # check nan in all_outs
        if torch.isnan(all_outs).any():
            raise ValueError(
                f"NaN in all_outs at layer {layer_idx}, please check the dtype for model inference!"
            )

        linears_to_scale = get_linears_for_fp8_scale(layer)

        # Step1: Get activation per-tensor scales for static quantization
        mean_scale, fp8_scales_dict = get_activation_scales_for_fp8(
            fp8_dtype, layer, linears_to_scale, all_inps, layer_kwargs
        )
        mean_scale_list.append(mean_scale)
        mean_input_scale = torch.mean(torch.stack(list(fp8_scales_dict.values())))

        # Step2: Get weight fp8 scales
        get_weight_scales_for_fp8(linears_to_scale, fp8_dtype, weight_per_tensor)
        for layer_info in linears_to_scale:
            linears = layer_info["linears"]
            if not isinstance(linears, list):
                linears = [linears]
            names = layer_info["names"]
            if not isinstance(names, list):
                names = [names]

            for name, m in zip(names, linears):
                if m.__class__.__name__ == "Linear":
                    fp8_qlinear = FP8QuantLinear.from_linear(
                        m,
                        name=name,
                        input_scale=fp8_scales_dict.get(f"{name}_input", mean_input_scale),
                        operators=OperatorsPluginRegistry.get("cuda").create(),
                    )
                elif m.__class__.__name__ == "BMMLinear":
                    fp8_qlinear = FP8BmmQuantLinear.from_linear(
                        m,
                        name=name,
                        input_scale=fp8_scales_dict[f"{name}_input"],
                        operators=OperatorsPluginRegistry.get("cuda").create(),
                    )
                else:
                    raise NotImplementedError(f"Unsupported Linear class: {m.__class__.__name__}")
                set_op_by_name(layer, name, fp8_qlinear)

        # Step3: Stat quantized layer error
        layer.to(device)

        err_batch = min(32, n_samples)
        q_output = get_module_outputs(layer, all_inps[:err_batch], kwargs=layer_kwargs)
        max_abs_error, max_rel_error = compute_quant_errors(
            all_outs[:err_batch], q_output, block_size=4
        )
        error_list.append((max_abs_error, max_rel_error))

        if DEBUG >= 1:
            logger.debug(
                f"Quant FwdErr: Abs={max_abs_error:.3f}, Rel={max_rel_error:.3%}, "
                f"Mean ActScales: {mean_scale:.3f} for layer_idx {layer_idx}"
            )
        else:
            pbar.set_description(
                f"Quant FwdErr: Abs={max_abs_error:.3f}, Rel={max_rel_error:.3%}, "
                f"Mean ActScales: {mean_scale:.3f} for layer_idx {layer_idx}"
            )

        layer.cpu()
        layers[layer_idx] = layer
        torch.cuda.empty_cache()

        # Swap for next layer inputs
        all_inps.data.copy_(all_outs.data)

    if quant_lm_head:
        # quantize lm_head
        lm_head_layer.to(device)
        output_device = torch.device("cpu")
        # This is to handle a potential issue where the vocabulary size of some LLMs
        # is so large that it could cause an Out Of Memory (OOM) error.
        all_outs = get_module_outputs(
            lm_head_layer, all_inps, is_seq_first=is_seq_first, output_device=output_device
        )

        linears_to_scale = [
            dict(prev_op=lm_head_layer[0], linears=[lm_head_layer[1]], names=[f"{lm_head_name}"])
        ]
        mean_scale, fp8_scales_dict = get_activation_scales_for_fp8(
            fp8_dtype,
            lm_head_layer,
            linears_to_scale,
            all_inps,
        )
        mean_scale_list.append(mean_scale)

        get_weight_scales_for_fp8(linears_to_scale, fp8_dtype, weight_per_tensor)
        fp8_qlinear = FP8QuantLinear.from_linear(
            lm_head_layer[1],
            input_scale=fp8_scales_dict[f"{lm_head_name}_input"],
            operators=OperatorsPluginRegistry.get("cuda").create(),
        )
        set_op_by_name(model, lm_head_name_full, fp8_qlinear)
        lm_head_layer[1] = fp8_qlinear

        err_batch = min(32, n_samples)
        q_output = get_module_outputs(
            lm_head_layer, all_inps[:err_batch], kwargs=layer_kwargs, output_device=output_device
        )
        max_abs_error, max_rel_error = compute_quant_errors(
            all_outs[:err_batch], q_output, block_size=4
        )
        error_list.append((max_abs_error, max_rel_error))

        logger.info(
            f"Quant FwdErr: Abs={max_abs_error:.3f}, Rel={max_rel_error:.3%}, "
            f"Mean ActScales: {mean_scale:.3f} for lm_head"
        )

    mean_abs_error = 0
    mean_rel_error = 0
    for abs_error, rel_error in error_list:
        mean_abs_error += abs_error
        mean_rel_error += rel_error
    mean_abs_error /= len(error_list)
    mean_rel_error /= len(error_list)
    mean_scale = sum(mean_scale_list) / len(mean_scale_list)
    logger.info(
        f"QLayerFwdErr(mean): "
        f"Abs={mean_abs_error:.3f}, "
        f"Rel={mean_rel_error:.3%}, "
        f"Activation scales(mean): {mean_scale:.3f}"
    )
    return mean_abs_error, mean_rel_error, mean_scale

@dataclass
class QuantizeArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    save_dir: str = field(
        metadata={"help": "Path to save quantized model"},
    )
    dtype: str = field(
        default="float16",
        metadata={"help": "dtype for the pretrained model inference"},
    )
    fp8_dtype: str = field(
        default="e4m3",
        metadata={
            "help": "Format of the fp8 datatype, choose from 'e4m3' and 'e5m2.'. Default is 'e4m3'."
        },
    )
    weight_per_tensor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to quantize weight by per_tensor."}
    )
    safetensors: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save quantized model in safetensors format"},
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Seed for sampling the calibration data"}
    )
    calib_set: Optional[str] = field(
        default="c4",
        metadata={
            "help": "Path to calibration data or dataset identifier from huggingface.co/dataset"
        },
    )
    download_mode: Optional[str] = field(
        default="reuse_dataset_if_exists",
        metadata={
            "help": "Download mode for the calibration data, "
            '"reuse_dataset_if_exists","reuse_cache_if_exists", "force_redownload".'
        },
    )
    loading_script: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the script for loading the calibration data"},
    )
    n_samples: Optional[int] = field(
        default=128, metadata={"help": "Number of samples for calibration"}
    )
    seq_len: Optional[int] = field(
        default=1024, metadata={"help": "Sequence length for calibration"}
    )
    exclude_layers: Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "Layer name in transformers_layers to exclude from quantization."},
    )
    quant_lm_head: Optional[bool] = field(
        default=False, metadata={"help": "Whether to quantize the lm_head layer."}
    )
    lm_head_name: Optional[str] = field(
        default="lm_head", metadata={"help": "Name of the lm_head layer."}
    )

    def __post_init__(self):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if self.fp8_dtype in ["e4m3", "e5m2"]:
            self.fp8_dtype = torch.float8_e4m3fn if self.fp8_dtype == "e4m3" else torch.float8_e5m2
        else:
            raise ValueError("Invalid fp8_dtype, choose from 'e4m3' and 'e5m2'.")

        self.dtype = getattr(torch, self.dtype)

        if HF_MODEL_BASE_DIR.value != "" and not os.path.exists(self.model_name_or_path):
            self.model_name_or_path = os.path.join(HF_MODEL_BASE_DIR.value, self.model_name_or_path)


def main():
    parser = HfArgumentParser((QuantizeArguments, KVQuantizeArguments))
    (args, kv_args) = parser.parse_args_into_dataclasses()

    logger.info("Loading model...")
    model = load_hf_model(args.model_name_or_path, args.dtype)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    logger.info(f"Loading calibration set {args.calib_set}...")
    calib_loader = CalibDataLoader()
    dataloader = calib_loader.get_dataset(
        args.calib_set,
        tokenizer,
        args.seed,
        args.n_samples,
        args.seq_len,
        args.download_mode,
        args.loading_script,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    quant_config = {
        "quant_type": QuantType.WEIGHT_ACTIVATION_FP8.value,
    }

    if not kv_args.kv_quant_only:
        tick = time.time()
        quant_abs_err, quant_rel_err, mean_scale = run_fp8_quantize(
            model,
            dataloader,
            fp8_dtype=args.fp8_dtype,
            weight_per_tensor=args.weight_per_tensor,
            seq_len=args.seq_len,
            exclude_layers=args.exclude_layers,
            quant_lm_head=args.quant_lm_head,
            lm_head_name=args.lm_head_name,
            device="cuda",
        )
        logger.info("Quantization takes {:.2f} minutes".format((time.time() - tick) / 60))
        quant_config |= {
            "bits": 8,
            "fp8_dtype": "e4m3" if args.fp8_dtype == torch.float8_e4m3fn else "e5m2",
            "exclude_layers": args.exclude_layers,
            "quant_abs_error": quant_abs_err,
            "quant_rel_error": quant_rel_err,
            "activation_mean_scale": mean_scale,
            "load_format": "safetensors" if args.safetensors else "pt",
            "quant_lm_head": args.quant_lm_head,
            "lm_head_name": args.lm_head_name,
        }
        logger.info(f"Save quantized model to {save_dir}...")
        save_model(model, save_dir, args.safetensors)

    if kv_args.kv_quant_pass or kv_args.kv_quant_only:
        logger.info("Running kv cache quantizing...")
        assert kv_args.kv_dtype == torch.float8_e4m3fn, (
            f"Expected kv_dtype to be 'torch.float8_e4m3fn', but got '{kv_args.kv_dtype}'. "
            "Only fp8_e4m3 requires offline scale computation."
        )
        kv_dtype = "float8_e4m3"

        quant_config |= {
            "kv_dtype": kv_dtype,
            "kv_bits": kv_args.kv_bit,
            "kv_symm_q": kv_args.kv_symm_q,
        }

        tick = time.time()
        kv_scales, k_quant_err, v_quant_err = kv_cache_quantize(
            model,
            dataloader,
            kv_bit=kv_args.kv_bit,
            kv_dtype=kv_args.kv_dtype,
            kv_symm_q=kv_args.kv_symm_q,
            seq_len=args.seq_len,
            device="cuda",
        )
        logger.info("KV cache quantization takes {:.2f} minutes".format((time.time() - tick) / 60))

        quant_config |= {
            "k_quant_err": k_quant_err,
            "v_quant_err": v_quant_err,
        }

        # Save kv qparams
        torch.save(kv_scales, save_dir / "kv_scales.pth")

    # Write q_config.json
    with open(save_dir / "quantize_config.json", "w") as f:
        f.write(json.dumps(quant_config))

    copy_hf_config_files(args.model_name_or_path, save_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
