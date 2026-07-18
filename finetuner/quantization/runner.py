from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from finetuner.core.artifacts import atomic_write_json
from finetuner.quantization.specs import QuantizationBackend, QuantizationConfig


def _resolve_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required executable {name!r} was not found on PATH")
    return executable


def build_quantization_commands(
    model_path: str, output_dir: str, config: QuantizationConfig
) -> list[list[str]]:
    """Build argv arrays (never shell strings) for a validated conversion plan."""
    config.require_valid()
    model = str(Path(model_path).resolve())
    output = str(Path(output_dir).resolve())
    backend = QuantizationBackend(config.backend)

    if backend == QuantizationBackend.OPENVINO:
        command = [
            _resolve_executable("optimum-cli"),
            "export",
            "openvino",
            "--model",
            model,
            "--weight-format",
            f"int{config.bits}",
            "--group-size",
            str(config.group_size),
        ]
        command.append("--sym" if config.scheme == "symmetric" else "--asym")
        if config.calibration_dataset:
            command.extend(["--dataset", config.calibration_dataset])
        command.append(output)
        return [command]

    if backend == QuantizationBackend.ONNX:
        intermediate = str(Path(output) / "fp32")
        quantized = str(Path(output) / "quantized")
        return [
            [
                _resolve_executable("optimum-cli"),
                "export",
                "onnx",
                "--model",
                model,
                "--task",
                "text-generation-with-past",
                intermediate,
            ],
            [
                _resolve_executable("optimum-cli"),
                "onnxruntime",
                "quantize",
                "--onnx_model",
                intermediate,
                "--avx2",
                "--per_channel",
                "-o",
                quantized,
            ],
        ]

    if backend == QuantizationBackend.GGUF:
        root = Path(config.llama_cpp_path).expanduser().resolve() if config.llama_cpp_path else None
        converter = root / "convert_hf_to_gguf.py" if root else None
        quantizer_names = ["llama-quantize.exe", "llama-quantize"]
        quantizer = next(
            (str(root / name) for name in quantizer_names if root and (root / name).exists()),
            None,
        ) or next((shutil.which(name) for name in quantizer_names if shutil.which(name)), None)
        if converter is None or not converter.exists():
            raise RuntimeError(
                "GGUF conversion requires llama_cpp_path containing convert_hf_to_gguf.py"
            )
        if quantizer is None:
            raise RuntimeError("llama-quantize was not found in llama_cpp_path or PATH")
        intermediate = str(Path(output) / "model-f16.gguf")
        final = str(Path(output) / f"model-q{config.bits}.gguf")
        python = _resolve_executable("python")
        quant_type = {2: "Q2_K", 3: "Q3_K_M", 4: "Q4_K_M", 5: "Q5_K_M", 6: "Q6_K", 8: "Q8_0"}[
            config.bits
        ]
        return [
            [python, str(converter), model, "--outfile", intermediate, "--outtype", "f16"],
            [quantizer, intermediate, final, quant_type],
        ]

    if backend == QuantizationBackend.AWQ:
        return []
    raise AssertionError(backend)


def quantize_model(
    model_path: str,
    output_dir: str,
    config: QuantizationConfig,
    log: Callable[[str], None] | None = None,
) -> str:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if QuantizationBackend(config.backend) == QuantizationBackend.AWQ:
        return _quantize_awq(model_path, output, config, log)
    commands = build_quantization_commands(model_path, str(output), config)
    for command in commands:
        if log:
            log(f"Running quantization step: {Path(command[0]).name}")
        subprocess.run(command, check=True, text=True)
    manifest = {
        "schema_version": 1,
        "source_model": str(Path(model_path).resolve()),
        "output_dir": str(output.resolve()),
        "quantization": config.to_dict(),
        "commands": [[Path(cmd[0]).name, *cmd[1:]] for cmd in commands],
    }
    atomic_write_json(output / "quantization_manifest.json", manifest)
    return str(output.resolve())


def _quantize_awq(
    model_path: str,
    output: Path,
    config: QuantizationConfig,
    log: Callable[[str], None] | None,
) -> str:
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("AWQ requires the optional 'llm-awq' package") from exc
    if log:
        log("Loading model for activation-aware weight quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    quant_config = {
        "w_bit": config.bits,
        "q_group_size": config.group_size,
        "zero_point": config.scheme == "asymmetric",
        "version": "GEMM",
    }
    model.quantize(tokenizer, quant_config=quant_config, calib_data=config.calibration_dataset)
    model.save_quantized(str(output), safetensors=True, shard_size="4GB")
    tokenizer.save_pretrained(str(output))
    atomic_write_json(
        output / "quantization_manifest.json",
        {
            "schema_version": 1,
            "source_model": str(Path(model_path).resolve()),
            "output_dir": str(output.resolve()),
            "quantization": config.to_dict(),
        },
    )
    return str(output.resolve())
