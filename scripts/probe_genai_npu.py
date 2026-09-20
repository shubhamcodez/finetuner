"""Load the 0.5B genai package onto Hexagon and generate a few tokens."""

from __future__ import annotations

import time
from pathlib import Path

import onnxruntime as ort
import onnxruntime_genai as og
import onnxruntime_qnn as qnn

MODEL_DIR = Path.home() / ".finetuner" / "bench" / "models" / "qwen2.5-0.5b-instruct-onnx-genai"

ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
if hasattr(og, "register_execution_provider_library"):
    og.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())

print("ORT devices:")
for device in ort.get_ep_devices():
    print(" ", device.ep_name, device.device.type, device.device.metadata)

config = og.Config(str(MODEL_DIR))
config.clear_providers()
config.append_provider("qnn")
config.set_provider_option("qnn", "backend_path", qnn.get_qnn_htp_path())
config.set_provider_option("qnn", "htp_performance_mode", "burst")
config.set_provider_option("qnn", "enable_htp_fp16_precision", "1")
if hasattr(config, "set_decoder_provider_options_hardware_device_type"):
    config.set_decoder_provider_options_hardware_device_type("qnn", "npu")
    print("set hardware device type npu")
if hasattr(config, "set_decoder_provider_options_hardware_device_id"):
    try:
        config.set_decoder_provider_options_hardware_device_id("qnn", 1093682224)
        print("set hardware device id")
    except Exception as exc:
        print("device id failed", exc)

started = time.perf_counter()
model = og.Model(config)
print(f"loaded in {1000 * (time.perf_counter() - started):.0f} ms")
tokenizer = og.Tokenizer(model)
tokens = tokenizer.encode("Hello")
params = og.GeneratorParams(model)
params.set_search_options(do_sample=False, max_length=len(tokens) + 8)
generator = og.Generator(model, params)
prefill = time.perf_counter()
generator.append_tokens(tokens)
print("prefill ms", 1000 * (time.perf_counter() - prefill))
decode = time.perf_counter()
n = 0
while not generator.is_done() and n < 8:
    generator.generate_next_token()
    n += 1
print("decode", n, "tokens in", 1000 * (time.perf_counter() - decode), "ms")
