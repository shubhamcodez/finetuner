# Finetuner

Finetuner is a local desktop workbench for reproducible LLM post-training. Training, preference
optimization, knowledge distillation, evaluation, representation analysis, target-aware compression,
and inference-engine optimization each run as their own tool.

**Using the app:** start with the [user guide](docs/USER_GUIDE.md). It covers install, the first
run, every tab, dataset formats, hardware pairing, and troubleshooting. Architecture and research
notes live in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/RESEARCH.md](docs/RESEARCH.md).

## Product capabilities

- Independent tools: train, distill, evaluate, analyze, quantize, and optimize inference without a
  shared DAG
- SFT, DPO, GRPO, PPO, KTO, reward-model, ORPO, and RLOO trainers through TRL
- Real preference-schema validation; synthetic negative responses are opt-in and deterministic
- Sequence knowledge distillation across model families plus experimental logit/GKD techniques for
  tokenizer-compatible teacher/student pairs
- Domain-selective distillation for computer science, mathematics, optimization, science, reasoning,
  safety, arbitrary custom topics, or all data
- GGUF, OpenVINO INT4/INT8, ONNX Runtime INT8, and AWQ deployment pipelines with an explicit
  backend/device compatibility matrix
- Adaptive inference: after a model is loaded, Finetuner detects this device and asks whether to
  optimize. Yes picks the strongest specialist (vLLM/llama.cpp CUDA, llama.cpp Vulkan/Metal/CPU,
  OpenVINO, or QNN/HTP) and then serves the model. No serves the artifact as-is. Either way the
  model is bound at http://127.0.0.1:1234. One-button NVIDIA / AMD / NPU overrides still exist.
- PCA, t-SNE, or UMAP hidden-state projections by layer, activation norms, attention entropy, and
  cross-layer centered-kernel alignment (CKA)
- Atomic run manifests containing stage status, duration, metrics, configuration digest, and artifact
  lineage
- CPU, RAM, NVIDIA GPU, NPU, and TPU monitoring; Hugging Face/local model management; benchmark comparison

Finetuner does not claim that one artifact runs optimally on every accelerator. Deployment is planned
against a concrete runtime and device: GGUF for broad CPU/GPU support, OpenVINO for supported Intel
CPU/GPU/NPU systems, ONNX Runtime INT8 for CPU, and AWQ for compatible NVIDIA inference stacks.
Serving uses that specialist when you optimize, or a portable CPU recipe when you skip. The live
server is always http://127.0.0.1:1234 (llama-server / vLLM / TGI when the CLI is present, otherwise
a built-in OpenAI-compatible fallback). Unsupported combinations are rejected before conversion or
compilation.

## Development

```powershell
cd E:\finetuner
.\scripts\install_gpu.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
.\.venv\Scripts\python.exe -m finetuner
```

Use a CUDA build of PyTorch for GPU training. Quantized artifacts can target non-CUDA devices, but the
current local trainer requires NVIDIA CUDA. Optional deployment toolchains are installed separately:

```powershell
pip install -e ".[openvino]"       # Intel CPU/GPU/NPU
pip install -e ".[onnx]"           # ONNX Runtime INT8 CPU
pip install llm-awq                 # NVIDIA AWQ
```

GGUF conversion requires a local llama.cpp checkout/build selected in the Deploy tab. Inference
optimization writes a serve/compile plan without those toolchains; enable Compile only when
TensorRT-LLM, OpenVINO, or ONNX Runtime is installed and you want a materialized engine.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest --cov=finetuner --cov-report=term-missing
```

Tests live in `tests/` and avoid model downloads. Heavy GPU/model integration tests should use explicit
small checkpoints in a dedicated CI job.

## Reproducibility and secrets

Each tool run writes `manifest.json` incrementally and atomically. The manifest captures the action
and a digest of the redacted project configuration. Hugging Face tokens are held in memory or
read from `HF_TOKEN`; they are intentionally not written to `config.json` or run manifests.

See the [user guide](docs/USER_GUIDE.md), [architecture](docs/ARCHITECTURE.md), and
[research rationale](docs/RESEARCH.md).

## Build the Windows installer

PyInstaller cannot cross-compile Windows binaries. Run the script on the architecture you
want to ship. Each run writes a native setup EXE and a portable zip.

```powershell
.\scripts\build_windows.ps1              # this PC (x64 or ARM64)
.\scripts\build_windows.ps1 -Arch x64
.\scripts\build_windows.ps1 -Arch arm64
```

Outputs in `dist\`:

- `Finetuner-Setup-x64.exe` or `Finetuner-Setup-arm64.exe` — Inno Setup installer
- `Finetuner-windows-x64.zip` or `Finetuner-windows-arm64.zip` — portable folder

x64 uses a CUDA PyTorch wheel when one is available, then falls back to CPU. ARM64 never
installs CUDA. After both native folders exist, build a single picker:

```powershell
.\scripts\build_windows.ps1 -Universal
```

That produces `Finetuner-Setup-universal.exe`, which installs the x64 payload on x64 Windows
and the ARM64 payload on ARM64 Windows (Snapdragon X Elite, etc.). Inno Setup 6.3 or newer
is required (`-InstallInno` will try to install it). 32-bit Windows is not supported.
