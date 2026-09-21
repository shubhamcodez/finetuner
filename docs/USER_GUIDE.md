# Inferna user guide

Inferna is a local desktop workbench for post-training language models. Each job is an independent tool: train, distill, evaluate, analyze, quantize, or optimize inference. Tools share the model queue and, when needed, the dataset on **Data & Train**. They do not wait on each other.

This guide is for people using the app. For internals see [ARCHITECTURE.md](ARCHITECTURE.md). For method rationale see [RESEARCH.md](RESEARCH.md).

## What you need

- Windows 10/11 with Python 3.10 or newer
- An NVIDIA GPU and a CUDA build of PyTorch for local training
- Disk space for model downloads (hundreds of MB to several GB per model)
- Optional: a Hugging Face token for gated models (`HF_TOKEN` or the token field on **Data & Train**)

Quantized artifacts can target CPU, AMD GPU, Apple GPU, Intel GPU/NPU, or Qualcomm NPU. The trainer itself still requires NVIDIA CUDA.

## Install and launch

From the project directory:

```powershell
.\scripts\install_gpu.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
.\.venv\Scripts\python.exe -m finetuner
```

Optional extras, only if you will use those backends:

```powershell
pip install -e ".[openvino]"    # Intel CPU / GPU / NPU
pip install -e ".[onnx]"        # ONNX Runtime INT8 / Qualcomm QNN
pip install llm-awq             # NVIDIA AWQ
```

GGUF conversion needs a local [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout selected on **Deployment**.

To pin a desktop shortcut after the venv exists:

```powershell
.\scripts\create_desktop_shortcut.ps1
```

Use the Windows installer that matches this PC: `Inferna-Setup-x64.exe` on Intel/AMD,
`Inferna-Setup-arm64.exe` on Snapdragon / ARM64, or `Inferna-Setup-universal.exe` if you
have the combined package. The installer is per-user by default (`%LOCALAPPDATA%\Programs\Inferna`)
and does not require Administrator. To build it:

```powershell
.\scripts\build_windows.ps1              # native installer for this PC
.\scripts\build_windows.ps1 -Universal   # after both x64 and arm64 payloads exist
```

Inno Setup 6.3+ is required for the `.exe` installer. The script also writes a portable zip
under `dist\`. 32-bit Windows is not supported.

## Workspace

The window has a header status badge, product tabs, and a **Run Console** at the bottom.

| Tab | What it does |
|---|---|
| **Project** | Readiness overview. Open or run any ready tool from its card. |
| **Models** | Queue Hugging Face or local models. Most tools run on this queue. |
| **Finetune** | Dataset, method, rewards, and optional LoRA settings. |
| **Distillation** | Teacher → student transfer. Uses its own teacher/student fields, not the queue. |
| **Evaluation** | Benchmarks (MMLU, GSM8K, HellaSwag, ARC Challenge). |
| **Analysis** | Hidden-state projections, activation norms, attention entropy, CKA. |
| **Deployment** | Quantize a queued model for a concrete backend and device. |
| **Inference** | Detect device, optional optimize, then serve on port 1234. |
| **Results** | Per-model scores and artifact links from the latest run. |
| **System** | Live CPU, RAM, NVIDIA GPU, NPU, and TPU utilization. |

Settings save automatically to `%LOCALAPPDATA%\.finetuner\config.json`. Hugging Face tokens are not written there; they stay in memory or come from `HF_TOKEN`.

### Running a tool

1. Open the tab for the tool you want, or use **Run** on a ready **Project** card.
2. Fix any “Needs attention” items. Inferna blocks a start until required fields are valid.
3. Click **Run …** on the tab, **Start Run** in the console, or **Run** on the Project card.
4. Watch the **Run Console**. Cancel stops the queue; it does not roll back a finished model.

Only one tool runs at a time. Closing the window while a job is running asks you to cancel first.

### Where files go

| Path | Contents |
|---|---|
| `%LOCALAPPDATA%\.finetuner\config.json` | Saved project settings (no token) |
| `%LOCALAPPDATA%\.finetuner\models\` | Hugging Face downloads |
| `%LOCALAPPDATA%\.finetuner\runs\` | Per-run artifacts and `manifest.json` |
| `%LOCALAPPDATA%\.finetuner\cache\` | Local cache |

Each completed tool writes a `manifest.json` with stage status, duration, metrics, a redacted config digest, and artifact lineage.

## First run

This is the shortest path that exercises the product without a large download.

1. Launch Inferna. Two small instruct models are already queued: `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2-0.5B-Instruct`.
2. On **Models**, select one row and click **Download Selected**. Wait until **Local Path** is filled. Inferna detects this device and asks **Optimize for your device?**
3. Choose **Optimize and serve** or **Serve without optimizing**. The model is bound at `http://127.0.0.1:1234`.
4. On **Data & Train**, pick **Bundled sample (smoke test)** or another preset and check **Use offline sample only**.
5. Leave the method on **SFT**.
6. Click **Run finetune**.
7. Open **Results** when the console says the run finished. Policy output is under `runs\`.

For a real experiment, download a model, pick a full preset (not the bundled sample), and raise **Max steps** under advanced settings.

## Models

Queued models are the input for training, evaluation, analysis, deployment, and inference. Distillation does **not** use the queue; it uses the teacher and student fields on its own tab.

**Add Model** accepts:

- **Hugging Face** — the name field lists current Hugging Face trending text-generation models. Pick one, or type any repo id such as `Qwen/Qwen2.5-0.5B-Instruct`. Download it here, or let a later run pull it.
- **Local Path** — a Hugging Face checkpoint folder, a folder of `.gguf` / `.onnx` weights, or a single GGUF/ONNX file.

After the weights are on disk, Inferna detects this machine and asks whether to optimize before serving on port 1234. **Not now** only queues the model.

Give gated models a token on **Data & Train** (advanced settings) or set `HF_TOKEN` in the environment.

To feed a trained or quantized artifact into another tool, add that folder as a **Local Path** model and run the next tool on the queue.

## Data & Train

Training needs at least one queued model and a dataset.

### Datasets

Choose one source. A custom path or Hugging Face id clears the preset.

| Source | When to use |
|---|---|
| Ready-made preset | Fast start aligned with a benchmark |
| Hugging Face trending | Current text-generation / conversational datasets from the Hub |
| Offline sample only | No network; small bundled JSONL (eval presets only) |
| Local JSONL/JSON | Your own data |
| Hugging Face dataset id | A public or gated dataset, e.g. `tatsu-lab/alpaca` |

Eval-aligned presets:

| Preset | Pairs with eval | Notes |
|---|---|---|
| GSM8K (Math) | GSM8K | Grade-school word problems |
| MMLU (Knowledge) | MMLU | Multitask multiple choice |
| HellaSwag (Commonsense) | HellaSwag | Sentence completion |
| ARC Challenge (Science) | ARC Challenge | Science exam questions |
| Bundled sample | GSM8K | Tiny mix for smoke tests |

The dataset list also includes current Hugging Face trending finetune sets (Alpaca, Dolly, UltraChat, SmolTalk, and whatever is trending now). Those load from the Hub and do not have an offline sample.

**Auto-enable matching eval** adds the paired benchmark to **Evaluation** when you pick an eval-aligned preset.

### Dataset formats

Local JSONL is accepted in several shapes. One object per line:

Instruction / output (Alpaca-style):

```json
{"instruction": "Solve the problem.", "input": "What is 2+2?", "output": "4"}
```

Plain text with a response marker:

```json
{"text": "### Instruction:\nSolve.\n### Input:\n2+2\n### Response:\n4"}
```

Chat messages:

```json
{"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]}
```

Preference methods need real pairs unless you explicitly allow synthetic negatives:

```json
{"prompt": "2+2?", "chosen": "4", "rejected": "5"}
```

KTO needs binary labels:

```json
{"prompt": "2+2?", "completion": "4", "label": true}
```

Online methods (GRPO, PPO, RLOO) need a `prompt` column, or SFT-style text that can be split on `### Response:`.

### Quality recipe

Leave **Quality recipe** checked. That path does full-weight SFT on the real transformer, formats every example with the tokenizer chat template, and computes loss only on assistant tokens. That is what actually moves GSM8K / HellaSwag / ARC.

**Use LoRA** stays off unless you turn it on. Full SFT is the default when the device can hold the model (a 0.5B instruct checkpoint does). Enable LoRA or QLoRA only when you want a PEFT adapter.

The NPU/TPU logit adapter is a frozen-decoder experiment. It does not update hidden states, so benchmark gains stay near zero. Use it only when you explicitly pick **NPU engine** / **TPU engine** and turn Quality recipe off.

### Training methods

| Method | Data it expects | Notes |
|---|---|---|
| **SFT** | Instruction–response pairs | Default starting point |
| **DPO** | `prompt` / `chosen` / `rejected` | Preference pairs; set **Beta** |
| **KTO** | `prompt` / `completion` / `label` | Binary feedback; set **Beta** |
| **ORPO** | Preference pairs | Experimental in TRL |
| **Reward Model** | Preference pairs | Trains a Bradley–Terry scorer |
| **GRPO** | Prompts + reward | On-policy rollouts; set **GRPO gens** |
| **RLOO** | Prompts + reward | Online REINFORCE leave-one-out |
| **PPO** | Prompts + reward model | Requires a **Reward model** id on CUDA/TRL. The NPU/TPU engine uses the selected reward function instead. |
| **NPU SFT / TPU SFT** | Instruction–response pairs | Shortcuts for SFT on the accelerator engine. |

### NPU / TPU engine

Set **Accelerator** to **NPU engine**, **TPU engine**, **CPU engine**, or **Auto**. Every Finetune method (SFT, DPO, KTO, ORPO, Reward, GRPO, RLOO, PPO) then runs on a frozen ONNX decoder plus a small logit LoRA:

- The NPU (Qualcomm Hexagon HTP via QNN, or ONNX CPU if QNN is missing) only produces base logits.
- A Cloud or Coral TPU is used when one is visible; otherwise the adapter trains on the host.
- There is **no backpropagation through Hexagon or the TPU**. The trainable A/B matrices live outside the frozen graph.

Point **ONNX artifact** at a GenAI/QNN package, or leave it blank to use `~/.finetuner/bench/models/qwen2.5-0.5b-instruct-onnx-genai`. QLoRA is disabled on this path.

**Allow synthetic negative preferences** is off by default and marked research-only. Turning it on invents rejected answers from SFT text. Do not use that for production preference runs.

Reward functions for GRPO / PPO / RLOO:

- **Exact match** — 1.0 when the completion matches ground truth (GSM8K-style normalization)
- **Partial match** — fraction of ground-truth tokens found
- **Format + match** — exact match plus a bonus for `### Response` / `####`
- **Length penalty** — exact match minus a penalty for long completions
- **HF reward model** — a Hugging Face sequence-classification model id

PPO always needs a reward-model id. The HF reward function also needs one.

### Hyperparameters

Click **Show advanced settings** for:

- Max steps, learning rate, batch size, gradient accumulation, max sequence length
- **Use LoRA** — off by default. Full SFT updates every weight. Turn this on for PEFT.
- LoRA rank, alpha, and optional target modules (blank = all linear layers; ignored unless LoRA is on)
- **Use QLoRA (4-bit)** — requires LoRA; keeps adapter training on a smaller GPU
- Hugging Face token

Start with the defaults on a 0.5B instruct model (full SFT, LoRA off). Raise steps and batch only after a short run succeeds. If the device cannot hold the full model plus optimizer, turn **Use LoRA** on.

## Distillation

Distillation transfers a **teacher** into a smaller **student**. Pick both from downloaded models, queued models, or a typed Hugging Face id / path. They must be different.

The dataset still comes from **Data & Train**.

| Technique | Use when | Constraint |
|---|---|---|
| Sequence KD (portable) | Different families or tokenizers; black-box teacher | Teacher generates answers; student is SFT’d on them |
| Logit KL (experimental) | Same tokenizer / vocab | Teacher and student in memory together |
| On-policy GKD (experimental) | You want the teacher to score student rollouts | Same tokenizer family; student-generated fraction is used |
| Reverse-KL / MiniLLM-style (experimental) | Mode-seeking transfer | Same tokenizer family |

**Knowledge scope** filters the dataset:

- **All / general** — every row
- **Selected fields** — computer science, mathematics, optimization, science, reasoning, safety
- **Custom topic** — comma-separated terms such as `compiler optimization`

Matching first looks at `domain`, `subject`, `category`, `topic`, `field`, and `tags`, then keyword search over prompt/text fields. It is not a classifier.

Set **Maximum samples** and **Teacher temperature**, then **Run distillation**.

## Evaluation

Select one or more benchmarks and **Max samples per eval**. Then **Run evaluation** on the queued models.

| Benchmark | What it measures |
|---|---|
| MMLU | Broad knowledge (5-shot subset) |
| GSM8K | Grade-school math (8-shot) |
| HellaSwag | Commonsense completion (10-shot) |
| ARC Challenge | Hard science questions (25-shot) |

Scores appear on **Results**. Highest score per task is highlighted. Keep **Max samples** small while iterating; full splits take much longer.

## Analysis

Analysis reads hidden states from queued models on the **Data & Train** dataset.

- **PCA** — fast, deterministic global variance
- **t-SNE** — local neighborhoods (not global geometry)
- **UMAP** — optional nonlinear view (needs the `analysis` extra)
- **Mean token** vs **Last token** pooling
- **Maximum samples** caps how many rows are encoded

After a run, pick a **Layer** to plot. Hover a point for its label and text. The summary shows activation-norm mean/std and, when available, attention entropy.

You can also **Load Analysis...** and open an existing `representations.json`. Clusters are correlational; they do not prove a human-like topic module.

## Deployment

Quantization builds a **target-specific** artifact. It is not a universal converter. Choose backend, device, and bit width, then **Run quantization**.

| Backend | Bits | Typical devices | Extra setup |
|---|---|---|---|
| GGUF / llama.cpp | 2, 3, 4, 5, 6, 8 | CPU, NVIDIA, AMD, Intel GPU, Apple GPU | llama.cpp path |
| OpenVINO IR | 4, 8 | Intel CPU / GPU / NPU | `pip install -e ".[openvino]"` |
| ONNX Runtime | 8 | CPU, or Qualcomm NPU (QNN/HTP) | `pip install -e ".[onnx]"` |
| AWQ | 4 | NVIDIA GPU | `pip install llm-awq` and calibration data |

**Detect This Device** lists what this machine reports. Unsupported backend/device pairs are rejected before conversion.

To serve a quantized model later, queue the output folder as a local model and open **Inference**.

## Inference

Quantization chooses the weight format. This tab chooses **how the model is served**. Loading a model already offers that choice; this tab is the manual override.

The live endpoint is always **`http://127.0.0.1:1234`** (`/health`, `/v1/models`, `/v1/chat/completions`). **Serve without optimizing** binds the current artifact as-is. **Optimize and serve** / **Run best engine for this machine** picks the specialist, converts if needed, then binds the same port. **Stop server** releases it.

Engines and the artifacts they consume:

| Engine | Consumes | Typical device |
|---|---|---|
| llama.cpp | GGUF (HF still needs conversion) | CPU and most GPUs |
| vLLM | Hugging Face or AWQ | NVIDIA GPU |
| TensorRT-LLM | Hugging Face or AWQ | NVIDIA GPU (ahead-of-time compile) |
| OpenVINO | OpenVINO IR or HF | Intel CPU / GPU / NPU |
| ONNX Runtime | ONNX | CPU, CUDA, DirectML/ROCm, or QNN/HTP |
| TGI | Hugging Face or AWQ | NVIDIA GPU or CPU serving |

Tune KV-cache dtype, max context, max batch, tensor parallel, GPU memory fraction, speculative tokens, prefix cache, CUDA graphs, and flash attention. **Compile or cache an engine artifact** is only enabled for engines that build ahead of time (TensorRT-LLM, OpenVINO, ONNX Runtime).

**Run best engine for this machine** detects hardware and picks the strongest specialist that can actually run the current artifact:

- NVIDIA GPU → vLLM (or llama.cpp CUDA if vLLM is missing)
- AMD GPU → llama.cpp (Vulkan/HIP)
- Intel NPU / GPU → OpenVINO
- Qualcomm Hexagon NPU → QNN/HTP **only** if the model is a context-binary / QNN genai package
- Otherwise → llama.cpp on CPU (the portable floor; on Snapdragon X Elite this was the fastest decode we measured)

Overrides: **Run on NVIDIA GPU**, **Run on AMD GPU**, **Run on NPU**. **Recommend** with **Best for this machine** fills the same choice without starting a run. A recipe also updates **Deployment** so the weight format matches the engine.

The run writes `inference_plan.json` (serve/compile plan) and an optional compiled engine plus a `device_bind` probe. **Load Plan...** opens a previous plan.

There is no single kernel that is fastest on Hexagon, CUDA, and Apple GPU. The adaptive engine is a dispatcher, not a universal graph. Generic ONNX INT4/INT8 is not a Qualcomm HTP package. vLLM does not load GGUF. llama.cpp does not load OpenVINO IR.

## Results and System

**Results** compares the latest run across queued models: eval scores plus Policy / Analysis / Deployment / Inference readiness. Double-click a **Ready** cell to open that artifact (analysis plot or inference plan).

**System** polls CPU, RAM, NVIDIA GPU, NPU, and TPU once per second so you can see whether a run is compute-bound or memory-bound.

## Hardware pairing

Plan conversion and serving against the same device.

| You have | Quantize with | Serve with |
|---|---|---|
| NVIDIA GPU | GGUF or AWQ | llama.cpp, vLLM, TensorRT-LLM, or TGI |
| AMD GPU | GGUF | llama.cpp |
| Apple GPU | GGUF | llama.cpp |
| Intel CPU / GPU / NPU | OpenVINO | OpenVINO |
| Qualcomm Hexagon NPU | ONNX (QNN/HTP recipe) | ONNX Runtime QNN/HTP |
| Generic CPU | GGUF or ONNX INT8 | llama.cpp or ONNX Runtime |

Generic ONNX INT8 is not a Qualcomm NPU artifact. vLLM does not load GGUF. llama.cpp does not load OpenVINO IR.

## Reproducibility and secrets

- Config, results, analysis, and manifests use atomic JSON writes.
- Run manifests are local provenance, not an experiment tracker or audit log.
- Tokens belong in `HF_TOKEN` or the in-memory field. Do not put them in shared `config.json` copies.
- Cancellation is checked before a tool starts. Mid-trainer cooperative cancel and resumable distributed checkpoints are not implemented yet.

## Troubleshooting

**“Tool Not Ready” / Needs attention**  
Read the message and open the tab it names. Common causes: empty model queue, no dataset, PPO without a reward-model id, preference method without real pairs, missing teacher/student, or an illegal backend/device pair.

**Download failed**  
Check network access and, for gated repos, `HF_TOKEN`. Only Hugging Face rows can be downloaded from **Models**.

**CUDA still not available**  
Re-run `.\scripts\install_gpu.ps1`. Training needs a CUDA PyTorch wheel (`cu128` in the current installer). CPU PyTorch will not train.

**Out of GPU memory**  
Turn on LoRA or QLoRA, keep batch size 1, and raise gradient accumulation instead. Use a 0.5B–1.5B student. Lower max sequence length.

**DPO / KTO / ORPO rejects the dataset**  
Provide real `prompt`/`chosen`/`rejected` (or KTO labels). Synthetic negatives stay off unless you accept the research-only checkbox.

**GGUF conversion fails**  
Set **llama.cpp path** to a checkout that contains the convert scripts / built binaries.

**Compile does nothing useful**  
Leave **Compile** unchecked to write a serve plan only. Enable it only when TensorRT-LLM, OpenVINO, or ONNX Runtime is installed and you want a materialized engine.

**Analysis plot is empty**  
The run must finish, or load a `representations.json` with `schema_version` 1 and a `layers` list. UMAP needs `pip install -e ".[analysis]"`.

**I want to serve a trained adapter**  
Queue the policy folder from `runs\` as a local model, then run **Deployment** and **Inference** for the device you will actually use.
