# Research rationale and public evidence

This document distinguishes public evidence from inference. Frontier labs do not publish complete
production recipes, data mixtures, reward systems, or infrastructure. Finetuner therefore exposes
method families and composable stages rather than advertising a guessed “OpenAI/xAI/Anthropic recipe.”

## Publicly described post-training patterns

| Organization | Publicly described techniques | Product implication |
|---|---|---|
| OpenAI | InstructGPT describes demonstration SFT, a preference reward model, and PPO. Deliberative alignment describes synthetic specification-aware reasoning data followed by reward-model RL. | Classic RLHF template; custom synthetic-data/distillation stages; verifier/reward hooks. |
| Anthropic | Constitutional AI describes self-critique/revision SFT and reinforcement learning from AI feedback (RLAIF). | Sequence distillation supports critique/revision data; reward stages accept model-based preferences. |
| Google DeepMind | Gemini 1 describes curated prompts, SFT, reward modeling, and RLHF. GKD trains on a mixture that can include student-generated sequences to reduce train/deployment distribution mismatch. | Staged workflows and experimental on-policy GKD with an explicit student-generated fraction. |
| xAI | Public Grok material describes large-scale RL, human feedback, verifiable rewards, model grading, synthetic data, and long-running agentic rollouts. Exact objectives are not disclosed. | Online GRPO/RLOO-style building blocks, external verifier hooks, and long-stage manifests—without claiming method identity. |

Primary sources:

- OpenAI, [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- OpenAI, [Deliberative alignment](https://openai.com/index/deliberative-alignment/)
- Anthropic, [Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- Google DeepMind, [Gemini: A Family of Highly Capable Multimodal Models](https://deepmind.google/gemini/gemini_1_report.pdf)
- xAI, [Grok 4 model card](https://data.x.ai/2025-08-20-grok-4-model-card.pdf)

## Optimization families

- **SFT** is the stable starting point for demonstrations, instruction following, and teacher-generated
  sequences.
- **Reward model + PPO (classic RLHF)** is appropriate when a learned scalar reward should generalize
  beyond fixed preference pairs, but it is operationally expensive and vulnerable to reward hacking.
- **DPO** directly optimizes chosen/rejected pairs without a separate learned reward model. It is
  simpler than PPO but still depends heavily on pair quality, reference-policy choice, and coverage.
- **KTO** consumes desirable/undesirable binary feedback when pairs are unavailable.
- **ORPO** combines supervised likelihood and a reference-free odds-ratio preference objective. TRL
  currently labels its implementation experimental.
- **GRPO/RLOO** are online methods suited to sampled completions and programmatic/model-based graders.
  They must be monitored for reward exploitation, mode collapse, and divergence from the reference.

The application requires real preference schemas by default. Its synthetic-negative option is clearly
marked research-only because naive corrupted answers create shortcuts and can contaminate conclusions.

## Distillation choices

1. **Sequence-level KD** generates teacher responses and trains the student with SFT. It works across
   unrelated tokenizers and black-box teachers, is easy to audit, and transfers behavior rather than
   the full output distribution.
2. **Forward logit KL** transfers soft token probabilities but requires vocabulary/token alignment and
   simultaneous teacher/student memory.
3. **On-policy GKD** includes student-generated sequences, asking the teacher to score behavior from the
   distribution the student will actually visit. The student-generated fraction is configurable.
4. **Reverse-KL/MiniLLM-style objectives** are mode-seeking and can avoid assigning too much student
   probability to low-probability teacher regions, but may reduce diversity.

Primary sources:

- Kim and Rush, [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- Google DeepMind, [On-Policy Distillation of Language Models](https://deepmind.google/research/publications/48050/)
- Gu et al., [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)

Every generated sequence dataset receives a content digest and manifest with teacher, scope, decoding
parameters, and seed. Domain selection first uses dataset metadata (`domain`, `subject`, `category`,
`topic`, `field`, `tags`) and then transparent keyword matching; it is not an opaque classifier.

## Quantization and device deployment

Quantization is a target-specific accuracy/latency/size tradeoff, not a universal conversion:

- GGUF/llama.cpp supports multiple integer widths and broad CPU, CUDA, HIP, Metal, Vulkan, and SYCL
  backends.
- OpenVINO/NNCF supports INT8 and INT4 weight compression for compatible Intel CPU/GPU devices and
  supported Intel NPUs.
- ONNX Runtime exposes many execution providers, but a generic ONNX INT8 graph is not automatically a
  valid NPU artifact. Finetuner currently limits generic ONNX quantization to CPU and refuses to claim a
  Qualcomm NPU path without a QNN/QAIRT-specific recipe.
- AWQ uses calibration activations and currently targets compatible NVIDIA serving stacks.

Primary sources:

- [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
- OpenVINO, [LLM weight compression](https://docs.openvino.ai/nightly/openvino-workflow/model-optimization-guide/weight-compression.html)
- ONNX Runtime, [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- Hugging Face Optimum ONNX, [Quantization](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/quantization)

## Representation and activation analysis

Finetuner extracts mean- or last-token hidden states from selected layers and provides:

- PCA for deterministic global variance structure
- t-SNE for local-neighborhood exploration (never interpreted as faithful global geometry)
- UMAP as an optional nonlinear neighborhood view
- activation-norm summaries for scale/drift checks
- attention entropy when the model/backend exposes attention tensors
- linear CKA for representation similarity across layers

These plots are correlational. Clusters do not prove that a model contains a human-like topic module,
and attention weights are not causal explanations. Sparse autoencoders, feature attribution, activation
patching, and steering interventions are stronger future extensions because they can isolate features
and test causal influence. Anthropic’s public work uses sparse dictionary learning and UMAP feature
neighborhoods at much larger scale: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).
