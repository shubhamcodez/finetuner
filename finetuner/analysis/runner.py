from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from finetuner.analysis.artifacts import LayerAnalysis, RepresentationPoint, write_analysis_artifact
from finetuner.analysis.config import AnalysisConfig
from finetuner.analysis.reducers import linear_cka, reduce_2d
from finetuner.training.common import load_raw_dataset


def _text_and_label(row: dict[str, Any], index: int) -> tuple[str, str]:
    text = next(
        (str(row[key]) for key in ("text", "prompt", "question", "instruction") if row.get(key)),
        str(row),
    )
    label = next(
        (
            str(row[key])
            for key in ("topic", "subject", "domain", "category", "label")
            if row.get(key)
        ),
        "unlabeled",
    )
    return text, label


def analyze_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    config: AnalysisConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> str:
    config = config or AnalysisConfig()
    config.require_valid()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_raw_dataset(dataset_path)
    count = min(len(dataset), config.max_samples)
    if count < 2:
        raise ValueError("Representation analysis requires at least two samples")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    layer_vectors: dict[int, list[np.ndarray]] = {}
    attention_entropy: dict[int, list[float]] = {}
    samples: list[tuple[str, str]] = []

    for index in range(count):
        text, label = _text_and_label(dataset[index], index)
        samples.append((text, label))
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=config.max_length
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        try:
            with torch.inference_mode():
                output = model(
                    **encoded,
                    output_hidden_states=True,
                    output_attentions=config.include_attention_entropy,
                    return_dict=True,
                )
        except (NotImplementedError, ValueError) as exc:
            if not config.include_attention_entropy:
                raise
            if log and index == 0:
                log(f"Attention tensors unavailable for this architecture ({exc}); continuing.")
            with torch.inference_mode():
                output = model(
                    **encoded,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
        hidden_states = output.hidden_states
        selected_layers = config.layers or sorted(
            {0, len(hidden_states) // 2, len(hidden_states) - 1}
        )
        for layer in selected_layers:
            resolved = layer if layer >= 0 else len(hidden_states) + layer
            if not 0 <= resolved < len(hidden_states):
                raise ValueError(f"Layer {layer} is outside 0..{len(hidden_states) - 1}")
            values = hidden_states[resolved][0].float()
            pooled = values[-1] if config.pooling == "last" else values.mean(dim=0)
            layer_vectors.setdefault(resolved, []).append(pooled.cpu().numpy())
            attentions = getattr(output, "attentions", None)
            if attentions and resolved > 0 and resolved - 1 < len(attentions):
                attention = attentions[resolved - 1][0].float().clamp_min(1e-9)
                entropy = -(attention * attention.log()).sum(dim=-1).mean().item()
                attention_entropy.setdefault(resolved, []).append(entropy)
        if log and (index + 1) % 10 == 0:
            log(f"Representation extraction: {index + 1}/{count}")

    analyses: list[LayerAnalysis] = []
    matrices: list[np.ndarray] = []
    for layer, vectors in sorted(layer_vectors.items()):
        matrix = np.stack(vectors)
        matrices.append(matrix)
        points_2d, metadata = reduce_2d(
            matrix, config.reducer, seed=config.seed, perplexity=config.perplexity
        )
        norms = np.linalg.norm(matrix, axis=1)
        points = [
            RepresentationPoint(
                sample_id=str(index),
                label=samples[index][1],
                text=samples[index][0][:500],
                x=float(points_2d[index, 0]),
                y=float(points_2d[index, 1]),
            )
            for index in range(count)
        ]
        entropy = attention_entropy.get(layer, [])
        analyses.append(
            LayerAnalysis(
                layer=layer,
                points=points,
                activation_norm_mean=float(norms.mean()),
                activation_norm_std=float(norms.std()),
                attention_entropy_mean=float(np.mean(entropy)) if entropy else None,
                reducer_metadata=metadata,
            )
        )

    cka = [[linear_cka(left, right) for right in matrices] for left in matrices]
    destination = Path(output_dir) / "representations.json"
    if log:
        log(f"Representation analysis saved to {destination}")
    return write_analysis_artifact(
        destination,
        model=model_path,
        reducer=config.reducer,
        pooling=config.pooling,
        layers=analyses,
        cka=cka,
    )
