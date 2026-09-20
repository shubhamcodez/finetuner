"""Launch the selected engine on a fixed local port (default 1234)."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from finetuner.core.paths import app_data_dir
from finetuner.inference.devices import apply_best_runtime
from finetuner.inference.http_server import CallbackBackend, GenerationBackend, start_builtin_server
from finetuner.inference.runner import (
    build_serve_command,
    detect_source_format,
    optimize_inference_engine,
)
from finetuner.inference.specs import (
    InferenceEngine,
    InferenceOptimizationConfig,
    SourceFormat,
    get_engine_spec,
)
from finetuner.quantization.specs import DeviceTarget

DEFAULT_SERVE_PORT = 1234
DEFAULT_SERVE_HOST = "127.0.0.1"


@dataclass
class ServeStatus:
    url: str
    port: int
    engine: str
    target: str
    optimized: bool
    backend: str
    model_path: str
    pid: int | None = None
    detail: str = ""


@dataclass
class _LiveServer:
    status: ServeStatus
    process: subprocess.Popen[str] | None = None
    httpd: Any = None
    thread: threading.Thread | None = None
    log: Callable[[str], None] = field(default=lambda _message: None)


_LIVE: _LiveServer | None = None


def serve_url(port: int = DEFAULT_SERVE_PORT) -> str:
    return f"http://{DEFAULT_SERVE_HOST}:{port}"


def current_server() -> ServeStatus | None:
    return _LIVE.status if _LIVE else None


def portable_serve_config(
    model_path: str, base: InferenceOptimizationConfig | None = None
) -> InferenceOptimizationConfig:
    """Unoptimized serve: keep the artifact as-is and use a portable engine."""
    source = detect_source_format(model_path)
    port = base.serve_port if base else DEFAULT_SERVE_PORT
    toolchain = base.toolchain_path if base else ""
    extras = dict(base.extra_options or {}) if base else {}
    extras["optimized"] = False
    if source == SourceFormat.ONNX:
        extras.setdefault("execution_provider", "CPUExecutionProvider")
        return InferenceOptimizationConfig(
            engine=InferenceEngine.ONNXRUNTIME.value,
            target=DeviceTarget.CPU.value,
            source_format=SourceFormat.ONNX.value,
            serve_port=port,
            toolchain_path=toolchain,
            extra_options=extras,
        )
    if source == SourceFormat.OPENVINO_IR:
        return InferenceOptimizationConfig(
            engine=InferenceEngine.OPENVINO.value,
            target=DeviceTarget.CPU.value,
            source_format=SourceFormat.OPENVINO_IR.value,
            serve_port=port,
            toolchain_path=toolchain,
            extra_options=extras,
        )
    source_format = (
        SourceFormat.GGUF.value if source == SourceFormat.GGUF else SourceFormat.HF.value
    )
    return InferenceOptimizationConfig(
        engine=InferenceEngine.LLAMACPP.value,
        target=DeviceTarget.CPU.value,
        source_format=source_format,
        serve_port=port,
        toolchain_path=toolchain,
        extra_options=extras,
    )


def find_llama_server(toolchain_path: str = "") -> str:
    names = ("llama-server.exe", "llama-server")
    root = Path(toolchain_path).expanduser().resolve() if toolchain_path else None
    if root:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    bench = app_data_dir() / "bench"
    if bench.is_dir():
        matches = sorted(bench.rglob("llama-server.exe")) + sorted(bench.rglob("llama-server"))
        if matches:
            return str(matches[0])
    return ""


def _tool_exists(command: list[str]) -> bool:
    if not command:
        return False
    binary = Path(command[0])
    return binary.exists() or shutil.which(command[0]) is not None


def resolve_launch_command(
    model_path: str, output_dir: str, config: InferenceOptimizationConfig
) -> list[str]:
    """Return a specialist CLI argv when one can actually be executed."""
    try:
        command = build_serve_command(model_path, output_dir, config)
    except (ValueError, AssertionError):
        command = []
    if command and _tool_exists(command):
        return command
    source = detect_source_format(model_path)
    if source == SourceFormat.GGUF or config.engine == InferenceEngine.LLAMACPP.value:
        server = find_llama_server(config.toolchain_path)
        if server:
            model = Path(model_path)
            weights = str(model.resolve())
            if model.is_dir():
                matches = sorted(model.glob("*.gguf"))
                if matches:
                    weights = str(matches[0].resolve())
            if Path(weights).suffix.lower() == ".gguf":
                return [
                    server,
                    "-m",
                    weights,
                    "-c",
                    str(config.max_context),
                    "--host",
                    DEFAULT_SERVE_HOST,
                    "--port",
                    str(config.serve_port),
                ]
    return []


def _maybe_convert(
    model_path: str,
    output_dir: Path,
    project_config: Any,
    log: Callable[[str], None],
) -> str:
    source = detect_source_format(model_path)
    spec = get_engine_spec(project_config.inference.engine)
    needs_gguf = spec.engine == InferenceEngine.LLAMACPP and source == SourceFormat.HF
    needs_ov = spec.engine == InferenceEngine.OPENVINO and source == SourceFormat.HF
    if not (needs_gguf or needs_ov):
        if source in spec.source_formats:
            return model_path
        if spec.engine == InferenceEngine.LLAMACPP and source == SourceFormat.GGUF:
            return model_path
    try:
        from finetuner.quantization.runner import quantize_model

        converted = quantize_model(
            model_path,
            str(output_dir / "quantize"),
            project_config.quantization,
            log,
        )
        log(f"Converted {source.value} weights for {spec.name}: {converted}")
        return converted
    except Exception as exc:
        log(f"Could not convert for {spec.name}: {exc}")
        return model_path


def prepare_serve_model(
    model_path: str,
    output_dir: str,
    project_config: Any,
    *,
    optimize: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[str, InferenceOptimizationConfig, str]:
    """Apply the device recipe when asked; otherwise keep a portable serve config."""
    logger = log or (lambda _message: None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if optimize:
        choice = apply_best_runtime(project_config, model_path=model_path, run_on_device=True)
        logger(choice.reason)
        for skipped in choice.skipped:
            logger(skipped)
        served_model = _maybe_convert(model_path, output, project_config, logger)
        config = project_config.inference
        config.serve_port = config.serve_port or DEFAULT_SERVE_PORT
        extras = dict(config.extra_options or {})
        extras["optimized"] = True
        config.extra_options = extras
        try:
            plan_dir = optimize_inference_engine(served_model, str(output / "optimize"), config, logger)
        except ValueError as exc:
            logger(f"Device plan skipped ({exc}); serving the original weights without that step")
            served_model = model_path
            config = portable_serve_config(model_path, config)
            plan_dir = str(output.resolve())
        return served_model, config, plan_dir
    config = portable_serve_config(model_path, project_config.inference)
    project_config.inference = config
    return model_path, config, str(output.resolve())


def _load_generation_backend(model_path: str, _config: InferenceOptimizationConfig) -> GenerationBackend:
    source = detect_source_format(model_path)

    def _transformers() -> GenerationBackend:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model_id = Path(model_path).name

        def generate(prompt: str, max_tokens: int) -> str:
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=max_tokens)
            return tokenizer.decode(output[0], skip_special_tokens=True)

        return CallbackBackend(generate, model_id)

    if source == SourceFormat.ONNX:
        try:
            import onnxruntime_genai as og

            model = og.Model(model_path)
            tokenizer = og.Tokenizer(model)

            def generate(prompt: str, max_tokens: int) -> str:
                params = og.GeneratorParams(model)
                params.set_search_options(max_length=max_tokens)
                generator = og.Generator(model, params)
                generator.append_tokens(tokenizer.encode(prompt))
                while not generator.is_done():
                    generator.generate_next_token()
                return tokenizer.decode(generator.get_sequence(0))

            return CallbackBackend(generate, Path(model_path).name)
        except Exception:
            pass
    try:
        return _transformers()
    except Exception as exc:
        raise RuntimeError(
            "No serve backend is available for this artifact. Install transformers "
            "for Hugging Face weights, or add llama-server and a GGUF file."
        ) from exc


def _wait_for_port(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((DEFAULT_SERVE_HOST, port), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _pump_process_output(process: subprocess.Popen[str], log: Callable[[str], None]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())


def stop_server() -> None:
    global _LIVE
    live = _LIVE
    _LIVE = None
    if live is None:
        return
    if live.httpd is not None:
        try:
            live.httpd.shutdown()
            live.httpd.server_close()
        except Exception:
            pass
    if live.process is not None and live.process.poll() is None:
        live.process.terminate()
        try:
            live.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            live.process.kill()
    if live.thread is not None and live.thread.is_alive():
        live.thread.join(timeout=2)


def start_bound_server(
    model_path: str,
    output_dir: str,
    config: InferenceOptimizationConfig,
    *,
    optimized: bool,
    log: Callable[[str], None] | None = None,
) -> ServeStatus:
    """Bind an already-chosen recipe on the configured serve port."""
    global _LIVE
    logger = log or (lambda _message: None)
    stop_server()
    port = config.serve_port or DEFAULT_SERVE_PORT
    command = resolve_launch_command(model_path, output_dir, config)
    if command:
        logger(f"Serving with {Path(command[0]).name} on {serve_url(port)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
            creationflags=_creationflags(),
        )
        pump = threading.Thread(target=_pump_process_output, args=(process, logger), daemon=True)
        pump.start()
        if process.poll() is not None or not _wait_for_port(port, timeout=90):
            output = ""
            if process.stdout is not None:
                try:
                    output = process.stdout.read()
                except Exception:
                    output = ""
            if process.poll() is None:
                process.terminate()
            raise RuntimeError(
                f"Serve process exited before binding {serve_url(port)}. {output}".strip()
            )
        status = ServeStatus(
            url=serve_url(port),
            port=port,
            engine=config.engine,
            target=config.target,
            optimized=optimized,
            backend=Path(command[0]).name,
            model_path=model_path,
            pid=process.pid,
            detail=" ".join(
                Path(part).name if index == 0 else part for index, part in enumerate(command)
            ),
        )
        _LIVE = _LiveServer(status=status, process=process, thread=pump, log=logger)
        logger(f"Model is serving at {status.url}")
        return status

    logger(f"No specialist CLI; starting built-in server on {serve_url(port)}")
    backend = _load_generation_backend(model_path, config)
    httpd = start_builtin_server(DEFAULT_SERVE_HOST, port, backend)

    def _serve() -> None:
        with httpd:
            httpd.serve_forever()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    if not _wait_for_port(port, timeout=10):
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass
        raise RuntimeError(f"Built-in server did not bind {serve_url(port)}")
    status = ServeStatus(
        url=serve_url(port),
        port=port,
        engine=config.engine,
        target=config.target,
        optimized=optimized,
        backend="builtin",
        model_path=model_path,
        detail="OpenAI-compatible /v1/chat/completions",
    )
    _LIVE = _LiveServer(status=status, httpd=httpd, thread=thread, log=logger)
    logger(f"Model is serving at {status.url}")
    return status


def launch_inference_server(
    model_path: str,
    output_dir: str,
    project_config: Any,
    *,
    optimize: bool = False,
    log: Callable[[str], None] | None = None,
) -> ServeStatus:
    """Prepare (optional optimize) and bind the model on the configured serve port."""
    logger = log or (lambda _message: None)
    served_model, config, plan_dir = prepare_serve_model(
        model_path,
        output_dir,
        project_config,
        optimize=optimize,
        log=logger,
    )
    return start_bound_server(
        served_model, plan_dir, config, optimized=optimize, log=logger
    )


def launch_from_plan(
    plan_dir: str,
    project_config: Any,
    *,
    log: Callable[[str], None] | None = None,
) -> ServeStatus:
    """Start serving from an existing optimize plan without re-planning."""
    import json

    logger = log or (lambda _message: None)
    root = Path(plan_dir)
    plan_path = root / "inference_plan.json" if root.is_dir() else root
    if not plan_path.exists():
        raise FileNotFoundError(f"No inference plan at {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    model_path = str(plan.get("source_model") or "")
    if not model_path:
        raise ValueError("inference plan is missing source_model")
    if plan.get("optimization"):
        restored = InferenceOptimizationConfig.from_dict(plan["optimization"])
        restored.serve_port = (
            int(plan.get("serve_port") or project_config.inference.serve_port)
            or DEFAULT_SERVE_PORT
        )
        project_config.inference = restored
    extras = dict(project_config.inference.extra_options or {})
    extras["optimized"] = True
    project_config.inference.extra_options = extras
    return start_bound_server(
        model_path,
        str(plan_path.parent),
        project_config.inference,
        optimized=True,
        log=logger,
    )


def resolved_model_path(model: Any) -> str:
    from finetuner.core.job import ModelSource
    from finetuner.core.model_catalog import find_downloaded_model

    if getattr(model, "source", None) == ModelSource.LOCAL:
        return str(model.identifier)
    if getattr(model, "output_path", ""):
        return str(model.output_path)
    return find_downloaded_model(model.identifier) or ""


# Silence unused-import noise if Windows needs the module later.
def _creationflags() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return 0
