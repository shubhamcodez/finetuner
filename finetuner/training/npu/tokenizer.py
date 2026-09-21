from __future__ import annotations

from pathlib import Path


class NpuTokenizer:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._encode = None
        self._decode = None
        self._bind()

    def _bind(self) -> None:
        tokenizer_json = self.model_dir / "tokenizer.json"
        try:
            from tokenizers import Tokenizer

            if tokenizer_json.is_file():
                inner = Tokenizer.from_file(str(tokenizer_json))
                self._encode = lambda text: inner.encode(text).ids
                self._decode = lambda ids: inner.decode(ids)
                return
        except Exception:
            pass
        try:
            import onnxruntime_genai as og

            inner = og.Tokenizer(og.Model(str(self.model_dir)))
            self._encode = lambda text: list(inner.encode(text))
            self._decode = lambda ids: inner.decode(ids)
            return
        except Exception:
            pass
        raise RuntimeError(
            f"Cannot tokenize {self.model_dir}. Install `tokenizers` or `onnxruntime-genai`."
        )

    def encode(self, text: str) -> list[int]:
        return [int(item) for item in self._encode(text or "")]

    def decode(self, ids: list[int]) -> str:
        return str(self._decode([int(item) for item in ids]))
