from __future__ import annotations

from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[int, str], None]
LogCallback = Callable[[str], None]


def _make_tqdm_class(on_progress: ProgressCallback):
    from huggingface_hub.utils import tqdm as hf_tqdm

    class HubProgressTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("leave", False)
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0:
                pct = min(100, int(self.n / self.total * 100))
            else:
                pct = 0
            desc = str(self.desc or "Downloading...")
            on_progress(pct, desc)

    return HubProgressTqdm


def download_hf_model(
    repo_id: str,
    dest: Path | str,
    token: str | None = None,
    on_progress: ProgressCallback | None = None,
    on_log: LogCallback | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    dest_path = Path(dest)
    if on_log:
        on_log(f"Downloading {repo_id} to {dest_path}...")

    kwargs: dict = {
        "repo_id": repo_id,
        "local_dir": str(dest_path),
        "token": token or None,
    }
    if on_progress:
        kwargs["tqdm_class"] = _make_tqdm_class(on_progress)

    snapshot_download(**kwargs)
    if on_progress:
        on_progress(100, "Download complete")
    return dest_path
