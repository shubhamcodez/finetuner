from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from finetuner.core.hf_download import download_hf_model
from finetuner.core.paths import model_download_path


class DownloadWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)
    progress_text = Signal(str)
    progress_percent = Signal(int, str)

    def __init__(self, repo_id: str, token: str = "", parent=None) -> None:
        super().__init__(parent)
        self.repo_id = repo_id
        self.token = token

    def run(self) -> None:
        try:
            dest = model_download_path(self.repo_id)

            def on_progress(percent: int, desc: str) -> None:
                self.progress_percent.emit(percent, desc)

            def on_log(msg: str) -> None:
                self.progress_text.emit(msg)

            path = download_hf_model(
                repo_id=self.repo_id,
                dest=dest,
                token=self.token or None,
                on_progress=on_progress,
                on_log=on_log,
            )
            self.finished_ok.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))
