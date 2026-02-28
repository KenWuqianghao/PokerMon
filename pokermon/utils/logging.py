"""TensorBoard logging utilities."""

from __future__ import annotations

from pathlib import Path


class TBLogger:
    """Thin wrapper around TensorBoard SummaryWriter."""

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self._writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(log_dir))

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def scalars(self, main_tag: str, values: dict[str, float], step: int) -> None:
        """Log multiple scalars under one tag."""
        if self._writer:
            self._writer.add_scalars(main_tag, values, step)

    def flush(self) -> None:
        """Flush pending writes."""
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        """Close the logger."""
        if self._writer:
            self._writer.close()
