"""RFSD recommendation model package with lazy public imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .main import RFSDConfig

if TYPE_CHECKING:
    from .data import RFSDData
    from .model import RFSD
    from .trainer import TrainingResult

__all__ = ["RFSD", "RFSDConfig", "RFSDData", "TrainingResult", "train_rfsd"]


def __getattr__(name: str):
    if name == "RFSD":
        from .model import RFSD

        return RFSD
    if name == "RFSDData":
        from .data import RFSDData

        return RFSDData
    if name in {"TrainingResult", "train_rfsd"}:
        from .trainer import TrainingResult, train_rfsd

        return {"TrainingResult": TrainingResult, "train_rfsd": train_rfsd}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
