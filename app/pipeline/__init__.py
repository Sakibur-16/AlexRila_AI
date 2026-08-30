"""Pipeline orchestration.

:class:`~app.pipeline.receipt_pipeline.ReceiptPipeline` is the single entry
point from document bytes to a :class:`~app.schemas.pipeline.PipelineResult`.
"""

from __future__ import annotations

from app.pipeline.receipt_pipeline import ReceiptPipeline

__all__ = ["ReceiptPipeline"]
