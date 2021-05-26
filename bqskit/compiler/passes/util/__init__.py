"""This package implements utility passes."""
from __future__ import annotations

from bqskit.compiler.passes.util.compress import CompressPass
from bqskit.compiler.passes.util.inject import InjectDataPass

__all__ = ['CompressPass', 'InjectDataPass']
