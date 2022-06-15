"""This package implements multistart generators."""
from __future__ import annotations

from bqskit.ir.opt.multistartgens.diagonal import DiagonalStartGenerator
from bqskit.ir.opt.multistartgens.random import RandomStartGenerator

__all__ = ['DiagonalStartGenerator', 'RandomStartGenerator']
