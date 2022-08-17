"""
This subpackage implements Layout algorithms.

Layout algorithms are responsible for finding the best initial mapping of
logical to physical qudits.
"""
from __future__ import annotations

from bqskit.passes.mapping.layout.sabre import GeneralizedSabreLayoutPass

__all__ = ['GeneralizedSabreLayoutPass']
