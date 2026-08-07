"""
This package implements most OPENQASM 2.0 reading and writing features.

References:
    Andrew W. Cross, Lev S. Bishop, John A. Smolin, Jay M. Gambetta
    "Open Quantum Assembly Language" [arXiv:1707.03429].
"""
from __future__ import annotations

from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language

__all__ = ['OPENQASM2Language']
