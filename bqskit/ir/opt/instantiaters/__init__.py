"""This package contains instantiation implementations."""
# from bqskitrs import CeresInstantiater
from __future__ import annotations
from bqskit.ir.opt.instantiaters.qfactor import QFactor
from bqskit.ir.opt.instantiaters.minimization import Minimization


instantiater_order = [QFactor, Minimization]


__all__ = [
    "instantiater_order",
    "QFactor",
    "Minimization"
]
