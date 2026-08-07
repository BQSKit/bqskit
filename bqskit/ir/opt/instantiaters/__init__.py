"""This package contains instantiation implementations."""
# from bqskitrs import CeresInstantiater
from __future__ import annotations

from bqskit.ir.opt.instantiaters.minimization import Minimization
from bqskit.ir.opt.instantiaters.qfactor import QFactor


instantiater_order = [Minimization, QFactor]


__all__ = [
    'instantiater_order',
    'QFactor',
    'Minimization',
]
