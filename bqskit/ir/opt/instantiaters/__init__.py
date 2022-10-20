"""This package contains instantiation implementations."""
# from bqskitrs import CeresInstantiater
from __future__ import annotations

from bqskit.ir.opt.instantiaters.minimization import Minimization
from bqskit.ir.opt.instantiaters.qfactor import QFactor
from bqskit.ir.opt.instantiaters.qfactor_einsum import QFactor_einsum


instantiater_order = [QFactor, Minimization, QFactor_einsum]


__all__ = [
    'instantiater_order',
    'QFactor',
    'QFactor_einsum'
    'Minimization',
]
