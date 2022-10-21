"""This package contains instantiation implementations."""
# from bqskitrs import CeresInstantiater
from __future__ import annotations

from bqskit.ir.opt.instantiaters.minimization import Minimization
from bqskit.ir.opt.instantiaters.qfactor import QFactor
from bqskit.ir.opt.instantiaters.qfactor_einsum import QFactor_einsum
from bqskit.ir.opt.instantiaters.qfactor_jax import QFactor_jax


instantiater_order = [QFactor, Minimization, QFactor_einsum, QFactor_jax]


__all__ = [
    'instantiater_order',
    'QFactor',
    'QFactor_einsum',
    'QFactor_jax',
    'Minimization',
]
