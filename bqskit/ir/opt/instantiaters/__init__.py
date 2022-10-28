"""This package contains instantiation implementations."""
# from bqskitrs import CeresInstantiater
from __future__ import annotations

from bqskit.ir.opt.instantiaters.minimization import Minimization
from bqskit.ir.opt.instantiaters.qfactor import QFactor
from bqskit.ir.opt.instantiaters.qfactor_jax_batched_jit import QFactor_jax_batched_jit


instantiater_order = [QFactor, Minimization, QFactor_jax_batched_jit]


__all__ = [
    'instantiater_order',
    'QFactor',
    'qfactor_jax_batched_jit',
    'Minimization',
]
