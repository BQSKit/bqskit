"""This package contains instantiation implementations."""
# from bqskitrs import CeresInstantiater
from __future__ import annotations

from bqskit.ir.opt.instantiaters.minimization import Minimization
from bqskit.ir.opt.instantiaters.qfactor import QFactor
from bqskit.ir.opt.instantiaters.qfactor_einsum import QFactor_einsum
from bqskit.ir.opt.instantiaters.qfactor_jax import QFactor_jax
from bqskit.ir.opt.instantiaters.qfactor_jax_batched import QFactor_jax_batched
from bqskit.ir.opt.instantiaters.qfactor_jax_batched_jit import QFactor_jax_batched_jit


instantiater_order = [QFactor, Minimization, QFactor_einsum, QFactor_jax, QFactor_jax_batched, QFactor_jax_batched_jit]


__all__ = [
    'instantiater_order',
    'QFactor',
    'QFactor_einsum',
    'QFactor_jax',
    'qfactor_jax_batched',
    'qfactor_jax_batched_jit',
    'Minimization',
]
