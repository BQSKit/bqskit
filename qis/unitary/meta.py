"""This module implements the UnitaryMeta metaclass."""
from __future__ import annotations

import abc
from typing import Any


class UnitaryMeta(abc.ABCMeta):
    """
    The UnitaryMeta metaclass.

    Necessary to provide isinstance checks for composed classes.
    """

    def __instancecheck__(cls, instance: Any) -> bool:
        """
        Check if an instance is a `Unitary` instance.

        Additional checks for DifferentiableUnitary and
        LocallyOptimizableUnitary. We check if the object has
        the is_differentiable or is_locally_optimizable callable, an
        instance method that maps nothing to a bool. If the object has
        the method, then it must return true for isinstance to pass.

        This can be used with composed classes to implement
        conditional inheritance.
        """
        if cls.__name__ == 'DifferentiableUnitary':
            if hasattr(instance, 'is_differentiable'):
                if not instance.is_differentiable():
                    return False

        if cls.__name__ == 'LocallyOptimizableUnitary':
            if hasattr(instance, 'is_locally_optimizable'):
                if not instance.is_locally_optimizable():
                    return False

        return super().__instancecheck__(instance)
