"""This module tests the UnitaryMeta's isinstance checks."""
from __future__ import annotations

import numpy as np

from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.meta import UnitaryMeta
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TestIsLocallyOptimizable:

    def test_normal_inheritance(self) -> None:
        class test_class(LocallyOptimizableUnitary):
            def get_unitary(self, p: RealVector = []) -> UnitaryMatrix:
                pass

            def optimize(self, env_matrix: np.ndarray) -> list[float]:
                pass

        assert isinstance(test_class, UnitaryMeta)
        assert isinstance(test_class(), LocallyOptimizableUnitary)

    def test_conditional_inheritance_true(self) -> None:
        class test_class(LocallyOptimizableUnitary):
            def get_unitary(self, p: RealVector = []) -> UnitaryMatrix:
                pass

            def optimize(self, env_matrix: np.ndarray) -> list[float]:
                pass

            def is_locally_optimizable(self) -> bool:
                return True

        assert isinstance(test_class, UnitaryMeta)
        assert isinstance(test_class(), LocallyOptimizableUnitary)

    def test_conditional_inheritance_false(self) -> None:
        class test_class(LocallyOptimizableUnitary):
            def get_unitary(self, p: RealVector = []) -> UnitaryMatrix:
                pass

            def optimize(self, env_matrix: np.ndarray) -> list[float]:
                pass

            def is_locally_optimizable(self) -> bool:
                return False

        assert isinstance(test_class, UnitaryMeta)
        assert not isinstance(test_class(), LocallyOptimizableUnitary)


class TestIsDifferentiable:

    def test_normal_inheritance(self) -> None:
        class test_class(DifferentiableUnitary):
            def get_unitary(self, p: RealVector = []) -> UnitaryMatrix:
                pass

            def get_grad(self, params: RealVector = []) -> np.ndarray:
                pass

        assert isinstance(test_class, UnitaryMeta)
        assert isinstance(test_class(), DifferentiableUnitary)

    def test_conditional_inheritance_true(self) -> None:
        class test_class(DifferentiableUnitary):
            def get_unitary(self, p: RealVector = []) -> UnitaryMatrix:
                pass

            def get_grad(self, params: RealVector = []) -> np.ndarray:
                pass

            def is_differentiable(self) -> bool:
                return True

        assert isinstance(test_class, UnitaryMeta)
        assert isinstance(test_class(), DifferentiableUnitary)

    def test_conditional_inheritance_false(self) -> None:
        class test_class(DifferentiableUnitary):
            def get_unitary(self, p: RealVector = []) -> UnitaryMatrix:
                pass

            def get_grad(self, params: RealVector = []) -> np.ndarray:
                pass

            def is_differentiable(self) -> bool:
                return False

        assert isinstance(test_class, UnitaryMeta)
        assert not isinstance(test_class(), DifferentiableUnitary)
