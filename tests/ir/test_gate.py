from __future__ import annotations

import numpy as np
import pytest

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TestBasicGate:

    def test_get_name(self, gate: Gate) -> None:
        assert isinstance(gate.get_name(), str)

    def test_get_num_params(self, gate: Gate) -> None:
        assert isinstance(gate.get_num_params(), int)
        assert gate.get_num_params() >= 0

    def test_get_num_params_constant(self, constant_gate: Gate) -> None:
        assert constant_gate.get_num_params() == 0

    def test_get_num_params_parameterized(self, param_gate: Gate) -> None:
        assert param_gate.get_num_params() != 0

    def test_get_radixes(self, gate: Gate) -> None:
        assert isinstance(gate.get_radixes(), tuple)
        assert all(isinstance(radix, int) for radix in gate.get_radixes())
        assert all(radix > 0 for radix in gate.get_radixes())

    def test_get_radixes_qubit(self, qubit_gate: Gate) -> None:
        assert all(radix == 2 for radix in qubit_gate.get_radixes())

    def test_get_radixes_qutrit(self, qutrit_gate: Gate) -> None:
        assert all(radix == 3 for radix in qutrit_gate.get_radixes())

    def test_get_size(self, gate: Gate) -> None:
        assert isinstance(gate.get_size(), int)
        assert gate.get_size() > 0

    def test_gate_size_matches_radixes(self, gate: Gate) -> None:
        assert len(gate.get_radixes()) == gate.get_size()

    def test_get_dim(self, gate: Gate) -> None:
        assert isinstance(gate.get_dim(), int)
        assert gate.get_dim() > 0

    def test_get_qasm_name(self, gate: Gate) -> None:
        try:
            qasm_name = gate.get_qasm_name()
        except AttributeError:
            return
        except BaseException:
            assert False, 'Unexpected error on gate.get_qasm_name() call.'

        assert isinstance(qasm_name, str)

    def test_get_qasm_gate_def(self, gate: Gate) -> None:
        try:
            qasm_gate_def = gate.get_qasm_gate_def()
        except AttributeError:
            return
        except BaseException:
            assert False, 'Unexpected error on gate.get_qasm_gate_def() call.'

        assert isinstance(qasm_gate_def, str)

    def test_get_unitary(self, gate: Gate) -> None:
        params = np.random.rand(gate.get_num_params())
        utry = gate.get_unitary(params)
        assert isinstance(utry, UnitaryMatrix)

    def test_unitary_dim_match(self, gate: Gate) -> None:
        params = np.random.rand(gate.get_num_params())
        utry = gate.get_unitary(params)
        assert utry.get_shape() == (gate.get_dim(), gate.get_dim())

    # def test_get_grad(self, gate: Gate) -> None:
    #     grads = gate.get_grad([0] * gate.get_num_params())
    #     assert isinstance(grads, np.ndarray)

    #     num_params = gate.get_num_params()
    #     dim = gate.get_dim()
    #     shapes_match = grads.shape == (num_params, dim, dim)
    #     empty_shape_and_no_params = grads.shape[0] == 0 and num_params == 0
    #     assert shapes_match or empty_shape_and_no_params

    # def test_get_unitary_and_grad(self, gate: Gate) -> None:
    #     params = np.random.rand(gate.get_num_params())
    #     grad1 = gate.get_grad(params)
    #     utry1 = gate.get_unitary(params)
    #     utry2, grad2 = gate.get_unitary_and_grad(params)
    #     assert np.allclose(grad1, grad2)
    #     assert np.allclose(utry1, utry2)

    # def test_optimize_valid(self, gate: Gate) -> None:
    #     try:
    #         env_matrix = np.random.rand(gate.get_dim(), gate.get_dim())
    #         optimal_params = gate.optimize(env_matrix)
    #     except NotImplementedError:
    #         return
    #     except BaseException:
    #         assert False, 'Unexpected error on gate.optimize() call.'

    #     assert isinstance(optimal_params, list)
    #     assert len(optimal_params) == gate.get_num_params()
    #     assert all(isinstance(p, float) for p in optimal_params)

    # def test_optimize_invalid(self, param_gate: Gate) -> None:
    #     with pytest.raises(Exception):
    #         param_gate.optimize(1)  # type: ignore
    #     with pytest.raises(Exception):
    #         param_gate.optimize([])  # type: ignore
    #     with pytest.raises(Exception):
    #         env_matrix = np.random.rand(param_gate.get_dim())
    #         optimal_params = param_gate.optimize(env_matrix)  # noqa
    #     with pytest.raises(Exception):
    #         env_matrix = np.random.rand(param_gate.get_dim(), 1)
    #         optimal_params = param_gate.optimize(env_matrix)  # noqa

    def test_is_qubit_gate(self, qubit_gate: Gate) -> None:
        assert qubit_gate.is_qubit_only()
        assert not qubit_gate.is_qutrit_only()

    def test_is_qutrit_gate(self, qutrit_gate: Gate) -> None:
        assert qutrit_gate.is_qutrit_only()
        assert not qutrit_gate.is_qubit_only()

    def test_is_constant_parameterized(self, gate: Gate) -> None:
        assert gate.is_constant() or gate.is_parameterized()

    def test_is_constant_gate(self, constant_gate: Gate) -> None:
        assert constant_gate.is_constant()
        assert not constant_gate.is_parameterized()

    def test_is_parameterized_gate(self, param_gate: Gate) -> None:
        assert not param_gate.is_constant()
        assert param_gate.is_parameterized()

    def test_check_parameters_valid(self, gate: Gate) -> None:
        gate.check_parameters(np.random.rand(gate.get_num_params()))
        gate.check_parameters([0] * gate.get_num_params())

    def test_check_parameters_invalid(self, gate: Gate) -> None:
        with pytest.raises(TypeError):
            gate.check_parameters('a')  # type: ignore
        with pytest.raises(TypeError):
            gate.check_parameters(1)  # type: ignore
        if gate.is_parameterized():
            with pytest.raises(TypeError):
                error_list = ['a'] * gate.get_num_params()
                gate.check_parameters(error_list)  # type: ignore
        with pytest.raises(ValueError):
            gate.check_parameters(np.random.rand(gate.get_num_params() + 1))

    def test_with_frozen_params(self, gate: Gate) -> None:
        num_params = gate.get_num_params()
        frozen_params = {i: float(j) for i, j in enumerate(range(num_params))}
        frozen_gate = gate.with_frozen_params(frozen_params)
        assert isinstance(frozen_gate, FrozenParameterGate)
        assert frozen_gate.frozen_params == frozen_params

    def test_with_all_frozen_params(self, gate: Gate) -> None:
        frozen_gate = gate.with_all_frozen_params([0] * gate.get_num_params())
        assert isinstance(frozen_gate, FrozenParameterGate)
        frozen_params = {i: 0 for i in range(gate.get_num_params())}
        assert frozen_gate.frozen_params == frozen_params

    def test_repr(self, gate: Gate) -> None:
        assert isinstance(gate.__repr__(), str)
