from __future__ import annotations

import pickle

import numpy as np
import pytest

from bqskit.ir.gate import Gate
from bqskit.ir.gates.composed.frozenparam import FrozenParameterGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TestBasicGate:
    def test_get_name(self, gate: Gate) -> None:
        assert isinstance(gate.name, str)

    def test_get_num_params(self, gate: Gate) -> None:
        assert isinstance(gate.num_params, int)
        assert gate.num_params >= 0

    def test_get_num_params_constant(self, constant_gate: Gate) -> None:
        assert constant_gate.num_params == 0

    def test_get_num_params_parameterized(self, param_gate: Gate) -> None:
        assert param_gate.num_params != 0

    def test_get_radixes(self, gate: Gate) -> None:
        assert isinstance(gate.radixes, tuple)
        assert all(isinstance(radix, int) for radix in gate.radixes)
        assert all(radix > 0 for radix in gate.radixes)

    def test_get_radixes_qubit(self, qubit_gate: Gate) -> None:
        assert all(radix == 2 for radix in qubit_gate.radixes)

    def test_get_radixes_qutrit(self, qutrit_gate: Gate) -> None:
        assert all(radix == 3 for radix in qutrit_gate.radixes)

    def test_get_size(self, gate: Gate) -> None:
        assert isinstance(gate.num_qudits, int)
        assert gate.num_qudits > 0

    def test_gate_size_matches_radixes(self, gate: Gate) -> None:
        assert len(gate.radixes) == gate.num_qudits

    def test_get_dim(self, gate: Gate) -> None:
        assert isinstance(gate.dim, int)
        assert gate.dim > 0

    def test_get_qasm_name(self, gate: Gate) -> None:
        try:
            qasm_name = gate.qasm_name
        except AttributeError:
            return
        except BaseException:
            assert False, 'Unexpected error on gate.qasm_name call.'

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
        params = np.random.rand(gate.num_params)
        utry = gate.get_unitary(params)
        assert isinstance(utry, UnitaryMatrix)

    def test_unitary_dim_match(self, gate: Gate) -> None:
        params = np.random.rand(gate.num_params)
        utry = gate.get_unitary(params)
        assert utry.shape == (gate.dim, gate.dim)

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
        gate.check_parameters(np.random.rand(gate.num_params))
        gate.check_parameters([0] * gate.num_params)

    def test_check_parameters_invalid(self, gate: Gate) -> None:
        with pytest.raises(TypeError):
            gate.check_parameters('a')  # type: ignore
        with pytest.raises(TypeError):
            gate.check_parameters(1)  # type: ignore
        if gate.is_parameterized():
            with pytest.raises(TypeError):
                error_list = ['a'] * gate.num_params
                gate.check_parameters(error_list)  # type: ignore
        with pytest.raises(ValueError):
            gate.check_parameters(np.random.rand(gate.num_params + 1))

    def test_with_frozen_params(self, gate: Gate) -> None:
        num_params = gate.num_params
        frozen_params = {i: float(j) for i, j in enumerate(range(num_params))}
        frozen_gate = gate.with_frozen_params(frozen_params)
        assert isinstance(frozen_gate, FrozenParameterGate)
        assert frozen_gate.frozen_params == frozen_params

    def test_with_all_frozen_params(self, gate: Gate) -> None:
        frozen_gate = gate.with_all_frozen_params([0] * gate.num_params)
        assert isinstance(frozen_gate, FrozenParameterGate)
        frozen_params = {i: 0 for i in range(gate.num_params)}
        assert frozen_gate.frozen_params == frozen_params

    def test_repr(self, gate: Gate) -> None:
        assert isinstance(gate.__repr__(), str)

    def test_use_as_key(self, gate: Gate) -> None:
        test_dict = {}
        test_dict[gate] = 0
        assert gate in test_dict

    def test_pickle(self, gate: Gate) -> None:
        params = [0] * gate.num_params
        utry = gate.get_unitary(params)
        pickled_utry = pickle.loads(pickle.dumps(gate)).get_unitary(params)
        assert utry == pickled_utry

    def test_get_inverse(self, gate: Gate) -> None:
        from bqskit.ir.gates.constantgate import ConstantGate
        from bqskit.qis.unitary.differentiable import DifferentiableUnitary
        from bqskit.ir.gates.qubitgate import QubitGate
        from bqskit.ir.operation import Operation
        if isinstance(gate, QubitGate):
            if isinstance(gate, ConstantGate):
                assert hasattr(gate, 'get_inverse')
                inv_gate = gate.get_inverse()
                assert inv_gate._qasm_name, \
                    'All inverses should have an effective _qasm_name'
                iden = np.identity(gate.dim)
                inv_gate = gate.get_inverse()
                supposed_to_be_iden = (
                    inv_gate.get_unitary() @ gate.get_unitary()
                )
                dist = supposed_to_be_iden.get_distance_from(iden, 1)
                assert dist < 1e-10

            if isinstance(gate, DifferentiableUnitary):
                assert hasattr(gate, 'get_inverse')
                inv_gate = gate.get_inverse()
                assert inv_gate._qasm_name, \
                    'All inverses should have an effective _qasm_name'
                op = Operation(
                    gate, list(range(gate.num_qudits)),
                    np.random.rand(gate.num_params),
                )
                inv_op = op.get_inverse()
                iden = np.identity(gate.dim)
                supposed_to_be_iden = (inv_op.get_unitary() @ op.get_unitary())
                dist = supposed_to_be_iden.get_distance_from(iden, 1)
                assert dist < 1e-10
