from __future__ import annotations

from typing import Callable
from typing import Sequence

import numpy as np
import numpy.typing as npt

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.passes.util.converttovar import ToVariablePass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


def test_single_qubit_general_gate_conversion(
        compiler: Compiler,
        single_qubit_general_gate: GeneralGate,
        gen_random_utry_np:
        Callable[[int | Sequence[int]], npt.NDArray[np.complex128]],
) -> None:

    circuit = Circuit(1)
    gate_utry = UnitaryMatrix(gen_random_utry_np(2))
    circuit.append_gate(
        single_qubit_general_gate, (0,),
        single_qubit_general_gate.calc_params(gate_utry),
    )

    circuit = compiler.compile(circuit, [ToVariablePass()])
    dist = circuit.get_unitary().get_distance_from(gate_utry)
    assert dist <= 1e-7  # Is this good enough??
    assert isinstance(circuit[0][0].gate, VariableUnitaryGate)


def test_skiping_non_general_gate(
        compiler: Compiler,
        single_qubit_gate: Gate,
) -> None:
    circuit = Circuit(1)
    params = np.random.random(single_qubit_gate.num_params)

    circuit.append_gate(single_qubit_gate, (0,), params)

    circuit = compiler.compile(
        circuit, [ToVariablePass(convert_all_single_qudit_gates=False)],
    )
    dist = circuit.get_unitary().get_distance_from(
        single_qubit_gate.get_unitary(params),
    )

    assert dist <= 1e-7  # Is this good enough??
    if isinstance(single_qubit_gate, GeneralGate):
        assert isinstance(circuit[0][0].gate, VariableUnitaryGate)
    else:
        assert isinstance(circuit[0][0].gate, type(single_qubit_gate))


def test_all_single_qubit_gate_conversion(
        compiler: Compiler,
        single_qubit_gate: Gate,
) -> None:

    circuit = Circuit(1)
    params = np.random.random(single_qubit_gate.num_params)

    circuit.append_gate(single_qubit_gate, (0,), params)

    circuit = compiler.compile(
        circuit, [ToVariablePass(convert_all_single_qudit_gates=True)],
    )

    dist = circuit.get_unitary().get_distance_from(
        single_qubit_gate.get_unitary(params),
    )

    assert dist <= 1e-7  # Is this good enough??
    assert isinstance(circuit[0][0].gate, VariableUnitaryGate)


def test_single_qutrit_general_gate_conversion(
        compiler: Compiler,
        single_qutrit_general_gate: GeneralGate,
        gen_random_utry_np:
        Callable[[int | Sequence[int]], npt.NDArray[np.complex128]],
) -> None:

    circuit = Circuit(1, [3])
    gate_utry = UnitaryMatrix(gen_random_utry_np(3))
    circuit.append_gate(
        single_qutrit_general_gate, (0,),
        single_qutrit_general_gate.calc_params(gate_utry),
    )

    circuit = compiler.compile(circuit, [ToVariablePass()])
    dist = circuit.get_unitary().get_distance_from(gate_utry)
    assert dist <= 1e-7  # Is this good enough??
    assert isinstance(circuit[0][0].gate, VariableUnitaryGate)


def test_all_single_qutrit_gate_conversion(
        compiler: Compiler,
        single_qutrit_gate: Gate,
) -> None:

    circuit = Circuit(1, [3])
    params = np.random.random(single_qutrit_gate.num_params)

    circuit.append_gate(single_qutrit_gate, (0,), params)

    circuit = compiler.compile(
        circuit, [ToVariablePass(convert_all_single_qudit_gates=True)],
    )

    dist = circuit.get_unitary().get_distance_from(
        single_qutrit_gate.get_unitary(params),
    )

    assert dist <= 1e-7  # Is this good enough??
    assert isinstance(circuit[0][0].gate, VariableUnitaryGate)


def test_all_gates_conversion(
        compiler: Compiler,
        gate: Gate,
) -> None:

    circuit = Circuit(gate.num_qudits, gate.radixes)
    params = np.random.random(gate.num_params)

    circuit.append_gate(gate, range(gate.num_qudits), params)

    circuit = compiler.compile(
        circuit, [ToVariablePass(convert_all_single_qudit_gates=True)],
    )

    dist = circuit.get_unitary().get_distance_from(
        gate.get_unitary(params),
    )

    assert dist <= 1e-7  # Is this good enough??
    if gate.num_qudits == 1:
        assert isinstance(circuit[0][0].gate, VariableUnitaryGate)
    else:
        assert isinstance(circuit[0][0].gate, type(gate))
