"""This module implemenets Quantinuum QPU models."""
from __future__ import annotations

from bqskit.compiler.machine import MachineModel
from bqskit.ir.gate import Gate
from bqskit.ir.gates.constant.zz import ZZGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u1q import U1qPi2Gate
from bqskit.ir.gates.parameterized.u1q import U1qPiGate

quantinuum_gate_set: set[Gate] = {U1qPiGate, U1qPi2Gate, RZGate(), ZZGate()}

H1_1Model = MachineModel(20, None, quantinuum_gate_set)
H1_2Model = MachineModel(20, None, quantinuum_gate_set)
