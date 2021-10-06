# type: ignore
"""This module implements a OPENQASM parse tree visitor."""
from __future__ import annotations

import logging
from typing import Any
from typing import NamedTuple

import lark
import numpy as np
from lark import Visitor

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates.composed.controlled import ControlledGate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.constant.ch import CHGate
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.gates.constant.cy import CYGate
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.sdg import SdgGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rxx import RXXGate
from bqskit.ir.gates.parameterized.ry import RYGate
from bqskit.ir.gates.parameterized.ryy import RYYGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.rzz import RZZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate
from bqskit.ir.gates.parameterized.u2 import U2Gate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.lang.language import LangException
from bqskit.ir.operation import Operation
from bqskit.utils.typing import is_iterable


_logger = logging.getLogger(__name__)


class QubitReg(NamedTuple):
    """Definition of a Qubit Register."""
    name: str
    size: int


class GateDef(NamedTuple):
    """Definition of a QASM Gate."""
    qasm_name: str
    num_params: int
    num_vars: int
    gate: Gate


class OPENQASMVisitor(Visitor):
    """
    The OPENQASMVisitor class.

    This class defines a complete visitor for a parse tree produced by
    OPENQASMParser.
    """

    def __init__(self) -> None:
        self.op_list: list[Operation] = []
        self.qubit_regs: list[QubitReg] = []
        self.gate_defs: dict[str, GateDef] = {}
        self.fill_gate_defs()
        self.gate_def_parsing_obj = None

    def get_circuit(self) -> Circuit:
        """Retrieve the circuit generated after walking a parse tree."""
        num_qubits = sum(qubit_reg.size for qubit_reg in self.qubit_regs)
        if num_qubits == 0:
            raise LangException('No qubit registers defined.')
        circuit = Circuit(num_qubits)
        circuit.extend(self.op_list)
        return circuit

    def fill_gate_defs(self) -> None:
        """Prefills gate definitions with built-in gates."""
        self.gate_defs['U'] = GateDef('U', 3, 1, U3Gate())
        self.gate_defs['u'] = GateDef('u', 3, 1, U3Gate())
        self.gate_defs['u3'] = GateDef('u3', 3, 1, U3Gate())
        self.gate_defs['u2'] = GateDef('u2', 2, 1, U2Gate())
        self.gate_defs['u1'] = GateDef('u1', 1, 1, U1Gate())
        self.gate_defs['cx'] = GateDef('cx', 0, 2, CXGate())
        self.gate_defs['cy'] = GateDef('cy', 0, 2, CYGate())
        self.gate_defs['cz'] = GateDef('cz', 0, 2, CZGate())
        self.gate_defs['ch'] = GateDef('ch', 0, 2, CHGate())
        self.gate_defs['swap'] = GateDef('swap', 0, 2, SwapGate())
        self.gate_defs['id'] = GateDef('id', 0, 1, IdentityGate(1))
        self.gate_defs['x'] = GateDef('x', 0, 1, XGate())
        self.gate_defs['y'] = GateDef('y', 0, 1, YGate())
        self.gate_defs['z'] = GateDef('z', 0, 1, ZGate())
        self.gate_defs['h'] = GateDef('h', 0, 1, HGate())
        self.gate_defs['s'] = GateDef('s', 0, 1, SGate())
        self.gate_defs['sdg'] = GateDef('sdg', 0, 1, SdgGate())
        self.gate_defs['t'] = GateDef('t', 0, 1, TGate())
        self.gate_defs['tdg'] = GateDef('tdg', 0, 1, TdgGate())
        self.gate_defs['rx'] = GateDef('rx', 1, 1, RXGate())
        self.gate_defs['ry'] = GateDef('ry', 1, 1, RYGate())
        self.gate_defs['rz'] = GateDef('rz', 1, 1, RZGate())
        self.gate_defs['sx'] = GateDef('sx', 0, 1, SXGate())
        self.gate_defs['sxdg'] = GateDef('sxdg', 0, 1, DaggerGate(SXGate()))
        self.gate_defs['rxx'] = GateDef('rxx', 1, 2, RXXGate())
        self.gate_defs['ryy'] = GateDef('ryy', 1, 2, RYYGate())
        self.gate_defs['rzz'] = GateDef('rzz', 1, 2, RZZGate())
        self.gate_defs['cu1'] = GateDef('cu1', 1, 2, ControlledGate(U1Gate()))
        self.gate_defs['cu2'] = GateDef('cu2', 2, 2, ControlledGate(U2Gate()))
        self.gate_defs['cu3'] = GateDef('cu3', 3, 2, ControlledGate(U3Gate()))

    def qreg(self, args: Any) -> None:
        reg_name = args.children[0]
        if any(reg_name == reg.name for reg in self.qubit_regs):
            raise LangException('Qubit register redeclared: %s.' % reg_name)
        reg_size = int(args.children[1])
        reg = QubitReg(reg_name, reg_size)
        _logger.debug('Qubit register %s declared with size %d.' % reg)
        self.qubit_regs.append(reg)

    def gate(self, args: Any) -> None:
        gate_name = args.children[0]
        param_list = None if len(args.children) == 2 else args.children[1]
        if len(args.children) == 2:
            var_list = args.children[1]
        else:
            var_list = args.children[2]

        if gate_name not in self.gate_defs:
            raise LangException('Unrecognized gate: %s.' % gate_name)

        gate_def = self.gate_defs[gate_name]

        # Calculate params
        params = []
        if param_list is not None:
            if is_iterable(eval_explist(param_list)):
                params = [float(i) for i in eval_explist(param_list)]
            else:
                params = [float(eval_explist(param_list))]

        if len(params) != gate_def.num_params:
            raise LangException(
                'Expected %d params got %d params for gate %s.'
                % (gate_def.num_params, len(params), gate_name),
            )

        # Calculate location
        location = []
        for reg_name, reg_idx in qubits_from_list(var_list):
            outer_idx = 0
            for reg in self.qubit_regs:
                if reg.name == reg_name:
                    location.append(outer_idx + reg_idx)
                    break
                outer_idx += reg.num_qudits

        if len(location) != gate_def.num_vars:
            raise LangException(
                'Gate acts on %d qubits, got %d qubit variables.'
                % (gate_def.num_vars, len(location)),
            )

        # Calculate operation
        self.op_list.append(Operation(gate_def.gate, location, params))

    def cxgate(self, args: Any) -> None:
        self.gate(args)

    def ugate(self, args: Any) -> None:
        self.gate(args)

    def gatedecl(self, args: Any) -> None:
        _logger.warning('Ignoring unsupported qasm feature: gate declaration.')

    def creg(self, args: Any) -> None:
        _logger.warning('Ignoring unsupported qasm feature: classic register.')

    def measure(self, args: Any) -> None:
        _logger.warning('Ignoring unsupported qasm feature: measure.')

    def reset(self, args: Any) -> None:
        _logger.warning('Ignoring unsupported qasm feature: reset.')


eval_locals = {
    'pi': np.pi,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'ln': np.log,
    'exp': np.exp,
}


def eval_exp_recurse(tree: lark.Tree) -> Any:
    if isinstance(tree, lark.Token):
        return str(tree)
    code = ''
    for op in tree.children:
        if isinstance(op, lark.Token):
            code += str(op)
            continue
        if op.data == 'unaryexp':
            unaryop = op.children[0]
            code += f'{unaryop.children[0]}({eval_exp_recurse(op.children[1])})'
            continue
        elif op.data == 'usub':
            code += '-'
        elif op.data == 'pow':
            base = eval_exp_recurse(op.children[0])
            exp = eval_exp_recurse(op.children[1])
            code += f'{base}**{exp}'
            continue
        code += ' '.join(map(eval_exp_recurse, op.children))
    return code


def eval_exp(tree: lark.Tree) -> Any:
    return eval(eval_exp_recurse(tree), {}, eval_locals)


def eval_explist(tree: lark.Tree) -> Any:
    if isinstance(tree, list):
        tree = tree[0]
    if tree.data == 'explist':
        if len(tree.children) == 1:
            return eval_exp(tree.children[0])
        list_vals = eval_explist(tree.children[0])
        exp_vals = eval_exp(tree.children[1])
        if isinstance(list_vals, tuple):
            return (*list_vals, exp_vals)
        return list_vals, exp_vals
    elif tree.data == 'exp':
        return eval_exp(tree)
    else:
        raise Exception()


def qubits_from_list(tree: lark.Tree) -> list[tuple[str, int]]:
    if tree.data == 'anylist':
        return qubits_from_list(tree.children[0])
    elif tree.data == 'mixedlist':
        q2 = (str(tree.children[-2]), int(tree.children[-1]))
        if isinstance(tree.children[0], lark.Tree):
            q1 = qubits_from_list(tree.children[0])
            return [*q1, q2]
        return [q2]
    else:
        raise Exception()
