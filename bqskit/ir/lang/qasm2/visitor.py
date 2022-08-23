"""This module implements a OPENQASM parse tree visitor."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import List
from typing import NamedTuple
from typing import Tuple

import lark
import numpy as np
from lark import Visitor

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.composed.controlled import ControlledGate
from bqskit.ir.gates.composed.daggergate import DaggerGate
from bqskit.ir.gates.constant.ccx import CCXGate
from bqskit.ir.gates.constant.ch import CHGate
from bqskit.ir.gates.constant.cs import CSGate
from bqskit.ir.gates.constant.ct import CTGate
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.gates.constant.cy import CYGate
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.constant.iswap import ISwapGate
from bqskit.ir.gates.constant.itoffoli import IToffoliGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.sdg import SdgGate
from bqskit.ir.gates.constant.sqrtcnot import SqrtCNOTGate
from bqskit.ir.gates.constant.sqrtiswap import SqrtISwapGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.constant.sycamore import SycamoreGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.xx import XXGate
from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.constant.yy import YYGate
from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.constant.zz import ZZGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.gates.parameterized.cp import CPGate
from bqskit.ir.gates.parameterized.crx import CRXGate
from bqskit.ir.gates.parameterized.cry import CRYGate
from bqskit.ir.gates.parameterized.crz import CRZGate
from bqskit.ir.gates.parameterized.fsim import FSIMGate
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
from bqskit.ir.lang.qasm2.parser import parse
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation


_logger = logging.getLogger(__name__)


class QubitReg(NamedTuple):
    """Definition of a Qubit Register."""
    name: str
    size: int


class ClassicalReg(NamedTuple):
    """Definition of a Classical Bit Register."""
    name: str
    size: int


@dataclass
class GateDef:
    """Definition of a QASM Gate."""
    qasm_name: str
    num_params: int
    num_vars: int
    gate: Gate

    def build_op(self, loc: CircuitLocation, params: list[float]) -> Operation:
        """Build an operation for this gate."""
        return Operation(self.gate, loc, params)


@dataclass
class CustomGateDef:
    """Definition of a Custom QASM Gate."""
    qasm_name: str
    num_params: int
    num_vars: int
    gate_def_list: list[GateDef | CustomGateDef]
    loc_list: list[CircuitLocation]
    param_exp_list: list[list[lark.Tree | float]]

    def build_op(self, loc: CircuitLocation, params: list[float]) -> Operation:
        """Build an operation for this gate."""
        cgc = Circuit(self.num_vars)
        for i in range(len(self.gate_def_list)):
            sublocation = self.loc_list[i]
            subparams = self.evaluate_param_exps(self.param_exp_list[i], params)
            cgc.append(self.gate_def_list[i].build_op(sublocation, subparams))
        gate = CircuitGate(cgc, True)
        return Operation(gate, loc, gate._circuit.params)

    def evaluate_param_exps(
        self,
        param_exps: list[lark.Tree | float],
        params: list[float],
    ) -> list[float]:
        """Substitute params into the place holders in param_exp and
        evaluate."""
        eval_params = []
        for exp in param_exps:
            if isinstance(exp, float):
                eval_params.append(exp)
            else:
                subbed_exp = self.replace_param_indices(exp, params)
                eval_params.append(float(eval_exp(subbed_exp)))
        return eval_params

    def replace_param_indices(
        self,
        exp: lark.Tree,
        params: list[float],
    ) -> lark.Tree:
        """Return a new tree with parameter indices replaced with values."""
        if isinstance(exp, lark.Token):
            if exp.type == 'PARAM_IDX':
                return lark.Token('REAL', params[int(exp)])
            else:
                return exp
        children = [self.replace_param_indices(c, params) for c in exp.children]
        return lark.Tree(exp.data, children)


class OPENQASMVisitor(Visitor):
    """
    The OPENQASMVisitor class.

    This class defines a complete visitor for a parse tree produced by
    OPENQASMParser.
    """

    def __init__(self) -> None:
        """Initialize state for the OPENQASMVisitor."""
        self.op_list: list[Operation] = []
        self.qubit_regs: list[QubitReg] = []
        self.gate_defs: dict[str, GateDef] = {}
        self.classical_regs: list[ClassicalReg] = []
        self.gate_def_parsing_obj: Any = None
        self.custom_gate_defs: dict[str, CustomGateDef] = {}
        self.measurements: dict[int, tuple[str, int]] = {}
        self.fill_gate_defs()

    def get_circuit(self) -> Circuit:
        """Retrieve the circuit generated after walking a parse tree."""
        num_qubits = sum(qubit_reg.size for qubit_reg in self.qubit_regs)
        if num_qubits == 0:
            raise LangException('No qubit registers defined.')
        circuit = Circuit(num_qubits)
        circuit.extend(self.op_list)

        # Add measurements
        if len(self.measurements) > 0:
            cregs = cast(List[Tuple[str, int]], self.classical_regs)
            mph = MeasurementPlaceholder(cregs, self.measurements)
            circuit.append_gate(mph, list(self.measurements.keys()))

        return circuit

    def fill_gate_defs(self) -> None:
        """Prefills gate definitions with built-in gates."""
        # Parameterized Gates
        self.gate_defs['p'] = GateDef('p', 1, 1, RZGate())
        self.gate_defs['cp'] = GateDef('cp', 1, 2, CPGate())
        self.gate_defs['crx'] = GateDef('crx', 1, 2, CRXGate())
        self.gate_defs['cry'] = GateDef('cry', 1, 2, CRYGate())
        self.gate_defs['crz'] = GateDef('crz', 1, 2, CRZGate())
        self.gate_defs['fsim'] = GateDef('fsim', 2, 2, FSIMGate())
        self.gate_defs['rx'] = GateDef('rx', 1, 1, RXGate())
        self.gate_defs['rxx'] = GateDef('rxx', 1, 2, RXXGate())
        self.gate_defs['ry'] = GateDef('ry', 1, 1, RYGate())
        self.gate_defs['ryy'] = GateDef('ryy', 1, 2, RYYGate())
        self.gate_defs['rz'] = GateDef('rz', 1, 1, RZGate())
        self.gate_defs['rzz'] = GateDef('rzz', 1, 2, RZZGate())
        self.gate_defs['u1'] = GateDef('u1', 1, 1, U1Gate())
        self.gate_defs['u2'] = GateDef('u2', 2, 1, U2Gate())
        self.gate_defs['u3'] = GateDef('u3', 3, 1, U3Gate())
        self.gate_defs['U'] = GateDef('U', 3, 1, U3Gate())
        self.gate_defs['u'] = GateDef('u', 3, 1, U3Gate())
        self.gate_defs['cu1'] = GateDef('cu1', 1, 2, ControlledGate(U1Gate()))
        self.gate_defs['cu2'] = GateDef('cu2', 2, 2, ControlledGate(U2Gate()))
        self.gate_defs['cu3'] = GateDef('cu3', 3, 2, ControlledGate(U3Gate()))

        # Constant Gates
        self.gate_defs['ccx'] = GateDef('ccx', 0, 3, CCXGate())
        self.gate_defs['ch'] = GateDef('ch', 0, 2, CHGate())
        self.gate_defs['cs'] = GateDef('cs', 0, 2, CSGate())
        self.gate_defs['ct'] = GateDef('ct', 0, 2, CTGate())
        self.gate_defs['cx'] = GateDef('cx', 0, 2, CXGate())
        self.gate_defs['CX'] = GateDef('CX', 0, 2, CXGate())
        self.gate_defs['cy'] = GateDef('cy', 0, 2, CYGate())
        self.gate_defs['cz'] = GateDef('cz', 0, 2, CZGate())
        self.gate_defs['h'] = GateDef('h', 0, 1, HGate())
        self.gate_defs['id'] = GateDef('id', 0, 1, IdentityGate(1))
        self.gate_defs['iswap'] = GateDef('iswap', 0, 2, ISwapGate())
        self.gate_defs['iccx'] = GateDef('iccx', 0, 3, IToffoliGate())
        self.gate_defs['s'] = GateDef('s', 0, 1, SGate())
        self.gate_defs['sdg'] = GateDef('sdg', 0, 1, SdgGate())
        self.gate_defs['csx'] = GateDef('csx', 0, 2, SqrtCNOTGate())
        self.gate_defs['cv'] = GateDef('cv', 0, 2, SqrtCNOTGate())
        self.gate_defs['sqisw'] = GateDef('sqisw', 0, 2, SqrtISwapGate())
        self.gate_defs['swap'] = GateDef('swap', 0, 2, SwapGate())
        self.gate_defs['sx'] = GateDef('sx', 0, 1, SXGate())
        self.gate_defs['syc'] = GateDef('syc', 0, 2, SycamoreGate())
        self.gate_defs['t'] = GateDef('t', 0, 1, TGate())
        self.gate_defs['tdg'] = GateDef('tdg', 0, 1, TdgGate())
        self.gate_defs['x'] = GateDef('x', 0, 1, XGate())
        self.gate_defs['xx'] = GateDef('xx', 0, 2, XXGate())
        self.gate_defs['y'] = GateDef('y', 0, 1, YGate())
        self.gate_defs['yy'] = GateDef('yy', 0, 2, YYGate())
        self.gate_defs['z'] = GateDef('z', 0, 1, ZGate())
        self.gate_defs['zz'] = GateDef('zz', 0, 2, ZZGate())
        self.gate_defs['sxdg'] = GateDef('sxdg', 0, 1, DaggerGate(SXGate()))

    def qreg(self, tree: lark.Tree) -> None:
        """Qubit register node visitor."""
        reg_name = tree.children[0]
        if any(reg_name == reg.name for reg in self.qubit_regs):
            raise LangException('Qubit register redeclared: %s.' % reg_name)
        reg_size = int(tree.children[1])
        reg = QubitReg(reg_name, reg_size)
        _logger.debug('Qubit register %s declared with size %d.' % reg)
        self.qubit_regs.append(reg)

    def gate(self, tree: lark.Tree) -> None:
        """Apply a normal gate statement to the circuit."""
        # Parse parameters
        if len(tree.children) == 3:
            exp_list = self.flatten_exps(tree.children[1])
            params = [float(eval_exp(exp)) for exp in exp_list]
        else:
            params = []

        # Parse location
        location = []
        for reg_name, reg_idx in qubits_from_list(tree.children[-1]):
            outer_idx = 0
            for reg in self.qubit_regs:
                if reg.name == reg_name:
                    location.append(outer_idx + reg_idx)
                    break
                outer_idx += reg.size
        location = CircuitLocation(location)

        if any(q in self.measurements for q in location):
            raise LangException(
                'BQSKit currently does not support mid-circuit measurements.'
                ' Unable to apply a gate on the same qubit where a measurement'
                ' has been previously made.',
            )

        # Parse gate object
        gate_name = str(tree.children[0])
        if gate_name in self.gate_defs:
            gate_def: GateDef | CustomGateDef = self.gate_defs[gate_name]
        elif gate_name in self.custom_gate_defs:
            gate_def = self.custom_gate_defs[gate_name]
        else:
            raise LangException('Unrecognized gate: %s.' % gate_name)

        if len(params) != gate_def.num_params:
            raise LangException(
                'Expected %d params got %d params for gate %s.'
                % (gate_def.num_params, len(params), gate_name),
            )

        if len(location) != gate_def.num_vars:
            raise LangException(
                'Gate acts on %d qubits, got %d qubit variables.'
                % (gate_def.num_vars, len(location)),
            )

        # Build operation and add to circuit
        self.op_list.append(gate_def.build_op(location, params))

    def cxgate(self, tree: lark.Tree) -> None:
        """CX gate node visitor."""
        control = tree.children[0].children
        target = tree.children[1].children
        cname, cidx = str(control[0]), int(control[1])
        tname, tidx = str(target[0]), int(target[1])

        outer_idx = 0
        cloc: int | None = None
        tloc: int | None = None
        for reg in self.qubit_regs:
            if reg.name == cname:
                cloc = outer_idx + cidx
            if reg.name == tname:
                tloc = outer_idx + tidx
            outer_idx += reg.size

        if cloc is None:
            raise LangException(f'Qubit register not found: {cname}')

        if tloc is None:
            raise LangException(f'Qubit register not found: {tname}')

        if cloc == tloc:
            raise LangException('CX control and target qubit cannot be equal.')

        loc = CircuitLocation([cloc, tloc])
        self.op_list.append(Operation(CXGate(), loc))

    def ugate(self, tree: lark.Tree) -> None:
        """U gate node visitor."""
        exp_list = self.flatten_exps(tree.children[0])
        params = [float(eval_exp(exp)) for exp in exp_list]
        qubit = tree.children[1].children
        name, idx = str(qubit[0]), int(qubit[1])

        outer_idx = 0
        location: int | None = None
        for reg in self.qubit_regs:
            if reg.name == name:
                location = outer_idx + idx
                break
            outer_idx += reg.size

        if location is None:
            raise LangException(f'Qubit register not found: {name}')

        location = CircuitLocation(location)
        self.op_list.append(self.gate_defs['U'].build_op(location, params))

    def gatep(self, tree: lark.Tree) -> None:
        """Apply a gate to a currently-being-built gate declaration."""
        # Parse parameters or store their expression
        param_exps = []
        if len(tree.children) == 3:
            exp_list = self.flatten_exps(tree.children[1])
            for exp in exp_list:
                if self.has_param_variable(exp):
                    param_exps.append(self.replace_param_ids(exp))
                else:
                    param_exps.append(float(eval_exp(exp)))

        self.gate_def_parsing_obj['param_exp_list'].append(param_exps)

        # Parse location
        location: list[int] = []
        iter = tree.children[-1].children[0]
        while iter:
            qname = str(iter.children[-1])
            qid = self.gate_def_parsing_obj['qubits'].index(qname)
            location.insert(0, qid)
            iter = None if len(iter.children) != 2 else iter.children[0]

        self.gate_def_parsing_obj['loc_list'].append(location)

        # Parse gate object
        gate_name = str(tree.children[0])
        if gate_name in self.gate_defs:
            gate_def: GateDef | CustomGateDef = self.gate_defs[gate_name]
        elif gate_name in self.custom_gate_defs:
            gate_def = self.custom_gate_defs[gate_name]
        else:
            raise LangException('Unrecognized gate: %s.' % gate_name)

        if len(param_exps) != gate_def.num_params:
            raise LangException(
                'Expected %d params got %d params for gate %s.'
                % (gate_def.num_params, len(param_exps), gate_name),
            )

        if len(location) != gate_def.num_vars:
            raise LangException(
                'Gate acts on %d qubits, got %d qubit variables.'
                % (gate_def.num_vars, len(location)),
            )

        self.gate_def_parsing_obj['gate_def_list'].append(gate_def)

    def cxgatep(self, tree: lark.Tree) -> None:
        """CX gate node visitor (while building custom gate)."""
        control = tree.children[0].children
        target = tree.children[1].children
        cid = self.gate_def_parsing_obj['qubits'].index(str(control[0]))
        tid = self.gate_def_parsing_obj['qubits'].index(str(target[0]))

        if cid == tid:
            raise LangException('CX control and target qubit cannot be equal.')

        loc = CircuitLocation([cid, tid])
        self.gate_def_parsing_obj['param_exp_list'].append([])
        self.gate_def_parsing_obj['loc_list'].append(loc)
        self.gate_def_parsing_obj['gate_def_list'].append(self.gate_defs['CX'])

    def ugatep(self, tree: lark.Tree) -> None:
        """U gate node visitor (while building custom gate)."""
        param_exps = []
        exp_list = self.flatten_exps(tree.children[0])
        for exp in exp_list:
            if self.has_param_variable(exp):
                param_exps.append(self.replace_param_ids(exp))
            else:
                param_exps.append(float(eval_exp(exp)))

        qubit = tree.children[1].children
        qid = self.gate_def_parsing_obj['qubits'].index(str(qubit[0]))

        location = CircuitLocation(qid)
        self.gate_def_parsing_obj['param_exp_list'].append(param_exps)
        self.gate_def_parsing_obj['loc_list'].append(location)
        self.gate_def_parsing_obj['gate_def_list'].append(self.gate_defs['U'])

    def rbracket(self, tree: lark.Tree) -> None:
        """Finish a gate declaration block."""
        qasm_name = self.gate_def_parsing_obj['name']
        num_params = self.gate_def_parsing_obj['num_params']
        num_vars = self.gate_def_parsing_obj['num_qubits']
        gate_def_list = self.gate_def_parsing_obj['gate_def_list']
        loc_list = self.gate_def_parsing_obj['loc_list']
        param_exp_list = self.gate_def_parsing_obj['param_exp_list']
        self.custom_gate_defs[qasm_name] = CustomGateDef(
            qasm_name,
            num_params,
            num_vars,
            gate_def_list,
            loc_list,
            param_exp_list,
        )
        self.gate_def_parsing_obj = None

    def gatedecl(self, tree: lark.Tree) -> None:
        """Start a gate declaration block."""
        # parse gate name
        self.gate_def_parsing_obj = {'name': str(tree.children[0])}

        # parse parameters
        if len(tree.children) == 2:
            self.gate_def_parsing_obj['num_params'] = 0
            self.gate_def_parsing_obj['params'] = []
        else:
            params: list[str] = []
            iter = tree.children[1]
            while iter:
                if isinstance(iter, lark.Tree):
                    if len(iter.children) == 2:
                        params.insert(0, str(iter.children[1]))
                        iter = iter.children[0]
                    else:
                        params.insert(0, str(iter.children[0]))
                        iter = None
                else:
                    params.insert(0, str(iter))
                    iter = None

            self.gate_def_parsing_obj['num_params'] = len(params)
            self.gate_def_parsing_obj['params'] = params

        # parse qubits
        qubits: list[str] = []
        iter = tree.children[-1]
        while iter:
            if isinstance(iter, lark.Tree):
                if len(iter.children) == 2:
                    qubits.insert(0, str(iter.children[1]))
                    iter = iter.children[0]
                else:
                    qubits.insert(0, str(iter.children[0]))
                    iter = None
            else:
                qubits.insert(0, str(iter))
                iter = None

        self.gate_def_parsing_obj['num_qubits'] = len(qubits)
        self.gate_def_parsing_obj['qubits'] = qubits
        self.gate_def_parsing_obj['gate_def_list'] = []
        self.gate_def_parsing_obj['loc_list'] = []
        self.gate_def_parsing_obj['param_exp_list'] = []

    def flatten_exps(self, explist: lark.Tree) -> list[lark.Tree]:
        """Flatten an explist tree into a list of exp subtrees."""
        exp_list: list[lark.Tree] = []
        iter = explist
        while iter:
            exp_list.insert(0, iter.children[-1])
            iter = None if len(iter.children) == 1 else iter.children[0]
        return exp_list

    def has_param_variable(self, exp: lark.Tree) -> bool:
        """Return true is exp is bound to gatedecl param variable."""
        if isinstance(exp, lark.Token):
            return str(exp) in self.gate_def_parsing_obj['params']
        return any(self.has_param_variable(c) for c in exp.children)

    def replace_param_ids(self, exp: lark.Tree) -> lark.Tree:
        """Return a new tree with parameter ids replaces with indices."""
        if isinstance(exp, lark.Token):
            if str(exp) in self.gate_def_parsing_obj['params']:
                index = self.gate_def_parsing_obj['params'].index(str(exp))
                return lark.Token('PARAM_IDX', index)
            else:
                return exp
        children = [self.replace_param_ids(c) for c in exp.children]
        return lark.Tree(exp.data, children)

    def incstmt(self, tree: lark.Tree) -> None:
        """Include statement node."""
        file_name = str(tree.children[0])
        if file_name[0] == "\"":
            file_name = file_name[1:]
        if file_name[-1] == "\"":
            file_name = file_name[:-1]

        if not os.path.isfile(file_name):
            if file_name != 'qelib1.inc':
                # Suppress warnings over qelib1.inc due to high frequency
                _logger.warning(
                    f'Unable to find {file_name} used in a QASM include'
                    ' statement. Ignoring include file and continuing.',
                )
            return

        with open(file_name) as f:
            filedata = f.read()
            if not filedata.startswith('OPENQASM'):
                filedata = 'OPENQASM 2.0;\n' + filedata
            tree = parse(filedata)
            visitor = OPENQASMVisitor()
            visitor.visit_topdown(tree)
            self.op_list.extend(visitor.op_list)
            self.qubit_regs.extend(visitor.qubit_regs)
            self.custom_gate_defs.update(visitor.custom_gate_defs)

    def creg(self, tree: lark.Tree) -> None:
        """Classical bit register node visitor."""
        reg_name = tree.children[0]
        if any(reg_name == reg.name for reg in self.classical_regs):
            raise LangException('Classical register redeclared: %s.' % reg_name)
        reg_size = int(tree.children[1])
        reg = ClassicalReg(reg_name, reg_size)
        _logger.debug('Classical register %s declared with size %d.' % reg)
        self.classical_regs.append(reg)

    def measure(self, tree: lark.Tree) -> None:
        """Measure statement node visitor."""
        qubit_childs = tree.children[0].children
        class_childs = tree.children[1].children
        qubit_reg_name = str(qubit_childs[0])
        class_reg_name = str(class_childs[0])
        if not any(r.name == qubit_reg_name for r in self.qubit_regs):
            raise LangException(
                f'Measuring undefined qubit register: {qubit_reg_name}',
            )

        if not any(r.name == class_reg_name for r in self.classical_regs):
            raise LangException(
                f'Measuring undefined classical register: {class_reg_name}',
            )

        if len(qubit_childs) == 1 and len(class_childs) == 1:
            for name, size in self.qubit_regs:
                if qubit_reg_name == name:
                    qubit_size = size
                    break
            for name, size in self.classical_regs:
                if class_reg_name == name:
                    class_size = size
                    break
            if qubit_size != class_size:
                raise LangException(
                    'Size mismatch between qubit and classical register'
                    f' during measure operation: {qubit_size} != {class_size}.',
                )

            outer_idx = 0
            for name, size in self.qubit_regs:
                if name == qubit_reg_name:
                    break
                outer_idx += size

            for i in range(qubit_size):
                self.measurements[outer_idx + i] = (class_reg_name, i)

        elif len(qubit_childs) == 2 and len(class_childs) == 2:
            qubit_index = int(qubit_childs[1])
            class_index = int(class_childs[1])

            # Convert qubit_index to global index
            outer_idx = 0
            for name, size in self.qubit_regs:
                if name == qubit_reg_name:
                    qubit_index = outer_idx + qubit_index
                    break
                outer_idx += size

            self.measurements[qubit_index] = (class_reg_name, class_index)

        else:
            raise LangException(
                'Invalid measurement: either a single qubit is being measured '
                'to a full classical register or a qubit register is being '
                'measured to a single classical bit.',
            )

    def reset(self, tree: lark.Tree) -> None:
        """Reset statement node visitor."""
        raise LangException('BQSKit currently does not support resets.')


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
    elif tree.data == 'argument':
        return [(str(tree.children[0]), int(tree.children[1]))]
    else:
        raise Exception()
