"""Regression coverage for issue #306: undefined QASM gate names.

`CSGate` was fixed to override `get_qasm_gate_def()` because `cs` is not part
of the standard `qelib1.inc` gate library that BQSKit's QASM exporter assumes
is available -- so a `cs`-containing circuit round-tripped fine through
BQSKit's own (lenient) parser, but was rejected by other QASM consumers (e.g.
Qiskit) with `QASM2ParseError: 'cs' is not defined in this scope`.

This module systematically checks the same property across every no-arg-
constructible constant gate, not just `CSGate` -- so a future gate
introducing the same bug is caught immediately, rather than needing its own
bug report first.
"""

from __future__ import annotations

import re

import pytest

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import constant as constant_gates
from bqskit.ir.lang.qasm2 import OPENQASM2Language

# The gate names defined by qelib1.inc, the standard OpenQASM 2 gate library
# every QASM 2 file produced by BQSKit's exporter includes. A gate whose
# QASM name isn't one of these (and isn't a call to one of these with fixed
# parameters, e.g. `rxx(pi/2)`) must supply its own definition via
# `get_qasm_gate_def()`.
_QELIB1_GATE_NAMES = frozenset(
    {
        'c3sqrtx',
        'c3x',
        'c4x',
        'ccx',
        'ch',
        'cp',
        'crx',
        'cry',
        'crz',
        'cswap',
        'csx',
        'cu',
        'cu1',
        'cu3',
        'cx',
        'cy',
        'cz',
        'h',
        'id',
        'p',
        'rc3x',
        'rccx',
        'rx',
        'rxx',
        'ry',
        'rz',
        'rzz',
        's',
        'sdg',
        'swap',
        'sx',
        'sxdg',
        't',
        'tdg',
        'u',
        'u0',
        'u1',
        'u2',
        'u3',
        'x',
        'y',
        'z',
    },
)

# Found live (2026-09) to have the exact same undefined-gate bug as #306,
# pre-dating this test: BQSKit exports each with no gate definition at all,
# and Qiskit's own QASM2 parser rejects the re-import with
# "'<name>' is not defined in this scope". Unlike `CSGate` (a simple
# controlled-phase definition via the qelib1.inc `cu1` gate), some of these
# (`BGate`, `IToffoliGate`, `SqrtISwapGate`, `SycamoreGate`) have no similarly
# trivial qelib1.inc-based decomposition, so fixing them is separate, larger
# follow-up work and out of scope for the #306 fix itself.
#
# Each is asserted via `xfail(strict=True)`: if a future fix makes one of
# these start passing, the test *fails* (not silently passes) until the
# gate's name is removed from this set -- so this known-gap list can't
# silently go stale.
_KNOWN_UNDEFINED_GATE_NAMES = frozenset(
    {
        'BGate',
        'CTGate',
        'IToffoliGate',
        'SqrtISwapGate',
        'SqrtTGate',
        'SycamoreGate',
        'YYGate',
    },
)


def _bare_qasm_name(qasm_name: str) -> str:
    """`'ryy(pi/2)'` -> `'ryy'`; `'cs'` -> `'cs'`."""
    m = re.match(r'^[a-zA-Z_]\w*', qasm_name)
    assert m is not None, f'unexpected qasm_name shape: {qasm_name!r}'
    return m.group(0)


def _gates_needing_a_definition() -> list[tuple[str, Gate]]:
    """Every no-arg-constructible, qubit-only constant gate whose QASM name
    is not a qelib1.inc builtin, paired with a ready-to-use instance.

    Gates that need constructor arguments (`PermutationGate`,
    `ConstantUnitaryGate`, ...) or are qudit-only (`ClockGate`, ...) are
    skipped -- they don't have a single canonical "the gate" to check here
    and would need bespoke handling of their own.
    """
    found = []
    for name in constant_gates.__all__:
        cls = getattr(constant_gates, name)
        try:
            instance = cls()
        except TypeError:
            continue
        try:
            qasm_name = instance.qasm_name
        except AttributeError:
            continue
        if _bare_qasm_name(qasm_name) not in _QELIB1_GATE_NAMES:
            found.append((name, instance))
    return found


def _as_param(name: str, instance: Gate) -> pytest.param:
    marks = []
    if name in _KNOWN_UNDEFINED_GATE_NAMES:
        marks.append(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{name} has the same undefined-QASM-gate bug as #306 "
                    'but is not fixed yet (tracked separately)'
                ),
            ),
        )
    return pytest.param(instance, id=name, marks=marks)


_GATES_NEEDING_A_DEFINITION = _gates_needing_a_definition()


def test_found_the_expected_gates() -> None:
    # Sanity check on the scan itself: if a new constant gate is added with
    # a QASM name not in qelib1.inc, it must show up here one way or another
    # (as a passing case below, or explicitly added to the known-gap set
    # above) -- this pins the current, fully-accounted-for set so that can't
    # happen silently.
    found = {name for name, _ in _GATES_NEEDING_A_DEFINITION}
    already_fixed = {'CSGate', 'ECRGate', 'IdentityGate', 'ISwapGate'}
    assert found == _KNOWN_UNDEFINED_GATE_NAMES | already_fixed


@pytest.mark.parametrize(
    'gate',
    [
        _as_param(name, instance)
        for name, instance in _GATES_NEEDING_A_DEFINITION
    ],
)
def test_qasm_export_is_reimportable_by_qiskit(gate: Gate) -> None:
    # This intentionally checks only that Qiskit's QASM2 parser *accepts* the
    # exported circuit (the actual #306 bug class: a reference to an
    # undefined gate name) -- not that the reimported unitary numerically
    # matches BQSKit's. Those are different questions: e.g. `ECRGate` already
    # has a working `get_qasm_gate_def()` (parses fine, excluded from
    # `_KNOWN_UNDEFINED_GATE_NAMES` above) but was found, independently of
    # this fix, to disagree with Qiskit's own `ECRGate` unitary by
    # convention (max `get_distance_from`) even natively (not through QASM
    # at all) -- a separate, pre-existing question for `ECRGate` itself, out
    # of scope for this test and for #306.
    pytest.importorskip('qiskit')
    from qiskit import QuantumCircuit
    from qiskit.qasm2.exceptions import QASM2ParseError

    num_qudits = gate.num_qudits
    circuit = Circuit(num_qudits)
    circuit.append_gate(gate, tuple(range(num_qudits)))
    output_qasm = OPENQASM2Language().encode(circuit)

    try:
        QuantumCircuit.from_qasm_str(output_qasm)
    except QASM2ParseError as e:
        pytest.fail(f"Qiskit rejected the exported QASM: {e}")
