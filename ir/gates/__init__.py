"""
BQSKit Gates (:mod:`bqskit.ir.gates`)
=====================================

.. rubric:: Gate Base Classes

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:
    :template: autosummary/gate.rst

    ComposedGate
    QubitGate
    QutritGate
    ConstantGate

.. rubric:: Constant Gates

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:
    :template: autosummary/gate.rst

    BGate
    CCXGate
    RCCXGate
    RC3XGate
    ToffoliGate
    CHGate
    CPIGate
    CSGate
    CSUMGate
    CTGate
    CNOTGate
    CXGate
    CYGate
    CZGate
    HGate
    IdentityGate
    ISwapGate
    IToffoliGate
    PermutationGate
    SGate
    SdgGate
    SqrtCNOTGate
    SwapGate
    SqrtXGate
    SqrtISwapGate
    SXGate
    SycamoreGate
    TGate
    TdgGate
    ConstantUnitaryGate
    XGate
    XXGate
    YGate
    YYGate
    ZGate
    ZZGate

.. rubric:: Parameterized Gates

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:
    :template: autosummary/gate.rst

    CCPGate
    CPGate
    CRXGate
    CRYGate
    CRZGate
    CUGate
    FSIMGate
    PauliGate
    PhasedXZGate
    RXGate
    RXXGate
    RYGate
    RYYGate
    RZGate
    RZZGate
    U1Gate
    U1qGate
    U1qPiGate
    U1qPi2Gate
    U2Gate
    U3Gate
    U8Gate
    VariableUnitaryGate

.. rubric:: Composed Gates

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:
    :template: autosummary/gate.rst

    ControlledGate
    DaggerGate
    FrozenParameterGate
    TaggedGate
    VariableLocationGate

.. rubric:: Special Gates

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:

    CircuitGate
    MeasurementPlaceholder
    BarrierPlaceholder
"""
from __future__ import annotations

from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.composed import *  # noqa
from bqskit.ir.gates.composed import __all__ as composed_all
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.qubit import *  # noqa
from bqskit.ir.gates.qubit import __qubit_all_constant__, __qubit_all_parameterized__
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.gates.qutrit import *  # noqa
from bqskit.ir.gates.qutrit import __qutrit_all_constant__, __qutrit_all_parameterized__
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.ir.gates.qutritgate import QutritGate

__all__ = composed_all + __qubit_all_constant__ + __qutrit_all_constant__ + __qubit_all_parameterized__ + __qutrit_all_parameterized__
__all__ += ['ComposedGate', 'QubitGate', 'QutritGate', 'ConstantGate']
__all__ += ['CircuitGate', 'MeasurementPlaceholder']

