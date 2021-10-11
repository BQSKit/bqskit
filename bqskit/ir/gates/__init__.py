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

    CHGate
    CPIGate
    CSUMGate
    CNOTGate
    CXGate
    CYGate
    CZGate
    HGate
    IdentityGate
    SGate
    SdgGate
    SqrtCNOTGate
    SwapGate
    SqrtXGate
    SXGate
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

    PauliGate
    RXGate
    RXXGate
    RYGate
    RYYGate
    RZGate
    RZZGate
    U1Gate
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
"""
from __future__ import annotations

from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.composed import *  # noqa
from bqskit.ir.gates.composed import __all__ as composed_all
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.constant import *  # noqa
from bqskit.ir.gates.constant import __all__ as constant_all
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.parameterized import *  # noqa
from bqskit.ir.gates.parameterized import __all__ as parameterized_all
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.ir.gates.qutritgate import QutritGate

__all__ = composed_all + constant_all + parameterized_all
__all__ += ['ComposedGate', 'QubitGate', 'QutritGate', 'ConstantGate']
__all__ += ['CircuitGate']
