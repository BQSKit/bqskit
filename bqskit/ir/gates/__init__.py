"""
BQSKit Gates (:mod:`bqskit.ir.gates`)
=====================================

.. rubric:: Constant Gates

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:
    :template: autosummary/gate.rst

    BGate
    CCXGate
    ToffoliGate
    CHGate
    ClockGate
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
    PDGate
    PermutationGate
    MargolusGate
    RC3XGate
    RCCXGate
    SGate
    SdgGate
    ShiftGate
    SqrtCNOTGate
    SqrtISwapGate
    SubSwapGate
    SwapGate
    SqrtXGate
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
    CKMGate
    CKMdgGate
    CPGate
    ArbitraryCPhaseGate
    CRXGate
    CRYGate
    CRZGate
    CUGate
    FSIMGate
    PauliGate
    PhasedXZGate
    RSU3Gate
    RXGate
    RXXGate
    RYGate
    RYYGate
    RZGate
    RZZGate
    U1Gate
    U1qGate
    U1qPi2Gate
    U1qPiGate
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
    EmbeddedGate
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

.. rubric:: Gate Base Classes

.. autosummary::
    :toctree: autogen
    :recursive:
    :nosignatures:
    :template: autosummary/gate.rst

    ComposedGate
    QubitGate
    QutritGate
    QuditGate
    ConstantGate
    GeneralGate
"""
from __future__ import annotations

from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.composed import *  # noqa
from bqskit.ir.gates.composed import __all__ as composed_all
from bqskit.ir.gates.composedgate import ComposedGate
from bqskit.ir.gates.constant import *  # noqa
from bqskit.ir.gates.constant import __all__ as constant_all
from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.generalgate import GeneralGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.gates.parameterized import *  # noqa
from bqskit.ir.gates.parameterized import __all__ as parameterized_all
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.ir.gates.qutritgate import QutritGate
from bqskit.ir.gates.quditgate import QuditGate

__all__ = composed_all + constant_all + parameterized_all
__all__ += ['ComposedGate', 'ConstantGate']
__all__ += ['QubitGate', 'QutritGate', 'QuditGate']
__all__ += ['CircuitGate', 'MeasurementPlaceholder', 'BarrierPlaceholder']
__all__ += ['GeneralGate']

# TODO: Implement the rest of the gates in:
# https://pubs.aip.org/aip/jmp/article-abstract/56/3/032202/763827

# TODO: Implement generalization of CZ and CZD
# https://arxiv.org/abs/2206.07216
