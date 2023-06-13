"""This package defines constant gates."""
from __future__ import annotations

from bqskit.ir.gates.qubit.constant.b import BGate
from bqskit.ir.gates.qubit.constant.ccx import CCXGate
from bqskit.ir.gates.qubit.constant.ccx import ToffoliGate
from bqskit.ir.gates.qubit.constant.ch import CHGate
from bqskit.ir.gates.qubit.constant.cpi import CPIGate
from bqskit.ir.gates.qubit.constant.cs import CSGate
from bqskit.ir.gates.qubit.constant.csum import CSUMGate
from bqskit.ir.gates.qubit.constant.ct import CTGate
from bqskit.ir.gates.qubit.constant.cx import CNOTGate
from bqskit.ir.gates.qubit.constant.cx import CXGate
from bqskit.ir.gates.qubit.constant.cy import CYGate
from bqskit.ir.gates.qubit.constant.cz import CZGate
from bqskit.ir.gates.qubit.constant.h import HGate
from bqskit.ir.gates.qubit.constant.identity import IdentityGate
from bqskit.ir.gates.qubit.constant.iswap import ISwapGate
from bqskit.ir.gates.qubit.constant.itoffoli import IToffoliGate
from bqskit.ir.gates.qubit.constant.permutation import PermutationGate
from bqskit.ir.gates.qubit.constant.rccx import MargolusGate
from bqskit.ir.gates.qubit.constant.rccx import RC3XGate
from bqskit.ir.gates.qubit.constant.rccx import RCCXGate
from bqskit.ir.gates.qubit.constant.s import SGate
from bqskit.ir.gates.qubit.constant.sdg import SdgGate
from bqskit.ir.gates.qubit.constant.sqrtcnot import SqrtCNOTGate
from bqskit.ir.gates.qubit.constant.sqrtiswap import SqrtISwapGate
from bqskit.ir.gates.qubit.constant.swap import SwapGate
from bqskit.ir.gates.qubit.constant.sx import SqrtXGate
from bqskit.ir.gates.qubit.constant.sx import SXGate
from bqskit.ir.gates.qubit.constant.sycamore import SycamoreGate
from bqskit.ir.gates.qubit.constant.t import TGate
from bqskit.ir.gates.qubit.constant.tdg import TdgGate
from bqskit.ir.gates.qubit.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.qubit.constant.x import XGate
from bqskit.ir.gates.qubit.constant.xx import XXGate
from bqskit.ir.gates.qubit.constant.y import YGate
from bqskit.ir.gates.qubit.constant.yy import YYGate
from bqskit.ir.gates.qubit.constant.z import ZGate
from bqskit.ir.gates.qubit.constant.zz import ZZGate

__all__ = [
    'CCXGate',
    'RCCXGate',
    'RC3XGate',
    'MargolusGate',
    'ToffoliGate',
    'CHGate',
    'CPIGate',
    'CSGate',
    'CSUMGate',
    'CTGate',
    'CNOTGate',
    'CXGate',
    'CYGate',
    'CZGate',
    'HGate',
    'IdentityGate',
    'ISwapGate',
    'IToffoliGate',
    'PermutationGate',
    'SGate',
    'SdgGate',
    'SqrtCNOTGate',
    'SwapGate',
    'SqrtXGate',
    'SqrtISwapGate',
    'SXGate',
    'SycamoreGate',
    'TGate',
    'TdgGate',
    'ConstantUnitaryGate',
    'XGate',
    'XXGate',
    'YGate',
    'YYGate',
    'ZGate',
    'ZZGate',
    'BGate',
]
