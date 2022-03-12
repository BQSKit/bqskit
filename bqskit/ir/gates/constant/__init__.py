"""This package defines constant gates."""
from __future__ import annotations

from bqskit.ir.gates.constant.ch import CHGate
from bqskit.ir.gates.constant.cpi import CPIGate
from bqskit.ir.gates.constant.csum import CSUMGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.gates.constant.cy import CYGate
from bqskit.ir.gates.constant.cz import CZGate
from bqskit.ir.gates.constant.h import HGate
from bqskit.ir.gates.constant.identity import IdentityGate
from bqskit.ir.gates.constant.iswap import ISwapGate
from bqskit.ir.gates.constant.permutation import PermutationGate
from bqskit.ir.gates.constant.s import SGate
from bqskit.ir.gates.constant.sdg import SdgGate
from bqskit.ir.gates.constant.sqrtcnot import SqrtCNOTGate
from bqskit.ir.gates.constant.sqrtiswap import SqrtISwapGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.constant.sx import SXGate
from bqskit.ir.gates.constant.t import TGate
from bqskit.ir.gates.constant.tdg import TdgGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.constant.x import XGate
from bqskit.ir.gates.constant.xx import XXGate
from bqskit.ir.gates.constant.y import YGate
from bqskit.ir.gates.constant.yy import YYGate
from bqskit.ir.gates.constant.z import ZGate
from bqskit.ir.gates.constant.zz import ZZGate

__all__ = [
    'CHGate',
    'CPIGate',
    'CSUMGate',
    'CNOTGate',
    'CXGate',
    'CYGate',
    'CZGate',
    'HGate',
    'IdentityGate',
    'ISwapGate',
    'PermutationGate',
    'SGate',
    'SdgGate',
    'SqrtCNOTGate',
    'SwapGate',
    'SqrtXGate',
    'SqrtISwapGate',
    'SXGate',
    'TGate',
    'TdgGate',
    'ConstantUnitaryGate',
    'XGate',
    'XXGate',
    'YGate',
    'YYGate',
    'ZGate',
    'ZZGate',
]
