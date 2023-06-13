"""This package defines constant gates."""
from __future__ import annotations

from bqskit.ir.gates.qutrit.constant.ccx import CCX01Gate
from bqskit.ir.gates.qutrit.constant.ccx import CCX02Gate
from bqskit.ir.gates.qutrit.constant.ccx import CCX12Gate
from bqskit.ir.gates.qutrit.constant.ccx import CCXGate
from bqskit.ir.gates.qutrit.constant.ch import CHGate
from bqskit.ir.gates.qutrit.constant.cpi import CPIGate
from bqskit.ir.gates.qutrit.constant.cs import CSGate
from bqskit.ir.gates.qutrit.constant.csum import CSUMGate
from bqskit.ir.gates.qutrit.constant.ct import CTGate
from bqskit.ir.gates.qutrit.constant.cx import CNOTGate
from bqskit.ir.gates.qutrit.constant.cx import CX01Gate
from bqskit.ir.gates.qutrit.constant.cx import CX02Gate
from bqskit.ir.gates.qutrit.constant.cx import CX12Gate
from bqskit.ir.gates.qutrit.constant.cx import CXGate
from bqskit.ir.gates.qutrit.constant.cz import CZ0Gate
from bqskit.ir.gates.qutrit.constant.cz import CZ1Gate
from bqskit.ir.gates.qutrit.constant.cz import CZ2Gate
from bqskit.ir.gates.qutrit.constant.cz import CZGate
from bqskit.ir.gates.qutrit.constant.h import HGate
from bqskit.ir.gates.qutrit.constant.identity import IdentityGate
from bqskit.ir.gates.qutrit.constant.itoffoli import IToffoliGate
from bqskit.ir.gates.qutrit.constant.permutation import PermutationGate
from bqskit.ir.gates.qutrit.constant.s import SGate
from bqskit.ir.gates.qutrit.constant.sdg import SdgGate
from bqskit.ir.gates.qutrit.constant.swap import SwapGate
from bqskit.ir.gates.qutrit.constant.t import TGate
from bqskit.ir.gates.qutrit.constant.tdg import TdgGate
from bqskit.ir.gates.qutrit.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.qutrit.constant.x import X01Gate
from bqskit.ir.gates.qutrit.constant.x import X02Gate
from bqskit.ir.gates.qutrit.constant.x import X12Gate
from bqskit.ir.gates.qutrit.constant.x import XGate
from bqskit.ir.gates.qutrit.constant.xx import X01X01Gate
from bqskit.ir.gates.qutrit.constant.xx import X01X02Gate
from bqskit.ir.gates.qutrit.constant.xx import X02X01Gate
from bqskit.ir.gates.qutrit.constant.xx import X02X02Gate
from bqskit.ir.gates.qutrit.constant.xx import XXGate
from bqskit.ir.gates.qutrit.constant.z import Z0Gate
from bqskit.ir.gates.qutrit.constant.z import Z1Gate
from bqskit.ir.gates.qutrit.constant.z import Z2Gate
from bqskit.ir.gates.qutrit.constant.z import ZGate
from bqskit.ir.gates.qutrit.constant.zz import Z0Z0Gate
from bqskit.ir.gates.qutrit.constant.zz import Z1Z1Gate
from bqskit.ir.gates.qutrit.constant.zz import Z2Z2Gate
from bqskit.ir.gates.qutrit.constant.zz import ZZGate
# from bqskit.ir.gates.qutrit.constant.sx import SqrtXGate
# from bqskit.ir.gates.qutrit.constant.sqrtcnot import SqrtXGCate

__all__ = [
    'CCXGate',
    'CCX01Gate',
    'CCX02Gate',
    'CCX12Gate',
    'CHGate',
    'CPIGate',
    'CSUMGate',
    'CTGate',
    'CNOTGate',
    'CXGate',
    'CX01Gate',
    'CX02Gate',
    'CX12Gate',
    'CZGate',
    'CZ1Gate',
    'CZ2Gate',
    'CZ0Gate',
    'HGate',
    'IToffoliGate',
    'IdentityGate',
    'PermutationGate',
    'SGate',
    'SdgGate',
    'SqrtXGate',
    'SwapGate',
    'SqrtXGate',
    'TGate',
    'TdgGate',
    'ConstantUnitaryGate'
    'XGate',
    'X01Gate',
    'X02Gate',
    'X12Gate',
    'XXGate',
    'X01X01Gate',
    'X02X02Gate',
    'X01X02Gate',
    'X02X01Gate',
    'ZGate',
    'Z0Gate',
    'Z1Gate',
    'Z2Gate',
    'CZGate',
    'ZZGate',
    'Z0Z0Gate',
    'Z1Z1Gate',
    'Z2Z2Gate',
]
