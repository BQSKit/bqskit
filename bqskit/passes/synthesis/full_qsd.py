"""This module implements the Quantum Shannon Decomposition for one level."""
from __future__ import annotations

import copy
import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir.operation import Operation
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.parameterized.mcry import MCRYGate
from bqskit.ir.gates.parameterized.mcrz import MCRZGate
from bqskit.ir.gates.parameterized.cun import CUNGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.location import CircuitLocation
from bqskit.ir.gates import CircuitGate
from bqskit.runtime import get_runtime
from scipy.linalg import cossin, diagsvd, schur
import numpy as np
import time
from bqskit.passes.alias import PassAlias
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.control import WhileLoopPass
from bqskit.passes.control import WidthPredicate
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.control import IfThenElsePass
from bqskit.passes.partitioning import ScanPartitioner
from bqskit.passes.synthesis import QSDPass
from bqskit.passes.control import ChangePredicate
from bqskit.passes.control import ManyQuditGatesPredicate

_logger = logging.getLogger(__name__)


class FullQSDPass(PassAlias):
    """
    A pass performing one round of decomposition from the QSD algorithm.

    References:
        C.C. Paige, M. Wei,
        History and generality of the CS decomposition,
        Linear Algebra and its Applications,
        Volumes 208–209,
        1994,
        Pages 303-326,
        ISSN 0024-3795,
        https://doi.org/10.1016/0024-3795(94)90446-4.
    """

    cs_time = 0
    schur_time = 0
    create_circ_time = 0
    append_circ_time = 0
    init_time = 0
    replace_time = 0

    def __init__(
            self,
            start_from_left: bool = True,
            min_qudit_size: int = 2,
            instantiation_options = {},
        ) -> None:
            """
            Construct a single level of the QSDPass.

            Args:
                start_from_left (bool): Determines where the scan starts
                    attempting to remove gates from. If True, scan goes left
                    to right, otherwise right to left. (Default: True)

                min_qudit_size (int): Performs a decomposition on all gates
                    with widht > min_qudit_size
            """

            self.start_from_left = start_from_left
            self.min_qudit_size = min_qudit_size
            instantiation_options = {"method":"qfactor"}
            instantiation_options.update(instantiation_options)
            # scan = ScanningGateRemovalPass(start_from_left=start_from_left, instantiate_options=instantiation_options)
            qsd = QSDPass(min_qudit_size=min_qudit_size)
            self.passes: list[BasePass] = [
                WhileLoopPass(
                    ChangePredicate(),
                    [qsd]
                ),
            ]

    def get_passes(self) -> list[BasePass]:
        """Return the passes to be run, see :class:`PassAlias` for more."""
        return self.passes
