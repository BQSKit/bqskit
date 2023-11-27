"""This module implements the Quantum Shannon Decomposition for one level."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass

from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.passes.alias import PassAlias
from bqskit.passes.processing import ScanningGateRemovalPass, TreeScanningGateRemovalPass
from bqskit.passes.synthesis.qsd import QSDPass
from bqskit.passes.synthesis.mgdp import MGDPass
from bqskit.ir.circuit import Circuit
from bqskit.passes.partitioning import DepthPartitioner
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.util import UnfoldPass

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
            tree_depth: int = 1,
            partition_depth = None, 
            instantiate_options = {},
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
            instantiation_options.update(instantiate_options)
            self.scan = TreeScanningGateRemovalPass(start_from_left=start_from_left, instantiate_options=instantiation_options, tree_depth=tree_depth)
            # self.scan = ScanningGateRemovalPass(start_from_left=start_from_left, instantiate_options=instantiation_options)
            self.qsd = QSDPass(min_qudit_size=min_qudit_size)
            self.mgd = MGDPass()
            if partition_depth:
                self.depth_partition = DepthPartitioner(depth=partition_depth)
            else:
                self.depth_partition = None

    def get_passes(self) -> list[BasePass]:
          return super().get_passes()

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        passes = []
        start_num = max(x.num_qudits for x in circuit.operations())
        for _ in range(self.min_qudit_size, start_num):
            passes.append(self.qsd)
            if self.depth_partition:
                passes.append(self.depth_partition)
                for_each_pass = ForEachBlockPass([self.scan])
                passes.append(for_each_pass)
                passes.append(UnfoldPass())
            else:
                passes.append(self.scan)

        for _ in range(1, start_num):
            passes.append(self.mgd)
            if self.depth_partition:
                passes.append(self.depth_partition)
                for_each_pass = ForEachBlockPass([self.scan])
                passes.append(for_each_pass)
                passes.append(UnfoldPass())
            else:
                passes.append(self.scan)

        # print(passes)

        await Workflow(passes).run(circuit, data)