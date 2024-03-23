"""This module implements the PAMVerificationSequence and helper passes."""
from __future__ import annotations

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import PermutationGate
from bqskit.ir.gates import TaggedGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.passes.alias import PassAlias
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.mapping.pam import PAMBlockResultDict
from bqskit.passes.mapping.routing.pam import PAMRoutingPass
from bqskit.passes.partitioning import QuickPartitioner
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import Sequence


class TagPAMBlockDataPass(BasePass):
    """Tag the blocks with the PAM block data."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if PAMRoutingPass.out_data_key not in data:
            raise RuntimeError('PAMRoutingPass must be run to verify results.')

        block_datas: PAMBlockResultDict = data[PAMRoutingPass.out_data_key]
        for block_point, block_data in block_datas.items():
            op = circuit[block_point]
            tagged_gate = TaggedGate(op.gate, block_data)
            circuit.replace_gate(
                block_point,
                tagged_gate,
                op.location,
                op.params,
            )

        del data[PAMRoutingPass.out_data_key]


class CalculatePAMErrorsPass(BasePass):
    """Calculates error of a panel consisting of blocks tagged with PAM data."""

    @staticmethod
    def get_opp_perm(in_perm: Sequence[int]) -> Sequence[int]:
        return tuple(in_perm.index(i) for i in range(len(in_perm)))

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # calculate approximate (current) panel unitary
        current_unitary = circuit.get_unitary()

        # calculate exact panel unitary
        exact_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        for op in circuit:
            if isinstance(op.gate, TaggedGate):
                pi = op.gate.tag['pre_perm']
                pf = op.gate.tag['post_perm']
                in_utry = op.gate.tag['original_utry']
                n = in_utry.num_qudits
                exact_circuit.append_gate(
                    PermutationGate(
                        n, CalculatePAMErrorsPass.get_opp_perm(pi),
                    ), op.location,
                )
                exact_circuit.append_gate(
                    ConstantUnitaryGate(in_utry), op.location,
                )
                exact_circuit.append_gate(
                    PermutationGate(
                        n, CalculatePAMErrorsPass.get_opp_perm(pf),
                    ), op.location,
                )

            else:
                exact_circuit.append_gate(op.gate, op.location, op.params)

        exact_unitary = exact_circuit.get_unitary()

        # calculate error
        data.update_error_mul(current_unitary.get_distance_from(exact_unitary))


class UnTagPAMBlockDataPass(BasePass):
    """Untag the blocks with the PAM block data."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, TaggedGate):
                circuit.replace_gate(
                    (cycle, op.location[0]),
                    op.gate.gate,
                    op.location,
                    op.params,
                )


class PAMVerificationSequence(PassAlias):
    """Calculates the error of a PAM sequence."""

    def __init__(self, error_sim_size: int = 8) -> None:
        """
        Construct a PAMVerificationPass.

        Args:
            error_sim_size (int): The block size to use during error
                calculations.
        """
        if not is_integer(error_sim_size):
            raise TypeError(f'Expected integer, got {type(error_sim_size)}.')

        if error_sim_size < 2:
            raise ValueError('Expected positive integer greater than 1.')

        self.error_sim_size = error_sim_size

    def get_passes(self) -> list[BasePass]:
        """Return the aliased passes, see :class:`PassAlias` for more info."""
        return [
            TagPAMBlockDataPass(),
            QuickPartitioner(self.error_sim_size),
            ForEachBlockPass(CalculatePAMErrorsPass()),
            UnfoldPass(),
            UnTagPAMBlockDataPass(),
        ]
