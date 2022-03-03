"""This module implements the QuestRunner CircuitRunner."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import dual_annealing

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.exec.results import RunnerResults
from bqskit.exec.runner import CircuitRunner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.partitioning import QuickPartitioner
from bqskit.passes.synthesis import LEAPSynthesisPass

_logger = logging.getLogger(__name__)


class QuestRunner(CircuitRunner):
    """
    Simulate a circuit using the Quest algorithm.

    References:
        Patel, Tirthak, et al. "Robust and Resource-Efficient Quantum Circuit
        Approximation." arXiv preprint arXiv:2108.12714 (2021).
    """

    def __init__(
        self,
        sub_runner: CircuitRunner,
        block_size: int = 3,
        compiler: Compiler | None = None,
        weit: float = 0.5,
        pert: int = 100,
        approx_threshold: float | None = None,
        sample_size: int = 16,
    ) -> None:
        """
        Run the QUEST algorithm using `sub_runner` to execute circuits.

        Args:
            sub_runner (CircuitRunner): The CircuitRunner to execute the
                approximated circuits.

            block_size (int): The maximum number of qudits in each
                block. Defaults to 3.

            compiler (Compiler | None): The compiler object used
                to partitioning and synthesize the circuit. If left
                as None, then a standard compiler will be created.

            weit (float): The weight set on number of CNOTs versus
                approximation dissimilarity. Default 0.5.

            pert (int): The percentile of sample distances used to judge
                approximation quality.

            approx_threshold (float | None): The maximum allowed error
                in the approximation. If left as None, then it will be
                computed based on the number of blocks.

            sample_size (int): The number of approximations to generate.
        """
        self.sub_runner = sub_runner
        self.block_size = block_size
        self.compiler = compiler if compiler is not None else Compiler()
        self.weit = weit
        self.pert = pert
        self.approx_threshold = approx_threshold
        self.sample_size = sample_size

    def run(self, circuit: Circuit) -> RunnerResults:
        """Execute the circuit, see CircuitRunner.run for more info."""

        # 1. Compile the circuit
        synthesis_pass = LEAPSynthesisPass(store_partial_solutions=True)
        task = CompilationTask(
            circuit.copy(), [
                QuickPartitioner(self.block_size),
                ForEachBlockPass(synthesis_pass),
            ],
        )
        blocked_circuit = self.compiler.compile(task)

        # 2. Gather partial solutions
        data = self.compiler.analyze(task, ForEachBlockPass.key)
        psols, pts = self.parse_data(blocked_circuit, data)
        # psols: psols[i] = list[[circuit, dist]] -> block i's partial solutions
        # pts: pts[i] = CircuitPoint -> block i's locations

        # 3. Approximate circuit
        approx_circuits = self.approximate_circuit(blocked_circuit, psols, pts)

        # 4. Run circuits
        results = [self.sub_runner.run(c) for c in approx_circuits]

        # 5. Average and return results
        probs = np.sum(np.array([result.probs for result in results]), axis=0)
        probs /= self.sample_size
        return RunnerResults(circuit.num_qudits, circuit.radixes, probs)

    def parse_data(
        self,
        blocked_circuit: Circuit,
        data: dict[Any, Any],
    ) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint]]:
        """Parse the data outputed from synthesis."""
        block_data = data[0]

        psols: list[list[tuple[Circuit, float]]] = []
        pts: list[CircuitPoint] = []

        for block in block_data:
            pts.append(block['point'])
            exact_block = blocked_circuit[pts[-1]].gate._circuit.copy()  # type: ignore  # noqa
            exact_block.set_params(blocked_circuit[pts[-1]].params)
            exact_utry = exact_block.get_unitary()
            psols.append([(exact_block, 0.0)])

            if 'psols' not in block:
                continue

            for depth, psol_list in block['psols'].items():
                for psol in psol_list:
                    dist = psol[0].get_unitary().get_distance_from(exact_utry)
                    psols[-1].append((psol[0], dist))

        return psols, pts

    def approximate_circuit(
        self,
        circuit: Circuit,
        psols: list[list[tuple[Circuit, float]]],
        pts: list[CircuitPoint],
    ) -> list[Circuit]:
        """Use partial block solutions to compute circuit approximations."""
        num_blocks = len(pts)
        approx_threshold = self.approx_threshold or num_blocks / 10
        bounds = [[0, len(psol_list)] for psol_list in psols]

        psol_configs: list[list[int]] = []
        approx_circuits: list[Circuit] = []
        dists = [[psol[1] for psol in psol_list] for psol_list in psols]
        utrys = [
            [psol[0].get_unitary() for psol in psol_list]
            for psol_list in psols
        ]
        n_cxs = [
            [psol[0].count(CNOTGate()) for psol in psol_list]
            for psol_list in psols
        ]
        n_cx = sum(op.gate._circuit.count(CNOTGate()) for op in circuit)  # type: ignore  # noqa

        for _ in range(self.sample_size):
            x = []
            f = float('inf')
            for _ in range(3):
                result = dual_annealing(
                    annealing_objective,
                    args=(
                        approx_threshold,
                        n_cx,
                        psol_configs,
                        dists,
                        utrys,
                        n_cxs,
                        self.pert,
                        self.weit,
                    ),
                    x0=[0] * num_blocks,
                    bounds=bounds,
                    seed=int(time.time()),
                    initial_temp=5e4,
                    maxiter=10000,
                )

                if result.fun < f:
                    x = result.x
                    f = result.fun

            blocks = [int(x_e) for x_e in x]

            if blocks in psol_configs:
                _logger.debug('Generated a repeat approximate circuit.')
                break

            approx_dist = sum(
                psols[b][ind][1]
                for b, ind in enumerate(blocks)
            )

            if approx_dist > approx_threshold:
                _logger.debug('Approximate circuit has high distance.')
                break

            psol_configs.append(blocks)
            approx_circuit = self.assemble_circuit(circuit, blocks, psols, pts)
            approx_circuits.append(approx_circuit)
            _logger.info(
                'Generated approximate circuit with approximate distance:'
                f' {approx_dist}.',
            )
            _logger.info(
                f'Approximate circuit has {approx_circuit.count(CNOTGate())}'
                ' cnots.',
            )

        return approx_circuits

    def assemble_circuit(
        self,
        circuit: Circuit,
        blocks: list[int],
        psols: list[list[tuple[Circuit, float]]],
        pts: list[CircuitPoint],
    ) -> Circuit:
        """Assemble a circuit from a list of block indices."""
        circuit_gates = [
            CircuitGate(psol_list[b][0])
            for b, psol_list
            in zip(blocks, psols)
        ]
        locations = [circuit[pt].location for pt in pts]
        operations = [
            Operation(cg, loc, cg._circuit.params)
            for cg, loc
            in zip(circuit_gates, locations)
        ]
        copied_circuit = circuit.copy()
        copied_circuit.batch_replace(pts, operations)
        return copied_circuit


def annealing_objective(x: npt.NDArray[np.float64], *args: Any) -> float:
    """Compute the objective function used during approximation generation."""
    blocks = [int(ind) for ind in x]

    approx_threshold = args[0]
    upper_limit = args[1]
    psol_configs = args[2]
    dists = args[3]
    utrys = args[4]
    n_cxs = args[5]
    p = args[6]
    w = args[7]

    if blocks in psol_configs:
        return 1.2

    approx_dist = sum(dists[i][b] for i, b in enumerate(blocks))

    if approx_dist > approx_threshold:
        return 1.1

    n_cx = w * (sum(n_cxs[i][b] for i, b in enumerate(blocks)) / upper_limit)

    if len(psol_configs) == 0:
        return n_cx

    distances = [0] * len(psol_configs)

    for index, config in enumerate(psol_configs):
        distances[index] = sum(  # type: ignore
            utrys[i][config[i]].get_distance_from(utrys[i][b])
            < max(dists[i][config[i]], dists[i][b])
            for i, b in enumerate(blocks)
        ) / len(blocks)

    b_dv = (1 - w) * np.percentile(distances, p)

    return n_cx + b_dv
