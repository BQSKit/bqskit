"""This module defines a pre-built `compile` function using BQSKit."""
from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import RZGate
from bqskit.ir.gates import SqrtXGate
from bqskit.ir.gates import SwapGate
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.operation import Operation
from bqskit.ir.opt import HilbertSchmidtCostGenerator
from bqskit.ir.opt import ScipyMinimizer
from bqskit.passes import *
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bqskit.qis.unitary import UnitaryLike
    from bqskit.qis.state import StateLike


def compile(
    input: Circuit | UnitaryLike | StateLike,
    model: MachineModel | None = None,
    optimization_level: int = 1,
    approximation_level: int = 1,
    max_synthesis_size: int = 4,
    block_size: int = 3,
    verify_result: bool = False,
    *compiler_args: Any,
    **compiler_kwargs: Any,
) -> Circuit:
    """
    Compile a circuit, unitary, or state with a standard workflow.

    Args:
        input (Circuit | UnitaryLike | StateLike): The input to compile.

        model (MachineModel | None): A model of the target machine.
            Defaults to an all-to-all connected hardware with CNOTs and U3s
            as native gates. See :class:`MachineModel` for information
            on loading a preset model or creating a custom one.

        optimization_level (int): The degree of optimization in the workflow.
            The workflow will produce better circuits at the cost of
            performance with a higher number.  An optimization_level
            of 0 is not supported due to inherit optimization in any
            workflow containing synthesis. Valid inputs are 1, 2, 3, or 4.
            Be cautious, an optimization level of 4 can lead to extremely
            long runtimes with diminishing returns. (Default: 1)

        approximation_level (int): The degree of approximation in the workflow.
            A larger number allows for a greater degree of error in final
            solution. Valid inputs are 0, 1, 2, 3, 4. An approximation level
            of zero implies an exact circuit is desired, while a level of 4
            allows for very large error. (Default: 1)

        max_synthesis_size (int): The maximum size of a unitary to
            synthesize. Larger circuits will need to be partitioned.

        block_size (int): If a circuit is partitioned, it will be divided
            into circuits of at most this many number of qudits.

        compiler_args (Any): Passed directly to BQSKit compiler construction.
            Arguments for connecting to a dask cluster can go here.

        compiler_kwargs (Any): Passed directly to BQSKit compiler construction.
            Arguments for connecting to a dask cluster can go here.

    Returns:
        (Circuit): The compiled circuit.

    Examples:
        >>> from bqskit import Circuit, compile
        >>> circuit = Circuit.from_file('input.qasm')
        >>> compiled_circuit = compile(circuit)
    """
    if isinstance(input, Circuit):
        pass

    elif UnitaryMatrix.is_unitary(input):
        input = UnitaryMatrix(input)

    elif StateVector.is_pure_state(input):
        input = StateVector(input)

    else:
        raise TypeError(
            'Input is neither a circuit, a unitary, nor a state.'
            f' Got {type(input)}.',
        )

    assert isinstance(input, (Circuit, UnitaryMatrix, StateVector))

    if not all(r == 2 for r in input.radixes):
        raise ValueError(
            'Currently can only automatically build a workflow '
            'for qubit-only systems.',
        )  # TODO

    if model is None:
        model = MachineModel(input.num_qudits)

    if not isinstance(model, MachineModel):
        raise TypeError(f'Expected MachineModel for model, got {type(model)}.')

    if model.num_qudits < input.num_qudits:
        raise ValueError('Machine is too small for circuit.')

    if not all(r == 2 for r in model.radixes):
        raise ValueError(
            'Currently can only automatically build a workflow '
            'for qubit-only systems.',
        )  # TODO

    model_mq_gates = len([g for g in model.gate_set if g.num_qudits >= 2])

    if model_mq_gates == 0 and input.num_qudits > 1:
        raise ValueError('No entangling gates in native gate set.')

    if model_mq_gates >= 2:
        _logger.warning(
            'Multiple entangling gates in native gate set.\n'
            'Expect long compile times.',
        )

    if not is_integer(optimization_level):
        raise TypeError(
            'Expected integer for optimization_level'
            f', got{type(optimization_level)}.',
        )

    if optimization_level < 1 or optimization_level > 4:
        raise ValueError(
            'The optimization level should be either 1, 2, 3, or 4.'
            f' Got {optimization_level}.',
        )

    if not is_integer(approximation_level):
        raise TypeError(
            'Expected integer for approximation_level'
            f', got{type(approximation_level)}.',
        )

    if approximation_level < 0 or approximation_level > 4:
        raise ValueError(
            'The approximation level should be either 0, 1, 2, 3, or 4.'
            f' Got {approximation_level}.',
        )

    if not is_integer(max_synthesis_size):
        raise TypeError(
            'Expected integer for max_synthesis_size'
            f', got{type(max_synthesis_size)}.',
        )

    if max_synthesis_size < 2:
        raise ValueError(
            'The maximum synthesis size needs to be greater than 2.'
            f' Got {max_synthesis_size}.',
        )

    if not is_integer(block_size):
        raise TypeError(
            'Expected integer for block_size'
            f', got{type(block_size)}.',
        )

    if block_size < 2:
        raise ValueError(
            'The block size needs to be greater than 2.'
            f' Got {block_size}.',
        )

    if max_synthesis_size < block_size:
        raise ValueError(
            'The maximum synthesis size needs to be greater than the block '
            f' size. Got {max_synthesis_size} < {block_size}.',
        )

    if max_synthesis_size > 6 or block_size > 6:
        _logger.warning(
            'Large synthesis size limit or large block size.\n'
            'Expect long compile times.',
        )

    if any(g.num_qudits > block_size for g in model.gate_set):
        raise ValueError(
            'Unable to compile circuit to machine model with native gate'
            'larger than block_size.\nConsider adjusting block_size '
            'and max_synthesis_size or the machine model native set.',
        )

    if isinstance(input, Circuit):
        if input.num_qudits > max_synthesis_size:
            if any(
                g.num_qudits > max_synthesis_size
                and not isinstance(g, MeasurementPlaceholder)
                for g in input.gate_set
            ):
                raise ValueError(
                    'Unable to compile circuit with gate larger than'
                    ' max_synthesis_size.\nConsider adjusting it.',
                )

        task = _circuit_workflow(
            input,
            model,
            optimization_level,
            approximation_level,
            max_synthesis_size,
            block_size,
        )

    elif isinstance(input, UnitaryMatrix):
        utry = input

        if utry.num_qudits > max_synthesis_size:
            raise ValueError(
                'Cannot synthesize unitary larger than max_synthesis_size.\n'
                'Consider adjusting max_synthesis_size or checking the input.',
            )

        task = _circuit_workflow(
            Circuit.from_unitary(utry),
            model,
            optimization_level,
            approximation_level,
            max_synthesis_size,
            block_size,
        )

    elif isinstance(input, StateVector):
        state = input

        if state.num_qudits > max_synthesis_size:
            raise ValueError(
                'Cannot synthesize state larger than max_synthesis_size.\n'
                'Consider adjusting max_synthesis_size or checking the input.',
            )

        task = _stateprep_workflow(
            state,
            model,
            optimization_level,
            approximation_level,
            max_synthesis_size,
            block_size,
        )

    with Compiler(*compiler_args, **compiler_kwargs) as compiler:
        return compiler.compile(task)


def _circuit_workflow(
    circuit: Circuit,
    model: MachineModel,
    optimization_level: int = 1,
    approximation_level: int = 1,
    max_synthesis_size: int = 4,
    block_size: int = 3,
) -> CompilationTask:
    """Build standard workflow for circuit compilation."""
    workflow_builders = [
        _opt1_workflow,
        _opt2_workflow,
        _opt3_workflow,
        _opt4_workflow,
    ]
    workflow_builder = workflow_builders[optimization_level - 1]
    workflow = [UnfoldPass(), ExtractMeasurements()]
    workflow += workflow_builder(
        circuit,
        model,
        approximation_level,
        max_synthesis_size,
        block_size,
    )
    workflow += [RestoreMeasurements()]
    return CompilationTask(circuit, workflow)


def _opt1_workflow(
    circuit: Circuit,
    model: MachineModel,
    approximation_level: int = 1,
    max_synthesis_size: int = 4,
    block_size: int = 3,
) -> list[BasePass]:
    """Build Optimization Level 1 workflow for circuit compilation."""
    threshold = _get_threshold(approximation_level)
    layer_gen = _get_layer_gen(circuit, model)
    leap = LEAPSynthesisPass(
        success_threshold=threshold,
        layer_generator=layer_gen,
    )
    qfast = QFASTDecompositionPass(
        gate=PauliGate(max(g.num_qudits for g in model.gate_set)),
        success_threshold=threshold,
    )
    direct_synthesis = IfThenElsePass(
        WidthPredicate(4),
        leap,
        [qfast, ForEachBlockPass(leap), UnfoldPass()],
    )
    single_qudit_gate_rebase = _get_single_qudit_gate_rebase_pass(model)

    return [
        IfThenElsePass(
            WidthPredicate(max_synthesis_size + 1),

            [  # Direct Synthesis Branch
                SetModelPass(model),
                direct_synthesis,
                single_qudit_gate_rebase,
            ],

            [  # Partitioned Branch
                SetModelPass(model),
                QuickPartitioner(block_size),
                GreedyPlacementPass(),
                GeneralizedSabreLayoutPass(),
                GeneralizedSabreRoutingPass(),
                QuickPartitioner(block_size),
                ForEachBlockPass(
                    [direct_synthesis],
                    replace_filter=_gen_replace_filter(model),
                ),
                UnfoldPass(),
                single_qudit_gate_rebase,
            ],
        ),
    ]


def _opt2_workflow(
    circuit: Circuit,
    model: MachineModel,
    approximation_level: int = 1,
    max_synthesis_size: int = 4,
    block_size: int = 3,
) -> list[BasePass]:
    """Build Optimization Level 2 workflow for circuit compilation."""
    inst_ops = {'multistarts': 4, 'ftol': 5e-12, 'gtol': 1e-14}
    threshold = _get_threshold(approximation_level)
    layer_gen = _get_layer_gen(circuit, model)
    scan = ScanningGateRemovalPass(
        success_threshold=threshold,
    )
    leap = LEAPSynthesisPass(
        success_threshold=threshold,
        layer_generator=layer_gen,
        instantiate_options=inst_ops,
    )
    qfast = QFASTDecompositionPass(
        gate=PauliGate(max(g.num_qudits for g in model.gate_set)),
        success_threshold=threshold,
        instantiate_options=inst_ops,
    )
    direct_synthesis = IfThenElsePass(
        WidthPredicate(4),
        [leap, scan],
        [qfast, ForEachBlockPass([leap, scan]), UnfoldPass()],
    )
    single_qudit_gate_rebase = _get_single_qudit_gate_rebase_pass(model)

    return [
        IfThenElsePass(
            WidthPredicate(max_synthesis_size + 1),

            [  # Direct Synthesis Branch
                SetModelPass(model),
                direct_synthesis,
                single_qudit_gate_rebase,
            ],

            [  # Partitioned Branch
                SetModelPass(model),
                QuickPartitioner(block_size),
                GreedyPlacementPass(),
                GeneralizedSabreLayoutPass(),
                GeneralizedSabreRoutingPass(),
                QuickPartitioner(block_size),
                ForEachBlockPass(
                    [direct_synthesis],
                    replace_filter=_gen_replace_filter(model),
                ),
                UnfoldPass(),
                single_qudit_gate_rebase,
            ],
        ),
    ]


def _opt3_workflow(
    circuit: Circuit,
    model: MachineModel,
    approximation_level: int = 1,
    max_synthesis_size: int = 4,
    block_size: int = 3,
) -> list[BasePass]:
    """Build optimization Level 3 workflow for circuit compilation."""
    inst_ops = {
        'multistarts': 8,
        'method': 'minimization',
        'ftol': 5e-16,
        'gtol': 1e-15,
    }
    threshold = _get_threshold(approximation_level)
    layer_gen = _get_layer_gen(circuit, model)
    qsearch = QSearchSynthesisPass(
        success_threshold=threshold,
        layer_generator=layer_gen,
        instantiate_options=inst_ops,
    )
    leap = LEAPSynthesisPass(
        success_threshold=threshold,
        layer_generator=layer_gen,
        instantiate_options=inst_ops,
    )
    qsearch_notarget = QSearchSynthesisPass(
        success_threshold=threshold,
        layer_generator=FourParamGenerator(),
        instantiate_options=inst_ops,
    )
    leap_notarget = LEAPSynthesisPass(
        success_threshold=threshold,
        layer_generator=FourParamGenerator(),
        instantiate_options=inst_ops,
    )
    scan = ScanningGateRemovalPass(
        success_threshold=threshold,
        instantiate_options=inst_ops,
    )
    iter_del = WhileLoopPass(
        ChangePredicate(),
        [
            QuickPartitioner(block_size),
            ForEachBlockPass(scan),
            UnfoldPass(),
        ],
    )

    if not any(g.num_qudits > 2 for g in model.gate_set):
        tq_gates = [g for g in model.gate_set if g.num_qudits == 2]
        retarget: BasePass = Rebase2QuditGatePass(
            gate_in_circuit=CNOTGate(),
            new_gate=tq_gates,
            max_retries=3,
            success_threshold=threshold,
            instantiate_options=inst_ops,
        )
        direct_synthesis: BasePass = PassGroup(
            IfThenElsePass(
                WidthPredicate(4),
                [
                    ParallelDo(
                        [
                            [qsearch, scan],
                            [qsearch_notarget, retarget, scan],
                        ],
                        _less_tq_gates,
                    ),
                ],
                [
                    ParallelDo(
                        [
                            [leap, scan],
                            [leap_notarget, retarget, scan],
                        ],
                        _less_tq_gates,
                    ),
                ],
            ),
        )
        if len(tq_gates) == 1 and CNOTGate() in tq_gates:
            direct_synthesis = IfThenElsePass(
                WidthPredicate(4),
                [qsearch, scan],
                [leap, scan],
            )
        convert_swaps: BasePass = Rebase2QuditGatePass(SwapGate(), tq_gates, 3)

    else:
        direct_synthesis = IfThenElsePass(
            WidthPredicate(4),
            [qsearch, scan],
            [leap, scan],
        )
        convert_swaps = direct_synthesis

    single_qudit_gate_rebase = _get_single_qudit_gate_rebase_pass(model)

    return [
        IfThenElsePass(
            WidthPredicate(max_synthesis_size + 1),

            [  # Direct Synthesis Branch
                SetModelPass(model),
                direct_synthesis,
                single_qudit_gate_rebase,
            ],

            [  # Partitioned Branch
                SetModelPass(model),
                iter_del,
                GreedyPlacementPass(),
                QuickPartitioner(block_size),
                GeneralizedSabreLayoutPass(),
                GeneralizedSabreRoutingPass(),
                ForEachBlockPass([direct_synthesis]),
                UnfoldPass(),
                QuickPartitioner(block_size),
                ForEachBlockPass([convert_swaps]),
                UnfoldPass(),
                single_qudit_gate_rebase,
                iter_del,
            ],
        ),
    ]


def _opt4_workflow(
    circuit: Circuit,
    model: MachineModel,
    approximation_level: int = 1,
    max_synthesis_size: int = 4,
    block_size: int = 3,
) -> list[BasePass]:
    """Build optimization Level 4 workflow for circuit compilation."""
    raise NotImplementedError('Optimization level 4 is not yet ready.')


def _stateprep_workflow(
    state: StateVector,
    model: MachineModel | None = None,
    optimization_level: int = 1,
    approximation_level: int = 1,
    max_synthesis_size: int = 4,
    block_size: int = 3,
) -> CompilationTask:
    # TODO
    raise NotImplementedError('State preparation is not yet implemented.')


def _get_threshold(approximation_level: int) -> float:
    return [0.0, 1e-12, 1e-10, 1e-6, 1e-3][approximation_level]


def _get_layer_gen(circuit: Circuit, model: MachineModel) -> LayerGenerator:
    """Build a `model`-compliant layer generator."""
    tq_gates = [gate for gate in model.gate_set if gate.num_qudits == 2]
    mq_gates = [gate for gate in model.gate_set if gate.num_qudits > 2]

    if len(tq_gates) == 1 and len(mq_gates) == 0:
        if CNOTGate() in tq_gates:
            return FourParamGenerator()
        else:
            return SimpleLayerGenerator(tq_gates[0])

    return WideLayerGenerator(tq_gates + mq_gates)


def _get_single_qudit_gate_rebase_pass(model: MachineModel) -> BasePass:
    """Build a pass to convert single-qudit-gates to the native gate set."""
    sq_gates = [g for g in model.gate_set if g.num_qudits == 1]

    if len(sq_gates) == 0:
        return NOOPPass()

    if all(g.is_constant() for g in sq_gates):
        _logger.warn(
            'The standard workflow with BQSKit may have trouble'
            ' targeting gate sets containing no parameterized'
            ' single-qudit gates.',
        )
        # TODO: Implement better retargeting techniques for constant gate sets

    instantiate_options = {
        'method': 'minimization',
        'minimizer': ScipyMinimizer(),
        'cost_fn_gen': HilbertSchmidtCostGenerator(),
    }
    layer_generator = SingleQuditLayerGenerator(sq_gates, True)
    core_sq_rebase: BasePass = NOOPPass()
    if len(sq_gates) == 1:
        if sq_gates[0] == U3Gate():
            core_sq_rebase = U3Decomposition()

    elif len(sq_gates) == 2:
        if RZGate() in sq_gates and SqrtXGate() in sq_gates:
            core_sq_rebase = ZXZXZDecomposition()

    if isinstance(core_sq_rebase, NOOPPass):
        core_sq_rebase = QSearchSynthesisPass(
            layer_generator=layer_generator,
            heuristic_function=DijkstraHeuristic(),
            instantiate_options=instantiate_options,
        )

    return PassGroup([
        IfThenElsePass(
            NotPredicate(SinglePhysicalPredicate()),
            [
                LogPass('Retargeting single-qudit gates.'),
                GroupSingleQuditGatePass(),
                ForEachBlockPass([
                    IfThenElsePass(
                        NotPredicate(SinglePhysicalPredicate()),
                        core_sq_rebase,
                    ),
                ]),
                UnfoldPass(),
            ],
        ),
    ])


def _less_tq_gates(c1: Circuit, c2: Circuit) -> bool:
    """Determine is `c1` has less two qudit gates than `c2`."""
    c1_sq_counts = sum(c1.count(g) for g in c1.gate_set if g.num_qudits == 1)
    c1_tq_counts = sum(c1.count(g) for g in c1.gate_set if g.num_qudits == 2)
    c2_sq_counts = sum(c2.count(g) for g in c2.gate_set if g.num_qudits == 1)
    c2_tq_counts = sum(c2.count(g) for g in c2.gate_set if g.num_qudits == 2)
    return (c1_tq_counts, c1_sq_counts) < (c2_tq_counts, c2_sq_counts)


def _diff_gate_or_shorter_gates(org: Circuit, new: Circuit) -> bool:
    """Return true if new has a different 2q gate than org or is shorter."""
    org_mq_gates = [g for g in org.gate_set if g.num_qudits >= 2]
    if any(g not in new.gate_set for g in org_mq_gates):
        return True

    org_sq_counts = sum(org.count(g) for g in org.gate_set if g.num_qudits == 1)
    org_mq_counts = sum(org.count(g) for g in org.gate_set if g.num_qudits >= 2)
    new_sq_counts = sum(new.count(g) for g in new.gate_set if g.num_qudits == 1)
    new_mq_counts = sum(new.count(g) for g in new.gate_set if g.num_qudits >= 2)
    return (new_mq_counts, new_sq_counts) < (org_mq_counts, org_sq_counts)


def _gen_replace_filter(
    model: MachineModel,
) -> Callable[[Circuit, Operation], bool]:
    """Generate a replace filter for use during the standard workflow."""
    def _replace_filter(new: Circuit, old: Operation) -> bool:
        # return true if old doesn't satisfy model
        if not isinstance(old.gate, CircuitGate):
            return True

        org = old.gate._circuit

        if any(g not in model.gate_set for g in org.gate_set):
            return True

        if any(
            (old.location[e[0]], old.location[e[1]]) not in model.coupling_graph
            for e in org.coupling_graph
        ):
            return True

        # else pick shortest circuit
        org_sq_n = sum(org.count(g) for g in org.gate_set if g.num_qudits == 1)
        org_mq_n = sum(org.count(g) for g in org.gate_set if g.num_qudits >= 2)
        new_sq_n = sum(new.count(g) for g in new.gate_set if g.num_qudits == 1)
        new_mq_n = sum(new.count(g) for g in new.gate_set if g.num_qudits >= 2)
        return (new_mq_n, new_sq_n) < (org_mq_n, org_sq_n)
    return _replace_filter
