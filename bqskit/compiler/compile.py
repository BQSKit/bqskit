"""This module defines a standard `compile` function using BQSKit."""
from __future__ import annotations

import logging
import warnings
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

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
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.operation import Operation
from bqskit.ir.opt import HilbertSchmidtCostGenerator
from bqskit.ir.opt import ScipyMinimizer
from bqskit.passes import *
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
_logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from bqskit.qis.unitary import UnitaryLike
    from bqskit.qis.state import StateLike


def compile(
    input: Circuit | UnitaryLike | StateLike,
    model: MachineModel | None = None,
    optimization_level: int = 1,
    max_synthesis_size: int = 3,
    synthesis_epsilon: float = 1e-10,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    compiler: Compiler | None = None,
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
            (Default: 1)

        max_synthesis_size (int): The maximum size of a unitary to
            synthesize or instantiate. Larger circuits will be partitioned.
            Increasing this will most likely lead to better results with
            an exponential time trade-off. (Default: 3)

        synthesis_epsilon (float): The maximum distance between target
            and circuit unitary during any instantiation or synthesis
            algorithms. (Default: 1e-10)

        error_threshold (float | None): This parameter controls the
            verification mechanism in this compile function. By default,
            it is set to None, so no verification is done. If you set
            this to a float, the upper bound on compilation error
            is calculated. If the upper bound is larger than this number,
            a warning will be logged. (Default: None)

        error_sim_size (int): If an `error_threshold` is set, the error
            upper bound is calculated by simulating blocks of this size.
            As you increase `error_sim_size`, the upper bound on error
            becomes more accurate. (Default: 8)

        compiler (Compiler | None): Pass a :class:`Compiler` to prevent
            creating one. Save on startup time by passing a compiler in
            when calling `compile` multiple times. (Default: None)

        compiler_args (Any): Passed directly to BQSKit compiler construction.
            Arguments for connecting to a dask cluster can go here.
            See :class:`Compiler` for more info.

        compiler_kwargs (Any): Passed directly to BQSKit compiler construction.
            Arguments for connecting to a dask cluster can go here.
            See :class:`Compiler` for more info.

    Returns:
        (Circuit): The compiled circuit.

    Examples:
        >>> from bqskit import Circuit, compile
        >>> circuit = Circuit.from_file('input.qasm')
        >>> compiled_circuit = compile(circuit)
        >>> compiled_circuit.save('output.qasm')
    """
    # Check `input`
    try:
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
    except Exception as e:
        raise TypeError(
            'Unable to determine type of input.'
            ' Ensure that you are trying to compile a valid'
            ' circuit, unitary, or state.',
        ) from e

    assert isinstance(input, (Circuit, UnitaryMatrix, StateVector))

    if not all(r == 2 for r in input.radixes):
        raise ValueError(
            'Currently, can only automatically build a workflow '
            'for qubit-only systems.',
        )  # TODO

    # Check `model`
    if model is None:
        model = MachineModel(input.num_qudits)

    if not isinstance(model, MachineModel):
        raise TypeError(f'Expected MachineModel for model, got {type(model)}.')

    if model.num_qudits < input.num_qudits:
        raise ValueError('Machine is too small for circuit.')

    if not all(r == 2 for r in model.radixes):
        raise ValueError(
            'Currently, can only automatically build a workflow '
            'for qubit-only systems.',
        )  # TODO

    model_mq_gates = [g for g in model.gate_set if g.num_qudits >= 2]

    if len(model_mq_gates) == 0 and input.num_qudits > 1:
        raise ValueError('No entangling gates in native gate set.')

    if all(g.num_qudits > input.num_qudits for g in model.gate_set):
        raise ValueError(
            'Model gate set does not contain any entangling gates'
            ' that is less than or equal to input size.'
            f' Cannot compile {input.num_qudits}-qudit input without'
            f' {input.num_qudits}-qudit or smaller gates.',
        )

    # Check `optimization_level`
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

    if len(model_mq_gates) >= 2 and optimization_level > 2:
        _logger.warning(
            'Multiple entangling gates in native gate set.\n'
            'Expect longer compile times.',
        )

    # Check `max_synthesis_size`
    if not is_integer(max_synthesis_size):
        raise TypeError(
            'Expected integer for max_synthesis_size'
            f', got{type(max_synthesis_size)}.',
        )

    if max_synthesis_size < max(g.num_qudits for g in model.gate_set):
        value = max(g.num_qudits for g in model.gate_set)
        raise ValueError(
            'The maximum synthesis size needs to be greater than or equal to'
            f' the largest native gate size: {max_synthesis_size} < {value}.',
        )

    if max_synthesis_size > 6:
        _logger.warning('Large max synthesis size; expect long compile times.')

    # Check `synthesis_epsilon`
    if not is_real_number(synthesis_epsilon):
        raise TypeError(
            'Expected float for synthesis_epsilon'
            f', got {type(synthesis_epsilon)}.',
        )

    if synthesis_epsilon < 0 or synthesis_epsilon > 1:
        raise ValueError(
            'Out-of-bounds synthesis_epsilon, it must be between 0 and 1,'
            f' got {synthesis_epsilon}.',
        )

    # Check `error_threshold`
    if error_threshold is not None:
        if not is_real_number(error_threshold):
            raise TypeError(
                'Expected float for error_threshold'
                f', got {type(error_threshold)}.',
            )

        if error_threshold < 0 or error_threshold > 1:
            raise ValueError(
                'Out-of-bounds error_threshold, it must be between 0 and 1,'
                f' got {error_threshold}.',
            )

    # Check `error_sim_size`
    if not is_integer(error_sim_size):
        raise TypeError(
            'Expected integer for error_sim_size'
            f', got{type(error_sim_size)}.',
        )

    if error_sim_size < max_synthesis_size:
        raise ValueError(
            'Simulation size for error calculation cannot be less than the'
            f'maximum synthesis size: {error_sim_size} < {max_synthesis_size}.',
        )

    # Build workflow
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
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
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
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
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
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        )

    # Connect to or construct a Compiler
    managed_compiler = compiler is None

    if managed_compiler:
        compiler = Compiler(*compiler_args, **compiler_kwargs)

    elif not isinstance(compiler, Compiler):
        raise TypeError(
            'Expected Compiler object for compiler parameter'
            f', got {type(compiler)}.',
        )

    # Perform the compilation
    out = compiler.compile(task)

    # Log error if necessary
    if error_threshold is not None:
        error = compiler.analyze(task, 'error')
        nonsq_error = 1 - np.sqrt(max(1 - (error * error), 0))
        if nonsq_error > error_threshold:
            warnings.warn(
                'Upper bound on error is greater than set threshold:'
                f' {nonsq_error} > {error_threshold}.',
            )

    # Close managed compiler
    if managed_compiler:
        compiler.close()

    return out


def _circuit_workflow(
    circuit: Circuit,
    model: MachineModel,
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-10,
    max_synthesis_size: int = 4,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
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
        synthesis_epsilon,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
    )
    workflow += [RestoreMeasurements()]
    return CompilationTask(circuit, workflow)


def _opt1_workflow(
    circuit: Circuit,
    model: MachineModel,
    synthesis_epsilon: float = 1e-10,
    max_synthesis_size: int = 4,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build Optimization Level 1 workflow for circuit compilation."""
    layer_gen = _get_layer_gen(model)
    scan_mq = ScanningGateRemovalPass(
        success_threshold=synthesis_epsilon,
        collection_filter=_mq_gate_collection_filter,
    )
    qsearch = QSearchSynthesisPass(
        success_threshold=synthesis_epsilon,
        layer_generator=layer_gen,
    )
    leap = LEAPSynthesisPass(
        success_threshold=synthesis_epsilon,
        layer_generator=layer_gen,
        min_prefix_size=3,
    )
    direct_synthesis = IfThenElsePass(WidthPredicate(3), qsearch, leap)
    single_qudit_gate_rebase = _get_single_qudit_gate_rebase_pass(model)
    if circuit.num_qudits > 1:
        smallest_entangler_size = min(
            g.num_qudits for g in model.gate_set
            if g.num_qudits != 1
        )
        non_native_gates = [
            g for g in circuit.gate_set
            if g not in model.gate_set
        ]
        non_native_tq_gates = [
            g for g in non_native_gates
            if g.num_qudits == 2
        ]
        if SwapGate() not in model.gate_set:
            non_native_tq_gates.append(SwapGate())
        native_tq_gates = [g for g in model.gate_set if g.num_qudits == 2]

        all_gates = model.gate_set.union(circuit.gate_set)
        if any(g.num_qudits > 2 for g in all_gates):
            multi_qudit_gate_rebase: BasePass = direct_synthesis
        else:
            multi_qudit_gate_rebase = Rebase2QuditGatePass(
                non_native_tq_gates,
                native_tq_gates,
                max_depth=3,
                max_retries=5,
            )
    else:
        smallest_entangler_size = 1
        multi_qudit_gate_rebase = NOOPPass()

    return [
        SetModelPass(model),

        # Multi-qudit gate retargeting
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Retargeting multi-qudit gates.'),
                QuickPartitioner(max_synthesis_size),
                ExtendBlockSizePass(smallest_entangler_size),
                QuickPartitioner(error_sim_size),
                ForEachBlockPass(
                    ForEachBlockPass(
                        [
                            FillSingleQuditGatesPass(synthesis_epsilon),
                            IfThenElsePass(
                                NotPredicate(MultiPhysicalPredicate()),
                                multi_qudit_gate_rebase,
                                scan_mq,
                            ),
                        ],
                        replace_filter=_gen_replace_filter(model),
                    ),
                    calculate_error_bound=(error_threshold is not None),
                ),
                UnfoldPass(),
            ],
        ),

        # Mapping via Sabre
        LogPass('Mapping circuit.'),
        GreedyPlacementPass(),
        GeneralizedSabreLayoutPass(),
        GeneralizedSabreRoutingPass(),

        # Swap gate retargeting
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Retargeting swap gates.'),
                QuickPartitioner(max_synthesis_size),
                ExtendBlockSizePass(smallest_entangler_size),
                QuickPartitioner(error_sim_size),
                ForEachBlockPass(
                    ForEachBlockPass(
                        [
                            FillSingleQuditGatesPass(synthesis_epsilon),
                            IfThenElsePass(
                                NotPredicate(MultiPhysicalPredicate()),
                                multi_qudit_gate_rebase,
                            ),
                        ],
                        replace_filter=_gen_replace_filter(model),
                    ),
                    calculate_error_bound=(error_threshold is not None),
                ),
                UnfoldPass(),
            ],
        ),

        # Single-qudit gate retargeting
        LogPass('Retargeting single-qudit gates.'),
        single_qudit_gate_rebase,

        # Finalizing
        LogErrorPass(),
        ApplyPlacement(),
    ]


def _opt2_workflow(
    circuit: Circuit,
    model: MachineModel,
    synthesis_epsilon: float = 1e-10,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build Optimization Level 2 workflow for circuit compilation."""
    inst_ops = {'multistarts': 4, 'ftol': 5e-12, 'gtol': 1e-14}
    layer_gen = _get_layer_gen(model)
    scan_mq = ScanningGateRemovalPass(
        success_threshold=synthesis_epsilon,
        collection_filter=_mq_gate_collection_filter,
    )
    scan = ScanningGateRemovalPass(success_threshold=synthesis_epsilon)
    qsearch = QSearchSynthesisPass(
        success_threshold=synthesis_epsilon,
        layer_generator=layer_gen,
        instantiate_options=inst_ops,
    )
    leap = LEAPSynthesisPass(
        success_threshold=synthesis_epsilon,
        layer_generator=layer_gen,
        instantiate_options=inst_ops,
        min_prefix_size=5,
    )
    direct_synthesis = IfThenElsePass(WidthPredicate(3), qsearch, leap)
    single_qudit_gate_rebase = _get_single_qudit_gate_rebase_pass(model)
    if circuit.num_qudits > 1:
        smallest_entangler_size = min(
            g.num_qudits for g in model.gate_set
            if g.num_qudits != 1
        )
        non_native_gates = [
            g for g in circuit.gate_set
            if g not in model.gate_set
        ]
        non_native_tq_gates = [
            g for g in non_native_gates
            if g.num_qudits == 2
        ]
        if SwapGate() not in model.gate_set:
            non_native_tq_gates.append(SwapGate())
        native_tq_gates = [g for g in model.gate_set if g.num_qudits == 2]

        all_gates = model.gate_set.union(circuit.gate_set)
        if any(g.num_qudits > 2 for g in all_gates):
            multi_qudit_gate_rebase: BasePass = direct_synthesis
        else:
            multi_qudit_gate_rebase = Rebase2QuditGatePass(
                non_native_tq_gates,
                native_tq_gates,
                max_depth=3,
                max_retries=5,
            )
    else:
        smallest_entangler_size = 1
        multi_qudit_gate_rebase = NOOPPass()

    return [
        SetModelPass(model),

        # Multi-qudit gate retargeting
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Retargeting multi-qudit gates.'),
                QuickPartitioner(max_synthesis_size),
                ExtendBlockSizePass(smallest_entangler_size),
                QuickPartitioner(error_sim_size),
                ForEachBlockPass(
                    ForEachBlockPass(
                        [
                            FillSingleQuditGatesPass(synthesis_epsilon),
                            IfThenElsePass(
                                NotPredicate(MultiPhysicalPredicate()),
                                multi_qudit_gate_rebase,
                                scan_mq,
                            ),
                        ],
                        replace_filter=_gen_replace_filter(model),
                    ),
                    calculate_error_bound=(error_threshold is not None),
                ),
                UnfoldPass(),
            ],
        ),

        # Mapping via Sabre
        LogPass('Mapping circuit.'),
        GreedyPlacementPass(),
        GeneralizedSabreLayoutPass(),
        GeneralizedSabreRoutingPass(),

        # Swap gate retargeting
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Retargeting swap gates.'),
                QuickPartitioner(max_synthesis_size),
                ExtendBlockSizePass(smallest_entangler_size),
                QuickPartitioner(error_sim_size),
                ForEachBlockPass(
                    ForEachBlockPass(
                        [
                            FillSingleQuditGatesPass(synthesis_epsilon),
                            IfThenElsePass(
                                NotPredicate(MultiPhysicalPredicate()),
                                multi_qudit_gate_rebase,
                            ),
                        ],
                        replace_filter=_gen_replace_filter(model),
                    ),
                    calculate_error_bound=(error_threshold is not None),
                ),
                UnfoldPass(),
            ],
        ),

        # Single-qudit gate retargeting
        LogPass('Retargeting single-qudit gates.'),
        single_qudit_gate_rebase,

        # Optimization: Scanning gate deletion on blocks
        LogPass('Attempting to delete gates.'),
        QuickPartitioner(max_synthesis_size),
        ExtendBlockSizePass(smallest_entangler_size),
        QuickPartitioner(error_sim_size),
        ForEachBlockPass(
            ForEachBlockPass(scan),
            calculate_error_bound=(error_threshold is not None),
        ),
        UnfoldPass(),

        # Finalizing
        LogErrorPass(),
        ApplyPlacement(),
    ]


def _opt3_workflow(
    circuit: Circuit,
    model: MachineModel,
    synthesis_epsilon: float = 1e-10,
    max_synthesis_size: int = 4,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build Optimization Level 3 workflow for circuit compilation."""
    inst_ops = {
        'multistarts': 8,
        'method': 'minimization',
        'ftol': 5e-16,
        'gtol': 1e-15,
    }
    layer_gen = _get_layer_gen(model)
    scan = ScanningGateRemovalPass(success_threshold=synthesis_epsilon)
    scan_mq = ScanningGateRemovalPass(
        success_threshold=synthesis_epsilon,
        collection_filter=_mq_gate_collection_filter,
    )
    qsearch = QSearchSynthesisPass(
        success_threshold=synthesis_epsilon,
        layer_generator=layer_gen,
        instantiate_options=inst_ops,
    )
    leap = LEAPSynthesisPass(
        success_threshold=synthesis_epsilon,
        layer_generator=layer_gen,
        instantiate_options=inst_ops,
        min_prefix_size=7,
    )
    direct_synthesis = IfThenElsePass(WidthPredicate(4), qsearch, leap)
    single_qudit_gate_rebase = _get_single_qudit_gate_rebase_pass(model)
    if circuit.num_qudits > 1:
        smallest_entangler_size = min(
            g.num_qudits for g in model.gate_set
            if g.num_qudits != 1
        )
        non_native_gates = [
            g for g in circuit.gate_set
            if g not in model.gate_set
        ]
        non_native_tq_gates = [
            g for g in non_native_gates
            if g.num_qudits == 2
        ]
        if SwapGate() not in model.gate_set:
            non_native_tq_gates.append(SwapGate())
        native_tq_gates = [g for g in model.gate_set if g.num_qudits == 2]
        native_mq_gates = [g for g in model.gate_set if g.num_qudits >= 2]

        all_gates = model.gate_set.union(circuit.gate_set)
        if any(g.num_qudits > 2 for g in all_gates):
            multi_qudit_gate_rebase: BasePass = direct_synthesis
        else:
            multi_qudit_gate_rebase = Rebase2QuditGatePass(
                non_native_tq_gates,
                native_tq_gates,
                max_depth=3,
                max_retries=5,
            )
    else:
        smallest_entangler_size = 1
        multi_qudit_gate_rebase = NOOPPass()
        native_mq_gates = []

    return [
        SetModelPass(model),

        # Multi-qudit gate retargeting
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Retargeting multi-qudit gates.'),
                QuickPartitioner(max_synthesis_size),
                ExtendBlockSizePass(smallest_entangler_size),
                QuickPartitioner(error_sim_size),
                ForEachBlockPass(
                    ForEachBlockPass(
                        [
                            FillSingleQuditGatesPass(synthesis_epsilon),
                            IfThenElsePass(
                                NotPredicate(MultiPhysicalPredicate()),
                                multi_qudit_gate_rebase,
                                scan_mq,
                            ),
                        ],
                        replace_filter=_gen_replace_filter(model),
                    ),
                    calculate_error_bound=(error_threshold is not None),
                ),
                UnfoldPass(),
            ],
        ),

        # Mapping via Sabre
        LogPass('Mapping circuit.'),
        GreedyPlacementPass(),
        GeneralizedSabreLayoutPass(),
        GeneralizedSabreRoutingPass(),

        # Swap gate retargeting
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Retargeting swap gates.'),
                QuickPartitioner(max_synthesis_size),
                ExtendBlockSizePass(smallest_entangler_size),
                QuickPartitioner(error_sim_size),
                ForEachBlockPass(
                    ForEachBlockPass(
                        [
                            FillSingleQuditGatesPass(synthesis_epsilon),
                            IfThenElsePass(
                                NotPredicate(MultiPhysicalPredicate()),
                                multi_qudit_gate_rebase,
                            ),
                        ],
                        replace_filter=_gen_replace_filter(model),
                    ),
                    calculate_error_bound=(error_threshold is not None),
                ),
                UnfoldPass(),
            ],
        ),

        # Optimization: Iterative Resynthesis
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                WhileLoopPass(
                    GateCountPredicate(native_mq_gates),
                    [
                        LogPass('Resynthesizing blocks.'),
                        QuickPartitioner(max_synthesis_size),
                        ExtendBlockSizePass(smallest_entangler_size),
                        QuickPartitioner(error_sim_size),
                        ForEachBlockPass(
                            ForEachBlockPass(
                                direct_synthesis,
                                replace_filter=_gen_replace_filter(model),
                            ),
                            calculate_error_bound=(error_threshold is not None),
                        ),
                        UnfoldPass(),
                    ],
                ),
            ],
        ),

        # Single-qudit gate retargeting
        LogPass('Retargeting single-qudit gates.'),
        single_qudit_gate_rebase,

        # Optimization: Iterative gate deletion on blocks
        WhileLoopPass(
            ChangePredicate(),
            [
                LogPass('Attempting to delete gates.'),
                QuickPartitioner(max_synthesis_size),
                QuickPartitioner(error_sim_size),
                ForEachBlockPass(
                    ForEachBlockPass(scan),
                    calculate_error_bound=(error_threshold is not None),
                ),
                UnfoldPass(),
            ],
        ),

        # Finalizing
        LogErrorPass(),
        ApplyPlacement(),
    ]


def _opt4_workflow(
    circuit: Circuit,
    model: MachineModel,
    synthesis_epsilon: float = 1e-10,
    max_synthesis_size: int = 4,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build optimization Level 4 workflow for circuit compilation."""
    # TODO
    raise NotImplementedError('Optimization level 4 is not yet ready.')


def _stateprep_workflow(
    state: StateVector,
    model: MachineModel,
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-10,
    max_synthesis_size: int = 4,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> CompilationTask:
    """Build a workflow for state preparation."""
    # TODO
    raise NotImplementedError('State preparation is not yet implemented.')


def _get_layer_gen(model: MachineModel) -> LayerGenerator:
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

    elif len(sq_gates) >= 2:
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


def _gen_replace_filter(
    model: MachineModel,
) -> Callable[[Circuit, Operation], bool]:
    """Generate a replace filter for use during the standard workflow."""
    def _replace_filter(new: Circuit, old: Operation) -> bool:
        # return true if old doesn't satisfy model
        if not isinstance(old.gate, CircuitGate):
            return True

        org = old.gate._circuit
        org_mq_gates = [g for g in org.gate_set if g.num_qudits > 1]

        if any(g not in model.gate_set for g in org_mq_gates):
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


def _mq_gate_collection_filter(op: Operation) -> bool:
    """Return true if `op` is a multi-qudit operation."""
    return op.num_qudits > 1
