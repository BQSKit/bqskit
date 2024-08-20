"""This module defines a standard `compile` function using BQSKit."""
from __future__ import annotations

import logging
import warnings
from typing import Any
from typing import Literal
from typing import overload
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

from bqskit.compiler.compiler import Compiler
from bqskit.compiler.gateset import GateSet
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.compiler.registry import _workflow_registry
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.measure import MeasurementPlaceholder
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.operation import Operation
from bqskit.ir.opt import HilbertSchmidtCostGenerator
from bqskit.ir.opt import ScipyMinimizer
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.control.foreach import gen_replace_filter
from bqskit.passes.control.foreach import ReplaceFilterFn
from bqskit.passes.control.ifthenelse import IfThenElsePass
from bqskit.passes.control.predicates.change import ChangePredicate
from bqskit.passes.control.predicates.count import GateCountPredicate
from bqskit.passes.control.predicates.many import ManyQuditGatesPredicate
from bqskit.passes.control.predicates.multi import MultiPhysicalPredicate
from bqskit.passes.control.predicates.notpredicate import NotPredicate
from bqskit.passes.control.predicates.single import AllConstantSingleQuditGates
from bqskit.passes.control.predicates.single import HasGeneralSingleQuditGate
from bqskit.passes.control.predicates.single import NoSingleQuditGatesInModel
from bqskit.passes.control.predicates.single import SinglePhysicalPredicate
from bqskit.passes.control.predicates.single import ZXGatePredicate
from bqskit.passes.control.predicates.width import WidthPredicate
from bqskit.passes.control.whileloop import WhileLoopPass
from bqskit.passes.group import PassGroup
from bqskit.passes.mapping.apply import ApplyPlacement
from bqskit.passes.mapping.embed import EmbedAllPermutationsPass
from bqskit.passes.mapping.layout.pam import PAMLayoutPass
from bqskit.passes.mapping.layout.sabre import GeneralizedSabreLayoutPass
from bqskit.passes.mapping.placement.greedy import GreedyPlacementPass
from bqskit.passes.mapping.routing.pam import PAMRoutingPass
from bqskit.passes.mapping.routing.sabre import GeneralizedSabreRoutingPass
from bqskit.passes.mapping.setmodel import ExtractModelConnectivityPass
from bqskit.passes.mapping.setmodel import RestoreModelConnectivityPass
from bqskit.passes.mapping.setmodel import SetModelPass
from bqskit.passes.mapping.topology import SubtopologySelectionPass
from bqskit.passes.mapping.verify import PAMVerificationSequence
from bqskit.passes.measure import ExtractMeasurements
from bqskit.passes.measure import RestoreMeasurements
from bqskit.passes.noop import NOOPPass
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.retarget.auto import AutoRebase2QuditGatePass
from bqskit.passes.retarget.general import GeneralSQDecomposition
from bqskit.passes.rules.u3 import U3Decomposition
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition
from bqskit.passes.search.generators.single import SingleQuditLayerGenerator
from bqskit.passes.search.heuristics.dijkstra import DijkstraHeuristic
from bqskit.passes.synthesis.leap import LEAPSynthesisPass
from bqskit.passes.synthesis.pas import PermutationAwareSynthesisPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.passes.synthesis.target import SetTargetPass
from bqskit.passes.util.extend import ExtendBlockSizePass
from bqskit.passes.util.fill import FillSingleQuditGatesPass
from bqskit.passes.util.log import LogErrorPass
from bqskit.passes.util.log import LogPass
from bqskit.passes.util.random import SetRandomSeedPass
from bqskit.passes.util.unfold import UnfoldPass
from bqskit.qis.state import StateLike
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.state.system import StateSystemLike
from bqskit.qis.unitary import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_iterable
from bqskit.utils.typing import is_real_number

if TYPE_CHECKING:
    from bqskit.compiler.basepass import BasePass


_logger = logging.getLogger(__name__)


CompilationInputLike = Union[Circuit, UnitaryLike, StateLike, StateSystemLike]
CompilationInput = Union[Circuit, UnitaryMatrix, StateVector, StateSystem]


@overload
def compile(
    input: CompilationInputLike,
    model: MachineModel | None = ...,
    *,
    with_mapping: Literal[False] = ...,
    optimization_level: int = ...,
    max_synthesis_size: int = ...,
    synthesis_epsilon: float = ...,
    error_threshold: float | None = ...,
    error_sim_size: int = ...,
    compiler: Compiler | None = ...,
    seed: int | None = ...,
    **compiler_kwargs: Any,
) -> Circuit:
    ...


@overload
def compile(
    input: Sequence[CompilationInput],
    model: MachineModel | None = ...,
    *,
    with_mapping: Literal[False] = ...,
    optimization_level: int = ...,
    max_synthesis_size: int = ...,
    synthesis_epsilon: float = ...,
    error_threshold: float | None = ...,
    error_sim_size: int = ...,
    compiler: Compiler | None = ...,
    seed: int | None = ...,
    **compiler_kwargs: Any,
) -> list[Circuit]:
    ...


@overload
def compile(
    input: CompilationInputLike,
    model: MachineModel | None = ...,
    *,
    with_mapping: Literal[True],
    optimization_level: int = ...,
    max_synthesis_size: int = ...,
    synthesis_epsilon: float = ...,
    error_threshold: float | None = ...,
    error_sim_size: int = ...,
    compiler: Compiler | None = ...,
    seed: int | None = ...,
    **compiler_kwargs: Any,
) -> tuple[Circuit, tuple[int, ...], tuple[int, ...]]:
    ...


@overload
def compile(
    input: Sequence[CompilationInput],
    model: MachineModel | None = ...,
    *,
    with_mapping: Literal[True],
    optimization_level: int = ...,
    max_synthesis_size: int = ...,
    synthesis_epsilon: float = ...,
    error_threshold: float | None = ...,
    error_sim_size: int = ...,
    compiler: Compiler | None = ...,
    seed: int | None = ...,
    **compiler_kwargs: Any,
) -> list[tuple[Circuit, tuple[int, ...], tuple[int, ...]]]:
    ...


@overload
def compile(
    input: CompilationInputLike,
    model: MachineModel | None = ...,
    *,
    with_mapping: bool,
    optimization_level: int = ...,
    max_synthesis_size: int = ...,
    synthesis_epsilon: float = ...,
    error_threshold: float | None = ...,
    error_sim_size: int = ...,
    compiler: Compiler | None = ...,
    seed: int | None = ...,
    **compiler_kwargs: Any,
) -> Circuit | tuple[Circuit, tuple[int, ...], tuple[int, ...]]:
    ...


@overload
def compile(
    input: Sequence[CompilationInput],
    model: MachineModel | None = ...,
    *,
    with_mapping: bool,
    optimization_level: int = ...,
    max_synthesis_size: int = ...,
    synthesis_epsilon: float = ...,
    error_threshold: float | None = ...,
    error_sim_size: int = ...,
    compiler: Compiler | None = ...,
    seed: int | None = ...,
    **compiler_kwargs: Any,
) -> list[Circuit] | list[tuple[Circuit, tuple[int, ...], tuple[int, ...]]]:
    ...


def compile(
    input: CompilationInputLike | Sequence[CompilationInput],
    model: MachineModel | None = None,
    optimization_level: int = 1,
    max_synthesis_size: int = 3,
    synthesis_epsilon: float = 1e-8,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    compiler: Compiler | None = None,
    seed: int | None = None,
    with_mapping: bool = False,
    **compiler_kwargs: Any,
) -> (
    Circuit
    | tuple[Circuit, tuple[int, ...], tuple[int, ...]]
    | list[Circuit]
    | list[tuple[Circuit, tuple[int, ...], tuple[int, ...]]]
):
    """
    Compile a circuit, unitary, or state with a standard workflow.

    Args:
        input (CompilationInputLike | Sequence[CompilationInput]): The input
            or inputs to compile. If a single input is given, a single
            circuit will be returned. If an iterable of inputs is given,
            a list of circuits will be returned.

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
            algorithms. (Default: 1e-8)

        error_threshold (float | None): This parameter controls the
            verification mechanism in this compile function. By default,
            it is set to None, so no verification is done. If you set
            this to a float, the upper bound on compilation error
            is calculated. If the upper bound is larger than this number,
            a warning will be logged. (Default: None)

        error_sim_size (int): If an `error_threshold` is set, the error
            upper bound is calculated by simulating blocks of this size.
            As you increase `error_sim_size`, the upper bound on error
            becomes more accurate. This setting is ignored with direct
            synthesis compilations, i.e., when a state, system, or unitary
            are given as input. (Default: 8)

        compiler (Compiler | None): Pass a :class:`Compiler` to prevent
            creating one. Save on startup time by passing a compiler in
            when calling `compile` multiple times. (Default: None)

        seed (int | None): Set a seed for the compile function for
            better reproducibility. If left as None, will not set seed.

        with_mapping (bool): If True, three values will be returned
            instead of just the compiled circuit. The first value is the
            compiled circuit, the second value is the initial mapping,
            and the third value is the final mapping. The initial mapping
            is a tuple where `initial_mapping[i] = j` implies that logical
            qudit `i` in the input system starts on the physical qudit
            `j` in the output circuit. Likewise, the final mapping describes
            where the logical qudits are in the physical circuit at the end
            of execution. (Default: False)

        compiler_kwargs (Any): Passed directly to BQSKit compiler construction.
            Arguments for connecting to a cluster can go here.
            See :class:`Compiler` for more info.

    Returns:
        (Circuit): The compiled circuit.

    Examples:
        >>> from bqskit import Circuit, compile
        >>> circuit = Circuit.from_file('input.qasm')
        >>> compiled_circuit = compile(circuit)
        >>> compiled_circuit.save('output.qasm')

        >>> from bqskit import compile
        >>> from bqskit.qis.state import StateVector
        >>> state = StateVector.random(3)
        >>> compiled_circuit = compile(state)
        >>> compiled_circuit.save('output.qasm')

        >>> from bqskit import compile
        >>> from bqskit.qis.unitary import UnitaryMatrix
        >>> utry = UnitaryMatrix.random(3)
        >>> compiled_circuit = compile(utry)
        >>> compiled_circuit.save('output.qasm')

        >>> from bqskit import compile, Circuit
        >>> circuits = [Circuit.from_file(f'input{i}.qasm') for i in range(5)]
        >>> compiled_circuits = compile(circuits)
        >>> for i, circuit in enumerate(compiled_circuits):
        ...     circuit.save(f'output{i}.qasm')

        >>> from bqskit import compile, Circuit, MachineModel
        >>> from bqskit.ir.gates import CZGate, RZGate, SqrtXGate
        >>> target_gate_set = {CZGate(), RZGate(), SqrtXGate()}
        >>> circuit = Circuit.from_file('input.qasm')
        >>> model = MachineModel(circuit.num_qudits, gate_set=target_gate_set)
        >>> compiled_circuit = compile(circuit, model, optimization_level=2)

        You can also use pre-built models from the :obj:`~bqskit.ext` package
        for common hardware. For example, to compile to the H1-1 machine
        from Quantinuum:

        >>> from bqskit import compile, Circuit
        >>> from bqskit.ext import H1_1Model
        >>> circuit = Circuit.from_file('input.qasm')
        >>> compiled_circuit = compile(circuit, H1_1Model)

    Raises:
        ValueError: If the input is an empty iterable.

        ValueError: If the input (or any input) is larger than the model.

        ValueError: If either the input or the model has mixed radixes.

        ValueError: If the model has a mismatched radix with the circuit.

        ValueError: If the model doesn't contain any entangling gates and
            the input (or any input) is larger than one-qudit.

        ValueError: If the model does not contain any entangling gates
            that are less than or equal to the input size.

        ValueError: If optimization_level is not 1, 2, 3, or 4.

        ValueError: If the maximum synthesis size is less than the
            largest gate in the model.

        ValueError: If the maximum synthesis size is less than 2.

        ValueError: If the synthesis epsilon is not between 0 and 1.

        ValueError: The error simulation size is less than the maximum
            synthesis size.
    """
    # Check `model`
    if model is not None and not isinstance(model, MachineModel):
        raise TypeError(f'Expected MachineModel for model, got {type(model)}.')

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

    # Check `max_synthesis_size`
    if not is_integer(max_synthesis_size):
        raise TypeError(
            'Expected integer for max_synthesis_size'
            f', got{type(max_synthesis_size)}.',
        )

    if max_synthesis_size <= 1:
        raise ValueError(
            'The maximum synthesis size must be greater than or equal to 2.'
            f' Got {max_synthesis_size}.',
        )

    if model is not None:
        if max_synthesis_size < max(g.num_qudits for g in model.gate_set):
            value = max(g.num_qudits for g in model.gate_set)
            raise ValueError(
                'The maximum synthesis size needs to be greater than or equal'
                ' to the largest native gate size: '
                f'{max_synthesis_size} < {value}.',
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

    # Check `compiler`
    if compiler is not None and not isinstance(compiler, Compiler):
        raise TypeError(
            'Expected Compiler object for compiler parameter'
            f', got {type(compiler)}.',
        )

    # Check `seed`
    if seed is not None and not is_integer(seed):
        raise TypeError(f'Expected integer for seed, got {type(seed)}.')

    # check `with_mapping`
    if not isinstance(with_mapping, bool):
        raise TypeError(
            f'Expected bool for with_mapping, got {type(with_mapping)}.',
        )

    def type_and_check_input(input: CompilationInputLike) -> CompilationInput:
        """Check input to be valid and convert to a proper typed object."""
        try:
            if isinstance(input, Circuit):
                pass

            elif UnitaryMatrix.is_unitary(input):
                input = UnitaryMatrix(input)

            elif StateVector.is_pure_state(input):
                input = StateVector(input)

            elif StateSystem.is_state_system(input):
                input = StateSystem(input)

            else:
                raise TypeError(
                    'Input is neither a circuit, a unitary, a state system'
                    f', nor a state. Got {type(input)}.',
                )
        except Exception as e:
            raise TypeError(
                'Unable to determine type of input.'
                ' Ensure that you are trying to compile a valid'
                ' circuit, unitary, state, or state-system.',
            ) from e

        assert isinstance(
            input, (
                Circuit,
                UnitaryMatrix,
                StateVector,
                StateSystem,
            ),
        )

        if not all(r == input.radixes[0] for r in input.radixes):
            raise ValueError(
                'Currently, can only automatically build a workflow '
                'for same-level systems, such as qubit-only or qutrit-only'
                'systems. mixed-radix systems are not yet supported'
                'with the standard workflows.',
            )

        if model is not None:
            if model.num_qudits < input.num_qudits:
                raise ValueError('Machine is too small for this input.')

            if not all(r == input.radixes[0] for r in model.radixes):
                raise ValueError('Model has a mismatch in radixes with input.')

            if len(model.gate_set.multi_qudit_gates) == 0:
                if input.num_qudits > 1:
                    raise ValueError('No entangling gates in native gate set.')

            if all(g.num_qudits > input.num_qudits for g in model.gate_set):
                raise ValueError(
                    'Model gate set does not contain any entangling gates'
                    ' that is less than or equal to input size.'
                    f' Cannot compile {input.num_qudits}-qudit input without'
                    f' {input.num_qudits}-qudit or smaller gates.',
                )

        return input

    # Connect to or construct a Compiler
    managed_compiler = compiler is None

    if managed_compiler:
        compiler = Compiler(**compiler_kwargs)

    assert compiler is not None

    if (
        is_iterable(input)
        and not isinstance(input, Circuit)
        and not UnitaryMatrix.is_unitary(input)
        and not StateVector.is_pure_state(input)
        and not StateSystem.is_state_system(input)
    ):
        typed_inputs = [type_and_check_input(i) for i in input]  # type: ignore

        if len(typed_inputs) == 0:
            raise ValueError('Input iterable is empty.')

        # Build workflows
        workflows = [
            build_workflow(
                typed_input,
                model,
                optimization_level,
                synthesis_epsilon,
                max_synthesis_size,
                error_threshold,
                error_sim_size,
                seed,
            )
            for typed_input in typed_inputs
        ]
        in_circuits = [
            typed_input if isinstance(typed_input, Circuit) else Circuit(1)
            for typed_input in typed_inputs
        ]

        # Perform the compilations in parallel
        job_ids = [
            compiler.submit(in_circuit, workflow, True)
            for in_circuit, workflow in zip(in_circuits, workflows)
        ]
        results = [compiler.result(job_id) for job_id in job_ids]

        outs: list[Circuit] = []
        datas: list[PassData] = []
        for result in results:
            out, data = result
            outs.append(out)  # type: ignore
            datas.append(data)  # type: ignore

        # Log error if necessary
        if error_threshold is not None:
            for i, data in enumerate(datas):
                error = data.error
                nonsq_error = 1 - np.sqrt(max(1 - (error * error), 0))
                if nonsq_error > error_threshold:
                    warnings.warn(
                        'Upper bound on error is greater than set threshold:'
                        f' {nonsq_error} > {error_threshold}'
                        f' in compilation index {i}.',
                    )

        # Gather mapping data if necessary
        if with_mapping:
            pis = []
            pfs = []
            for typed_input, data in zip(typed_inputs, datas):
                default = list(range(typed_input.num_qudits))
                pi = data.get('initial_mapping', default)
                pf = data.get('final_mapping', default)
                pis.append(pi)
                pfs.append(pf)
            to_return: Any = list(zip(outs, pis, pfs))

        else:
            to_return = outs

    else:
        typed_input = type_and_check_input(input)  # type: ignore

        # Build workflow
        workflow = build_workflow(
            typed_input,
            model,
            optimization_level,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            seed,
        )
        if isinstance(typed_input, Circuit):
            in_circuit = typed_input

        else:
            in_circuit = Circuit(1)

        # Perform the compilation
        out, data = compiler.compile(in_circuit, workflow, True)

        # Log error if necessary
        if error_threshold is not None:
            error = data.error
            nonsq_error = 1 - np.sqrt(max(1 - (error * error), 0))
            if nonsq_error > error_threshold:
                warnings.warn(
                    'Upper bound on error is greater than set threshold:'
                    f' {nonsq_error} > {error_threshold}.',
                )

        # Gather mapping data if necessary
        if with_mapping:
            default = list(range(typed_input.num_qudits))
            pi = data.get('initial_mapping', default)
            pf = data.get('final_mapping', default)
            to_return = (out, pi, pf)

        else:
            to_return = out

    # Close managed compiler
    if managed_compiler:
        compiler.close()

    return to_return


def build_workflow(
    input: Circuit | UnitaryMatrix | StateVector | StateSystem,
    model: MachineModel | None = None,
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build a BQSKit Off-the-Shelf workflow, see :func:`compile` for info."""
    if model is None:
        model = MachineModel(input.num_qudits, radixes=input.radixes)

    # Use a registered workflow if model is found in the registry for a given
    # optimization_level
    for machine_or_gateset in _workflow_registry:
        if isinstance(machine_or_gateset, GateSet):
            gate_set = machine_or_gateset
        else:
            gate_set = machine_or_gateset.gate_set
        gs_match = gate_set == model.gate_set
        ol_found = optimization_level in _workflow_registry[machine_or_gateset]
        if gs_match and ol_found:
            return _workflow_registry[machine_or_gateset][optimization_level]

    if isinstance(input, Circuit):
        if input.num_qudits > max_synthesis_size:
            if any(
                g.num_qudits > max_synthesis_size
                and not isinstance(
                    g, (
                        MeasurementPlaceholder,
                        BarrierPlaceholder,
                    ),
                )
                for g in input.gate_set
            ):
                raise ValueError(
                    'Unable to compile circuit with gate larger than'
                    ' max_synthesis_size.\nConsider adjusting it.',
                )

        return _circuit_workflow(
            model,
            optimization_level,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            seed,
        )

    elif isinstance(input, UnitaryMatrix):
        if input.num_qudits > max_synthesis_size:
            raise ValueError(
                'Unable to compile unitary with size larger than'
                ' max_synthesis_size.\nConsider adjusting it.',
            )

        return _synthesis_workflow(
            input,
            model,
            optimization_level,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            seed,
        )

    elif isinstance(input, StateVector):
        if input.num_qudits > max_synthesis_size:
            raise ValueError(
                'Unable to compile states with size larger than'
                ' max_synthesis_size.\nConsider adjusting it.',
            )

        return _stateprep_workflow(
            input,
            model,
            optimization_level,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            seed,
        )

    elif isinstance(input, StateSystem):
        if input.num_qudits > max_synthesis_size:
            raise ValueError(
                'Unable to compile state systems with size larger than'
                ' max_synthesis_size.\nConsider adjusting it.',
            )

        return _statemap_workflow(
            input,
            model,
            optimization_level,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            seed,
        )

    raise TypeError(f'Unexpected input type: {type(input)}.')


def _circuit_workflow(
    model: MachineModel,
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build standard workflow for circuit compilation."""
    workflow_builders = [
        _opt1_workflow,
        _opt2_workflow,
        _opt3_workflow,
        _opt4_workflow,
    ]
    workflow_builder = workflow_builders[optimization_level - 1]
    workflow: list[BasePass] = [] if seed is None else [SetRandomSeedPass(seed)]
    workflow += [UnfoldPass(), ExtractMeasurements()]
    workflow += workflow_builder(
        model,
        synthesis_epsilon,
        max_synthesis_size,
        error_threshold,
        error_sim_size,
    )
    workflow += [RestoreMeasurements()]
    return Workflow(workflow, name='Off-the-Shelf Circuit Compilation')


def get_instantiate_options(optimization_level: int) -> dict[str, Any]:
    """
    Return good default instantiate options based on an optimization level.

    Args:
        optimization_level (int): The optimization level.

    Returns:
        (dict[str, Any]): The options.

    Raises:
        ValueError: If the optimization level is not 1, 2, 3, or 4.
    """
    if optimization_level == 1:
        return {
            'multistarts': 1,
            'ftol': 1e-6,
            'gtol': 1e-10,
            'diff_tol_r': 1e-4,
            'max_iters': 1000,
            'min_iters': 0,
        }

    elif optimization_level == 2:
        return {
            'multistarts': 2,
            'ftol': 5e-12,
            'gtol': 1e-14,
            'diff_tol_r': 1e-5,
            'max_iters': 10000,
            'min_iters': 100,
        }

    elif optimization_level == 3:
        return {
            'multistarts': 4,
            'ftol': 5e-16,
            'gtol': 1e-15,
            'diff_tol_r': 5e-5,
            'max_iters': 50000,
            'min_iters': 200,
        }

    elif optimization_level == 4:
        return {
            'multistarts': 8,
            'ftol': 5e-16,
            'gtol': 1e-15,
            'diff_tol_r': 1e-6,
            'max_iters': 100000,
            'min_iters': 1000,
        }

    else:
        raise ValueError(
            'Invalid optimization level, must be 1, 2, 3, or 4.'
            f' Got {optimization_level}.',
        )


def build_partitioning_workflow(
    inner_workflow: WorkflowLike,
    block_size: int = 3,
    error_sim_size: int | None = None,
    replace_filter_method: str = 'less-than-respecting-multi',
) -> Workflow:
    """
    Build standard partitioning workflow for circuit compilation.

    This includes additional steps than normally necessary for error upper
    bound analysis.

    Args:
        inner_workflow (WorkflowLike): The workflow to partition.

        block_size (int): The size of the blocks to partition into.

        error_sim_size (int | None): The size of the blocks to simulate
            errors on. If None, then no error analysis is performed.

        replace_filter_method (str): The method to use for replacing
            gates in the circuit. See :class:`ForEachBlockPass` for more.

    Returns:
        Workflow: The partitioning workflow.

    Raises:
        ValueError: If block_size < 2.

        ValueError: If error_sim_size < block_size.
    """
    if not is_integer(block_size):
        raise TypeError(
            f'Expected block_size to be int, got {type(block_size)}.',
        )

    if block_size < 2:
        raise ValueError(f'Expected block_size > 1, got {block_size}.')

    if error_sim_size is not None and not is_integer(block_size):
        raise TypeError(
            f'Expected int for error_sim_size, got {type(error_sim_size)}.',
        )

    if error_sim_size is not None and error_sim_size < block_size:
        raise ValueError(
            f'Expected error_sim_size >= block_size, got {error_sim_size}.',
        )

    pass_list = [QuickPartitioner(block_size), ExtendBlockSizePass()]

    if error_sim_size is None:
        pass_list += [
            ForEachBlockPass(
                inner_workflow,
                replace_filter=replace_filter_method,
            ),
        ]

    elif error_sim_size == block_size:
        pass_list += [
            ForEachBlockPass(
                inner_workflow,
                replace_filter=replace_filter_method,
                calculate_error_bound=True,
            ),
        ]

    else:
        pass_list += [
            QuickPartitioner(error_sim_size),
            ForEachBlockPass(
                ForEachBlockPass(
                    inner_workflow,
                    replace_filter=replace_filter_method,
                ),
                calculate_error_bound=True,
            ),
        ]

    pass_list += [UnfoldPass()]
    return Workflow(pass_list, name='Partitioning')


def build_standard_search_synthesis_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
) -> BasePass:
    """
    Build standard search-based synthesis pass for block-level compilation.

    Args:
        optimization_level (int): The optimization level. See :func:`compile`
            for more information.

        synthesis_epsilon (float): The maximum distance between target
            and circuit unitary allowed to declare successful synthesis.
            Set to 0 for exact synthesis. (Default: 1e-8)

    Returns:
        BasePass: The synthesis pass.

    Raises:
        ValueError: If the optimization level is not 1, 2, 3, or 4.

    Note:
        For larger circuits, this pass may take a very long time to run.
        If unitary synthesis is your ultimate goal, rather than circuit
        compilation, consider designing a custom instantiation-based
        synthesis method or using alternative synthesis techniques
        -- such as QFAST or QPredict -- for large unitaries.
    """
    if optimization_level not in [1, 2, 3, 4]:
        raise ValueError(
            'Invalid optimization level, must be 1, 2, 3, or 4.'
            f' Got {optimization_level}.',
        )

    if not is_real_number(synthesis_epsilon):
        raise TypeError(
            'Expected float for synthesis_epsilon'
            f', got {type(synthesis_epsilon)}.',
        )

    qsearch = QSearchSynthesisPass(
        success_threshold=synthesis_epsilon,
        instantiate_options=get_instantiate_options(optimization_level),
    )
    leap = LEAPSynthesisPass(
        success_threshold=synthesis_epsilon,
        min_prefix_size=[3, 4, 7, 9][optimization_level - 1],
        instantiate_options=get_instantiate_options(optimization_level),
    )
    return IfThenElsePass(WidthPredicate(3), qsearch, leap)


def build_multi_qudit_retarget_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> Workflow:
    """
    Build standard workflow for circuit multi-qudit gate set retargeting.

    Notes:
        - This workflow assumes that SetModelPass will be run earlier in the
          full workflow and doesn't add it in here.

        - For the most part, circuit connectivity isn't a concern during
          retargeting. However, if the circuit contains many-qudit (>= 3)
          gates, then the workflow will not preserve connectivity during
          the decomposition of those gates. If your input contains many-qudit
          gates, consider following this with a mapping workflow.
    """

    core_retarget_workflow = [
        FillSingleQuditGatesPass(),
        IfThenElsePass(
            NotPredicate(MultiPhysicalPredicate()),
            IfThenElsePass(
                ManyQuditGatesPredicate(),
                [
                    ExtractModelConnectivityPass(),
                    build_standard_search_synthesis_workflow(
                        optimization_level,
                        synthesis_epsilon,
                    ),
                    RestoreModelConnectivityPass(),
                ],
                AutoRebase2QuditGatePass(3, 5, synthesis_epsilon),
            ),
            ScanningGateRemovalPass(
                success_threshold=synthesis_epsilon,
                collection_filter=_mq_gate_collection_filter,
                instantiate_options=get_instantiate_options(optimization_level),
            ),
        ),
    ]

    return Workflow(
        [
            IfThenElsePass(
                NotPredicate(WidthPredicate(2)),
                [
                    LogPass('Retargeting multi-qudit gates.'),
                    build_partitioning_workflow(
                        core_retarget_workflow,
                        max_synthesis_size,
                        None if error_threshold is None else error_sim_size,
                    ),
                ],
            ),
        ], name='Multi Qudit Retargeting',
    )


def build_single_qudit_retarget_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> BasePass:
    """Build a pass to convert single-qudit-gates to the native gate set."""
    sq_synthesis = QSearchSynthesisPass(
        layer_generator=SingleQuditLayerGenerator(None, allow_repeats=True),
        heuristic_function=DijkstraHeuristic(),
        instantiate_options={
            'method': 'minimization',
            'minimizer': ScipyMinimizer(),
            'cost_fn_gen': HilbertSchmidtCostGenerator(),
        },
    )

    sq_gate_deletion = build_partitioning_workflow(
        ScanningGateRemovalPass(
            success_threshold=synthesis_epsilon,
            collection_filter=_sq_gate_collection_filter,
            instantiate_options=get_instantiate_options(optimization_level),
        ),
        max_synthesis_size,
        None if error_threshold is None else error_sim_size,
    )

    return Workflow(
        [
            IfThenElsePass(
                NotPredicate(SinglePhysicalPredicate()),
                [
                    LogPass('Retargeting single-qudit gates.'),
                    UnfoldPass(),
                    GroupSingleQuditGatePass(),
                    IfThenElsePass(
                        AllConstantSingleQuditGates(),
                        # TODO: Implement better retargeting techniques
                        # for constant gate sets
                        LogPass(
                            'The standard workflow with BQSKit may have trouble'
                            ' targeting gate sets containing no parameterized'
                            ' single-qudit gates.',
                            logging.WARNING,
                        ),
                    ),
                    IfThenElsePass(
                        NoSingleQuditGatesInModel(),
                        [
                            LogPass('Attempting to remove single-qudit gates.'),
                            sq_gate_deletion,
                            IfThenElsePass(
                                NotPredicate(SinglePhysicalPredicate()),
                                LogPass(
                                    'Unable to remove all single-qudit gates;'
                                    ' gate set may not be universal without'
                                    ' single-qudit gates. Consider changing'
                                    ' gate set, increasing optimization_level'
                                    ' or max_synthesis_size.',
                                    logging.WARNING,
                                ),
                            ),
                        ],
                        [
                            ForEachBlockPass(
                                IfThenElsePass(
                                    NotPredicate(SinglePhysicalPredicate()),
                                    [
                                        IfThenElsePass(
                                            HasGeneralSingleQuditGate(),
                                            GeneralSQDecomposition(),
                                            IfThenElsePass(
                                                ZXGatePredicate(),
                                                ZXZXZDecomposition(),
                                                sq_synthesis,
                                            ),
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                    UnfoldPass(),
                ],
            ),
        ], name='Single Qudit Retargeting',
    )


def build_sabre_mapping_workflow() -> Workflow:
    """
    Build standard workflow for circuit mapping.

    Note:
        - This workflow assumes that SetModelPass will be run earlier in the
          full workflow and doesn't add it in here.

        - It also assumes that ApplyPlacement will be run later in the full
          workflow and doesn't add it in here.
    """
    return Workflow(
        [
            LogPass('Mapping circuit.'),
            GreedyPlacementPass(),
            GeneralizedSabreLayoutPass(),
            GeneralizedSabreRoutingPass(),
        ], name='SABRE Mapping',
    )


def build_seqpam_mapping_optimization_workflow(
    optimization_level: int = 4,
    synthesis_epsilon: float = 1e-8,
    num_layout_passes: int = 3,
    block_size: int = 3,
    error_sim_size: int | None = None,
) -> Workflow:
    """
    Build a Sequential-Permutation-Aware Mapping and Optimizing Workflow.

    Note:
        - This workflow assumes that SetModelPass will be run earlier in the
          full workflow and doesn't add it in here.

        - This will apply the placement found during the workflow. The
        resulting circuit will be physically mapped.

    Args:
        optimization_level (int): The optimization level. See :func:`compile`
            for more information.

        synthesis_epsilon (float): The maximum distance between target
            and circuit unitary allowed to declare successful synthesis.
            Set to 0 for exact synthesis. (Default: 1e-8)

        num_layout_passes (int): The number of layout forward and backward
            passes to run. See :class:`PamLayoutPass` for more information.
            (Default: 3)

        block_size (int): The size of the blocks to partition into.
            Warning, the number of permutation evaluated increases
            factorially and the difficulty of each permutation increases
            exponentially with this. (Default: 3)

        error_sim_size (int | None): The size of the blocks to simulate
            errors on. If None, then no error analysis is performed.
            (Default: None)

    Raises:
        ValueError: If block_size < 2.

        ValueError: If error_sim_size < block_size.
    """
    if not is_integer(block_size):
        raise TypeError(
            f'Expected block_size to be int, got {type(block_size)}.',
        )

    if block_size < 2:
        raise ValueError(f'Expected block_size > 1, got {block_size}.')

    if error_sim_size is not None and not is_integer(block_size):
        raise TypeError(
            f'Expected int for error_sim_size, got {type(error_sim_size)}.',
        )

    if error_sim_size is not None and error_sim_size < block_size:
        raise ValueError(
            f'Expected error_sim_size >= block_size, got {error_sim_size}.',
        )

    qsearch = QSearchSynthesisPass(
        success_threshold=synthesis_epsilon,
        instantiate_options=get_instantiate_options(optimization_level),
    )

    leap = LEAPSynthesisPass(
        success_threshold=synthesis_epsilon,
        min_prefix_size=9,
        instantiate_options=get_instantiate_options(optimization_level),
    )

    if error_sim_size is not None:
        post_pam_seq: BasePass = PAMVerificationSequence(error_sim_size)
    else:
        post_pam_seq = NOOPPass()

    return Workflow(
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass('Caching permutation-aware synthesis results.'),
                ExtractModelConnectivityPass(),
                QuickPartitioner(block_size),
                ForEachBlockPass(
                    IfThenElsePass(
                        WidthPredicate(4),
                        EmbedAllPermutationsPass(
                            inner_synthesis=qsearch,
                            input_perm=True,
                            output_perm=False,
                            vary_topology=False,
                        ),
                        EmbedAllPermutationsPass(
                            inner_synthesis=leap,
                            input_perm=True,
                            output_perm=False,
                            vary_topology=False,
                        ),
                    ),
                ),
                LogPass('Preoptimizing with permutation-aware mapping.'),
                PAMRoutingPass(),
                post_pam_seq,
                UnfoldPass(),
                RestoreModelConnectivityPass(),

                LogPass('Recaching permutation-aware synthesis results.'),
                SubtopologySelectionPass(block_size),
                QuickPartitioner(block_size),
                ForEachBlockPass(
                    IfThenElsePass(
                        WidthPredicate(4),
                        EmbedAllPermutationsPass(
                            inner_synthesis=qsearch,
                            input_perm=False,
                            output_perm=True,
                            vary_topology=True,
                        ),
                        EmbedAllPermutationsPass(
                            inner_synthesis=leap,
                            input_perm=False,
                            output_perm=True,
                            vary_topology=True,
                        ),
                    ),
                ),
                LogPass('Performing permutation-aware mapping.'),
                ApplyPlacement(),
                PAMLayoutPass(num_layout_passes),
                PAMRoutingPass(0.1),
                post_pam_seq,
                ApplyPlacement(),
                UnfoldPass(),
            ],
        ),
        name='SeqPAM Mapping',
    )


def build_gate_deletion_optimization_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    iterative: bool = False,
) -> Workflow:
    """Build standard workflow for circuit gate deletion optimization."""
    core_workflow = Workflow(
        [
            LogPass('Attempting to delete gates.'),
            build_partitioning_workflow(
                ScanningGateRemovalPass(
                    success_threshold=synthesis_epsilon,
                    instantiate_options=get_instantiate_options(
                        optimization_level,
                    ),
                ),
                max_synthesis_size,
                None if error_threshold is None else error_sim_size,
            ),
        ],
        name='Gate Deletion Optimization',
    )

    if iterative:
        return Workflow(
            WhileLoopPass(ChangePredicate(), core_workflow),
            name='Iterative Gate Deletion Optimization',
        )

    return core_workflow


def build_resynthesis_optimization_workflow(
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    iterative: bool = False,
) -> Workflow:
    """Build standard workflow for circuit resynthesis optimization."""
    core_workflow = Workflow(
        [
            LogPass('Resynthesizing blocks.'),
            build_partitioning_workflow(
                IfThenElsePass(
                    NotPredicate(WidthPredicate(2)),
                    build_standard_search_synthesis_workflow(
                        optimization_level,
                        synthesis_epsilon,
                    ),
                ),
                max_synthesis_size,
                None if error_threshold is None else error_sim_size,
            ),
        ], name='Resynthesis Optimization',
    )

    if iterative:
        return Workflow(
            [
                WhileLoopPass(
                    GateCountPredicate('multi'),
                    core_workflow,
                ),
            ], name='Iterative Resynthesis Optimization',
        )

    return core_workflow


def _opt1_workflow(
    model: MachineModel,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build Optimization Level 1 workflow for circuit compilation."""
    return [
        # Initializing
        SetModelPass(model),

        build_multi_qudit_retarget_workflow(
            1,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_sabre_mapping_workflow(),

        build_multi_qudit_retarget_workflow(
            1,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_single_qudit_retarget_workflow(
            1,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        # Finalizing
        LogErrorPass(),
        ApplyPlacement(),
    ]


def _opt2_workflow(
    model: MachineModel,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build Optimization Level 2 workflow for circuit compilation."""
    return [
        # Initializing
        SetModelPass(model),

        build_multi_qudit_retarget_workflow(
            2,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_sabre_mapping_workflow(),


        build_multi_qudit_retarget_workflow(
            2,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_single_qudit_retarget_workflow(
            2,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_gate_deletion_optimization_workflow(
            2,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        # Finalizing
        LogErrorPass(),
        ApplyPlacement(),
    ]


def _opt3_workflow(
    model: MachineModel,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build Optimization Level 3 workflow for circuit compilation."""
    return [
        # Initializing
        SetModelPass(model),

        build_multi_qudit_retarget_workflow(
            3,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_sabre_mapping_workflow(),


        build_multi_qudit_retarget_workflow(
            3,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_resynthesis_optimization_workflow(
            3,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            True,
        ),

        build_single_qudit_retarget_workflow(
            3,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_gate_deletion_optimization_workflow(
            3,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            True,
        ),

        # Finalizing
        LogErrorPass(),
        ApplyPlacement(),
    ]


def _opt4_workflow(
    model: MachineModel,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
) -> list[BasePass]:
    """Build optimization Level 4 workflow for circuit compilation."""
    if max_synthesis_size > 3:
        _logger.warning(
            'It is currently recommended to set max_synthesis_size to 3'
            ' for optimization level 4. This may change in the future.',
        )

    return [
        SetModelPass(model),

        build_seqpam_mapping_optimization_workflow(
            4,
            synthesis_epsilon,
            block_size=max_synthesis_size,
            error_sim_size=None if error_threshold is None else error_sim_size,
        ),

        build_multi_qudit_retarget_workflow(
            4,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_resynthesis_optimization_workflow(
            4,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            True,
        ),

        build_single_qudit_retarget_workflow(
            4,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),

        build_gate_deletion_optimization_workflow(
            4,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
            True,
        ),

        # Finalizing
        LogErrorPass(),
    ]


def _synthesis_workflow(
    input: UnitaryMatrix,
    model: MachineModel,
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build a workflow for unitary synthesis."""
    if error_threshold is not None:
        if error_threshold < synthesis_epsilon:
            raise ValueError(
                'When performing direct synthesis, the error threshold'
                ' cannot be less than the synthesis epsilon.',
            )

    sq_synthesis = QSearchSynthesisPass(
        layer_generator=SingleQuditLayerGenerator(None, allow_repeats=True),
        heuristic_function=DijkstraHeuristic(),
        instantiate_options={
            'method': 'minimization',
            'minimizer': ScipyMinimizer(),
            'cost_fn_gen': HilbertSchmidtCostGenerator(),
        } if input.radixes == (2,) else {},
    )

    qsearch = QSearchSynthesisPass(
        success_threshold=synthesis_epsilon,
        instantiate_options=get_instantiate_options(optimization_level),
    )

    leap = LEAPSynthesisPass(
        success_threshold=synthesis_epsilon,
        min_prefix_size=[3, 4, 7, 9][optimization_level - 1],
        instantiate_options=get_instantiate_options(optimization_level),
    )

    if optimization_level < 4:
        synthesis: BasePass = IfThenElsePass(WidthPredicate(3), qsearch, leap)

    else:
        in_synthesis = qsearch if input.num_qudits <= 3 else leap
        synthesis = PermutationAwareSynthesisPass(inner_synthesis=in_synthesis)

    if input.num_qudits == 1:
        synthesis = sq_synthesis

    scan = ScanningGateRemovalPass(
        success_threshold=synthesis_epsilon,
        instantiate_options=get_instantiate_options(optimization_level),
    )

    workflow: list[BasePass] = [] if seed is None else [SetRandomSeedPass(seed)]
    workflow += [
        SetModelPass(model),
        SetTargetPass(input),
        synthesis,
        build_single_qudit_retarget_workflow(
            optimization_level,
            synthesis_epsilon,
            max_synthesis_size,
            error_threshold,
            error_sim_size,
        ),
        scan if optimization_level >= 2 else NOOPPass(),
    ]

    return Workflow(workflow, name='Off-the-Shelf Unitary Synthesis')


def _stateprep_workflow(
    state: StateVector,
    model: MachineModel,
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build a workflow for state preparation."""
    layer_gen = model.gate_set.build_mq_layer_generator()

    if optimization_level == 1:
        inst_ops = {
            'multistarts': 1,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer(),
        }
        synthesis: SynthesisPass = LEAPSynthesisPass(
            success_threshold=synthesis_epsilon,
            layer_generator=layer_gen,
            instantiate_options=inst_ops,
            min_prefix_size=3,
            cost=HilbertSchmidtCostGenerator(),
        )

    elif optimization_level == 2:
        inst_ops = {
            'multistarts': 4,
            'method': 'minimization',
            'ftol': 5e-12,
            'gtol': 1e-14,
        }
        synthesis = LEAPSynthesisPass(
            success_threshold=synthesis_epsilon,
            layer_generator=layer_gen,
            instantiate_options=inst_ops,
            min_prefix_size=5,
        )

    elif optimization_level == 3:
        inst_ops = {
            'multistarts': 8,
            'method': 'minimization',
            'ftol': 5e-16,
            'gtol': 1e-15,
        }
        if state.num_qudits > 3:
            synthesis = LEAPSynthesisPass(
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
                min_prefix_size=7,
            )
        else:
            synthesis = QSearchSynthesisPass(
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
            )

    elif optimization_level == 4:
        inst_ops = {
            'multistarts': 8,
            'method': 'minimization',
            'ftol': 5e-16,
            'gtol': 1e-15,
        }
        if state.num_qudits > 3:
            in_synthesis = LEAPSynthesisPass(
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
                min_prefix_size=7,
            )
        else:
            in_synthesis = QSearchSynthesisPass(  # type: ignore
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
            )
        synthesis = PermutationAwareSynthesisPass(inner_synthesis=in_synthesis)

    scan = ScanningGateRemovalPass(
        success_threshold=synthesis_epsilon,
        instantiate_options=inst_ops,
        cost=HilbertSchmidtCostGenerator(),
    )

    workflow: list[BasePass] = [] if seed is None else [SetRandomSeedPass(seed)]
    workflow += [
        SetModelPass(model),
        SetTargetPass(state),
        synthesis,
        scan if optimization_level >= 2 else NOOPPass(),
    ]

    return Workflow(workflow, name='Off-the-Shelf State Synthesis')


def _statemap_workflow(
    state: StateSystem,
    model: MachineModel,
    optimization_level: int = 1,
    synthesis_epsilon: float = 1e-8,
    max_synthesis_size: int = 3,
    error_threshold: float | None = None,
    error_sim_size: int = 8,
    seed: int | None = None,
) -> Workflow:
    """Build a workflow for state preparation."""
    layer_gen = model.gate_set.build_mq_layer_generator()

    if optimization_level == 1:
        inst_ops = {
            'multistarts': 1,
            'method': 'minimization',
        }
        synthesis: SynthesisPass = LEAPSynthesisPass(
            success_threshold=synthesis_epsilon,
            layer_generator=layer_gen,
            instantiate_options=inst_ops,
            min_prefix_size=3,
        )

    elif optimization_level == 2:
        inst_ops = {
            'multistarts': 4,
            'method': 'minimization',
            'ftol': 5e-12,
            'gtol': 1e-14,
        }
        synthesis = LEAPSynthesisPass(
            success_threshold=synthesis_epsilon,
            layer_generator=layer_gen,
            instantiate_options=inst_ops,
            min_prefix_size=5,
        )

    elif optimization_level == 3:
        inst_ops = {
            'multistarts': 8,
            'method': 'minimization',
            'ftol': 5e-16,
            'gtol': 1e-15,
        }
        if state.num_qudits > 3:
            synthesis = LEAPSynthesisPass(
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
                min_prefix_size=7,
            )
        else:
            synthesis = QSearchSynthesisPass(
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
            )

    elif optimization_level == 4:
        inst_ops = {
            'multistarts': 8,
            'method': 'minimization',
            'ftol': 5e-16,
            'gtol': 1e-15,
        }
        if state.num_qudits > 3:
            in_synthesis = LEAPSynthesisPass(
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
                min_prefix_size=7,
            )
        else:
            in_synthesis = QSearchSynthesisPass(  # type: ignore
                success_threshold=synthesis_epsilon,
                layer_generator=layer_gen,
                instantiate_options=inst_ops,
            )
        synthesis = PermutationAwareSynthesisPass(inner_synthesis=in_synthesis)

    scan = ScanningGateRemovalPass(
        success_threshold=synthesis_epsilon,
        instantiate_options=inst_ops,
    )

    workflow: list[BasePass] = [] if seed is None else [SetRandomSeedPass(seed)]
    workflow += [
        SetModelPass(model),
        SetTargetPass(state),
        synthesis,
        scan if optimization_level >= 2 else NOOPPass(),
    ]

    return Workflow(workflow, name='Off-the-Shelf State System Synthesis')


def _get_single_qudit_gate_rebase_pass(model: MachineModel) -> BasePass:
    """Build a pass to convert single-qudit-gates to the native gate set."""
    warnings.warn(
        'This single qudit rebase workflow has moved'
        ' to build_single_qudit_retarget_workflow. See Workflow for more'
        ' information. This function is deprecated and will be removed'
        ' in a future release.',
        DeprecationWarning,
    )

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


def _gen_replace_filter(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter for use during the standard workflow."""
    warnings.warn(
        'The replace filter generated by _gen_replace_filter has moved'
        ' to gen_replace_filter in foreach. See ForEachBlockPass for more'
        ' information. This function is deprecated and will be removed'
        ' in a future release.',
        DeprecationWarning,
    )
    return gen_replace_filter('less-then-respecting-multi', model)


def _mq_gate_collection_filter(op: Operation) -> bool:
    """Return true if `op` is a multi-qudit operation."""
    return op.num_qudits > 1


def _sq_gate_collection_filter(op: Operation) -> bool:
    """Return true if `op` is a multi-qudit operation."""
    return op.num_qudits == 1
