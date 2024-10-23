"""This module implements the ForEachBlockPass class."""
from __future__ import annotations

import functools
import logging
from typing import Callable, List
import pickle
from os.path import join, exists
from pathlib import Path

# from bqskit.compiler.basepass import _sub_do_work
from collections import Counter
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.runtime import get_runtime

_logger = logging.getLogger(__name__)


class ForEachBlockPass(BasePass):
    """
    A pass that executes other passes on each block in the circuit.

    This is a control pass that executes a workflow on every block in the
    circuit. This will be done in parallel.
    """

    key = 'ForEachBlockPass_data'
    """The key in data, where block data will be put."""

    pass_down_key_prefix = 'ForEachBlockPass_pass_down_'
    """If a key exists in the pass data with this prefix, pass it to blocks."""

    pass_down_block_specific_key_prefix = (
        'ForEachBlockPass_specific_pass_down_'
    )
    """
    Data specific to the processing of individual blocks in a partitioned
    circuit can be injected into the `PassData` in `run` by using this prefix.

    The expected type of the associated value is `dict[int, Any]`, where
    integer (sub-)keys correspond to block numbers in a partitioned quantum
    circuit.

    Pseudocode example for seed circuits:
        seeds = {block_id: [seed_circuit_a, seed_circuit_b, ...], ...}
        key = self.pass_down_block_specific_key_prefix + 'seed_circuits'
        seed_updater = UpdateDataPass(key, seeds)
        workflow = Workflow([..., seed_updater, ForEachBlockPass(...), ...])
    """

    def __init__(
        self,
        loop_body: WorkflowLike,
        calculate_error_bound: bool = False,
        error_cost_gen: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        collection_filter: Callable[[Operation], bool] | None = None,
        replace_filter: ReplaceFilterFn | str = 'always',
        batch_size: int | None = None,
        blocks_to_run: List[int] = [],
        allocate_error: bool = False,
        allocate_error_gate: Gate = CNOTGate(),
        allocate_skew_factor: int = 3
    ) -> None:
        """
        Construct a ForEachBlockPass.

        Args:
            loop_body (WorkflowLike): The workflow to execute on every block.

            calculate_error_bound (bool): If set to true, will calculate
                errors on blocks after running `loop_body` on them and
                use these block errors to calculate an upper bound on the
                full circuit error. (Default: False)

            collection_filter (Callable[[Operation], bool] | None):
                A predicate that determines which operations should have
                `loop_body` called on them. Called with each operation
                in the circuit. If this returns true, that operation will
                be formed into an individual circuit and passed through
                `loop_body`. Defaults to all CircuitGates,
                ConstantUnitaryGates, and VariableUnitaryGates.
                #TODO: address importability

            replace_filter (ReplaceFilterFn | str | None):
                A predicate that determines if the resulting circuit, after
                calling `loop_body` on a block, should replace the original
                operation. Called with the circuit output from `loop_body`
                and the original operation. If this returns true, the
                operation will be replaced with the new circuit.
                Defaults to always replace. If none is passed, will
                generate a replace filter always replaces. If a string is
                passed, will generate a replace filter corresponding to
                the string. The string should either be 'always', 'less-than',
                'less-than-multi', 'less-than-many', 'less-than-respecting',
                'less-than-respecting-multi', or 'less-than-respecting-many'.
                    - 'always' will always replace
                    - 'less-than' will replace if the new circuit has fewer
                        gates than the old circuit.
                    - 'less-than-multi' will replace if the new circuit has
                        fewer multi-qudit gates than the old circuit.
                    - 'less-than-many' will replace if the new circuit has
                        fewer many-qudit gates than the old circuit.
                    - 'less-than-respecting' will replace if the new circuit
                        has fewer gates than the old circuit or the old
                        doesn't respect the model (ignoring single-qudit
                        gate sets).
                    - 'less-than-respecting-multi' will replace if the new
                        circuit has fewer multi-qudit gates than the old
                        circuit or the old doesn't respect the model
                        (ignoring single-qudit gate sets).
                    - 'less-than-respecting-many' will replace if the new
                        circuit has fewer many-qudit gates than the old
                        circuit or the old doesn't respect the model
                        (ignoring single-qudit gate sets).
                    - 'less-than-respecting-fully' will replace if the new
                        circuit has fewer gates than the old circuit or
                        the old doesn't respect the model.
                    - 'less-than-respecting-fully-multi' will replace if
                        the new circuit has fewer multi-qudit gates than
                        the old circuit or the old doesn't respect the model.
                    - 'less-than-respecting-fully-many' will replace if
                        the new circuit has fewer many-qudit gates than
                        the old circuit or the old doesn't respect the model.
                Defaults to 'always'.  #TODO: address importability

            batch_size (int): (Deprecated).

            blocks_to_run (List[int]):
                A list of blocks to run the ForEachBlockPass body on. By default
                you run on all blocks. This is mainly used with checkpointing, 
                where some blocks have already finished while others have not.
        """
        if batch_size is not None:
            import warnings
            warnings.warn(
                'Batch size is no longer supported, this warning will'
                ' become an error in a future update.',
                DeprecationWarning,
            )

        self.calculate_error_bound = calculate_error_bound
        self.collection_filter = collection_filter or default_collection_filter
        self.replace_filter = replace_filter or default_replace_filter
        self.workflow = Workflow(loop_body)
        self.blocks_to_run = sorted(blocks_to_run)
        self.allocate_error = allocate_error
        self.allocate_error_gate = allocate_error_gate
        self.allocate_skew_factor = allocate_skew_factor
        self.error_cost_gen = error_cost_gen
        if not callable(self.collection_filter):
            raise TypeError(
                'Expected callable method that maps Operations to booleans for'
                f' collection_filter, got {type(self.collection_filter)}.',
            )

        if not isinstance(self.replace_filter, str):
            if not callable(self.replace_filter):
                raise TypeError(
                    'Expected either string representing a valid replacement'
                    ' filter or callable method that maps Circuit and'
                    ' Operations to bools for replace_filter'
                    f' , got {type(self.replace_filter)}.',
                )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # Get the callable replacement filter
        if isinstance(self.replace_filter, str):
            method = self.replace_filter
            replace_filter = gen_replace_filter(method, data.model)
        else:
            replace_filter = self.replace_filter

        should_checkpoint = 'checkpoint_dir' in data
        checkpoint_dir = data.get('checkpoint_dir', "")

        # Make room in data for block data
        if self.key not in data:
            data[self.key] = []

        # Collect blocks
        blocks: list[tuple[int, Operation]] = []
        if (len(self.blocks_to_run) == 0):
            # TODO: This is buggy, need to fix to work with collection filter
            self.blocks_to_run = list(range(circuit.num_operations))

        block_ids = self.blocks_to_run.copy()
        next_id = block_ids.pop(0)
        for i, (cycle, op) in enumerate(circuit.operations_with_cycles()):
            if self.collection_filter(op) and i == next_id:
                blocks.append((cycle, op))
                try:
                    next_id = block_ids.pop(0)
                except IndexError:
                    # No more blocks to run on
                    break

        # No blocks, no work
        if len(blocks) == 0:
            data[self.key].append([])
            return
        
        # print("NUMBER OF BLOCKS", len(blocks))
        # Get the machine model
        model = data.model
        coupling_graph = data.connectivity

        # Preprocess blocks
        subcircuits: list[Circuit] = []
        block_datas: list[PassData] = []
        block_gates = []
        for i, (cycle, op) in enumerate(blocks):
            # Check if checkpoint exists:
            # Need to zero pad block ids for consistency
            num_digits = len(str(circuit.num_operations))
            block_num = str(self.blocks_to_run[i]).zfill(num_digits)
            save_data_file = join(checkpoint_dir, f'block_{block_num}.data')
            save_circuit_file = join(checkpoint_dir, f'block_{block_num}.pickle')
            checkpoint_found = False
            if should_checkpoint and exists(save_data_file):
                _logger.debug(f'Loading block {i} from checkpoint.')
                try:
                    subcircuit = pickle.load(open(save_circuit_file, 'rb'))
                    block_data = pickle.load(open(save_data_file, 'rb'))
                    checkpoint_found = True
                except Exception as e:
                    print(f"Exception for file: {save_data_file}", e)
                    checkpoint_found = False
            
            if not checkpoint_found:
                # Form Subcircuit
                if isinstance(op.gate, CircuitGate):
                    subcircuit = op.gate._circuit.copy()
                    subcircuit.set_params(op.params)
                else:
                    subcircuit = Circuit.from_operation(op)

                # Form Submodel
                subradixes = [circuit.radixes[q] for q in op.location]
                subnumbering = {op.location[i]: i for i in range(len(op.location))}
                submodel = MachineModel(
                    len(op.location),
                    coupling_graph.get_subgraph(op.location, subnumbering),
                    model.gate_set,
                    subradixes,
                )

                # We are cubing here so that blocks with more CNOT gates 
                # are given more error budget

                # Form Subdata
                block_data: PassData = PassData(subcircuit)
                block_data['subnumbering'] = subnumbering
                block_data['model'] = submodel
                block_data['point'] = CircuitPoint(cycle, op.location[0])
                block_data['calculate_error_bound'] = self.calculate_error_bound
                block_data['block_num'] = block_num
                for key in data:
                    if key.startswith(self.pass_down_key_prefix):
                        block_data[key] = data[key]
                    elif key.startswith(
                        self.pass_down_block_specific_key_prefix,
                    ) and i in data[key]:
                        block_data[key] = data[key][i]
                block_data.seed = data.seed

                if should_checkpoint:
                    # Create checkpoint directory if it doesn't exist
                    # Dump initial subcircuit and block in it
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    pickle.dump(block_data, open(save_data_file, 'wb'))
                    pickle.dump(subcircuit, open(save_circuit_file, 'wb'))

            # Change next subdirectory
            if should_checkpoint:
                # Update checkpoint dir, circ file, and data file
                block_data["checkpoint_dir"] = join(checkpoint_dir, f'block_{block_num}')
                block_data["checkpoint_circ_file"] = save_circuit_file
                block_data["checkpoint_data_file"] = save_data_file
            subcircuits.append(subcircuit)
            block_datas.append(block_data)
            # TODO: This is expensive, need to find a better way to do this
            unfolded_circ = subcircuit.copy()
            unfolded_circ.unfold_all()
            skewed_gates = unfolded_circ.count(self.allocate_error_gate) ** self.allocate_skew_factor
            block_gates.append(skewed_gates)

        # Assign error as percentage of block
        total_gates = sum(block_gates)
        if self.allocate_error:
            for i in range(len(block_datas)):
                block_datas[i]["error_percentage_allocated"] = block_gates[i] * data.get("error_percentage_allocated", 1) / total_gates 

        # Do the work
        results = await get_runtime().map(
            _sub_do_work,
            [self.workflow] * len(subcircuits),
            subcircuits,
            block_datas,
            cost=self.error_cost_gen,
        )

        # Unpack results
        completed_subcircuits, completed_block_datas = zip(*results)

        # Postprocess blocks
        points: list[CircuitPoint] = []
        ops: list[Operation] = []
        error_sum = 0.0
        for i, (cycle, op) in enumerate(blocks):
            subcircuit = completed_subcircuits[i]
            block_data = completed_block_datas[i]

            # Mark Blocks to be Replaced
            if replace_filter(subcircuit, op):
                _logger.debug(f'Replacing block {i}.')
                points.append(CircuitPoint(cycle, op.location[0]))
                ops.append(
                    Operation(
                        CircuitGate(subcircuit, True),
                        op.location,
                        subcircuit.params,
                    ),
                )
                block_data['replaced'] = True

                # Calculate Error
                error_sum += block_data.error
            else:
                block_data['replaced'] = False

        # Replace blocks
        circuit.batch_replace(points, ops)

        # Record block data into pass data
        data[self.key].append(completed_block_datas)

        # Record error
        data.update_error_mul(error_sum)
        if self.calculate_error_bound:
            _logger.debug(f'New circuit error is {data.error}.')


async def _sub_do_work(
    workflow: Workflow,
    circuit: Circuit,
    data: PassData,
    cost: CostFunctionGenerator,
) -> tuple[Circuit, PassData]:
    """Execute a sequence of passes on circuit."""
    if 'calculate_error_bound' in data and data['calculate_error_bound']:
        old_utry = circuit.get_unitary()

    await workflow.run(circuit, data)

    if 'calculate_error_bound' in data and data['calculate_error_bound']:
        data.error = cost.calc_cost(circuit, old_utry)

    return circuit, data

def default_collection_filter(op: Operation) -> bool:
    return isinstance(
        op.gate, (
            CircuitGate,
            ConstantUnitaryGate,
            VariableUnitaryGate,
            PauliGate,
        ),
    )


def default_replace_filter(circuit: Circuit, op: Operation) -> bool:
    """Always replace."""
    # legacy name and style for backwards compatibility
    return True


def _less_than(new: Circuit, old: Operation) -> bool:
    """Return true if the new circuit has fewer gates."""
    if isinstance(old.gate, CircuitGate):
        return new.num_operations < old.gate._circuit.num_operations

    return True  # TODO: Re-evaluate always true when old is not a circuit


def _less_than_multi(new: Circuit, old: Operation) -> bool:
    """Return true if the new circuit has fewer multi-qudit gates."""
    if isinstance(old.gate, CircuitGate):
        org = old.gate._circuit
        omq = sum([c for g, c in org.gate_counts.items() if g.num_qudits > 1])
        osq = sum([c for g, c in org.gate_counts.items() if g.num_qudits == 1])
        nmq = sum([c for g, c in new.gate_counts.items() if g.num_qudits > 1])
        nsq = sum([c for g, c in new.gate_counts.items() if g.num_qudits == 1])
        return (nmq, nsq) < (omq, osq)

    return True


def _less_than_many(new: Circuit, old: Operation) -> bool:
    """Return true if the new circuit has fewer many-qudit gates."""
    if isinstance(old.gate, CircuitGate):
        org = old.gate._circuit
        omq = sum([c for g, c in org.gate_counts.items() if g.num_qudits > 2])
        otq = sum([c for g, c in org.gate_counts.items() if g.num_qudits == 2])
        osq = sum([c for g, c in org.gate_counts.items() if g.num_qudits == 1])
        nmq = sum([c for g, c in new.gate_counts.items() if g.num_qudits > 2])
        ntq = sum([c for g, c in new.gate_counts.items() if g.num_qudits == 2])
        nsq = sum([c for g, c in new.gate_counts.items() if g.num_qudits == 1])
        return (nmq, ntq, nsq) < (omq, otq, osq)

    return True


def _is_respecting(
    circuit: Circuit,
    location: CircuitLocation,
    model: MachineModel,
    fully: bool = False,
) -> bool:
    """
    Return true if the `circuit` respects the `model` at `location`.

    Args:
        circuit (Circuit): The circuit to check.

        location (CircuitLocation): The location to check.

        model (MachineModel): The machine model to check against.

        fully (bool): If set to true, will check if the circuit respects
            the model fully. If set to false, will ignore single-qudit
            gate sets. (Default: False)

    Returns:
        True if the circuit respects the model at the location. This implies
        that the circuit can be run on the machine at the location.
    """
    org_mq_gates = circuit.gate_set.multi_qudit_gates
    org_sq_gates = circuit.gate_set.single_qudit_gates

    if any(g not in model.gate_set for g in org_mq_gates):
        return False

    if fully and any(g not in model.gate_set for g in org_sq_gates):
        return False

    if any(
        (location[e[0]], location[e[1]]) not in model.coupling_graph
        for e in circuit.coupling_graph
    ):
        return False

    return True


def _less_than_fn_respecting(
    new: Circuit,
    old: Operation,
    model: MachineModel,
    fn: ReplaceFilterFn,
) -> bool:
    """Return true if the new circuit has fewer gates or the old doesn't respect
    the model."""
    if isinstance(old.gate, CircuitGate):
        if not _is_respecting(old.gate._circuit, old.location, model):
            if not _is_respecting(new, old.location, model):
                _logger.debug("New block doesn't respect model.")
            return True

        if not _is_respecting(new, old.location, model):
            _logger.debug("New block doesn't respect model.")
            return False

    return fn(new, old)


def _less_than_fn_respecting_fully(
    new: Circuit,
    old: Operation,
    model: MachineModel,
    fn: ReplaceFilterFn,
) -> bool:
    """Return true if the new circuit has fewer gates or the old doesn't respect
    the model."""
    if isinstance(old.gate, CircuitGate):
        if not _is_respecting(old.gate._circuit, old.location, model, True):
            if not _is_respecting(new, old.location, model, True):
                _logger.debug("New block doesn't respect model.")
            return True

        if not _is_respecting(new, old.location, model, True):
            _logger.debug("New block doesn't respect model.")
            return False

    return fn(new, old)


def gen_always(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that always replaces."""
    # legacy name and style for backwards compatibility
    return default_replace_filter


def gen_less_than(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    gates."""
    return _less_than


def gen_less_than_multi(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    multi-qudit gates."""
    return _less_than_multi


def gen_less_than_many(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    many-qudit gates."""
    return _less_than_many


def gen_less_than_rspt(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting,
        model=model,
        fn=_less_than,
    )


def gen_less_than_rspt_multi(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    multi-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting,
        model=model,
        fn=_less_than_multi,
    )


def gen_less_than_rspt_many(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    many-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting,
        model=model,
        fn=_less_than_many,
    )


def gen_less_than_rspt_fully(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting_fully,
        model=model,
        fn=_less_than,
    )


def gen_less_than_rspt_fully_multi(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    multi-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting_fully,
        model=model,
        fn=_less_than_multi,
    )


def gen_less_than_rspt_fully_many(model: MachineModel) -> ReplaceFilterFn:
    """Generate a replace filter that replaces if the new circuit has fewer
    many-qudit gates or the old doesn't respect the model."""
    return functools.partial(
        _less_than_fn_respecting_fully,
        model=model,
        fn=_less_than_many,
    )


def gen_replace_filter(method: str, model: MachineModel) -> ReplaceFilterFn:
    """
    Generate a replace filter for use during the standard workflow.

    Args:
        method (str): The method to use for the replace filter. See
            :class:`ForEachBlockPass` for more information.

        model (MachineModel): The machine model to potentially respect.

    Returns:
        A replace filter function.
    """
    replace_filters = {
        'always': gen_always,
        'less-than': gen_less_than,
        'less-than-multi': gen_less_than_multi,
        'less-than-many': gen_less_than_many,
        'less-than-respecting': gen_less_than_rspt,
        'less-than-respecting-multi': gen_less_than_rspt_multi,
        'less-than-respecting-many': gen_less_than_rspt_many,
        'less-than-respecting-fully': gen_less_than_rspt_fully,
        'less-than-respecting-fully-multi': gen_less_than_rspt_fully_multi,
        'less-than-respecting-fully-many': gen_less_than_rspt_fully_many,
    }

    if method not in replace_filters:
        raise ValueError(f'Unknown replace filter method {method}.')

    return replace_filters[method](model)


ReplaceFilterFn = Callable[[Circuit, Operation], bool]


class ClearAllBlockData(BasePass):
    """Clear all block data and passed down data from the pass data."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        for key in list(data.keys()):
            if key.startswith(ForEachBlockPass.key):
                del data[key]
            elif key.startswith(ForEachBlockPass.pass_down_key_prefix):
                del data[key]
