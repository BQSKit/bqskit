"""
The _workflow_registry enables MachineModel or GateSet specific workflows to be
registered for used in the `bqskit.compile` method.

The _workflow_registry maps MachineModels a dictionary of Workflows which
are indexed by optimization level. This object should not be accessed directly
by the user, but instead through the `register_workflow` function.

Example:
    model_t = SpecificMachineModel(num_qudits, radixes)
    workflow = [QuickPartitioner(3), NewFangledOptimization()]
    register_workflow(model_t, workflow, level)
    ...
    new_circuit = compile(circuit, model_t, optimization_level=level)
"""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.gateset import GateSet
from bqskit.compiler.gateset import GateSetLike
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike
from bqskit.ir.gate import Gate

_logger = logging.getLogger(__name__)


_workflow_registry: dict[MachineModel | GateSet, dict[int, Workflow]] = {}


def register_workflow(
    machine_or_gateset: MachineModel | GateSetLike,
    workflow: WorkflowLike,
    optimization_level: int = 1,
) -> None:
    """
    Register a workflow for a given machine model.

    Args:
        machine_or_gateset (MachineModel | GateSetLike): A MachineModel or
            GateSetLike to register the workflow under. If a circuit is
            compiled targeting this machine or gate set, the registered
            workflow will be used.

        workflow (list[BasePass]): The workflow or list of passes that whill
            be executed if the MachineModel in a call to `compile` matches
            `machine`. If `machine` is already registered, a warning will be
            logged.

        optimization_level (Optional[int]): The optimization level with
            which to register the workflow. If no level is provided, the
            Workflow will be registered as level 1. (Default: 1)

    Raises:
        TypeError: If `machine_or_gateset` is not a MachineModel or GateSet.

        TypeError: If `workflow` is not a list of BasePass objects.
    """
    if not isinstance(machine_or_gateset, MachineModel):
        if isinstance(machine_or_gateset, Gate):
            machine_or_gateset = [machine_or_gateset]
        if all(isinstance(g, Gate) for g in machine_or_gateset):
            machine_or_gateset = GateSet(machine_or_gateset)
        else:
            m = '`machine_or_gateset` must be a MachineModel or '
            m += f'GateSet, got {type(machine_or_gateset)}.'
            raise TypeError(m)

    workflow = Workflow(workflow)

    for p in workflow:
        if not isinstance(p, BasePass):
            m = 'All elements of `workflow` must be BasePass objects. Got '
            m += f'{type(p)}.'
            raise TypeError(m)

    global _workflow_registry
    new_workflow = {optimization_level: workflow}
    if machine_or_gateset in _workflow_registry:
        if optimization_level in _workflow_registry[machine_or_gateset]:
            m = f'Overwritting workflow for {machine_or_gateset} '
            m += f'at level {optimization_level}.'
            _logger.warn(m)
        _workflow_registry[machine_or_gateset].update(new_workflow)
    else:
        _workflow_registry[machine_or_gateset] = new_workflow


def clear_registry() -> None:
    """
    Clear the workflow registry.

    This will remove all registered workflows from the registry.
    """
    global _workflow_registry
    _workflow_registry.clear()
