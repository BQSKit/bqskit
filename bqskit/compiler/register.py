"""
The workflow_registry enables MachineModel specific workflows to be registered
for used in the `bqskit.compile` method.

The workflow_registry maps MachineModels a dictionary of Workflows which
are indexed by optimization level. This object should not be accessed directly
by the user, but instead through the `register_workflow` function.

Examples:
    model_t = SpecificMachineModel(num_qudits, radixes)
    workflow = [QuickPartitioner(3), NewFangledOptimization()]
    register_workflow(model_t, workflow, level)
    ...
    new_circuit = compile(circuit, model_t, optimization_level=level)
"""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.machine import MachineModel
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike


_logger = logging.getLogger(__name__)


workflow_registry: dict[MachineModel, dict[int, Workflow]] = {}


def register_workflow(
    machine: MachineModel,
    workflow: WorkflowLike,
    optimization_level: int = 1,
) -> None:
    """
    Register a workflow for a given machine model.

    Args:
        machine (MachineModel): The machine to register the workflow for.

        workflow (list[BasePass]): The workflow or list of passes that whill
            be executed if the MachineModel in a call to `compile` matches
            `machine`. If `machine` is already registered, a warning will be
            logged.

        optimization_level (Optional[int]): The optimization level with
            which to register the workflow. If no level is provided, the
            Workflow will be registered as level 1. (Default: 1)

    Raises:
        TypeError: If `machine` is not a MachineModel.

        TypeError: If `workflow` is not a list of BasePass objects.
    """
    if not isinstance(machine, MachineModel):
        m = f'`machine` must be a MachineModel, got {type(machine)}.'
        raise TypeError(m)

    workflow = Workflow(workflow)

    for p in workflow:
        if not isinstance(p, BasePass):
            m = 'All elements of `workflow` must be BasePass objects. Got '
            m += f'{type(p)}.'
            raise TypeError(m)

    global workflow_registry

    if machine in workflow_registry:
        if optimization_level in workflow_registry[machine]:
            m = f'Overwritting workflow for {machine} at level '
            m += f'{optimization_level}.'
            _logger.warn(m)
        workflow_registry[machine].update({optimization_level: workflow})
    else:
        workflow_registry[machine] = {optimization_level: workflow}
