"""Register MachineModel specific default workflows."""
from __future__ import annotations

import warnings

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.workflow import Workflow
from bqskit.compiler.workflow import WorkflowLike


_compile_circuit_registry: dict[MachineModel, dict[int, Workflow]] = {}
_compile_unitary_registry: dict[MachineModel, dict[int, Workflow]] = {}
_compile_stateprep_registry: dict[MachineModel, dict[int, Workflow]] = {}
_compile_statemap_registry: dict[MachineModel, dict[int, Workflow]] = {}


def model_registered_target_types(key: MachineModel) -> list[str]:
    """
    Return a list of target_types for which key is registered.

    Args:
        key (MachineModel): A MachineModel to check for.

    Returns:
        (list[str]): If `key` has been registered in any of the registry, the
            name of that target type will be contained in this list.
    """
    global _compile_circuit_registry
    global _compile_unitary_registry
    global _compile_stateprep_registry
    global _compile_statemap_registry
    registered_types = []
    if key in _compile_circuit_registry:
        registered_types.append('circuit')
    if key in _compile_unitary_registry:
        registered_types.append('unitary')
    if key in _compile_stateprep_registry:
        registered_types.append('stateprep')
    if key in _compile_statemap_registry:
        registered_types.append('statemap')
    return registered_types


def register_workflow(
    key: MachineModel,
    workflow: WorkflowLike,
    optimization_level: int,
    target_type: str = 'circuit',
) -> None:
    """
    Register a workflow for a given MachineModel.

    The _compile_registry enables MachineModel specific workflows to be
    registered for use in the `bqskit.compile` method. _compile_registry maps
    MachineModels a dictionary of Workflows which are indexed by optimization
    level. This object should not be accessed directly by the user, but
    instead through the `register_workflow` function.

    Args:
        key (MachineModel): A MachineModel to register the workflow under.
            If a circuit is compiled targeting this machine or gate set, the
            registered workflow will be used.

        workflow (list[BasePass]): The workflow or list of passes that will
            be executed if the MachineModel in a call to `compile` matches
            `key`. If `key` is already registered, a warning will be logged.

        optimization_level (int): The optimization level with which to
            register the workflow.

        target_type (str): Register a workflow for targets of this type. Must
            be 'circuit', 'unitary', 'stateprep', or 'statemap'.
            (Default: 'circuit')

    Example:
        model_t = SpecificMachineModel(num_qudits, radixes)
        workflow = [QuickPartitioner(3), NewFangledOptimization()]
        register_workflow(model_t, workflow, level, 'circuit')
        ...
        new_circuit = compile(circuit, model_t, optimization_level=level)

    Raises:
        Warning: If a workflow for a given optimization_level is overwritten.

        ValueError: If `target_type` is not 'circuit', 'unitary', 'stateprep',
            or 'statemap'.
    """
    if target_type not in ['circuit', 'unitary', 'stateprep', 'statemap']:
        m = 'target_type must be "circuit", "unitary", "stateprep", or '
        m += f'"statemap", got {target_type}.'
        raise ValueError(m)

    if target_type == 'circuit':
        global _compile_circuit_registry
        _compile_registry = _compile_circuit_registry
    elif target_type == 'unitary':
        global _compile_unitary_registry
        _compile_registry = _compile_unitary_registry
    elif target_type == 'stateprep':
        global _compile_stateprep_registry
        _compile_registry = _compile_stateprep_registry
    else:
        global _compile_statemap_registry
        _compile_registry = _compile_statemap_registry

    workflow = Workflow(workflow)

    new_workflow = {optimization_level: workflow}
    if key in _compile_registry:
        if optimization_level in _compile_registry[key]:
            m = f'Overwritting workflow for {key} at level '
            m += f'{optimization_level} for target type {target_type}.'
            m += 'If multiple Namespace packages are installed, ensure'
            m += 'that their __init__.py files do not attempt to'
            m += 'overwrite the same default Workflows.'
            warnings.warn(m)
        _compile_registry[key].update(new_workflow)
    else:
        _compile_registry[key] = new_workflow
