"""This module implements the various intermediate results classes."""
from __future__ import annotations

import logging
import pickle
from os import listdir
from os.path import exists, join, isdir
import shutil
from re import findall
from typing import cast, Sequence
from pathlib import Path

from copy import deepcopy

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.workflow import Workflow
from bqskit.passes.alias import PassAlias
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.ir.operation import Operation
from bqskit.passes.util.converttou3 import ToU3Pass
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


def contains_subdirectory_with_os_listdir(directory):
    for item in listdir(directory):
        item_path = join(directory, item)
        if isdir(item_path):
            return True
    return False

class SaveIntermediatePass(BasePass):
    """
    The SaveIntermediate class.

    The SaveIntermediatePass stores individual CircuitGates in pickle or qasm
    format.
    """

    def __init__(
        self,
        project_dir: str,
        save_as_qasm: bool = True,
        overwrite: bool = False
    ) -> None:
        """
        Constructor for the SaveIntermediatePass.

        Args:
            path_to_save_dir (str): Path to the directory in which inter-
                qasm for circuit blocks should be saved.

            project_name (str): Name of the project files.

        Raises:
            ValueError: If `path_to_save_dir` is not an existing directory.
        """
        if exists(project_dir):
            if not overwrite:
                _logger.error("Directory already exists!")
                return
            self.pathdir = project_dir
            if self.pathdir[-1] != '/':
                self.pathdir += '/'
        else:
            Path(project_dir).mkdir(parents=True, exist_ok=True)
            _logger.warning(
                f'Path {project_dir} does not exist',
            )
            self.pathdir = project_dir

        # mkdir(join(self.pathdir,self.projname))
        self.as_qasm = save_as_qasm

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        # Gather and enumerate CircuitGates to save
        blocks_to_save: list[tuple[int, Operation]] = []
        for enum, op in enumerate(circuit):
            if isinstance(op.gate, CircuitGate):
                blocks_to_save.append((enum, op))

        # Set up path and file names
        structure_file = join(self.pathdir, 'structure.pickle')
        num_digits = len(str(len(blocks_to_save)))

        structure_list: list[list[int]] = []
        # NOTE: Block numbers are gotten by iterating through the circuit so
        # there is no guarantee that the blocks were partitioned in that order.
        for enum, block in blocks_to_save:
            enum = str(enum).zfill(num_digits)  # type: ignore
            structure_list.append(block.location)  # type: ignore
            subcircuit = Circuit(block.num_qudits)
            subcircuit.append_gate(
                block.gate,
                list(
                    range(
                        block.num_qudits,
                    ),
                ),
                block.params,
            )
            subcircuit.unfold((0, 0))
            if self.as_qasm:
                await ToU3Pass().run(subcircuit, PassData(subcircuit))
                with open(join(self.pathdir, f'block_{enum}.qasm'), 'w') as f:
                    f.write(OPENQASM2Language().encode(subcircuit))
            else:
                with open(
                    join(self.pathdir, f'block_{enum}.pickle'), 'wb',
                ) as f:
                    pickle.dump(subcircuit, f)

        with open(structure_file, 'wb') as f:
            pickle.dump(structure_list, f)


class RestoreIntermediatePass(BasePass):
    def __init__(self, project_directory: str, load_blocks: bool = True, as_circuit_gate: bool = False):
        """
        Constructor for the RestoreIntermediatePass.

        Args:
            project_directory (str): Path to the checkpoint block files. This
                directory must also contain a valid "structure.pickle" file.

            load_blocks (bool): If True, blocks in the project directory will
                be loaded to the block_list during the constructor. Otherwise
                the user must explicitly call load_blocks() themselves. Defaults
                to True.

            as_circuit_gate (bool): If True, blocks are reloaded as a circuit 
            gate rather than a circuit.

        Raises:
            ValueError: If `project_directory` does not exist or if
                `structure.pickle` is invalid.
        """
        self.proj_dir = project_directory
        # self.block_list: list[str] = []
        self.as_circuit_gate = as_circuit_gate

        self.load_blocks = load_blocks

    def reload_blocks(proj_dir: str, structure: list) -> tuple[list[str], bool]:
        """
        Updates the `block_list` variable with the current contents of the
        `proj_dir`.
    
        Raises:
            ValueError: if there are more block files than indices in the
            `structure.pickle`.
        """
        files = sorted(listdir(proj_dir))
        # Files are of the form block_*.pickle or block_*.qasm
        block_list = [f for f in files if 'block_' in f]
        pickle_list = [f for f in block_list if ".pickle" in f]
        saved_as_qasm = False
        if len(pickle_list) == 0:
            saved_as_qasm = True
            block_list = [f for f in block_list if ".qasm" in f]
        else:
            block_list = pickle_list
        if len(block_list) > len(structure):
            raise ValueError(
                f'More block files ({len(block_list), len(pickle_list)}) than indices ({len(structure)}) in `{proj_dir}/structure.pickle` {block_list}',
            )
        return block_list, saved_as_qasm

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Perform the pass's operation, see BasePass for more info.

        Raises:
            ValueError: if a block file and the corresponding index in
                `structure.pickle` are differnt lengths.
        """

        if not exists(self.proj_dir):
            raise TypeError(
                f"Project directory '{self.proj_dir}' does not exist.",
            )
        if not exists(self.proj_dir + '/structure.pickle'):
            raise TypeError(
                f'Project directory `{self.proj_dir}` does not '
                'contain `structure.pickle`.',
            )

        with open(self.proj_dir + '/structure.pickle', 'rb') as f:
            structure = pickle.load(f)

        if not isinstance(structure, list):
            raise TypeError('The provided `structure.pickle` is not a list.')
        
        block_list, saved_as_qasm = RestoreIntermediatePass.reload_blocks(self.proj_dir, structure)

        # Get circuit from checkpoint, ignore previous circuit
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        for block in block_list:
            # Get block
            block_num = int(findall(r'\d+', block)[0])
            if saved_as_qasm:
                with open(join(self.proj_dir, block)) as f:
                    block_circ = OPENQASM2Language().decode(f.read())
            else:
                with open(join(self.proj_dir, block), "rb") as f:
                    block_circ = pickle.load(f)
            # Get location
            block_location = structure[block_num]
            if block_circ.num_qudits != len(block_location):
                raise ValueError(
                    f'{block} and `structure.pickle` locations are '
                    'different sizes.',
                )
            # Append to circuit
            try:
                new_circuit.append_circuit(block_circ, block_location, as_circuit_gate=self.as_circuit_gate)
            except Exception as e:
                print(self.proj_dir)
                raise e
        
        circuit.become(new_circuit)

class CheckpointRestartPass(BasePass):
    '''
    This pass is used to reload a checkpointed circuit. Checkpoints are useful
    to restart a workflow from a certain point in the event of a crash or 
    timeout.
    '''
    def __init__(self, checkpoint_dir: str, 
                 default_passes: BasePass | Sequence[BasePass]) -> None:
        """ 
        Args:
            checkpoint_dir (str): 
                Path to the directory containing the checkpointed circuit.
            default_passes (BasePass | Sequence[BasePass]): 
                The passes to run if the checkpoint does not exist. Typically,
                these will be the partitioning passes to set up the block 
                structure.
        """
        if not is_sequence(default_passes):
            default_passes = [cast(BasePass, default_passes)]

        if not isinstance(default_passes, list):
            default_passes = list(default_passes)

        self.checkpoint_dir = checkpoint_dir
        self.default_passes = default_passes
    
    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Set's the `checkpoint_dir` attribute and restores the circuit from the
        checkpoint if possible. If the checkpoint does not exist, the default
        passes are run.
        """
        # block_id = data.get("block_num", "0")
        data["checkpoint_dir"] = self.checkpoint_dir
        if not exists(join(self.checkpoint_dir, "circuit.pickle")):
            _logger.info("Checkpoint does not exist!")
            await Workflow(self.default_passes).run(circuit, data)
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            pickle.dump(circuit, open(join(self.checkpoint_dir, "circuit.pickle"), "wb"))
            pickle.dump(data, open(join(self.checkpoint_dir, "data.pickle"), "wb"))
        else:
            # Already checkpointed, restore
            _logger.info("Restoring from Checkpoint!")
            new_circuit = pickle.load(open(join(self.checkpoint_dir, "circuit.pickle"), "rb"))
            circuit.become(new_circuit)
            new_data = pickle.load(open(join(self.checkpoint_dir, "data.pickle"), "rb"))
            data.update(new_data)
