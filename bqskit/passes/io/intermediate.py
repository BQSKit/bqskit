"""This module implements the various intermediate results classes."""
from __future__ import annotations

import logging
import pickle
from os import listdir, mkdir
from os.path import exists, join
import shutil
from re import findall
from typing import cast, Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.passes.alias import PassAlias
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.ir.operation import Operation
from bqskit.passes.util.converttou3 import ToU3Pass
from bqskit.utils.typing import is_sequence

_logger = logging.getLogger(__name__)


class SaveIntermediatePass(BasePass):
    """
    The SaveIntermediate class.

    The SaveIntermediatePass stores individual CircuitGates in pickle or qasm
    format.
    """

    def __init__(
        self,
        path_to_save_dir: str,
        project_name: str | None = None,
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
        if exists(path_to_save_dir):
            self.pathdir = path_to_save_dir
            if self.pathdir[-1] != '/':
                self.pathdir += '/'
        else:
            raise ValueError(
                f'Path {path_to_save_dir} does not exist',
            )
        self.projname = project_name if project_name is not None \
            else 'unnamed_project'

        enum = 1
        if exists(join(self.pathdir,self.projname)):
            if overwrite:
                shutil.rmtree(join(self.pathdir,self.projname))
            else:
                while exists(join(self.pathdir, self.projname + f'_{enum}')):
                    enum += 1
                self.projname += f'_{enum}'
                _logger.warning(
                    f'Path {path_to_save_dir} already exists, '
                    f'saving to {self.pathdir + self.projname} '
                    'instead.',
                )

        mkdir(self.pathdir + self.projname)

        self.as_qasm = save_as_qasm

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        # Gather and enumerate CircuitGates to save
        blocks_to_save: list[tuple[int, Operation]] = []
        for enum, op in enumerate(circuit):
            if isinstance(op.gate, CircuitGate):
                blocks_to_save.append((enum, op))

        # Set up path and file names
        structure_file = self.pathdir + self.projname + '/structure.pickle'
        block_skeleton = self.pathdir + self.projname + '/block_'
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
            await ToU3Pass().run(subcircuit, PassData(subcircuit))
            if self.as_qasm:
                with open(block_skeleton + f'{enum}.qasm', 'w') as f:
                    f.write(OPENQASM2Language().encode(subcircuit))
            else:
                with open(
                    f'{block_skeleton}{enum}.pickle', 'wb',
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
        self.block_list: list[str] = []
        self.as_circuit_gate = as_circuit_gate
        # We will detect automatically if blocks are saved as qasm or pickle
        self.saved_as_qasm = False

        self.load_blocks = load_blocks

    def reload_blocks(self) -> None:
        """
        Updates the `block_list` variable with the current contents of the
        `proj_dir`.

        Raises:
            ValueError: if there are more block files than indices in the
            `structure.pickle`.
        """
        files = sorted(listdir(self.proj_dir))
        # Files are of the form block_*.pickle or block_*.qasm
        self.block_list = [f for f in files if 'block_' in f]
        pickle_list = [f for f in self.block_list if ".pickle" in f]
        if len(pickle_list) == 0:
            self.saved_as_qasm = True
            self.block_list = [f for f in self.block_list if ".qasm" in f]
        else:
            self.block_list = pickle_list
        if len(self.block_list) > len(self.structure):
            raise ValueError(
                'More block files than indices in `structure.pickle`',
            )

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
            self.structure = pickle.load(f)

        if not isinstance(self.structure, list):
            raise TypeError('The provided `structure.pickle` is not a list.')
        
        if self.load_blocks:
            self.reload_blocks()

        # Get circuit from checkpoint, ignore previous circuit
        new_circuit = Circuit(circuit.num_qudits, circuit.radixes)
        for block in self.block_list:
            # Get block
            block_num = int(findall(r'\d+', block)[0])
            if self.saved_as_qasm:
                with open(join(self.proj_dir, block)) as f:
                    block_circ = OPENQASM2Language().decode(f.read())
            else:
                with open(join(self.proj_dir, block), "rb") as f:
                    block_circ = pickle.load(f)
            # Get location
            block_location = self.structure[block_num]
            if block_circ.num_qudits != len(block_location):
                raise ValueError(
                    f'{block} and `structure.pickle` locations are '
                    'different sizes.',
                )
            # Append to circuit
            new_circuit.append_circuit(block_circ, block_location, as_circuit_gate=self.as_circuit_gate)
        
        circuit.become(new_circuit)
        # Check if the circuit has been partitioned, if so, try to replace
        # blocks

class CheckpointRestartPass(PassAlias):
    def __init__(self, base_checkpoint_dir: str, 
                 project_name: str,
                 default_passes: BasePass | Sequence[BasePass],
                 save_as_qasm: bool = True) -> None:
        """Group together one or more `passes`."""
        if not is_sequence(default_passes):
            default_passes = [cast(BasePass, default_passes)]

        if not isinstance(default_passes, list):
            default_passes = list(default_passes)

        full_checkpoint_dir = join(base_checkpoint_dir, project_name)
        
        # Check if checkpoint files exist
        if not exists(join(full_checkpoint_dir, "structure.pickle")):
            _logger.debug("Checkpoint does not exist!")
            save_pass = SaveIntermediatePass(base_checkpoint_dir, project_name, 
                                             save_as_qasm=save_as_qasm, overwrite=True)
            default_passes.append(save_pass)
            self.passes = default_passes
        else:
            # Already checkpointed, restore
            _logger.debug("Restoring from Checkpoint!")
            self.passes = [
                RestoreIntermediatePass(full_checkpoint_dir, as_circuit_gate=True)
            ]

    def get_passes(self) -> list[BasePass]:
        """Return the passes to be run, see :class:`PassAlias` for more."""
        return self.passes