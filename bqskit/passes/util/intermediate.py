"""This module implements the various intermediate results classes."""
from __future__ import annotations

import logging
import pickle
from os import listdir
from os import mkdir
from os.path import exists
from re import findall
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language
from bqskit.ir.operation import Operation
from bqskit.passes.util.converttou3 import ToU3Pass

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
        if exists(self.pathdir + self.projname):
            while exists(self.pathdir + self.projname + f'_{enum}'):
                enum += 1
            self.projname += f'_{enum}'
            _logger.warning(
                f'Path {path_to_save_dir} already exists, '
                f'saving to {self.pathdir + self.projname} '
                'instead.',
            )

        mkdir(self.pathdir + self.projname)

        self.as_qasm = save_as_qasm

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
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
            ToU3Pass().run(subcircuit, {})
            if self.as_qasm:
                with open(block_skeleton + f'{enum}.qasm', 'w') as f:
                    f.write(OPENQASM2Language().encode(subcircuit))
            else:
                with open(  # type: ignore
                    f'{block_skeleton}{enum}.pickle', 'wb',
                ) as f:
                    pickle.dump(subcircuit, f)  # type: ignore

        with open(structure_file, 'wb') as f:  # type: ignore
            pickle.dump(structure_list, f)  # type: ignore


class RestoreIntermediatePass(BasePass):
    def __init__(self, project_directory: str, load_blocks: bool = True):
        """
        Constructor for the RestoreIntermediatePass.

        Args:
            project_directory (str): Path to the checkpoint block files. This
                directory must also contain a valid "structure.pickle" file.

            load_blocks (bool): If True, blocks in the project directory will
                be loaded to the block_list during the constructor. Otherwise
                the user must explicitly call load_blocks() themselves. Defaults
                to True.

        Raises:
            ValueError: If `project_directory` does not exist or if
                `structure.pickle` is invalid.
        """
        self.proj_dir = project_directory
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

        self.block_list: list[str] = []
        if load_blocks:
            self.reload_blocks()

    def reload_blocks(self) -> None:
        """
        Updates the `block_list` variable with the current contents of the
        `proj_dir`.

        Raises:
            ValueError: if there are more block files than indices in the
            `structure.pickle`.
        """
        files = listdir(self.proj_dir)
        self.block_list = [f for f in files if 'block_' in f]
        if len(self.block_list) > len(self.structure):
            raise ValueError(
                'More block files than indicies in `structure.pickle`',
            )

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """
        Perform the pass's operation, see BasePass for more info.

        Raises:
            ValueError: if a block file and the corresponding index in
                `structure.pickle` are differnt lengths.
        """
        # If the circuit is empty, just append blocks in order
        if circuit.depth == 0:
            for block in self.block_list:
                # Get block
                block_num = int(findall(r'\d+', block)[0])
                with open(self.proj_dir + '/' + block) as f:
                    block_circ = OPENQASM2Language().decode(f.read())
                # Get location
                block_location = self.structure[block_num]
                if block_circ.num_qudits != len(block_location):
                    raise ValueError(
                        f'{block} and `structure.pickle` locations are '
                        'different sizes.',
                    )
                # Append to circuit
                circuit.append_circuit(block_circ, block_location)
        # Check if the circuit has been partitioned, if so, try to replace
        # blocks
