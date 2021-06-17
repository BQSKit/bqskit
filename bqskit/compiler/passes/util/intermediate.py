"""This module implements the RecordStatsPass class."""
from __future__ import annotations
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language

from typing import Any
from os.path import exists
from os import mkdir
import logging
import pickle

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passes.util.variabletou3 import VariableToU3Pass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate

_logger = logging.getLogger(__name__)


class SaveIntermediatePass(BasePass):
    """
    The SaveIntermediate class.

    The SaveIntermediatePass stores individual circuit gates as qasm files.
    """
    def __init__(
        self, 
        path_to_save_dir : str, 
        project_name : str | None = None,
    ) -> None:
        """
        Constructor for the SaveIntermediatePass.
        
        Args:
            path_to_save_dir (str): Path to the directory in which inter-
                qasm for circuit blocks should be saved.
            
            project_name (str): Name of the project files.
        """
        if exists(path_to_save_dir):
            self.pathdir = path_to_save_dir
            if self.pathdir[-1] != "/":
                self.pathdir += "/"
        else:
            _logger.warning(
                f"Path {path_to_save_dir} does not exist, "
                "saving to local directory"
            )
            self.pathdir = "./"
        self.projname = project_name if project_name is not None \
            else "unnamed_project"
        
        enum = 1
        if exists(self.pathdir + self.projname):
            while exists(self.pathdir + self.projname + f"_{enum}"):
                enum += 1
            self.projname += f"_{enum}"
        
        mkdir(self.pathdir + self.projname)
            

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """Perform the pass's operation, see BasePass for more info."""

        # Gather and enumerate CircuitGates to save
        blocks_to_save: list[tuple[int, CircuitGate]] = []
        for enum, op in enumerate(circuit):
            if isinstance(op.gate, CircuitGate):
                blocks_to_save.append((enum, op))
        
        # Set up path and file names
        structure_file = self.pathdir + self.projname + "/structure.pickle"
        block_skeleton = self.pathdir + self.projname + "/block_"

        structure_list : list[dict[int,int]] = []
        for enum, block in blocks_to_save:
            num_q = block.size
            qudit_map  = {num:qudit for num,qudit in enumerate(block.location)}
            structure_list.append(qudit_map)
            subcircuit = Circuit(num_q)
            subcircuit.append_gate(block.gate, list(range(num_q)), block.params)
            subcircuit.unfold((0,0))
            VariableToU3Pass().run(subcircuit, {})

            with open(block_skeleton + f"{enum}.qasm", "w") as f:
                f.write(OPENQASM2Language().encode(subcircuit))
        
        with open(structure_file, "wb") as f:
            pickle.dump(structure_list, f)
