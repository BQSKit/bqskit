"""This module defines the SimplePartitioner pass."""
from __future__ import annotations

from typing import Any, Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit

import networkx as nx
import os

class SimplePartitioner(BasePass):
    
    def __init__(
        self,
        circ: Circuit | None = None,
        coupling_map: Sequence[tuple[int, int]] | None = None,
        qubit_groups: Sequence[Sequence[int]] | None = None,
        reopt: bool = True,
        reopt_single: bool = True,
        verbosity: int = 0,
        timeout: int = 0,
        syn_tool: str = "sc",
        b_size: int | None = 3,
        cache: bool = True,
        cache_threshold: float = 1e-8,
        sc_threshold: float = 1e-10,
        compose_threshold: float = 2e-8,
        name: str = "new_circuit",
        block_details: bool = True,
        all_to_all: bool = False
    ) -> None:
        """
        Constructor for a SimplePartitioner based on the QGo algorithm.

        Args:
            circ (Circuit, optional): The Qiskit QuantumCircuit on which 
                parition is to be done. Defaults to None.
            coupling_map (Sequence[tuple[int, int]], optional): A list of 
                tuples that defines edges in the machines physical topology. 
                Defaults to None.
            qubit_groups (Sequence[Sequence[int]], optional): If qubit groups 
                have been precomputed, they can be passed here to save time. 
                Defaults to None.
            reopt (bool, optional): [description]. Defaults to True.
            reopt_single (, optional): [description]. Defaults to True.
            verbosity (int, optional): For differing levels of output while 
                running. Defaults to 0.
            timeout (int, optional): [description]. Defaults to 0.
            syn_tool (str, optional): For setting the synthesis tool used. 
                Defaults to "sc".
            b_size (int, optional): The block size of partitions to be made. 
                Defaults to 3.
            cache (bool, optional): [description]. Defaults to True.
            cache_threshold (float, optional): [description]. Defaults to 
                1e-8.
            sc_threshold (float, optional): [description]. Defaults to 1e-10.
            compose_threshold (float, optional): [description]. Defaults to 
                2e-8.
            name (str, optional): [description]. Defaults to "new_circuit".
            block_details (bool, optional): [description]. Defaults to True.
            all_to_all (bool, optional): [description]. Defaults to False.
        """
        self.qubit_array = []
        self.coupling_map = coupling_map
        self.num_q = 0
        if coupling_map is not None:
            self.num_q = self._num_device_qubits(coupling_map)
        self.qubit_groups = qubit_groups
        self.b_size = b_size
        self.name = name
        self.verbosity = verbosity
        self.used_qubits = set()
        
        ######################################################################
        # THESE VARIABLES MAY BE OBSOLETE                                    #
        ######################################################################
        #self.original_circuit = circ
        #self.new_circuit = QuantumCircuit(self.num_q, self.num_q)
        #self.cache = cache
        #self.cache_threshold = cache_threshold
        #self.sc_threshold = sc_threshold
        #self.compose_threshold = compose_threshold
        #self.block_details = block_details
        #self.all_to_all = all_to_all
        #self.syn_tool = syn_tool
        #self.timeout = timeout
        #self.reopt_single = reopt_single
        ## Variables for storing intermediate status
        #self.qasm_blocks = []
        #self.unitary_blocks = []
        #self.new_qasm_blocks = []
        #self.exe_time = []
        #self.distance = []
        #self.valid_blocks = []
        #self.hit_rate = 0
        #self.reopt = reopt

        self.project_name = "SimplePartitioner_%s_q%s_b%s_%s_%s" \
            % (self.name, self.num_q, self.b_size, str(sc_threshold), syn_tool)

        # Set up directory for output
        if not os.path.exists(self.project_name):
            os.mkdir(self.project_name)
        self.tmp_new_circ = self.project_name + "/tmp_new_circ"
        if not os.path.exists(self.tmp_new_circ):
            os.mkdir(self.tmp_new_circ)
        
        # Set up lists for partitioning
        for i in range(self.num_q):
            self.qubit_array.append([])
        
        ######################################################################
        # TODO: USE BQSKIT CIRCUIT                                           #
        ######################################################################
        if circ is not None:
            for i in range(len(circ.data)):
                #############
                print("")

    def _num_device_qubits(coupling_map):
        """Find the number of qubits used in the coupling_map.

        Args:
            coupling_map (Sequence[tuple[int,int]]): Edges in the physical
                qubit topology of the machine.

        Returns:
            (int): The number of qubits in the coupling_map.
        """
        nodes = set()
        for x,y in coupling_map:
            nodes.update([x,y])
        return len(nodes)

    def run(self, circuit: Circuit, data: dict[str, Any]) -> None:
        """
        Block gates into CircuitGates. Given the b_size, partition a 
        circuit into a series of CircuitGates (subcircuits represented as 
        immutable gates). Synthesis should be run on each of the CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.
            data (dict[str,Any]): Additional information for the partitioner.
                If "b_size", "circ", or "coupling_map" are specified, they
                will replace the version given in the constructor.
        """
        if "b_size" in data:
            self.b_size = data["b_size"]
        if "coupling_map" in data:
            self.coupling_map = data["coupling_map"]
        if "circ" in data:
            self.circ = data["circ"]
    

    