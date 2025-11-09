"""This module implements the SquanderSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.passdata import PassData

import math 
import numpy as np
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from bqskit.qis.graph import CouplingGraph
from squander import utils
from squander.gates import qgd_Circuit
from squander import N_Qubit_Decomposition_Tree_Search
from squander import N_Qubit_Decomposition_Tabu_Search
from typing import Any
_logger = logging.getLogger(__name__)


class SquanderSynthesisPass(SynthesisPass):
    """
    A pass implementing the Squander synthesis algorithm.

    """

    def __init__(
        self, success_threshold : float = 1e-8,
        max_layer: int = 20,
        squander_config: dict[str, Any] = {} 
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to 20. (Default: None)
                
             squander_config (dict[str, Any]): Configuration dictionary for
                 customizing the Squander synthesis behavior. Supported keys include:

                - "verbosity" (int): Verbosity level for logging and output.
                  Default is -1 (minimal output).

                - "strategy" (str): Search strategy to use. Options:
                  "Tabu_search" (default) or "Tree_search".

                - "optimization_tolerance" (float): Tolerance for optimization
                  convergence. Default is 1e-8.

                - "Cost_Function_Variant" (int): Variant of cost function to use.
                  Default is 3 to hilbert_schmidt test.

                - "optimizer_engine" (str): Optimization algorithm to employ,
                  e.g., "BFGS","ADAM","GRAD_dESCEND","ADAM_BATCHED","BFGS","AGENTS","COSINE","GRAD_DESCEND:PARAMETER_RULE","AGENTS_COMBINES","BAYES_OPT".
                  Default is 'BFGS'.


        Raises:
            ValueError: If `max_depth` is nonpositive.
        """

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                f', got {type(success_threshold)}',
            )

        if not isinstance(max_layer, int):
            raise TypeError(
                f'Expected max_layer to be an integer, got {type(max_layer)}.',
            )


        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                f'Expected max_layer to be positive, got {int(max_layer)}.',
            )

        self.success_threshold = success_threshold

        # cost calculator from BQSkit to verify squander sythesis
        self.bqskit_cost_calculator = HilbertSchmidtResidualsGenerator()

        # the maximum number of layers (CNOT gates) used in the tree search
        self.max_layer = max_layer

        self.squander_config = squander_config


        squander_config.setdefault("verbosity", -1)
        squander_config.setdefault("strategy", "Tabu_search")
        squander_config.setdefault("optimization_tolerance", 1e-8)
        squander_config.setdefault("Cost_Function_Variant",3)
        squander_config.setdefault("optimizer_engine",'BFGS')
        
        valid_strategies = ["Tabu_search", "Tree_search"]
        valid_strategy_variants = valid_strategies + [s.lower() for s in valid_strategies]



        if squander_config["strategy"] not in valid_strategy_variants:
            raise TypeError (
                 f"Invalid strategy: {strategy}",
            )
            
            
    def transform_circuit_from_squander_to_bqskit(self,
        Squander_circuit,
        parameters)-> Circuit:
        '''
        Convert a Squander circuit and its parameters into a BQSKit Circuit.

        Args:
            Squander_circuit: The circuit created with Squander.
            parameters: Array of gate parameters used by the circuit.

        Raises:
            TypeError:
                If `Squander_circuit` is not an instance of `qgd_Circuit`
                or if `parameters` is not a numpy ndarray.
        ValueError:
            If `parameters` is an empty array or if the circuit
            contains unsupported gates.
        
        
       
        
        '''
        qgd_type = getattr(qgd_Circuit, 'qgd_Circuit', None)
        if not isinstance(Squander_circuit, qgd_type):
            raise TypeError(f'Expected qgd_Circuit, got {type(Squander_circuit)}.')


        if not isinstance(parameters, np.ndarray):
            raise TypeError(
                f'Expected parameters to be a numpy.ndarray, got {type(parameters)}.'
            )

        expected_param_count = Squander_circuit.get_Parameter_Num()
        if parameters.size != expected_param_count:
            raise ValueError(
                f'Parameter size mismatch: expected {expected_param_count}, got {parameters.size}.'
            )
            

    #import all the gates
        from bqskit.ir.gates.constant.cx import CNOTGate
        from bqskit.ir.gates.parameterized.cry import CRYGate
        from bqskit.ir.gates.constant.cz import CZGate
        from bqskit.ir.gates.constant.ch import CHGate
        from bqskit.ir.gates.constant.sycamore import SycamoreGate
        from bqskit.ir.gates.parameterized.u3 import U3Gate 
        from bqskit.ir.gates.parameterized.rx import RXGate
        from bqskit.ir.gates.parameterized.ry import RYGate
        from bqskit.ir.gates.parameterized.rz import RZGate
        from bqskit.ir.gates.constant.x import XGate
        from bqskit.ir.gates.constant.y import YGate
        from bqskit.ir.gates.constant.z import ZGate
        from bqskit.ir.gates.constant.sx import SqrtXGate
        import squander
        
        qbit_num = Squander_circuit.get_Qbit_Num()
        circuit = Circuit(qbit_num)
        gates = Squander_circuit.get_Gates()
        # constructing quantum circuit
        for gate in gates:

            if isinstance( gate, squander.CNOT ):
                # adding CNOT gate to the quantum circuit
                control_qbit = qbit_num - gate.get_Control_Qbit()  - 1              
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1               
                circuit.append_gate(CNOTGate(), (control_qbit, target_qbit))
            
            elif isinstance( gate, squander.CRY ):
                # adding CNOT gate to the quantum circuit
                parameters_gate = gate.Extract_Parameters( parameters )
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1                
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1               
                circuit.append_gate(CRYGate() ,(control_qbit, target_qbit), parameters_gate)
            
            elif isinstance( gate, squander.CZ ):
                # adding CZ gate to the quantum circuit
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1               
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1                  
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))

            elif isinstance( gate, squander.CH ):    
                # adding CH gate to the quantum circuit
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1               
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1             
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))

            elif isinstance( gate, squander.SYC ):
                # Sycamore gate
                control_qbit = qbit_num - gate.get_Control_Qbit() - 1               
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1            
                circuit.append_gate(SycamoreGate(), (control_qbit, target_qbit))

            elif isinstance( gate, squander.U3 ):
                # adding U3 gate to the quantum circuit
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(U3Gate(),target_qbit, parameters_gate)   

            elif isinstance( gate, squander.RX ):
                # RX gate
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1        
                circuit.append_gate(RXGate(),( target_qbit), parameters_gate)   
            
            elif isinstance( gate, squander.RY ):
                # RY gate
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1   
                circuit.append_gate(RYGate(),(target_qbit), parameters_gate)

            elif isinstance( gate, squander.RZ ):
                # RZ gate
                parameters_gate = gate.Extract_Parameters( parameters )
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1    
                circuit.append_gate(RZGate(), (target_qbit), parameters_gate ) 
            
            elif isinstance( gate, squander.H ):
                # Hadamard gate
                circuit.h( gate.get_Target_Qbit() )    

            elif isinstance( gate, squander.X ):
                # X gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(XGate(), (target_qbit))

            elif isinstance( gate, squander.Y ):
                # Y gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(YGate(), (target_qbit))

            elif isinstance( gate, squander.Z ):
                # Z gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(ZGate(), (target_qbit))

            elif isinstance( gate, squander.SX ):
                # SX gate
                target_qbit = qbit_num - gate.get_Target_Qbit() - 1      
                circuit.append_gate(SqrtXGate(), (target_qbit))

            elif isinstance( gate, squander.Circuit ):
                # Sub-circuit gate
                raise ValueError("Qiskit export of circuits with subcircuit is not supported. Use Circuit::get_Flat_Circuit prior of exporting circuit.")  
            
            else:
                print(gate)
                raise ValueError("Unsupported gate in the circuit export.")

  
        return(circuit)




    async def synthesize(
        self,
        utry: UnitaryMatrix,
        data: PassData,
    ) -> Circuit:
        #"""Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # Initialize run-dependent options
        
        '''
        Synthesize a quantum circuit that approximates the target unitary.

	

	Args:
	    utry (UnitaryMatrix):
	        The target unitary matrix to synthesize.
	    data (PassData): 
	        PassData object containing synthesis context
	        and target hardware information.

	

	Raises:
	    TypeError: 
	        If `utry` is not a UnitaryMatrix instance or
	        `data` is not a PassData instance.
	    ValueError:
	        If synthesis fails to achieve the success threshold.'''
        
        
        
        if not isinstance(utry, UnitaryMatrix):
            raise TypeError(
                f'Expected utry to be a UnitaryMatrix, got {type(utry)}.'
            )

        if not isinstance(data, PassData):
            raise TypeError(
                f'Expected data to be a PassData, got {type(data)}.'
            )
    
    
        Umtx = utry.numpy
        qbit_num = data.target.num_qudits
        
        topology_list = data.connectivity
        num_qubits = topology_list.num_qudits
        original_edges = list(topology_list)  # list of (i, j)

        reversed_topology_list = []
        for i, j in original_edges:
            new_i = num_qubits - i - 1
            new_j = num_qubits - j - 1
            reversed_topology_list.append((new_i, new_j))
        
        if self.squander_config["strategy"] == "Tree_search":
            cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, topology= reversed_topology_list , config=self.squander_config, accelerator_num=0 )
        elif self.squander_config["strategy"] == "Tabu_search":
            cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, topology= reversed_topology_list , config=self.squander_config, accelerator_num=0 )

            

            
       
        cDecompose.set_Verbose( self.squander_config["verbosity"] )
        cDecompose.set_Cost_Function_Variant(self.squander_config["Cost_Function_Variant"])

    

        # adding new layer to the decomposition until threshold
        cDecompose.set_Optimizer( self.squander_config["optimizer_engine"] )

        # starting the decomposition
        cDecompose.Start_Decomposition()
            

        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()
   
        #Circuit_squander = self.transform_circuit_from_squander_to_bqskit( squander_circuit, parameters)
        Circuit_squander = cDecompose.get_Bqskit_Circuit()          
        dist             = self.bqskit_cost_calculator.calc_cost(Circuit_squander, utry)  
        
        #print( 'Squander dist: ', str(dist) )
           

        _logger.debug("The error of the decomposition with SQUANDER is "  + str(dist))            
           
            
        if dist >  self.success_threshold:
            _logger.debug('the squander decomposition error is bigger than the succes_treshold, with the value of:', dist)                
        else: 
            _logger.debug('Successful synthesis with squander.')

        '''
        squander_gates = {}

        for gate in Circuit_squander.gate_set:
            squander_gates[str(gate)] = Circuit_squander.count(gate)
 
        #print('squander: qbit_num = ', qbit_num, ' dist = ', dist, squander_gates, "\n")
        '''
        
           
        return(Circuit_squander) 

            
         

        return layer_gen
