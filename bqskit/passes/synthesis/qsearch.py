"""This module implements the QSearchSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any
import math 
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators import SimpleLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic
from bqskit.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number


_logger = logging.getLogger(__name__)


class QSearchSynthesisPass(SynthesisPass):
    """
    A pass implementing the QSearch A* synthesis algorithm.

    References:
        Davis, Marc G., et al. “Towards Optimal Topology Aware Quantum
        Circuit Synthesis.” 2020 IEEE International Conference on Quantum
        Computing and Engineering (QCE). IEEE, 2020.
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        layer_generator: LayerGenerator = SimpleLayerGenerator(),
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator): The successor function
                to guide node expansion.

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-10)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `max_depth` is nonpositive.
        """
        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        if not isinstance(layer_generator, LayerGenerator):
            raise TypeError(
                'Expected LayerGenerator, got %s.'
                % type(layer_generator),
            )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                'Expected max_layer to be an integer, got %s' % type(max_layer),
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                'Expected max_layer to be positive, got %d.' % int(max_layer),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': self.cost,
        }
        self.instantiate_options.update(instantiate_options)
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth

    def transform_circuit_from_squander_to_qsearch(self,
    cDecompose,
    qubitnum)-> Circuit:
    #import all the gates
        from bqskit.ir.gates.constant.cx import CNOTGate
        from bqskit.ir.gates.parameterized.cry import CRYGate
        from bqskit.ir.gates.constant.cz import CZGate
        from bqskit.ir.gates.constant.ch import CHGate
        from bqskit.ir.gates.constant.sycamore import SycamoreGate as SYC
        from bqskit.ir.gates.parameterized.u3 import U3Gate 
        from bqskit.ir.gates.parameterized.rx import RXGate
        from bqskit.ir.gates.parameterized.ry import RYGate
        from bqskit.ir.gates.parameterized.rz import RZGate
        from bqskit.ir.gates.constant.x import XGate
        from bqskit.ir.gates.constant.y import YGate
        from bqskit.ir.gates.constant.z import ZGate
        from bqskit.ir.gates.constant.sx import SqrtXGate
        import numpy as np
        
        
        circuit=Circuit(qubitnum)
        gates=cDecompose.get_Gates()
        for idx in range(len(gates)-1, -1, -1):
            
            gate = gates[idx]

            if gate.get("type") == "CNOT":
                # adding CNOT gate to the quantum circuit
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(CNOTGate(), (control_qbit, target_qbit))
                

            if gate.get("type") == "CRY":
                # adding CRY gate to the quantum circuit
                Theta=gate.get("Theta")     
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(CRYGate() ,(control_qbit, target_qbit),[THeta])

            elif gate.get("type") == "CZ":
                # adding CZ gate to the quantum circuit
            
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")   
                Theta = gate.get("Theta")                  
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))
               

            elif gate.get("type") == "CH":
                # adding CZ gate to the quantum circuit
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(CZGate(), (control_qbit, target_qbit))
               

            elif gate.get("type") == "SYC":
                # Sycamore gate
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(SycamoreGate(), (control_qbit, target_qbit))
               
                
            elif gate.get("type") == "U3":
                target_qbit=gate.get("target_qbit") 
                Theta = gate.get("Theta")     
                Lambda = gate.get("Lambda") 
                Phi = gate.get("Phi") 
                circuit.append_gate(U3Gate(),target_qbit,[Theta,Phi,Lambda])

            elif gate.get("type") == "RX":
                # RX gate               
                target_qbit = gate.get("target_qbit")                
                Theta = gate.get("Theta")                
                circuit.append_gate(RXGate(),( target_qbit),[Theta]) 
              

            elif gate.get("type") == "RY":
                # RY gate
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                Theta = gate.get("Theta")                
                circuit.append_gate(RYGate(),(target_qbit),[Theta])
                

            elif gate.get("type") == "RZ":
                # RZ gate
                       
                target_qbit = gate.get("target_qbit")                
                Phi = gate.get("Phi")                
                circuit.append_gate(RZGate(), (target_qbit),[Phi] )
               

            elif gate.get("type") == "X":
                # X gate
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(XGate(), (target_qbit))
               

            elif gate.get("type") == "Y":
                # Y gate
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(YGate(), (target_qbit))
             

            elif gate.get("type") == "Z":
                # Z gate
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(ZGate(), (target_qbit))

            elif gate.get("type") == "SX":
                # SX gate
                control_qbit = gate.get("control_qbit")                
                target_qbit = gate.get("target_qbit")                
                circuit.append_gate(RZGate(), (target_qbit))
       
        return(circuit)
    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        Umtx = utry.numpy
        from squander import N_Qubit_Decomposition_adaptive
        import numpy as np
        import numpy.linalg as LA
        from squander import utils
     
        qubitnum=math.floor(math.log2(utry.__len__()))
        if qubitnum > 2 :
            #try with qiskit
            
            
            cDecompose = N_Qubit_Decomposition_adaptive(Umtx.conj().T, level_limit_max=5,level_limit_min=0)
            cDecompose.set_Verbose(0)
            cDecompose.Start_Decomposition()
            
            
            quantum_circuit = cDecompose.get_Quantum_Circuit()
            
            
            decomposed_matrix_qskit = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
            product_matrix_qskit = np.dot(Umtx,decomposed_matrix_qskit.conj().T)
            phase_qskit = np.angle(product_matrix_qskit[0,0])
            product_matrix_qskit = product_matrix_qskit*np.exp(-1j*phase_qskit)
            shape_qskit = np.shape(product_matrix_qskit)
            product_matrix_qskit = np.eye(8)*2 - product_matrix_qskit - product_matrix_qskit.conj().T
            # the error of the decomposition
            decomposition_error_qskit = (np.real(np.trace(product_matrix_qskit)))/2
       
            print('The error of the decomposition in qskit is' + str(decomposition_error_qskit))
            
            
            
            Circuit2=self.transform_circuit_from_squander_to_qsearch(cDecompose,qubitnum)
            Unitarymatrix_bqskit=Circuit.get_unitary(Circuit2)	
            
            product_matrix = np.dot(Umtx,Unitarymatrix_bqskit.conj().T)
            phase = np.angle(product_matrix[0,0])
            product_matrix = product_matrix*np.exp(-1j*phase)
            shape=np.shape(product_matrix)
            product_matrix = np.eye(shape[0])*2 - product_matrix - product_matrix.conj().T
            decomposition_error = (np.real(np.trace(product_matrix)))/2
            print('\n\n\nThe error of the decomposition is ' + str(decomposition_error),"\n\n\n ")	
             #def save(self, filename: str) -> None:
       # """Save the circuit to a file."""
        #language = get_language(filename.split('.')[-1])
        #numpy.savetxt
            squander_decomposition_error=cDecompose.get_Decomposition_Error()
            if abs(squander_decomposition_error-decomposition_error)>10e-3:
                Circuit.save(Circuit2,"bad_circuit.qasm")
                np.savetxt("bad_Unitarybqskit.txt",Unitarymatrix_bqskit)
                np.savetxt("bad_umtx",Umtx)
                np.savetxt("decomposed_matrix_qskit",decomposed_matrix_qskit)
                #exit()
        "ide kell majd az összehasonlítás az umtx-el a Circuit 2"
        frontier = Frontier(utry, self.heuristic_function)

        # Seed the search with an initial layer
        initial_layer = self.layer_gen.gen_initial_layer(utry, data)
        initial_layer = self.execute(
            data,
            Circuit.instantiate,
            [initial_layer],
            target=utry,
            **self.instantiate_options,
        )[0]
        frontier.add(initial_layer, 0)

        # Track best circuit, initially the initial layer
        best_dist = self.cost.calc_cost(initial_layer, utry)
        best_circ = initial_layer
        best_layer = 0

        # Track partial solutions
        psols: dict[int, list[tuple[Circuit, float]]] = {}

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        if best_dist < self.success_threshold:
            _logger.debug('Successful synthesis.')
            return initial_layer

        # Main loop
        while not frontier.empty():
            top_circuit, layer = frontier.pop()

            # Generate successors
            successors = self.layer_gen.gen_successors(top_circuit, data)

            # Instantiate successors
            circuits = self.execute(
                data,
                Circuit.instantiate,
                successors,
                target=utry,
                **self.instantiate_options,
            )

            # Evaluate successors
            for circuit in circuits:
                dist = self.cost.calc_cost(circuit, utry)

                if dist < self.success_threshold:
                    _logger.debug('Successful synthesis.')
                    if self.store_partial_solutions:
                        data['psols'] = psols
                    return circuit

                if dist < best_dist:
                    _logger.debug(
                        'New best circuit found with %d layer%s and cost: %e.'
                        % (layer + 1, '' if layer == 0 else 's', dist),
                    )
                    best_dist = dist
                    best_circ = circuit
                    best_layer = layer

                if self.store_partial_solutions:
                    if layer not in psols:
                        psols[layer] = []

                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )
        if self.store_partial_solutions:
            data['psols'] = psols

        return best_circ
