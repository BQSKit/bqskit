"""
This module implements the Circuit class.

A circuit represents a quantum program composed of gate objects.
"""



from bqskit import qis

class Circuit( qis.Unitary ):
    """The Circuit class."""

    def __init__ ( self, num_qudits, qudit_radixes : Union[None, List[int]] = None ):
        """
        Circuit constructor. Builds an empty circuit with
        the specified number of qudits. By default, all qudits
        are qubits, but this can be changed with qudit_radixes parameter.

        Args:
            num_qudits (int): The number of qudits in this circuit.

            qudit_radixes (List[int]): A list with length equal
                to num_qudits. Each element specifies the base
                of a qudit. Defaults to qubits.
        
        Raises:
            ValueError: if num_qudits is non-positive.

        Examples:
            circ = Circuit(4) # Creates four-qubit empty circuit.
        """

        if not isinstance( num_qudits, int ):
            raise TypeError( "Invalid type for num_qudits: "
                             "expected int, got %s." % type( num_qudits ) )
        if num_qudits <= 0:
            raise ValueError( "Expected positive number for num_qudits." )

        self.num_qudits = num_qudits
        self.qudit_radixes = qudit_radixes or [ 2 for q in range( num_qudits ) ]
        self._circuit = [ [] for q in range( num_qudits ) ]
        self.gate_set = set()
    
    def get_num_params ( self ):
        pass

    def get_gate ( self, qudit, time_step ):
        pass

    def get_num_gates ( self ):
        pass

    def append_gate ( self, gate, qudits ):
        pass

    def remove_gate ( self, qudit, time_step ):
        pass

    def insert_gate ( self, gate, qudits, time_step ):
        pass
    
    def get_unitary ( self, params = None ):
        assert( params is None or len( params ) == num_params )
        pass

    def __iter__ ( self ):
        pass

    def __str__ ( self ):
        pass

    def __add__ ( self, rhs ):
        pass

    def __mul__ ( self, rhs ):
        pass

    def save ( self, filename ):
        pass

    @staticmethod
    def load ( filename ):
        pass
    