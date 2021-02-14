import abc

class ObjectiveFunction(abc.ABC):
    """
    An ObjectiveFunction in BQSKit is a real-valued function that maps
    unitary matrices to real numbers.
    """

    @abc.abstractmethod
    def cost( U: UnitaryMatrix ) -> float:
        """Returns the cost given a circuit's unitary."""
    
    @abc.abstractmethod
    def cost_grad( dU: Sequence[np.ndarray] ) -> list[float]:
        """Returns the gradient of the cost function, given the circuit's gradient."""