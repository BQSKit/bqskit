import abc

class StateVector:
    def __init__(self) -> None:
        pass
    
class StateVectorMap(abc.ABC):
    @abc.abstractmethod
    def get_statevector(self, in_state: StateVector) -> StateVector:
        pass