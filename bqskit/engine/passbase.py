import abc


class PassBaseClass (abc.ABC):

    def __init__(self):
        pass

    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def run(self, ) -> None: # LEFT OFF HERE
        pass
