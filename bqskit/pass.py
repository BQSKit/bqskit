import abc

class Pass ( abc.ABC ):
    
    def __init__ ( self ):
        pass
    
    @abc.abstractmethod
    def run ( self ):
        pass