from bqskit.utils.cachedclass import CachedClass

class Arg(CachedClass):
    def __init__(self, arg) -> None:
        self.arg = arg
    
class Kwarg(CachedClass):
    def __init__(self, default = 4) -> None:
        self.arg = default

def test_cachedclass_arg():
    a = Arg(2)
    assert a.arg == 2
    b = Arg(2)
    c = Arg(3)
    assert a is b
    assert a is not c

def test_cachedclass_kwarg():
    a = Kwarg(default=1)
    assert a.arg == 1
    b = Kwarg(default=1)
    c = Kwarg(default=0)
    assert a is b
    assert a is not c