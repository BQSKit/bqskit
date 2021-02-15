from __future__ import annotations

from bqskit.utils.cachedclass import CachedClass


class Arg(CachedClass):
    def __init__(self, arg: int) -> None:
        self.arg = arg


class Kwarg(CachedClass):
    def __init__(self, default: int = 4) -> None:
        self.arg = default


def test_cachedclass_arg() -> None:
    a = Arg(2)
    assert a.arg == 2
    b = Arg(2)
    c = Arg(3)
    assert a is b
    assert a is not c


def test_cachedclass_kwarg() -> None:
    a = Kwarg(default=1)
    assert a.arg == 1
    b = Kwarg(default=1)
    c = Kwarg(default=0)
    assert a is b
    assert a is not c
