from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.utils import typing


class TestIsIterable:
    def test_base_false(self) -> None:
        # Numbers are not iterable
        assert not typing.is_iterable(0)
        assert not typing.is_iterable(-1)
        assert not typing.is_iterable(1)
        assert not typing.is_iterable(1.0)
        assert not typing.is_iterable(1.0 + 2.0j)

        # Booleans are not iterable
        assert not typing.is_iterable(False)
        assert not typing.is_iterable(True)

        # Callables are not iterable
        assert not typing.is_iterable(lambda x: 0)

        # Modules are not iterable
        assert not typing.is_iterable(typing)

    def test_base_true(self) -> None:
        # Sequences are iterable
        assert typing.is_iterable('TestString')
        assert typing.is_iterable((0, 1))
        assert typing.is_iterable((0,))
        assert typing.is_iterable([])
        assert typing.is_iterable([0, 1])

        # Sets and Dicts are also iterable
        assert typing.is_iterable({0, 1})
        assert typing.is_iterable({0: 1, 1: 2})

    def test_gates(self, gate: Gate) -> None:
        # Gates are not iterable
        assert not typing.is_iterable(gate)

    def test_circuit(self, swap_circuit: Circuit) -> None:
        # Circuits are iterable
        assert typing.is_iterable(swap_circuit)
