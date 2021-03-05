"""This module implements the CircuitPoint class."""


from typing import NamedTuple, Union


class CircuitPoint(NamedTuple):
    """
    The CircuitPoint NamedTuple class.

    A CircuitPoint is a 2d-index into the Circuit data structure.
    """
    cycle: int
    qudit: int


CircuitPointLike = Union[tuple[int, int], CircuitPoint]
