"""This module implements the Workflow class."""
from __future__ import annotations

import copy
from typing import Iterable
from typing import Iterator
from typing import overload
from typing import Sequence
from typing import Union

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.utils.random import seed_random_sources
from bqskit.utils.typing import is_iterable


class Workflow(BasePass, Sequence[BasePass]):
    """A BQSKit workflow captures a quantum circuit compilation process."""

    def __init__(self, passes: WorkflowLike) -> None:
        """
        Initialize a workflow object from a sequence of passes.

        Raises:
            ValueError: If passes is empty.
        """
        if isinstance(passes, Workflow):
            self._passes: list[BasePass] = copy.deepcopy(passes._passes)
            return

        if isinstance(passes, BasePass):
            passes = [passes]

        if not is_iterable(passes):
            msg = f'Expected Pass or sequence of Passes, got {type(passes)}.'
            raise TypeError(msg)

        if not all(isinstance(p, BasePass) for p in passes):
            truth_list = [isinstance(p, BasePass) for p in passes]
            wrong_type = type(list(passes)[truth_list.index(False)])
            msg = f'Expected Pass or sequence of Passes, got {wrong_type}.'
            raise TypeError(msg)

        self._passes = list(passes)

        if len(self._passes) == 0:
            raise ValueError('Expected at least one pass in workflow.')

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        for pass_obj in self._passes:
            if data.seed is not None:
                seed_random_sources(data.seed)
            await pass_obj.run(circuit, data)

    # @staticmethod
    # def build_compile_flow() -> Workflow:
    #     pass

    # @staticmethod
    # def build_transpile_flow() -> Workflow:
    #     pass

    # @staticmethod
    # def build_synthesize_flow() -> Workflow:
    #     pass

    # @staticmethod
    # def build_map_flow() -> Workflow:
    #     pass

    # @staticmethod
    # def build_prepare_flow() -> Workflow:
    #     pass

    # def __add__
    # def pretty_print
    # def save

    def __len__(self) -> int:
        return self._passes.__len__()

    def __iter__(self) -> Iterator[BasePass]:
        return self._passes.__iter__()

    @overload
    def __getitem__(self, _key: int, /) -> BasePass:
        ...

    @overload
    def __getitem__(self, _key: slice, /) -> list[BasePass]:
        ...

    def __getitem__(self, _key: int | slice) -> BasePass | list[BasePass]:
        return self._passes.__getitem__(_key)


WorkflowLike = Union[Workflow, Iterable[BasePass], BasePass]
