"""This module implements the Workflow class."""
from __future__ import annotations

import copy
import logging
from typing import Iterable
from typing import Iterator
from typing import overload
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

from bqskit.compiler.basepass import BasePass
from bqskit.utils.random import seed_random_sources
from bqskit.utils.typing import is_iterable

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class Workflow(BasePass, Sequence[BasePass]):
    """A BQSKit workflow captures a quantum circuit compilation process."""

    def __init__(self, passes: WorkflowLike, name: str = '') -> None:
        """
        Initialize a workflow object from a sequence of passes.

        Args:
            passes (WorkflowLike): The passes to run in the workflow.

            name (str): The optional name of the workflow.

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

        if not isinstance(name, str):
            raise TypeError(f'Expected name to be str, got {type(name)}.')

        self._name = name
        self._passes = list(passes)

        if len(self._passes) == 0:
            raise ValueError('Expected at least one pass in workflow.')

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        for pass_obj in self._passes:
            if data.seed is not None:
                seed_random_sources(data.seed)
            _logger.debug(f'Running {pass_obj.name}')
            await pass_obj.run(circuit, data)

    def save(self, filename: str) -> None:
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> Workflow:
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def name(self) -> str:
        """The name of the pass."""
        return self._name or self.__class__.__name__

    def __str__(self) -> str:
        name_seq = f'Workflow: {self.name}\n\t'
        pass_strs = [
            f'{i}. {"Workflow: " + p.name if isinstance(p, Workflow) else p}'
            for i, p in enumerate(self._passes)
        ]
        return name_seq + '\n\t'.join(pass_strs)

    def __add__(self, other: WorkflowLike) -> Workflow:
        return Workflow(self._passes + Workflow(other)._passes)

    def __radd__(self, other: WorkflowLike) -> Workflow:
        return Workflow(Workflow(other)._passes + self._passes)

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
