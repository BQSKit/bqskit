from __future__ import annotations

from typing import cast
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.passes.alias import PassAlias
from bqskit.utils.typing import is_sequence


class PassGroup(PassAlias):
    """A pass that is a group of other passes."""

    def __init__(self, passes: BasePass | Sequence[BasePass]) -> None:
        """Group together one or more `passes`."""
        if not is_sequence(passes):
            passes = [cast(BasePass, passes)]

        if not isinstance(passes, list):
            passes = list(passes)

        self.passes: list[BasePass] = passes

        for p in self.passes:
            if not isinstance(p, BasePass):
                raise TypeError(f'Expected a Pass, got {type(p)}.')

    def get_passes(self) -> list[BasePass]:
        """Return the passes to be run, see :class:`PassAlias` for more."""
        return self.passes
