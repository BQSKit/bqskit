"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any
from typing import Callable

from dask.distributed import as_completed
from dask.distributed import Client
from dask.distributed import Future
from dask.distributed import rejoin
from dask.distributed import secede

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.qis.state.state import StateLike
from bqskit.qis.unitary.unitarymatrix import UnitaryLike
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

_logger = logging.getLogger(__name__)


class SynthesisPass(BasePass):
    """
    SynthesisPass class.

    The SynthesisPass is a base class that exposes an abstract
    synthesize function. Inherit from this class and implement the
    synthesize function to create a synthesis tool.

    A SynthesisPass will synthesize a new circuit targeting the input
    circuit's unitary.
    """

    @abstractmethod
    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """
        Synthesis abstract method to synthesize a UnitaryMatrix into a Circuit.

        Args:
            utry (UnitaryMatrix): The unitary to synthesize.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to provide auxillary information from
                previous passes. This function should never error based
                on what is in this dictionary.

        Note:
            This function should be self-contained and have no side effects.
        """

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        target_utry = circuit.get_unitary()
        circuit.become(self.synthesize(target_utry, data))

    def batched_instantiate(
        self,
        circuits: list[Circuit],
        target: UnitaryLike | StateLike,
        client: Client,
        **kwargs: Any,
    ) -> list[list[Future]]:
        """
        Batch instantiate `circuits`.

        Args:
            circuits (list[Circuit]): The circuit batch to instantiate.

            target (UnitaryLike | StateLike): The instantiation target.

            client (Client): The Dask client used to submit jobs.

            kwargs (Any): Other keyword arguments are passed directly to
                instantiate calls.

        Returns:
            (list[list[Future]]): The Dask Futures corresponding to the
                the instantiate jobs submitted. This returns a double list,
                where the other list is indexed by input circuits, and the
                inner list is indexed by multistart.
        """
        multistarts = 1
        if 'multistarts' in kwargs:
            multistarts = kwargs['multistarts']
            kwargs['multistarts'] = 1

        futures: list[list[Future]] = []

        client.scatter(circuits)

        for circuit in circuits:
            futures.append([])
            for i in range(multistarts):
                futures[-1].append(
                    client.submit(
                        Circuit.instantiate,
                        circuit,
                        pure=False,
                        target=target,
                        **kwargs,
                    ),
                )

        return futures

    def gather_best_results(
        self,
        futures: list[list[Future]],
        client: Client,
        fn: Callable[..., float],
        *args: Any,
        **kwargs: Any,
    ) -> list[Circuit]:
        """
        Gather best results from a `batched_instantiate` call.

        Args:
            futures (list[list[Future]]): The futures return from a
                `batched_instantiate` call.

            client (Client): The Dask client used to submit the jobs.

            fn (Callable[..., float]): The function used to sort
                instantiated circuits. This should take a circuit as
                the first parameter.

            args (Any): Arguments passed directly to fn.

            kwargs (Any): Keyword arguments passed directly to fn.

        Returns:
            (list[Circuit]): The resulting circuits. There is one circuit
                for each future list, i.e., `len(output) == len(futures)`.
        """

        score_futures: list[list[Future]] = []
        for future_list in futures:
            score_futures.append([])
            for future in future_list:
                score_futures[-1].append(
                    client.submit(
                        fn,
                        future,
                        *args,
                        **kwargs,
                    ),
                )

        flat_score_list = []
        for future_list in score_futures:
            flat_score_list.extend(future_list)

        secede()
        client.gather(flat_score_list)
        rejoin()

        best_circuit_futures = []
        for i, score_list in enumerate(score_futures):
            scores = client.gather(score_list)
            best_index = scores.index(min(scores))
            best_circuit_futures.append(futures[i][best_index])

        return client.gather(best_circuit_futures)

    def gather_first_results(
        self,
        futures: list[list[Future]],
        client: Client,
    ) -> list[Circuit]:
        """
        Gather first results from a `batched_instantiate` call.

        Args:
            futures (list[list[Future]]): The futures return from a
                `batched_instantiate` call.

            client (Client): The Dask client used to submit the jobs.

        Returns:
            (list[Circuit]): The resulting circuits. There is one circuit
                for each future list, i.e., `len(output) == len(futures)`.
                This call will return the first multistart instantiate
                job for each circuit.
        """
        all_futures = []
        future_to_index_map = {}
        for index, future_list in enumerate(futures):
            all_futures.extend(future_list)
            for future in future_list:
                future_to_index_map[future] = index

        secede()
        circuits = []
        indices_seen = set()
        for future, circuit in as_completed(all_futures, with_results=True):
            index = future_to_index_map[future]
            if index not in indices_seen:
                indices_seen.add(index)
                circuits.append((index, circuit))
                for other_future in futures[index]:
                    if not other_future.done():
                        client.cancel(other_future)
        rejoin()

        return [c for _, c in sorted(circuits, key=lambda x: x[0])]
