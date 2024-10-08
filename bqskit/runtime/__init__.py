"""
=======================================================
A Runtime for the Compile-time (:mod:`bqskit.runtime`)
=======================================================

.. currentmodule:: bqskit.runtime

This package provides an execution environment for compiler passes,
enabling them to parallelize and distribute their workload efficiently.

Launching a Server
##################

To parallelize BQSKit compilations, you can launch a BQSKit Runtime
server in either attached or detached mode, and submit a
:class:`~bqskit.compiler.task.CompilationTask` through a connected
:class:`~bqskit.compiler.compiler.Compiler`.

By default, a compiler will automatically start and connect to an attached
runtime server during initialization. In this mode, the compiler
starts up and shuts down the server automatically and transparently to
the end user. See :class:`~bqskit.runtime.attached.AttachedServer` for
more info.

A server running in attached mode is limited to a single machine. If you
would like to parallelize BQSKit across multiple nodes over a network,
you can launch a server independently in detached mode. Here, you first
start :class:`~bqskit.runtime.manager.Manager` processes on all machines.
Then link them all together by running a
:class:`~bqskit.runtime.detached.DetachedServer` process, which Compilers'
can connect to by passing the appropriate address in their constructor.

Upon installing BQSKit, two shell scripts are added to your environment
that can spawn managers and detached servers::

    bqskit-manager

and::

    bqskit-server <address_of_bqskit_managers>

Typically, you start managers first on all nodes in the desired cluster.
Then you can start the server with a comma-separated list of all managers
ip address and optionally ports. Once a server is started and has connected
to the requested managers, no more managers can be added. You can see
the `-h` option of each command or the
:func:`~bqskit.runtime.detached.start_server` and
:func:`~bqskit.runtime.manager.start_manager` entry points for
more information.

For more information on how to manage jobs submitted to a runtime server,
see the :obj:`~bqskit.compiler` documentation.

.. autosummary::
    :toctree: autogen
    :recursive:

    ~bqskit.runtime.attached.AttachedServer
    ~bqskit.runtime.detached.DetachedServer
    ~bqskit.runtime.manager.Manager
    ~bqskit.runtime.detached.start_server
    ~bqskit.runtime.manager.start_manager
    ~bqskit.runtime.worker.start_worker_rank

Parallelize Pass Computation
############################

When developing a BQSKit compiler pass, you can hook into the active
runtime server through the :func:`get_runtime` function. This returns a
:class:`RuntimeHandle`, which you can use to submit, map, wait on, and
cancel tasks in the execution environment.

For more information on how to design a custom pass, see the following
guide: :doc:`guides/custompass.md`.

.. autosummary::
    :toctree: autogen
    :recursive:

    ~bqskit.runtime.get_runtime
    ~bqskit.runtime.RuntimeHandle
    ~bqskit.runtime.future.RuntimeFuture

Standard Logging Interface
##########################

This system uses the standard python logging module to provide a familiar
logging interface. On client processes -- the ones where you create the
Compiler object -- you can configure python loggers like normal before
submitting tasks and have the entire system honor that configuration.
"""
from __future__ import annotations

import faulthandler
import os
from typing import Any
from typing import Callable
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bqskit.runtime.future import RuntimeFuture


# Enable low-level fault handling: system crashes print a minimal trace.
faulthandler.enable()
os.environ['RUST_BACKTRACE'] = '1'


# Control multi-threading in BLAS libraries.
def set_blas_thread_counts(i: int = 1) -> None:
    """
    Control number of threads used by numpy and others.

    Must be called before any numpy or other BLAS libraries are loaded.
    """
    os.environ['OMP_NUM_THREADS'] = str(i)
    os.environ['OPENBLAS_NUM_THREADS'] = str(i)
    os.environ['MKL_NUM_THREADS'] = str(i)
    os.environ['NUMEXPR_NUM_THREADS'] = str(i)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(i)


class RuntimeHandle(Protocol):
    """
    A structural type capturing BQSKit Runtime capabilities.

    A compiler pass designer can effortlessly parallelize certain aspects
    of compilation using the :func:`submit` and :func:`map` methods.
    These both will return :class:`~bqskit.runtime.future.RuntimeFuture`
    objects, which can be awaited on using the standard python `await`
    keyword. This can also be used to cancel tasks and wait on specific
    events.

    This should never be constructed directly and only accessed through
    the :func:`get_runtime` function, since this is a structural type
    definition and does not contain any implementation.
    """

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_name: str | None = None,
        log_context: dict[str, str] = {},
        **kwargs: Any,
    ) -> RuntimeFuture:
        """
        Submit a function to the runtime for execution.

        This method schedules the function `fn` to be executed by the
        runtime with the provided arguments `args` and keyword arguments
        `kwargs`. The execution may happen asynchronously.

        Args:
            fn (Callable[..., Any]): The function to be executed.

            *args (Any): Variable length argument list to be passed to
                the function `fn`.

            task_name (str | None): An optional name for the task, which
                can be used for logging or tracking purposes. Defaults to
                None, which will use the function name as the task name.

            log_context (dict[str, str]): A dictionary containing logging
                context information. All log messages produced by the fn
                and any children tasks will contain this context if the
                appropriate level (logging.DEBUG) is set on the logger.
                Defaults to an empty dictionary for no added context.

            **kwargs (Any): Arbitrary keyword arguments to be passed to
                the function `fn`.

        Returns:
            RuntimeFuture: An object representing the future result of
            the function execution. This can be used to retrieve the
            result by `await`ing it.

        Example:
            >>> from bqskit.runtime import get_runtime
            >>>
            >>> def add(x, y):
            ...     return x + y
            >>>
            >>> future = get_runtime().submit(add, 1, 2)
            >>> result = await future
            >>> print(result)
            3

        See Also:
            - :func:`map` for submitting multiple tasks in parallel.
            - :func:`cancel` for cancelling tasks.
            - :class:`~bqskit.runtime.future.RuntimeFuture` for more
                information on how to interact with the future object.
        """
        ...

    def map(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_name: Sequence[str | None] | str | None = None,
        log_context: Sequence[dict[str, str]] | dict[str, str] = {},
        **kwargs: Any,
    ) -> RuntimeFuture:
        """
        Map a function over a sequence of arguments and execute in parallel.

        This method schedules the function `fn` to be executed by the runtime
        for each set of arguments provided in `args`. Each invocation of `fn`
        will be executed potentially in parallel, depending on the runtime's
        capabilities and current load.

        Args:
            fn (Callable[..., Any]): The function to be executed.

            *args (Any): Variable length argument list to be passed to
                the function `fn`. Each argument is expected to be a
                sequence of arguments to be passed to a separate
                invocation. The sequences should be of equal length.

            task_name (Sequence[str | None] | str | None): An optional
                name for the task group, which can be used for logging
                or tracking purposes. Defaults to None, which will use
                the function name as the task name. If a string is
                provided, it will be used as the prefix for all task
                names. If a sequence of strings is provided, each task
                will be named with the corresponding string in the
                sequence.

            log_context (Sequence[dict[str, str]]) | dict[str, str]): A
                dictionary containing logging context information. All
                log messages produced by the `fn` and any children tasks
                will contain this context if the appropriate level
                (logging.DEBUG) is set on the logger. Defaults to an
                empty dictionary for no added context. Can be a sequence
                of contexts, one for each task, or a single context to be
                used for all tasks.

            **kwargs (Any): Arbitrary keyword arguments to be passed to
                each invocation of the function `fn`.

        Returns:
            RuntimeFuture: An object representing the future result of
            the function executions. This can be used to retrieve the
            results by `await`ing it, which will return a list.

        Example:
            >>> from bqskit.runtime import get_runtime
            >>>
            >>> def add(x, y):
            ...     return x + y
            >>>
            >>> args_list = [(1, 2, 3), (4, 5, 6)]
            >>> future = get_runtime().map(add, *args_list)
            >>> results = await future
            >>> print(results)
            [5, 7, 9]

        See Also:
            - :func:`submit` for submitting a single task.
            - :func:`cancel` for cancelling tasks.
            - :func:`next` for retrieving results incrementally.
            - :class:`~bqskit.runtime.future.RuntimeFuture` for more
                information on how to interact with the future object.
        """
        ...

    def cancel(self, future: RuntimeFuture) -> None:
        """
        Cancel all tasks associated with `future`.

        This is asynchronous and non-pre-emptive. The cancel will return
        immediately, the future it represents will become dead, and at some
        point in the future, all tasks and data associated with the cancelled
        one will be removed from the system.
        """
        ...

    def get_cache(self) -> dict[str, Any]:
        """
        Retrieve worker's local cache.

        In situations where a large or non-easily serializable object is
        needed during the execution of a custom pass, `get_cache` can be used
        to load and store data. Objects placed in cache are transient. Objects
        can be placed in cache to reduce reloading, but should not be expected
        to remain in cache forever.

        For example, say some CustomPassA needs a SomeBigScaryObject. Within
        its run() method, the object is constructed, then placed into cache.
        A later pass, CustomPassB, also needs to use SomeBigScaryObject. It
        checks that the object has been loaded by CustomPassA, then uses it.

        ```
            # Load helper method
            def load_or_retrieve_big_scary_object(file):
                worker_cache = get_runtime().get_cache()
                if 'big_scary_object' not in worker_cache:
                    # Expensive io operation...
                    big_scary_object = load_big_scary_object(file)
                    worker_cache['big_scary_object'] = big_scary_object
                big_scary_object = worker_cache['big_scary_object']
                return big_scary_object

            # In CustomPassA's .run() definition...
            # CustomPassA performs the load.
            big_scary_object = load_or_retrieve_big_scary_object(file)
            big_scary_object.use(...)

            # In CustomPassB's .run() definition...
            # CustomPassB saves time by retrieving big_scary_object from
            # cache after CustomPassA has loaded it.
            big_scary_object = load_or_retrieve_big_scary_object(file)
            unpicklable_object.use(...)
        ```
        """
        ...

    async def next(self, future: RuntimeFuture) -> list[tuple[int, Any]]:
        """
        Wait for and return the next batch of results from a map task.

        Returns:
            (list[tuple[int, Any]]): A list of the results that arrived
                while the task was waiting. Each result is paired with
                the index of its arguments in the original map call.
        """
        ...


def get_runtime() -> RuntimeHandle:
    """
    Return a handle on the active runtime.

    See :class:`RuntimeHandle` for more info.
    """
    from bqskit.runtime.worker import get_worker
    return get_worker()


default_server_port = 7472
default_manager_port = 7473
default_worker_port = 7474
__all__ = ['get_runtime']
