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
Then you can start the server with a comma seperated list of all managers
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

Parallelize Pass Computation
############################

When developing a BQSKit compiler pass, you can hook into the active
runtime server through the :func:`get_runtime` function. This returns a
:class:`RuntimeHandle`, which you can use to submit, map, wait on, and
cancel tasks in the execution environment.

For more information on how to design a custom pass, see this (TODO, sorry,
you can look at the source code of existing
`passes <https://github.com/BQSKit/bqskit/blob/master/bqskit/passes>`_
for a good example for the time being).

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
from typing import TYPE_CHECKING

# Enable low-level fault handling: system crashes print a minimal trace.
faulthandler.enable()


# Disable multi-threading in BLAS libraries.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['RUST_BACKTRACE'] = '1'


if TYPE_CHECKING:
    from bqskit.runtime.future import RuntimeFuture


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
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Submit a `fn` to the runtime."""
        ...

    def map(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> RuntimeFuture:
        """Map `fn` over the input arguments distributed across the runtime."""
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
__all__ = ['get_runtime']
