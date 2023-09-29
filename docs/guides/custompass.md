# Design a Custom Pass

Designing a custom pass in BQSKit is easy and allows you to completely
tailor a compilation workflow to your specific needs. As a reminder,
we happily accept contributions; if you feel that others could benefit
from a pass you have designed, please make a Pull Request on our
[Github page](github.com/bqskit/bqskit). We are happy to help get your
code out to other users if needed.

## Basic Idea

Every pass in BQSKit inherits from [`BasePass`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.compiler.BasePass.html#bqskit.compiler.BasePass) and implements the async [`run()`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.compiler.BasePass.run.html#bqskit.compiler.BasePass.run) method. The run method will take as input
a circuit (and the PassData, but we will get to that later) and either analyze
it, modify it, or both. This method should be entirely self-contained without
side-effects to the class, meaning the Pass object (`self`) should only be read
from and not written to. Any configuration for your Pass should be done
in an `__init__` method, and the Pass should be immutable afterward.

### Note about Importability

Since the BQSKit Runtime is built on top of Python's multiprocessing library,
which uses pickle, every object sent through the runtime must be pickle-able.
All code that is sent must be accessible and importable from all workers
in the Runtime. **Therefore, you cannot define a pass in the same Python
script (`__main__`) executed.** A simple workaround is to define your
pass in another module or file next to your script and import the pass.

## PassData

The [`PassData`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.compiler.PassData.html#bqskit.compiler.PassData) is an object wrapping a dictionary from strings to anything.
During the execution of a compilation workflow, one is created at the very
beginning with some default values, and this object is shared between all
the passes in the workflow. This allows the passes to communicate with one
another by reading from and writing to the PassData object. Some passes will
have features that can be accessed by leaving specific keys in the dictionary.

Some important reserved keys exist in the PassData, for example, the
`PassData.model`, which holds the current workflow's target [`MachineModel`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.compiler.MachineModel.html).

## Parallelization

Some passes may be very compute intensive and can benefit from parallelizing or
distributing their workload across the active Runtime. This can be done
by requesting a handle on the runtime and mapping work over it. For example,
the following code that performs some work under a loop:

```python
...
class MyPass(BasePass):
    async def run(circuit: Circuit, data: PassData) -> None:
        ...
        results = []
        for i in long_list:
            results.append(do_work(i))
        ...
```

can be parallelized to:

```python
...
from bqskit.runtime import get_runtime
...
class MyPass(BasePass):
    ...
    async def run(circuit: Circuit, data: PassData) -> None:
        ...
        results = await get_runtime().map(do_work, long_list)
        ...
```

There are other methods available, such as `cancel`, `next`, and `submit`.
See [`get_runtime`](https://bqskit.readthedocs.io/en/latest/source/autogen/bqskit.runtime.get_runtime.html#bqskit.runtime.get_runtime) for more info.
