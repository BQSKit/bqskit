import argparse

from bqskit import Backend
from bqskit import CompilationTask

# Simple Test
backend = Backend()
task1 = CompilationTask()
task2 = CompilationTask()
backend.submit(task1)
backend.submit(task2)
print(backend.status(task1))
print(backend.status(task2))
print(backend.result(task1))
print(backend.result(task2))
backend.close()

# Example workflow
# import bqskit

# task = bqskit.CompilationTask(...)
# bqskit.submit( task )
# bqskit.status( task ) Maybe default bqskit backend
# bqskit.result( task ) as static global

# backend = bqskit.Backend() # default local machine... spawns thread
# backend = bqskit.Backend("ip address") # connection other machine in future

# backend.submit( task ) <---
#task.submit( backend )

# ...

# if backend.status(task) == DONE:
# result = backend.result(task)
# do stuff
