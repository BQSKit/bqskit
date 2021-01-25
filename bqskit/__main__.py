import argparse


# Example workflow
# import bqskit

# task = bqskit.CompilationTask(...)
# bqskit.submit( task )
# bqskit.status( task ) Maybe default bqskit backend 
# bqskit.result( task ) as static global

# backend = bqskit.Backend() # default local machine... spawns thread
# backend = bqskit.Backend("ip address") # connection other machine in future

#backend.submit( task ) <---
#task.submit( backend )

# ...

# if backend.status(task) == DONE:
    # result = backend.result(task)
    # do stuff

