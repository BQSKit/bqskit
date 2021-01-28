import uuid

class CompilationTask():

    def __init__ ( self ):
        self.task_id = uuid.uuid4()

    # TODO:
    # def __init__(self, input_file, output_format = "QASM",
    #              passes = [], **kwargs):
    #     self.input_file = input_file
    #     self.output_format = output_format
    #     self.passes = []
    #     self.configuration = kwargs
