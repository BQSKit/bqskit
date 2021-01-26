from bqskit import qis


class FixedGate(Gate):

    def get_unitary(self):
        if not self.utry:
            raise Exception

        return self.utry
