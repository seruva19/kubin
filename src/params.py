from models.kandinsky import KandinskyCheckpoint


class KubinParams:
    def __init__(self, args):
        self.args = args
        self.checkpoint = KandinskyCheckpoint()

    def __getattr__(self, key):
        return getattr(self.args, key)
