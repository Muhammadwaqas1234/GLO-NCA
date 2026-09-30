

class Data_Container():
    r"""In-memory store of preprocessed samples."""

    def __init__(self):
        r"""Create an empty store."""
        self.data = {}

    def get_data(self, key):
        r"""Return a stored sample, or False when absent."""
        if key in self.data:
            return self.data[key]
        else:
            return False

    def set_data(self, key, data):
        r"""Store a processed sample."""
        self.data[key] = data