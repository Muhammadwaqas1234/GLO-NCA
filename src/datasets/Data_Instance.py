from collections import OrderedDict
import os


class Data_Container():
    r"""Bounded in-memory store for preprocessed volumes.

    This was an unbounded dict: every case loaded in a process stayed resident
    for the lifetime of that process. At 128^3 a case costs ~56 MB (4 modality
    volumes + 3 region masks, float32), so 898 training cases need ~49 GB and
    each DataLoader worker keeps its OWN copy. A worker was OOM-killed at
    16.3 GB RSS after roughly 298 cases.

    The store is now an LRU capped at ``GLO_RAM_CACHE_CASES`` entries
    (default 24, about 1.3 GB at 128^3). Durable reuse is the job of the
    on-disk preprocessing cache, which already exists and survives across
    epochs and processes; this layer only avoids repeated work within a short
    window. Set the variable to 0 to disable in-memory retention entirely.
    """

    DEFAULT_MAX_ENTRIES = 24

    def __init__(self, max_entries: int = None):
        if max_entries is None:
            try:
                max_entries = int(os.environ.get("GLO_RAM_CACHE_CASES",
                                                 self.DEFAULT_MAX_ENTRIES))
            except ValueError:
                max_entries = self.DEFAULT_MAX_ENTRIES
        self.max_entries = max(0, int(max_entries))
        self.data = OrderedDict()

    def get_data(self, key):
        r"""Return previously processed data, or False when absent.

        Returning False (not None) preserves the original contract: callers
        test the result for truthiness.
        """
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return False

    def set_data(self, key, data):
        r"""Store data, evicting the least recently used entry when full."""
        if self.max_entries == 0:
            return
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = data
        while len(self.data) > self.max_entries:
            self.data.popitem(last=False)
