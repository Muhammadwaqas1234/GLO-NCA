from collections import OrderedDict
import os


class Data_Container():
    r"""Bounded LRU store for preprocessed volumes (per process).

    Capped at GLO_RAM_CACHE_CASES entries (default 24, about 1.3 GB at 128³; 0 disables it).
    An unbounded dict OOM-killed workers; durable reuse is the on-disk preprocessing cache.
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
        r"""Return stored data, or False when absent (callers test truthiness)."""
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
