#!/usr/bin/env python
r"""Run glo_nca.cli.train from a checkout; after `pip install -e .` the same tool is `glo-train`."""
import sys

from glo_nca.cli.train import main

if __name__ == "__main__":
    sys.exit(main())
