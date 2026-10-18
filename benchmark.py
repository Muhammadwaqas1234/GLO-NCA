#!/usr/bin/env python
r"""Run glo_nca.cli.benchmark from a checkout; after `pip install -e .` the same tool is `glo-benchmark`."""
import sys

from glo_nca.cli.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
