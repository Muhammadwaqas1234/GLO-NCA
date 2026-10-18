#!/usr/bin/env python
r"""Run glo_nca.cli.compare_runs from a checkout; after `pip install -e .` the same tool is `glo-compare`."""
import sys

from glo_nca.cli.compare_runs import main

if __name__ == "__main__":
    sys.exit(main())
