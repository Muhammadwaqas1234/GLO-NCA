#!/usr/bin/env python
r"""Run glo_nca.cli.analyze from a checkout; after `pip install -e .` the same tool is `glo-analyze`."""
import sys

from glo_nca.cli.analyze import main

if __name__ == "__main__":
    sys.exit(main())
