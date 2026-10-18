#!/usr/bin/env python
r"""Run glo_nca.cli.evaluate from a checkout; after `pip install -e .` the same tool is `glo-evaluate`."""
import sys

from glo_nca.cli.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
