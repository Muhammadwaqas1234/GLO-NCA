#!/usr/bin/env python
r"""Run glo_nca.cli.make_figures from a checkout; after `pip install -e .` the same tool is `glo-figures`."""
import sys

from glo_nca.cli.make_figures import main

if __name__ == "__main__":
    sys.exit(main())
