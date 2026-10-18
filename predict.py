#!/usr/bin/env python
r"""Run glo_nca.cli.predict from a checkout; after `pip install -e .` the same tool is `glo-predict`."""
import sys

from glo_nca.cli.predict import main

if __name__ == "__main__":
    sys.exit(main())
