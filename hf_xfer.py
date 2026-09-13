#!/usr/bin/env python3
"""Shim: the implementation lives in hfhub.cli (import/export delegate to hfhub.xfer)."""
import sys
from hfhub.cli import xfer_main

if __name__ == "__main__":
    sys.exit(xfer_main())
