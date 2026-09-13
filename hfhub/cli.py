"""Console entry points for hfu and hf-xfer."""
import sys

from hfhub import hfu as _hfu
from hfhub import xfer as _xfer


def hfu_main() -> None:
    _hfu.main()


def xfer_main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _xfer.main(argv)
