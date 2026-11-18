"""Entry point for `python -m ctinyml ...` — delegates to tools/cli.py."""

import sys

from tools.cli import main

if __name__ == "__main__":
    sys.exit(main())
