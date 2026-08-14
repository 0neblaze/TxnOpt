"""Compatibility entry point for the packaged Level 1 campaign runner."""

import sys as _sys

from txnopt_evidence import level1_campaign_runner as _implementation

_sys.modules[__name__] = _implementation
main = _implementation.main


if __name__ == "__main__":
    raise SystemExit(main())
