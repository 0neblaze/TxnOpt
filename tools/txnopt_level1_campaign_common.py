"""Compatibility wrapper for the packaged Level 1 campaign contract.

The implementation lives in :mod:`txnopt_evidence.level1_campaign_common` so
that installed wheels do not import from the repository's ``tools`` tree.
This module remains a deliberately thin source-compatibility shim for existing
orchestration commands and tests.
"""

import sys as _sys

from txnopt_evidence import level1_campaign_common as _implementation

# Make legacy imports and monkeypatch targets resolve against the packaged
# implementation itself (including private helpers used by old tests).
_sys.modules[__name__] = _implementation
