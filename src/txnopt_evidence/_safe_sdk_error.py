"""Fail-closed formatting for errors from credential-bearing SDK boundaries."""

from __future__ import annotations


def safe_sdk_error_text(
    *,
    provider: str,
    operation: str,
    error: Exception,
) -> str:
    """Return diagnostics that never include an untrusted SDK message body.

    Cloud SDKs may interpolate temporary CAM-role credentials into ``str(error)``.
    Those values are not necessarily present in the process environment and cannot
    be exhaustively redacted.  The safe boundary therefore preserves only locally
    supplied context and the exception's type identity.
    """

    category = f"{type(error).__module__}.{type(error).__qualname__}"
    return f"{provider} SDK {operation} failed [{category}]"


__all__ = ["safe_sdk_error_text"]
