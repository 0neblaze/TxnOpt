from __future__ import annotations

import sys

from txnopt_evidence.level1_calibration import _run


def test_calibration_command_observes_peak_rss() -> None:
    payload, elapsed, peak_rss_bytes = _run(
        [
            sys.executable,
            "-c",
            "import json; data=bytearray(8_000_000); print(json.dumps({'ok': bool(data)}))",
        ]
    )

    assert payload == {"ok": True}
    assert elapsed > 0.0
    assert peak_rss_bytes >= 8_000_000
