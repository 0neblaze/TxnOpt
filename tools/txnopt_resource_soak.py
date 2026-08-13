"""Run the TxnOpt native-round resource soak and write a signed receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from txnopt_evidence.resource_soak import write_resource_soak_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-rounds", type=int, default=1_000)
    parser.add_argument("--maximum-rss-growth-kib", type=int, default=65_536)
    arguments = parser.parse_args()
    digest = write_resource_soak_receipt(
        arguments.output,
        rounds=arguments.rounds,
        workers=arguments.workers,
        warmup_rounds=arguments.warmup_rounds,
        maximum_rss_growth_kib=arguments.maximum_rss_growth_kib,
    )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
