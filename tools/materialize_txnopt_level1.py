"""Create the local Level 1 run plan; this tool never executes a run."""

from __future__ import annotations

import argparse
from pathlib import Path

from txnopt_evidence.campaign import materialize_level1_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--raw-output-root", type=Path, required=True)
    parser.add_argument("--fixed-work", type=int, required=True)
    parser.add_argument("--fixed-time-seconds", type=float, required=True)
    parser.add_argument("--max-rounds", type=int, required=True)
    parser.add_argument("--evrptw-max-candidates", type=int, required=True)
    parser.add_argument("--rcpsp-max-candidates", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    arguments = parser.parse_args()
    manifest = materialize_level1_plan(
        arguments.protocol,
        arguments.catalog,
        destination=arguments.destination,
        raw_output_root=arguments.raw_output_root,
        fixed_work=arguments.fixed_work,
        fixed_time_seconds=arguments.fixed_time_seconds,
        max_rounds=arguments.max_rounds,
        evrptw_max_candidates=arguments.evrptw_max_candidates,
        rcpsp_max_candidates=arguments.rcpsp_max_candidates,
        attempt=arguments.attempt,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
