"""Best-known solution values for the Schneider E-VRPTW benchmark.

This module provides the canonical BKS (best-known solution) data for all 92
Schneider et al. (2014) benchmark instances, along with a model compatibility
assessment comparing published objectives with our lexicographic
``(vehicle_count, total_distance, total_charging_time, charging_count)``
objective.

Data sources
------------
* Schneider et al. (2014), Table 5 (page 511):
  CPLEX optimal / best-upper-bound for 36 small instances (5/10/15 customers).
  VNS/TS results are also from the same paper; for RC204-15 the VNS/TS solution
  (384.86) improves on the CPLEX upper bound (407.45).

* Keskin & Çatay (2016), Table 2 (page 122):
  Updated BKS for all 56 large instances (100 customers), citing the original
  source (SSG, GS, or HPH) for each value.  These are the most up-to-date
  full-recharge BKS values as of 2016.

Model compatibility
--------------------
Published papers minimise a 2-component lexicographic objective
``(vehicle_count, total_distance)`` or a weighted sum
``fixed_cost * vehicle_count + total_distance`` (HPH).
Our model uses a 4-component lexicographic objective
``(vehicle_count, total_distance, total_charging_time, charging_count)``.
The objective mismatch means **no gap may be computed**; results are listed
separately as model-incompatible.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Source references
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Citation info for a BKS source paper."""

    abbreviation: str
    authors: str
    year: int
    title: str
    journal: str
    doi: str
    in_vor_collection: bool


SOURCE_REFERENCES: dict[str, SourceReference] = {
    "SSG": SourceReference(
        abbreviation="SSG",
        authors="Schneider, Stenger, Goeke",
        year=2014,
        title=(
            "The Electric Vehicle-Routing Problem with Time Windows and "
            "Recharging Stations"
        ),
        journal="Transportation Science",
        doi="10.1287/trsc.2013.0490",
        in_vor_collection=True,
    ),
    "GS": SourceReference(
        abbreviation="GS",
        authors="Goeke, Schneider",
        year=2015,
        title="Routing a mixed fleet of electric and conventional vehicles",
        journal="European Journal of Operational Research",
        doi="10.1016/j.ejor.2015.01.049",
        in_vor_collection=False,
    ),
    "HPH": SourceReference(
        abbreviation="HPH",
        authors="Hiermann, Puchinger, Ropke, Hartl",
        year=2016,
        title=(
            "The Electric Fleet Size and Mix Vehicle Routing Problem with "
            "Time Windows and Recharging Stations"
        ),
        journal="European Journal of Operational Research",
        doi="10.1016/j.ejor.2016.01.038",
        in_vor_collection=True,
    ),
    "KC": SourceReference(
        abbreviation="KC",
        authors="Keskin, Catay",
        year=2016,
        title=(
            "Partial Recharge Strategies for the Electric Vehicle Routing "
            "Problem with Time Windows"
        ),
        journal="Transportation Research Part C",
        doi="10.1016/j.trc.2016.01.013",
        in_vor_collection=True,
    ),
}


# ---------------------------------------------------------------------------
# BKS record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BKSRecord:
    """Best-known solution for one Schneider benchmark instance.

    ``bks_charging_time`` and ``bks_charging_count`` are always ``None``
    because published BKS tables report only vehicle count and total distance.
    """

    instance: str
    paper_name: str
    customer_count: int
    class_name: str
    bks_vehicles: int
    bks_distance: float
    source_ref: str
    compilation_ref: str
    source_table: str
    optimal_proven: bool
    bks_charging_time: float | None = None
    bks_charging_count: int | None = None


def _small(
    instance: str,
    paper_name: str,
    class_name: str,
    vehicles: int,
    distance: float,
    *,
    optimal: bool,
) -> BKSRecord:
    """Construct a small-instance BKSRecord from Schneider Table 5."""

    count = int(instance.rsplit("C", maxsplit=1)[-1])
    return BKSRecord(
        instance=instance,
        paper_name=paper_name,
        customer_count=count,
        class_name=class_name,
        bks_vehicles=vehicles,
        bks_distance=distance,
        source_ref="SSG",
        compilation_ref="SSG",
        source_table="Table 5",
        optimal_proven=optimal,
    )


def _large(
    paper_name: str,
    class_name: str,
    vehicles: int,
    distance: float,
    source_ref: str,
) -> BKSRecord:
    """Construct a large-instance BKSRecord from Keskin & Catay Table 2."""

    return BKSRecord(
        instance=f"{paper_name}_21",
        paper_name=paper_name,
        customer_count=100,
        class_name=class_name,
        bks_vehicles=vehicles,
        bks_distance=distance,
        source_ref=source_ref,
        compilation_ref="KC",
        source_table="Table 2",
        optimal_proven=False,
    )


# ---------------------------------------------------------------------------
# 36 small instances — Schneider et al. (2014) Table 5 (CPLEX column)
# ---------------------------------------------------------------------------

_SMALL_INSTANCES: tuple[BKSRecord, ...] = (
    # 5-customer (12 instances)
    _small("c101C5", "C101-5", "C1", 2, 257.75, optimal=True),
    _small("c103C5", "C103-5", "C1", 1, 176.05, optimal=True),
    _small("c206C5", "C206-5", "C2", 1, 242.55, optimal=True),
    _small("c208C5", "C208-5", "C2", 1, 158.48, optimal=True),
    _small("r104C5", "R104-5", "R1", 2, 136.69, optimal=True),
    _small("r105C5", "R105-5", "R1", 2, 156.08, optimal=True),
    _small("r202C5", "R202-5", "R2", 1, 128.78, optimal=True),
    _small("r203C5", "R203-5", "R2", 1, 179.06, optimal=True),
    _small("rc105C5", "RC105-5", "RC1", 2, 241.30, optimal=True),
    _small("rc108C5", "RC108-5", "RC1", 1, 253.93, optimal=True),
    _small("rc204C5", "RC204-5", "RC2", 1, 176.39, optimal=True),
    _small("rc208C5", "RC208-5", "RC2", 1, 167.98, optimal=True),
    # 10-customer (12 instances)
    _small("c101C10", "C101-10", "C1", 3, 393.76, optimal=True),
    _small("c104C10", "C104-10", "C1", 2, 273.93, optimal=True),
    _small("c202C10", "C202-10", "C2", 1, 304.06, optimal=True),
    _small("c205C10", "C205-10", "C2", 2, 228.28, optimal=True),
    _small("r102C10", "R102-10", "R1", 3, 249.19, optimal=True),
    _small("r103C10", "R103-10", "R1", 2, 207.05, optimal=True),
    _small("r201C10", "R201-10", "R2", 1, 241.51, optimal=True),
    _small("r203C10", "R203-10", "R2", 1, 218.21, optimal=True),
    _small("rc102C10", "RC102-10", "RC1", 4, 423.51, optimal=True),
    _small("rc108C10", "RC108-10", "RC1", 3, 345.93, optimal=True),
    _small("rc201C10", "RC201-10", "RC2", 1, 412.86, optimal=False),
    _small("rc205C10", "RC205-10", "RC2", 2, 325.98, optimal=True),
    # 15-customer (12 instances)
    _small("c103C15", "C103-15", "C1", 3, 384.29, optimal=False),
    _small("c106C15", "C106-15", "C1", 3, 275.13, optimal=True),
    _small("c202C15", "C202-15", "C2", 2, 383.62, optimal=False),
    _small("c208C15", "C208-15", "C2", 2, 300.55, optimal=False),
    _small("r102C15", "R102-15", "R1", 5, 413.93, optimal=False),
    _small("r105C15", "R105-15", "R1", 4, 336.15, optimal=False),
    _small("r202C15", "R202-15", "R2", 2, 358.00, optimal=False),
    _small("r209C15", "R209-15", "R2", 1, 313.24, optimal=False),
    _small("rc103C15", "RC103-15", "RC1", 4, 397.67, optimal=False),
    _small("rc108C15", "RC108-15", "RC1", 3, 370.25, optimal=False),
    _small("rc202C15", "RC202-15", "RC2", 2, 394.39, optimal=False),
    # RC204-15: VNS/TS found 384.86, better than CPLEX upper bound 407.45
    BKSRecord(
        instance="rc204C15",
        paper_name="RC204-15",
        customer_count=15,
        class_name="RC2",
        bks_vehicles=1,
        bks_distance=384.86,
        source_ref="SSG",
        compilation_ref="SSG",
        source_table="Table 5 (VNS/TS)",
        optimal_proven=False,
    ),
)


# ---------------------------------------------------------------------------
# 56 large instances — Keskin & Catay (2016) Table 2
# ---------------------------------------------------------------------------

_LARGE_INSTANCES: tuple[BKSRecord, ...] = (
    # C1 class (9 instances)
    _large("c101", "C1", 12, 1053.83, "SSG"),
    _large("c102", "C1", 11, 1051.38, "GS"),
    _large("c103", "C1", 10, 1034.86, "GS"),
    _large("c104", "C1", 10, 961.88, "GS"),
    _large("c105", "C1", 11, 1075.37, "SSG"),
    _large("c106", "C1", 11, 1057.65, "HPH"),
    _large("c107", "C1", 11, 1031.56, "SSG"),
    _large("c108", "C1", 10, 1095.66, "GS"),
    _large("c109", "C1", 10, 1033.67, "GS"),
    # C2 class (8 instances)
    _large("c201", "C2", 4, 645.16, "SSG"),
    _large("c202", "C2", 4, 645.16, "SSG"),
    _large("c203", "C2", 4, 644.98, "SSG"),
    _large("c204", "C2", 4, 636.43, "SSG"),
    _large("c205", "C2", 4, 641.13, "SSG"),
    _large("c206", "C2", 4, 638.17, "SSG"),
    _large("c207", "C2", 4, 638.17, "SSG"),
    _large("c208", "C2", 4, 638.17, "SSG"),
    # R1 class (12 instances)
    _large("r101", "R1", 18, 1663.04, "HPH"),
    _large("r102", "R1", 16, 1487.41, "GS"),
    _large("r103", "R1", 13, 1271.35, "GS"),
    _large("r104", "R1", 11, 1088.43, "SSG"),
    _large("r105", "R1", 14, 1442.35, "GS"),
    _large("r106", "R1", 13, 1324.10, "GS"),
    _large("r107", "R1", 12, 1150.95, "GS"),
    _large("r108", "R1", 11, 1050.04, "SSG"),
    _large("r109", "R1", 12, 1261.31, "GS"),
    _large("r110", "R1", 11, 1119.50, "GS"),
    _large("r111", "R1", 12, 1106.19, "SSG"),
    _large("r112", "R1", 11, 1016.63, "GS"),
    # R2 class (11 instances)
    _large("r201", "R2", 3, 1264.82, "SSG"),
    _large("r202", "R2", 3, 1052.32, "SSG"),
    _large("r203", "R2", 3, 895.54, "GS"),
    _large("r204", "R2", 2, 779.49, "GS"),
    _large("r205", "R2", 3, 987.36, "GS"),
    _large("r206", "R2", 3, 922.19, "GS"),
    _large("r207", "R2", 2, 845.26, "GS"),
    _large("r208", "R2", 2, 736.12, "GS"),
    _large("r209", "R2", 3, 867.05, "GS"),
    _large("r210", "R2", 3, 846.20, "GS"),
    _large("r211", "R2", 2, 827.89, "GS"),
    # RC1 class (8 instances)
    _large("rc101", "RC1", 16, 1726.91, "HPH"),
    _large("rc102", "RC1", 14, 1552.08, "HPH"),
    _large("rc103", "RC1", 13, 1350.09, "GS"),
    _large("rc104", "RC1", 11, 1227.25, "GS"),
    _large("rc105", "RC1", 14, 1475.31, "HPH"),
    _large("rc106", "RC1", 13, 1427.21, "GS"),
    _large("rc107", "RC1", 12, 1274.89, "SSG"),
    _large("rc108", "RC1", 11, 1197.83, "GS"),
    # RC2 class (8 instances)
    _large("rc201", "RC2", 4, 1444.94, "SSG"),
    _large("rc202", "RC2", 3, 1410.74, "GS"),
    _large("rc203", "RC2", 3, 1055.19, "GS"),
    _large("rc204", "RC2", 3, 884.80, "GS"),
    _large("rc205", "RC2", 3, 1273.55, "GS"),
    _large("rc206", "RC2", 3, 1188.63, "GS"),
    _large("rc207", "RC2", 3, 985.03, "GS"),
    _large("rc208", "RC2", 3, 836.29, "GS"),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

BEST_KNOWN_VALUES: tuple[BKSRecord, ...] = _SMALL_INSTANCES + _LARGE_INSTANCES

_BKS_INDEX: dict[str, BKSRecord] = {rec.instance: rec for rec in BEST_KNOWN_VALUES}

TOTAL_INSTANCES = len(BEST_KNOWN_VALUES)


def get_bks(instance_name: str) -> BKSRecord | None:
    """Return the BKS record for *instance_name*, or ``None`` if not found."""

    return _BKS_INDEX.get(instance_name)


def get_all_bks() -> tuple[BKSRecord, ...]:
    """Return all 92 BKS records."""

    return BEST_KNOWN_VALUES


# ---------------------------------------------------------------------------
# Model compatibility assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompatibilityDimension:
    """One dimension of model compatibility comparison."""

    dimension: str
    our_model: str
    published_model: str
    compatible: bool
    notes: str


@dataclass(frozen=True, slots=True)
class CompatibilityAssessment:
    """Full model compatibility assessment."""

    dimensions: tuple[CompatibilityDimension, ...]
    overall_compatible: bool
    summary: str


COMPATIBILITY_ASSESSMENT: CompatibilityAssessment = CompatibilityAssessment(
    dimensions=(
        CompatibilityDimension(
            dimension="charging_model",
            our_model="full_recharge: t = (Q - b) * g",
            published_model="full_recharge (SSG, GS, HPH)",
            compatible=True,
            notes=(
                "All three source papers use full recharge with linear "
                "charging function.  Our model matches."
            ),
        ),
        CompatibilityDimension(
            dimension="objective_function",
            our_model=(
                "lexicographic(vehicle_count, total_distance, "
                "total_charging_time, charging_count)"
            ),
            published_model=(
                "SSG/GS: lexicographic(vehicle_count, total_distance); "
                "HPH: weighted_sum(fixed_cost * vehicle_count + "
                "total_distance)"
            ),
            compatible=False,
            notes=(
                "Our 4-component lexicographic objective adds "
                "total_charging_time and charging_count as tertiary and "
                "quaternary criteria not present in any published "
                "objective.  HPH uses a weighted sum rather than strict "
                "lexicographic ordering.  Gap computation is not permitted."
            ),
        ),
        CompatibilityDimension(
            dimension="distance_metric",
            our_model="unrounded Euclidean (math.hypot)",
            published_model="Euclidean, Solomon convention (possibly rounded)",
            compatible=False,
            notes=(
                "Solomon-derivative instances traditionally compute Euclidean "
                "distances with truncation or rounding to two decimal "
                "places.  Our solver uses math.hypot without rounding.  "
                "Even small rounding differences can cause distance values "
                "to disagree at the last decimal, making direct comparison "
                "unreliable."
            ),
        ),
        CompatibilityDimension(
            dimension="vehicle_parameters",
            our_model="Schneider benchmark instances (unchanged)",
            published_model="Schneider benchmark instances (unchanged)",
            compatible=True,
            notes=(
                "All sources use the same 92 instance files from "
                "Schneider et al. (2014) with identical vehicle battery "
                "capacity, load capacity, consumption rate, and inverse "
                "refueling rate."
            ),
        ),
        CompatibilityDimension(
            dimension="time_windows",
            our_model="Schneider-rewritten Solomon time windows (unchanged)",
            published_model="Schneider-rewritten Solomon time windows (unchanged)",
            compatible=True,
            notes=(
                "All sources use the same time windows as defined in the "
                "Schneider benchmark instance files."
            ),
        ),
    ),
    overall_compatible=False,
    summary=(
        "Model is NOT fully compatible.  The objective function and "
        "distance metric differ.  Per roadmap policy, no gap is computed.  "
        "BKS values are listed as reference only."
    ),
)


def assess_compatibility() -> CompatibilityAssessment:
    """Return the model compatibility assessment."""

    return COMPATIBILITY_ASSESSMENT
