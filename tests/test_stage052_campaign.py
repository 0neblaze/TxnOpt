from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from evrptw.stage052_campaign import (
    PILOT_INSTANCE_NAMES,
    PILOT_SEEDS,
    AcceptedGlobalBest,
    AnytimeCheckpoint,
    ArchiveTransferCompletedError,
    ArchiveTransferError,
    BatchArchiver,
    BatchManifest,
    BatchPlan,
    BenchmarkCampaignConfig,
    BenchmarkPreflightObservation,
    CampaignManifest,
    FileSystemArchiveIO,
    ManifestIntegrityError,
    PilotStorageObservation,
    ShardPlan,
    StorageRoot,
    StorageRootLocator,
    SystemLoadWindow,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    directory_file_count,
    load_campaign_manifest,
)


def _pilot_observations(
    *, small_bytes: int = 1_000, large_30_second_bytes: int = 2_000
) -> tuple[PilotStorageObservation, ...]:
    return tuple(
        PilotStorageObservation(
            family=family,
            customer_count=customer_count,
            budget_seconds=30,
            compressed_bytes=small_bytes,
        )
        for customer_count in (5, 10, 15)
        for family in ("C", "R", "RC")
    ) + tuple(
        PilotStorageObservation(
            family=family,
            customer_count=100,
            budget_seconds=30,
            compressed_bytes=large_30_second_bytes,
        )
        for family in ("C", "R", "RC")
    )


def test_campaign_objective_keys_require_shared_canonical_precision() -> None:
    with pytest.raises(ValueError, match="canonical objective precision"):
        AcceptedGlobalBest(
            completed_at_seconds=1.0,
            iteration=1,
            objective_key=(2, 100.0000000004, 1.0, 1),
        )


def test_formal_campaign_plan_has_exact_scope_order_and_pilot_estimates() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    )

    plan = config.build_plan(_pilot_observations())

    assert len(config.instances) == 92
    assert config.seeds == tuple(range(2014, 2024))
    assert len(plan.shards) == 920
    assert plan.axis_count == 2_040
    assert plan.declared_solver_seconds == 229_200
    assert plan.checkpoint_count == 10_400
    assert tuple(
        (shard.customer_count, shard.instance, shard.seed) for shard in plan.shards
    ) == tuple(sorted((shard.customer_count, shard.instance, shard.seed) for shard in plan.shards))
    assert {shard.estimated_bytes for shard in plan.shards if shard.customer_count < 100} == {1_500}
    assert {shard.estimated_bytes for shard in plan.shards if shard.customer_count == 100} == {
        39_000
    }
    assert {shard.max_iterations for shard in plan.shards if shard.customer_count < 100} == {1_000}
    assert {shard.max_iterations for shard in plan.shards if shard.customer_count == 100} == {None}
    assert config.to_dict()["storage_policy_version"] == "artifact-storage-v2"
    assert config.to_dict()["screening_schema_version"] == "screening_decisions_v3"
    assert config.to_dict()["selected_backend"] == "native_cpu"
    assert config.to_dict()["selected_exact_backend"] == "cpu_batch"


def test_pilot_campaign_uses_exact_stage0_scope_and_shared_manifest_contract(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    )

    plan = config.build_plan()

    assert config.scope == "pilot"
    assert config.seeds == PILOT_SEEDS
    assert tuple(item.instance for item in config.instances) == PILOT_INSTANCE_NAMES
    assert len(plan.shards) == 36
    assert plan.axis_count == 36
    assert plan.declared_solver_seconds == 1_080
    assert plan.checkpoint_count == 144
    assert plan.scope == "pilot"
    assert len(plan.batches) == 3
    assert all(shard.budgets_seconds == (30,) for shard in plan.shards)
    assert all(shard.scope == "pilot" for shard in plan.shards)
    assert all(shard.estimated_bytes == 2 * 1024**3 for shard in plan.shards)
    expected_customer_counts = {item.instance: item.customer_count for item in config.instances}
    assert tuple(
        (shard.customer_count, shard.instance, shard.seed) for shard in plan.shards
    ) == tuple(
        sorted(
            (expected_customer_counts[instance], instance, seed)
            for instance in PILOT_INSTANCE_NAMES
            for seed in PILOT_SEEDS
        )
    )

    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={
            "transfer_staging": 82 * 1024**3,
            "internal_archive": 200 * 1024**3 + plan.estimated_bytes,
        },
    )
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256="a" * 64,
        prerequisite_review_sha256="b" * 64,
    )

    assert campaign.scope == "pilot"
    assert campaign.to_dict()["scope"] == "pilot"
    assert CampaignManifest.from_dict(campaign.to_dict()) == campaign

    with pytest.raises(ValueError, match="formal campaign instance scope"):
        replace(config, scope="formal")


def test_campaign_contract_can_bind_an_independently_promoted_cuda_backend() -> None:
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="cuda",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )

    assert config.selected_backend == "cuda"
    assert config.to_dict()["selected_backend"] == "cuda"


def test_formal_campaign_uses_sequential_next_fit_without_splitting_shards() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )

    plan = config.build_plan(
        _pilot_observations(
            small_bytes=1_000_000_000,
            large_30_second_bytes=100_000_000,
        )
    )

    assert tuple(batch.batch_id for batch in plan.batches) == tuple(
        f"batch{ordinal:04d}" for ordinal in range(1, len(plan.batches) + 1)
    )
    assert tuple(shard.shard_id for batch in plan.batches for shard in batch.shards) == tuple(
        shard.shard_id for shard in plan.shards
    )
    assert len(plan.batches) > 1
    assert all(batch.estimated_bytes <= 24 * 1024**3 for batch in plan.batches)
    for batch, following in zip(plan.batches, plan.batches[1:], strict=False):
        assert batch.estimated_bytes + following.shards[0].estimated_bytes > 24 * 1024**3


def test_formal_campaign_storage_limits_cannot_be_weakened() -> None:
    with pytest.raises(ValueError, match="storage byte limits are fixed"):
        BenchmarkCampaignConfig.formal(
            run_label="stage05.2_benchmark_attempt02",
            staging_root_alias="transfer_staging",
            archive_root_aliases=("internal_archive",),
            selected_backend="native_cpu",
            selected_exact_backend="cpu_batch",
            selected_workers=2,
            native_profile="stage05.2-native-kernels-v1",
            batch_target_bytes=10_000,
        )


def test_formal_campaign_rejects_an_indivisible_shard_over_two_gib() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )
    observations = list(_pilot_observations())
    observations[-1] = PilotStorageObservation(
        family="RC",
        customer_count=100,
        budget_seconds=30,
        compressed_bytes=120_000_000,
    )

    with pytest.raises(ValueError, match="shard estimate exceeds 2 GiB"):
        config.build_plan(tuple(observations))


def test_shard_plan_encodes_small_iteration_cap_and_large_wall_clock_only() -> None:
    with pytest.raises(ValueError, match="small shard requires 1000 iterations"):
        ShardPlan(
            "shard0001",
            "c101C5",
            2014,
            5,
            "C",
            (30,),
            (1, 5, 10, 30, 60, 120, 300),
            1,
            None,
        )
    with pytest.raises(ValueError, match="large shard must be wall-clock only"):
        ShardPlan(
            "shard0001",
            "c101_21",
            2014,
            100,
            "C",
            (30, 60, 300),
            (1, 5, 10, 30, 60, 120, 300),
            1,
            1_000,
        )


def _root_locator(tmp_path: Path) -> StorageRootLocator:
    local = tmp_path / "stage052_storage_roots.local.toml"
    local.write_text(
        """
[roots.transfer_staging]
absolute_path = "/Volumes/TRANSFER/project/results"
device_uuid = "transfer-device"
filesystem = "exfat"

[roots.transfer_archive]
absolute_path = "/Volumes/TRANSFER/project/archive"
device_uuid = "transfer-device"
filesystem = "exfat"

[roots.internal_archive]
absolute_path = "/Users/example/project-results"
device_uuid = "internal-device"
filesystem = "apfs"
""".strip(),
        encoding="utf-8",
    )
    return StorageRootLocator.from_toml(local)


def test_storage_root_locator_keeps_absolute_paths_out_of_tracked_payload(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)

    assert locator.resolve("transfer_staging").absolute_path == Path(
        "/Volumes/TRANSFER/project/results"
    )
    tracked = locator.tracked_payload()
    encoded = json.dumps(tracked)
    assert "/Volumes/TRANSFER" not in encoded
    assert "/Users/example" not in encoded
    assert tracked["roots"]["internal_archive"] == {
        "device_uuid": "internal-device",
        "filesystem": "apfs",
    }


def test_storage_root_locator_refreshes_operational_volume_telemetry(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)

    refreshed = locator.with_observed_volumes(
        lambda _path: VolumeIdentity("observed-device", "xfs"),
        ("internal_archive",),
    )

    assert refreshed.resolve("internal_archive").volume == VolumeIdentity(
        "observed-device", "xfs"
    )
    assert refreshed.resolve("transfer_staging") == locator.resolve(
        "transfer_staging"
    )

    observed = {
        "/Volumes/TRANSFER/project/results": VolumeIdentity("transfer-device", "exfat"),
        "/Volumes/TRANSFER/project/archive": VolumeIdentity("transfer-device", "exfat"),
        "/Users/example/project-results": VolumeIdentity("internal-device", "apfs"),
    }
    locator.verify_all(lambda path: observed[str(path)])
    observed["/Users/example/project-results"] = VolumeIdentity("tampered", "apfs")
    with pytest.raises(RuntimeError, match="volume identity mismatch"):
        locator.verify_all(lambda path: observed[str(path)])


def test_campaign_capacity_preserves_external_workspace_and_internal_reserve(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )
    plan = config.build_plan(_pilot_observations())
    external_floor = 82 * 1024**3
    internal_free = plan.estimated_bytes

    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={
            "transfer_staging": external_floor,
            "transfer_archive": external_floor,
            "internal_archive": internal_free,
        },
    )

    assert len(capacity.assignments) == len(plan.batches)
    assert {assignment.root_alias for assignment in capacity.assignments} == {"internal_archive"}

    with pytest.raises(RuntimeError, match="staging capacity"):
        config.plan_archive_roots(
            plan,
            locator,
            free_bytes_by_alias={
                "transfer_staging": external_floor - 1,
                "transfer_archive": external_floor - 1,
                "internal_archive": internal_free,
            },
        )
    with pytest.raises(RuntimeError, match="campaign archive projection"):
        config.plan_archive_roots(
            plan,
            locator,
            free_bytes_by_alias={
                "transfer_staging": external_floor,
                "transfer_archive": external_floor,
                "internal_archive": internal_free - 1,
            },
        )


def test_campaign_preflight_treats_power_and_load_as_telemetry() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )
    valid = BenchmarkPreflightObservation(
        power_source="AC Power",
        low_power_mode_enabled=False,
        windows=(
            SystemLoadWindow(100.0, 30.0, 3.5, 0.75),
            SystemLoadWindow(130.0, 30.0, 4.0, 0.99),
        ),
    )

    config.validate_preflight(valid)
    config.validate_preflight(
        BenchmarkPreflightObservation(
            "Battery Power",
            True,
            (valid.windows[0], SystemLoadWindow(130.0, 30.0, 40.0, 8.0)),
        )
    )

    with pytest.raises(RuntimeError, match="consecutive"):
        config.validate_preflight(
            BenchmarkPreflightObservation(
                "AC Power",
                False,
                (valid.windows[0], SystemLoadWindow(131.0, 30.0, 3.0, 0.5)),
            )
        )


def test_anytime_checkpoints_use_last_complete_global_best_at_each_boundary() -> None:
    checkpoints = AnytimeCheckpoint.for_axis(
        instance="c101C5",
        seed=2014,
        axis_budget_seconds=30,
        initial_objective_key=(3, 120.0, 4.0, 2),
        accepted_global_bests=(
            AcceptedGlobalBest(1.0, 3, (2, 110.0, 3.0, 2)),
            AcceptedGlobalBest(5.0001, 8, (2, 100.0, 2.0, 1)),
            AcceptedGlobalBest(11.0, 15, (2, 90.0, 1.0, 1)),
        ),
        max_iterations_completed_at_seconds=12.0,
        final_objective_key=(2, 90.0, 1.0, 1),
    )

    assert tuple(item.checkpoint_seconds for item in checkpoints) == (1, 5, 10, 30)
    assert checkpoints[0].source == "accepted_global_best"
    assert checkpoints[0].objective_key == (2, 110.0, 3.0, 2)
    assert checkpoints[1].source == "accepted_global_best"
    assert checkpoints[1].objective_key == (2, 110.0, 3.0, 2)
    assert checkpoints[2].objective_key == (2, 100.0, 2.0, 1)
    assert checkpoints[3].source == "final_incumbent_carry_forward"
    assert checkpoints[3].objective_key == (2, 90.0, 1.0, 1)


def test_anytime_checkpoints_use_verified_initial_incumbent_before_improvement() -> None:
    checkpoints = AnytimeCheckpoint.for_axis(
        instance="r101_21",
        seed=2023,
        axis_budget_seconds=60,
        initial_objective_key=(20, 2_000.0, 20.0, 10),
        accepted_global_bests=(AcceptedGlobalBest(10.0001, 2, (19, 1_900.0, 19.0, 9)),),
    )

    assert tuple(item.checkpoint_seconds for item in checkpoints) == (1, 5, 10, 30, 60)
    assert checkpoints[2].source == "verified_initial_incumbent"
    assert checkpoints[2].objective_key == (20, 2_000.0, 20.0, 10)
    assert checkpoints[3].source == "accepted_global_best"


def test_anytime_carry_forward_rejects_an_unverified_final_incumbent() -> None:
    with pytest.raises(ValueError, match="final incumbent does not match"):
        AnytimeCheckpoint.for_axis(
            instance="c101C5",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(3, 120.0, 4.0, 2),
            accepted_global_bests=(),
            max_iterations_completed_at_seconds=12.0,
            final_objective_key=(1, 1.0, 0.0, 0),
        )

    with pytest.raises(ValueError, match="small instances"):
        AnytimeCheckpoint.for_axis(
            instance="r101_21",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(20, 2_000.0, 20.0, 10),
            accepted_global_bests=(),
            max_iterations_completed_at_seconds=12.0,
            final_objective_key=(20, 2_000.0, 20.0, 10),
        )


def test_anytime_history_rejects_a_non_improving_global_best() -> None:
    with pytest.raises(ValueError, match="strict objective improvement"):
        AnytimeCheckpoint.for_axis(
            instance="c101C5",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(2, 100.0, 1.0, 1),
            accepted_global_bests=(AcceptedGlobalBest(1.0, 1, (2, 100.0, 1.0, 1)),),
        )


def test_anytime_history_allows_multiple_global_bests_in_one_iteration() -> None:
    checkpoints = AnytimeCheckpoint.for_axis(
        instance="c101C5",
        seed=2014,
        axis_budget_seconds=30,
        initial_objective_key=(3, 120.0, 4.0, 2),
        accepted_global_bests=(
            AcceptedGlobalBest(1.0001, 7, (2, 110.0, 3.0, 2)),
            AcceptedGlobalBest(1.1, 7, (2, 100.0, 2.0, 1)),
        ),
    )

    assert checkpoints[0].objective_key == (3, 120.0, 4.0, 2)
    assert checkpoints[1].objective_key == (2, 100.0, 2.0, 1)

    with pytest.raises(ValueError, match="non-decreasing iteration"):
        AnytimeCheckpoint.for_axis(
            instance="c101C5",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(3, 120.0, 4.0, 2),
            accepted_global_bests=(
                AcceptedGlobalBest(1.0, 7, (2, 110.0, 3.0, 2)),
                AcceptedGlobalBest(1.1, 6, (2, 100.0, 2.0, 1)),
            ),
        )


def test_campaign_and_batch_manifests_are_path_free_and_completion_gated(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    )
    plan = config.build_plan(_pilot_observations())
    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={
            "transfer_staging": 82 * 1024**3,
            "internal_archive": 200 * 1024**3 + plan.estimated_bytes,
        },
    )
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256="a" * 64,
        prerequisite_review_sha256="b" * 64,
    )

    encoded = json.dumps(campaign.to_dict())
    assert "/Volumes/TRANSFER" not in encoded
    assert "/Users/example" not in encoded
    assert campaign.shard_count == 920
    assert campaign.axis_count == 2_040
    assert campaign.checkpoint_count == 10_400
    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_bytes = json.dumps(campaign.to_dict(), sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.with_suffix(".sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "\n",
        encoding="utf-8",
    )
    assert load_campaign_manifest(manifest_path).to_dict() == campaign.to_dict()
    manifest_path.write_bytes(manifest_bytes + b"\n")
    with pytest.raises(ManifestIntegrityError, match="checksum mismatch"):
        load_campaign_manifest(manifest_path)
    tampered_batch = replace(
        campaign.batches[0],
        shard_ids=("shard9999", *campaign.batches[0].shard_ids[1:]),
    )
    with pytest.raises(ValueError, match="exact shard IDs"):
        replace(campaign, batches=(tampered_batch, *campaign.batches[1:]))
    with pytest.raises(RuntimeError, match="all batches must be archived"):
        campaign.mark_complete()

    batch = campaign.batches[0].mark_verified(
        checksum_sha256="c" * 64,
        actual_bytes=plan.batches[0].estimated_bytes,
        row_count=10_400,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="d" * 64,
        persistence_attribution_sha256="f" * 64,
        control_persistence_seconds=1.0,
        persistence_ratio=0.1,
        shard_manifest_sha256_by_id={
            shard_id: "e" * 64 for shard_id in campaign.batches[0].shard_ids
        },
        shard_actual_bytes_by_id={shard_id: 1 for shard_id in campaign.batches[0].shard_ids},
    )
    assert batch.to_dict()["root_alias"] == "transfer_staging"
    assert batch.to_dict()["volume_identity"] == {
        "device_uuid": "transfer-device",
        "filesystem": "exfat",
    }
    campaign = campaign.with_batch(batch)
    with pytest.raises(RuntimeError, match="all batches must be archived"):
        campaign.mark_complete()

    archived = batch.mark_archived(
        root_alias="internal_archive",
        volume=VolumeIdentity("internal-device", "apfs"),
        transfer_mode="cross_volume_verified_copy",
        archive_transfer_seconds=1.0,
    )
    completed = (
        campaign.with_batch(archived)
        .with_batch_persistence_envelope(archived.batch_id, "9" * 64)
        .mark_complete()
    )
    assert completed.status == "complete"

    failed = campaign.with_batch(batch.mark_failed("checksum mismatch")).mark_failed(
        "batch0001 failed"
    )
    assert failed.status == "failed"
    assert failed.batches[0].status == "failed"
    assert failed.batches[0].checksum_sha256 == "c" * 64


def test_batch_manifest_rejects_actual_bytes_above_32_gib_hard_cap(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    plan = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    ).build_plan(_pilot_observations())
    batch = BatchManifest.planned(
        run_label=plan.run_label,
        plan=plan.batches[0],
        staging_root=locator.resolve("transfer_staging"),
        archive_root_alias="internal_archive",
    )

    with pytest.raises(ValueError, match="batch actual bytes exceed 32 GiB"):
        batch.mark_verified(
            checksum_sha256="c" * 64,
            actual_bytes=32 * 1024**3 + 1,
            row_count=1,
            physical_schema="screening_decisions_v3",
            resource_summary_sha256="d" * 64,
            persistence_attribution_sha256="f" * 64,
            control_persistence_seconds=1.0,
            persistence_ratio=0.1,
            shard_manifest_sha256_by_id={shard_id: "e" * 64 for shard_id in batch.shard_ids},
            shard_actual_bytes_by_id={shard_id: 1 for shard_id in batch.shard_ids},
        )


def test_batch_manifest_rejects_actual_shard_above_two_gib_hard_cap(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    shard = ShardPlan(
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        customer_count=5,
        family="C",
        budgets_seconds=(30,),
        checkpoint_seconds=(1, 5, 10, 30, 60, 120, 300),
        estimated_bytes=1,
        max_iterations=1_000,
    )
    batch = BatchManifest.planned(
        run_label="stage05.2_benchmark_attempt02",
        plan=BatchPlan("batch0001", (shard,), 1),
        staging_root=locator.resolve("transfer_staging"),
        archive_root_alias="internal_archive",
    )

    with pytest.raises(ValueError, match="shard actual bytes exceed 2 GiB"):
        batch.mark_verified(
            checksum_sha256="c" * 64,
            actual_bytes=2 * 1024**3 + 1,
            row_count=1,
            physical_schema="screening_decisions_v3",
            resource_summary_sha256="d" * 64,
            persistence_attribution_sha256="f" * 64,
            control_persistence_seconds=1.0,
            persistence_ratio=0.1,
            shard_manifest_sha256_by_id={"shard0001": "e" * 64},
            shard_actual_bytes_by_id={"shard0001": 2 * 1024**3 + 1},
        )


@pytest.mark.parametrize("tamper", ("batch", "shard"))
def test_batch_manifest_parser_rejects_resigned_hard_cap_bypass(
    tmp_path: Path,
    tamper: str,
) -> None:
    locator = _root_locator(tmp_path)
    shard = ShardPlan(
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        customer_count=5,
        family="C",
        budgets_seconds=(30,),
        checkpoint_seconds=(1, 5, 10, 30, 60, 120, 300),
        estimated_bytes=1,
        max_iterations=1_000,
    )
    planned = BatchManifest.planned(
        run_label="stage05.2_benchmark_attempt02",
        plan=BatchPlan("batch0001", (shard,), 1),
        staging_root=locator.resolve("transfer_staging"),
        archive_root_alias="internal_archive",
    )
    verified = planned.mark_verified(
        checksum_sha256="c" * 64,
        actual_bytes=1024,
        row_count=1,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="d" * 64,
        persistence_attribution_sha256="f" * 64,
        control_persistence_seconds=1.0,
        persistence_ratio=0.1,
        shard_manifest_sha256_by_id={"shard0001": "e" * 64},
        shard_actual_bytes_by_id={"shard0001": 512},
    )
    payload = verified.to_dict()
    if tamper == "batch":
        payload["actual_bytes"] = 40 * 1024**3
    else:
        payload["actual_bytes"] = 3 * 1024**3
        payload["shard_actual_bytes_by_id"] = {"shard0001": 3 * 1024**3}

    with pytest.raises(ValueError, match="hard cap"):
        BatchManifest.from_dict(payload)


def _verified_archive_batch(
    tmp_path: Path, *, same_volume: bool
) -> tuple[StorageRootLocator, BatchManifest, Path, Path]:
    source_root_path = tmp_path / "staging"
    destination_root_path = tmp_path / "archive"
    source_root_path.mkdir()
    destination_root_path.mkdir()
    source_volume = VolumeIdentity("external-device", "exfat")
    destination_volume = source_volume if same_volume else VolumeIdentity("internal-device", "apfs")
    locator = StorageRootLocator(
        {
            "staging": StorageRoot("staging", source_root_path, source_volume),
            "archive": StorageRoot("archive", destination_root_path, destination_volume),
        }
    )
    shard = ShardPlan(
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        customer_count=5,
        family="C",
        budgets_seconds=(30,),
        checkpoint_seconds=(1, 5, 10, 30, 60, 120, 300),
        estimated_bytes=len(b"verified-evidence"),
        max_iterations=1_000,
    )
    batch = BatchManifest.planned(
        run_label="stage05.2_benchmark_attempt02",
        plan=BatchPlan("batch0001", (shard,), len(b"verified-evidence")),
        staging_root=locator.resolve("staging"),
        archive_root_alias="archive",
    )
    source = source_root_path / batch.logical_path
    destination = destination_root_path / batch.logical_path
    source.mkdir(parents=True)
    (source / "payload.bin").write_bytes(b"verified-evidence")
    batch = batch.mark_verified(
        checksum_sha256=directory_checksum(source),
        actual_bytes=len(b"verified-evidence"),
        row_count=1,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="d" * 64,
        persistence_attribution_sha256="f" * 64,
        control_persistence_seconds=1.0,
        persistence_ratio=0.1,
        shard_manifest_sha256_by_id={"shard0001": "e" * 64},
        shard_actual_bytes_by_id={"shard0001": len(b"verified-evidence")},
    )
    return locator, batch, source, destination


def test_batch_payload_checksum_excludes_only_the_top_level_manifest_envelope(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "batch0001"
    nested_control = batch_dir / "shard0001" / "control"
    nested_control.mkdir(parents=True)
    (batch_dir / "payload.bin").write_bytes(b"payload")
    (nested_control / "shard_manifest.json").write_bytes(b"nested manifest")

    checksum_before = directory_checksum(batch_dir)
    bytes_before = directory_byte_count(batch_dir)
    files_before = directory_file_count(batch_dir)
    (batch_dir / "batch_manifest.json").write_bytes(b"self-referencing envelope")
    (batch_dir / "batch_manifest.sha256").write_text("0" * 64 + "\n", encoding="utf-8")

    assert directory_checksum(batch_dir) == checksum_before
    assert directory_byte_count(batch_dir) == bytes_before
    assert directory_file_count(batch_dir) == files_before == 2
    (nested_control / "shard_manifest.json").write_bytes(b"tampered nested manifest")
    assert directory_checksum(batch_dir) != checksum_before


def test_same_volume_archive_uses_atomic_rename(tmp_path: Path) -> None:
    locator, batch, source, destination = _verified_archive_batch(tmp_path, same_volume=True)
    clock_values = iter((10.0, 12.5))

    archived = BatchArchiver(locator, clock=lambda: next(clock_values)).archive(batch)

    assert archived.status == "archived"
    assert archived.transfer_mode == "same_volume_atomic_rename"
    assert archived.archive_transfer_seconds == 2.5
    assert not source.exists()
    assert destination.is_dir()
    assert directory_checksum(destination) == batch.checksum_sha256


def test_cross_volume_archive_verifies_incoming_before_deleting_source(
    tmp_path: Path,
) -> None:
    locator, batch, source, destination = _verified_archive_batch(tmp_path, same_volume=False)

    archived = BatchArchiver(locator).archive(batch)

    assert archived.status == "archived"
    assert archived.transfer_mode == "cross_volume_verified_copy"
    assert not source.exists()
    assert destination.is_dir()
    assert not destination.with_name(f"{destination.name}.incoming").exists()


@pytest.mark.parametrize("same_volume", (True, False))
def test_archive_reports_final_destination_after_trailing_fsync_failure(
    tmp_path: Path,
    same_volume: bool,
) -> None:
    locator, batch, source, destination = _verified_archive_batch(
        tmp_path,
        same_volume=same_volume,
    )

    class TrailingFsyncFailureIO(FileSystemArchiveIO):
        def fsync_directory(self, path: Path) -> None:
            if not source.exists() and (
                (same_volume and path == destination.parent)
                or (not same_volume and path == source.parent)
            ):
                raise OSError("injected trailing fsync failure")
            super().fsync_directory(path)

    with pytest.raises(ArchiveTransferCompletedError) as caught:
        BatchArchiver(locator, io=TrailingFsyncFailureIO()).archive(batch)

    assert caught.value.batch.status == "archived"
    assert caught.value.destination == destination
    assert not source.exists()
    assert destination.is_dir()
    assert directory_checksum(destination) == batch.checksum_sha256


def test_cross_volume_archive_tampering_retains_source_and_incoming(
    tmp_path: Path,
) -> None:
    locator, batch, source, destination = _verified_archive_batch(tmp_path, same_volume=False)

    class TamperingIO(FileSystemArchiveIO):
        def fsync_tree(self, path: Path) -> None:
            super().fsync_tree(path)
            if path.name.endswith(".incoming"):
                (path / "payload.bin").write_bytes(b"tampered")

    with pytest.raises(ArchiveTransferError, match="incoming checksum mismatch"):
        BatchArchiver(locator, io=TamperingIO()).archive(batch)

    incoming = destination.with_name(f"{destination.name}.incoming")
    assert batch.status == "verified"
    assert source.is_dir()
    assert incoming.is_dir()
    assert not destination.exists()
