#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "native_sha256.hpp"
#include "native_kernel_protocol.hpp"
#include "native_solver_kernels.hpp"

namespace evrptw::native_search {

struct RouteBatchViewV2 final {
    std::span<const std::int64_t> offsets;
    std::span<const std::int64_t> indices;

    [[nodiscard]] std::size_t route_count() const noexcept {
        return offsets.empty() ? 0U : offsets.size() - 1U;
    }

    [[nodiscard]] std::span<const std::int64_t> route(
        const std::size_t row) const {
        if (row >= route_count()) {
            throw std::out_of_range("native route batch row is out of range");
        }
        return indices.subspan(
            static_cast<std::size_t>(offsets[row]),
            static_cast<std::size_t>(offsets[row + 1] - offsets[row]));
    }

    void validate(const std::string_view name) const {
        if (offsets.empty() || offsets.front() != 0
            || offsets.back() != static_cast<std::int64_t>(indices.size())) {
            throw std::invalid_argument(
                std::string(name) + " boundary is invalid");
        }
        for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
            if (offsets[row] < 0 || offsets[row] > offsets[row + 1]) {
                throw std::invalid_argument(
                    std::string(name) + " offsets must be monotonic");
            }
        }
    }
};

struct AcceptanceOutcomeV2 final {
    std::int64_t accepted = 0;
    std::int64_t improved_global_best = 0;
    std::int64_t vehicle_reduction = 0;

    [[nodiscard]] std::array<std::int64_t, 3> values() const noexcept {
        return {accepted, improved_global_best, vehicle_reduction};
    }
};

struct ThreeLaneTerminationStateV2 final {
    std::int64_t reason = 0;
    std::int64_t exact_budget = 0;
    std::int64_t started = 0;
    std::int64_t completed = 0;
    std::int64_t interrupted = 0;
    std::int64_t completed_iterations = 0;
};

struct Stage04BoundaryStateV2 final {
    std::array<std::int64_t, 4> statuses{};
    std::array<double, 8> old_new_weights{};
    std::array<std::int64_t, 4> calls_at_boundary{};
    std::array<double, 4> rewards_at_boundary{};
    std::array<std::int64_t, 7> control_status{};
    double reheat_floor = 0.0;
};

struct ConstraintIterationOutcomeV2 final {
    std::int64_t operation = 0;
    std::int64_t probe_seed = 0;
    std::int64_t candidate_feasible = 0;
    std::int64_t accepted = 0;
    std::int64_t improved_global_best = 0;
    std::int64_t vehicle_reduction = 0;
    std::int64_t iteration = -1;
};

struct DynamicRemovalSelectionV2 final {
    std::int64_t tier = 0;
    std::int64_t requested_count = 0;
    std::int64_t lower_bound = 0;
    std::int64_t upper_bound = 0;
    std::int64_t stagnation_iterations = 0;
    std::int64_t trigger = 0;
    std::int64_t global_best_reset = 0;
    std::int64_t iteration = -1;

    [[nodiscard]] std::array<std::int64_t, 7> values() const noexcept {
        return {
            tier,
            requested_count,
            lower_bound,
            upper_bound,
            stagnation_iterations,
            trigger,
            global_best_reset,
        };
    }
};

struct ConstraintRemovalResultV2 final {
    std::vector<std::int64_t> partial_offsets;
    std::vector<std::int64_t> partial_indices;
    std::vector<std::int64_t> removed_indices;
    std::vector<std::int64_t> score_nodes;
    std::vector<double> score_values;
    std::vector<std::int64_t> score_routes;
    std::array<std::int64_t, 3> metadata{};
    std::int64_t iteration = -1;

    void validate() const {
        if (partial_offsets.empty() || partial_offsets.front() != 0
            || partial_offsets.back()
                != static_cast<std::int64_t>(partial_indices.size())
            || score_nodes.size() != score_values.size()
            || score_nodes.size() != score_routes.size()
            || metadata[0] < 0 || metadata[0] > 2
            || metadata[2] < 0
            || metadata[2]
                != static_cast<std::int64_t>(removed_indices.size())) {
            throw std::logic_error(
                "native constraint removal result is inconsistent");
        }
        for (std::size_t index = 0; index + 1 < partial_offsets.size(); ++index) {
            if (partial_offsets[index] < 0
                || partial_offsets[index] > partial_offsets[index + 1]) {
                throw std::logic_error(
                    "native constraint removal offsets are not monotonic");
            }
        }
        for (const auto score : score_values) {
            if (!std::isfinite(score)) {
                throw std::logic_error(
                    "native constraint removal score is not finite");
            }
        }
    }
};

struct RepairResultV2 final {
    std::vector<std::int64_t> route_offsets;
    std::vector<std::int64_t> route_indices;
    std::array<std::int64_t, 7> counters{};
    std::int64_t iteration = -1;

    void validate() const {
        if (route_offsets.empty() || route_offsets.front() != 0
            || route_offsets.back()
                != static_cast<std::int64_t>(route_indices.size())
            || counters[0] < 0 || counters[0] > 2) {
            throw std::logic_error(
                "native candidate-control repair result is inconsistent");
        }
        for (std::size_t index = 0; index + 1 < route_offsets.size(); ++index) {
            if (route_offsets[index] < 0
                || route_offsets[index] > route_offsets[index + 1]
                || (counters[0] == 0
                    && route_offsets[index] == route_offsets[index + 1])) {
                throw std::logic_error(
                    "native candidate-control repair offsets are not monotonic");
            }
        }
        for (std::size_t index = 1; index < counters.size(); ++index) {
            if (counters[index] < 0) {
                throw std::logic_error(
                    "native candidate-control repair counter is negative");
            }
        }
        if (counters[3] != counters[4] + counters[5]
            || (counters[0] == 0 && counters[6] != 0)
            || (counters[0] == 0 && route_offsets.size() < 2)
            || (counters[0] != 0
                && (route_offsets.size() != 1 || !route_indices.empty()))) {
            throw std::logic_error(
                "native candidate-control repair status is inconsistent");
        }
    }
};

struct InsertionPlanPoolV2 final {
    std::vector<std::int64_t> plan_offsets;
    std::vector<std::int64_t> route_offsets;
    std::vector<std::int64_t> route_indices;
    std::vector<std::int64_t> metadata;

    [[nodiscard]] std::size_t plan_count() const noexcept {
        return plan_offsets.empty() ? 0 : plan_offsets.size() - 1;
    }

    void validate() const {
        constexpr auto maximum_i64_size =
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        if (plan_offsets.empty() || route_offsets.empty()
            || plan_count() > maximum_i64_size
            || route_offsets.size() - 1 > maximum_i64_size
            || route_indices.size() > maximum_i64_size
            || plan_offsets.front() != 0 || route_offsets.front() != 0
            || plan_offsets.back()
                != static_cast<std::int64_t>(route_offsets.size() - 1)
            || route_offsets.back()
                != static_cast<std::int64_t>(route_indices.size())
            || metadata.size() % 2 != 0
            || metadata.size() / 2 != plan_count()) {
            throw std::logic_error(
                "native insertion plan pool is inconsistent");
        }
        for (std::size_t route = 0; route + 1 < route_offsets.size(); ++route) {
            if (route_offsets[route] < 0
                || route_offsets[route] >= route_offsets[route + 1]
                || route_offsets[route + 1]
                    > static_cast<std::int64_t>(route_indices.size())) {
                throw std::logic_error(
                    "native insertion route offsets are not strictly monotonic");
            }
        }
        for (std::size_t plan = 0; plan < plan_count(); ++plan) {
            if (plan_offsets[plan] < 0
                || plan_offsets[plan] >= plan_offsets[plan + 1]
                || plan_offsets[plan + 1]
                    > static_cast<std::int64_t>(route_offsets.size() - 1)) {
                throw std::logic_error(
                    "native insertion plan offsets are not strictly monotonic");
            }
        }
        for (std::size_t plan = 0; plan < plan_count(); ++plan) {
            const auto target = metadata[plan * 2];
            const auto position = metadata[plan * 2 + 1];
            const auto plan_route_count =
                plan_offsets[plan + 1] - plan_offsets[plan];
            if (target < 0 || target >= plan_route_count || position < 0) {
                throw std::logic_error(
                    "native insertion plan metadata is invalid");
            }
            const auto target_route = plan_offsets[plan] + target;
            const auto target_size =
                route_offsets[static_cast<std::size_t>(target_route + 1)]
                - route_offsets[static_cast<std::size_t>(target_route)];
            if (position >= target_size) {
                throw std::logic_error(
                    "native insertion position is outside its target route");
            }
        }
    }
};

struct RouteMergeCandidatePoolV2 final {
    std::vector<std::int64_t> candidate_offsets;
    std::vector<std::int64_t> candidate_indices;
    std::vector<std::int64_t> metadata;
    std::array<std::int64_t, 2> pruning{};
    std::int64_t input_route_count = 0;

    [[nodiscard]] std::size_t candidate_count() const noexcept {
        return candidate_offsets.empty() ? 0 : candidate_offsets.size() - 1;
    }

    void validate() const {
        constexpr auto maximum_i64_size =
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        if (candidate_offsets.empty() || candidate_offsets.front() != 0
            || candidate_count() > maximum_i64_size
            || candidate_indices.size() > maximum_i64_size
            || candidate_offsets.back()
                != static_cast<std::int64_t>(candidate_indices.size())
            || metadata.size() % 5 != 0
            || metadata.size() / 5 != candidate_count()
            || pruning[0] < 0 || pruning[1] < 0
            || input_route_count < 2) {
            throw std::logic_error(
                "native route-merge candidate pool is inconsistent");
        }
        for (std::size_t candidate = 0; candidate < candidate_count(); ++candidate) {
            if (candidate_offsets[candidate] < 0
                || candidate_offsets[candidate]
                    >= candidate_offsets[candidate + 1]
                || candidate_offsets[candidate + 1]
                    > static_cast<std::int64_t>(candidate_indices.size())) {
                throw std::logic_error(
                    "native route-merge candidate offsets are invalid");
            }
        }
        for (std::size_t candidate = 0; candidate < candidate_count(); ++candidate) {
            const auto left = metadata[candidate * 5];
            const auto right = metadata[candidate * 5 + 1];
            const auto source = metadata[candidate * 5 + 2];
            const auto target = metadata[candidate * 5 + 3];
            const auto position = metadata[candidate * 5 + 4];
            const auto pair_matches =
                (source == left && target == right)
                || (source == right && target == left);
            const auto candidate_size =
                candidate_offsets[candidate + 1]
                - candidate_offsets[candidate];
            if (left < 0 || right <= left || right >= input_route_count
                || !pair_matches || position < 0
                || position >= candidate_size) {
                throw std::logic_error(
                    "native route-merge candidate metadata is invalid");
            }
        }
    }
};

struct ChangedCandidatePoolV1 final {
    std::vector<std::int64_t> changed_route_indices;
    std::vector<std::int64_t> change_offsets;
    std::vector<std::int64_t> change_indices;
    std::vector<std::int64_t> removed_offsets;
    std::vector<std::int64_t> removed_indices;
    std::int64_t operation = -1;
    std::int64_t input_route_count = 0;

    [[nodiscard]] std::size_t candidate_count() const noexcept {
        return changed_route_indices.size() / 2;
    }

    void validate() const {
        constexpr auto maximum_i64_size =
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        if (change_offsets.empty() || removed_offsets.empty()
            || changed_route_indices.size() % 2 != 0
            || candidate_count() > maximum_i64_size
            || change_indices.size() > maximum_i64_size
            || removed_indices.size() > maximum_i64_size
            || change_offsets.size() - 1 != changed_route_indices.size()
            || removed_offsets.size() - 1 != candidate_count()
            || change_offsets.front() != 0 || removed_offsets.front() != 0
            || change_offsets.back()
                != static_cast<std::int64_t>(change_indices.size())
            || removed_offsets.back()
                != static_cast<std::int64_t>(removed_indices.size())
            || operation < 0 || operation > 2 || input_route_count <= 0) {
            throw std::logic_error(
                "native changed-candidate pool is inconsistent");
        }
        for (std::size_t change = 0; change + 1 < change_offsets.size(); ++change) {
            if (change_offsets[change] < 0
                || change_offsets[change] >= change_offsets[change + 1]
                || change_offsets[change + 1]
                    > static_cast<std::int64_t>(change_indices.size())) {
                throw std::logic_error(
                    "native changed-candidate route offsets are invalid");
            }
        }
        for (std::size_t candidate = 0; candidate < candidate_count(); ++candidate) {
            const auto left = changed_route_indices[candidate * 2];
            const auto right = changed_route_indices[candidate * 2 + 1];
            if (left < 0 || right <= left || right >= input_route_count
                || removed_offsets[candidate] < 0
                || removed_offsets[candidate] > removed_offsets[candidate + 1]
                || removed_offsets[candidate + 1]
                    > static_cast<std::int64_t>(removed_indices.size())) {
                throw std::logic_error(
                    "native changed-candidate metadata is invalid");
            }
            const auto removed_count =
                removed_offsets[candidate + 1] - removed_offsets[candidate];
            const auto expected_removed_count =
                operation == 0 ? 1 : (operation == 1 ? 2 : 0);
            if (removed_count != expected_removed_count) {
                throw std::logic_error(
                    "native changed-candidate removal identity is invalid");
            }
        }
    }
};

struct CandidatePlanPoolV1 final {
    std::vector<std::int64_t> plan_offsets;
    std::vector<std::int64_t> route_offsets;
    std::vector<std::int64_t> route_indices;
    std::int64_t routes_per_plan = 0;

    [[nodiscard]] std::size_t plan_count() const noexcept {
        return plan_offsets.empty() ? 0 : plan_offsets.size() - 1;
    }

    void validate() const {
        constexpr auto maximum_i64_size =
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        if (plan_offsets.empty() || route_offsets.empty()
            || plan_count() > maximum_i64_size
            || route_offsets.size() - 1 > maximum_i64_size
            || route_indices.size() > maximum_i64_size
            || plan_offsets.front() != 0 || route_offsets.front() != 0
            || plan_offsets.back()
                != static_cast<std::int64_t>(route_offsets.size() - 1)
            || route_offsets.back()
                != static_cast<std::int64_t>(route_indices.size())
            || routes_per_plan <= 0) {
            throw std::logic_error(
                "native candidate-plan pool is inconsistent");
        }
        for (std::size_t route = 0; route + 1 < route_offsets.size(); ++route) {
            if (route_offsets[route] < 0
                || route_offsets[route] >= route_offsets[route + 1]
                || route_offsets[route + 1]
                    > static_cast<std::int64_t>(route_indices.size())) {
                throw std::logic_error(
                    "native candidate-plan route offsets are invalid");
            }
        }
        for (std::size_t plan = 0; plan < plan_count(); ++plan) {
            if (plan_offsets[plan] < 0
                || plan_offsets[plan] >= plan_offsets[plan + 1]
                || plan_offsets[plan + 1]
                    > static_cast<std::int64_t>(route_offsets.size() - 1)
                || plan_offsets[plan + 1] - plan_offsets[plan]
                    != routes_per_plan) {
                throw std::logic_error(
                    "native candidate-plan offsets are invalid");
            }
        }
    }
};

struct ScreenBatchInputV2 final {
    std::span<const std::int64_t> node_kind;
    std::span<const double> demand;
    std::span<const double> ready_time;
    std::span<const double> due_date;
    std::span<const double> service_time;
    std::span<const double> distance;
    std::span<const std::uint8_t> reachable;
    std::span<const double> vehicle;
    std::span<const std::int64_t> route_offsets;
    std::span<const std::int64_t> route_indices;
    std::span<const std::int64_t> candidate_ids;
    std::span<const double> options;
    std::span<const double> incremental;
    std::span<const std::int64_t> negative_offsets;
    std::span<const std::int64_t> negative_indices;
    std::span<const std::int64_t> negative_reason_codes;
    std::int64_t worker_count = 0;
};

struct ScreenBatchResultV2 final {
    std::vector<std::int64_t> candidate_ids;
    std::vector<std::int64_t> statuses;
    std::vector<std::int64_t> duplicate_of;
    std::vector<std::int64_t> codes;
    std::vector<double> metrics;
    std::array<std::int64_t, 5> counters{};
    std::string digest;

    [[nodiscard]] std::size_t candidate_count() const noexcept {
        return candidate_ids.size();
    }

    void validate() const {
        constexpr auto maximum_i64_size =
            static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        if (candidate_count() > maximum_i64_size
            || statuses.size() != candidate_count()
            || duplicate_of.size() != candidate_count()
            || candidate_count() > std::numeric_limits<std::size_t>::max() / 16
            || codes.size() != candidate_count() * 16
            || candidate_count() > std::numeric_limits<std::size_t>::max() / 15
            || metrics.size() != candidate_count() * 15
            || counters[0] != static_cast<std::int64_t>(candidate_count())
            || counters[1] < 0 || counters[2] < 0 || counters[3] < 0
            || counters[4] < 0 || counters[1] > counters[0]
            || counters[2] > counters[0] || counters[3] > counters[1]
            || counters[4] > counters[1]
            || counters[1] != counters[0] - counters[2]
            || counters[3] != counters[1] - counters[4]
            || digest.size() != 64
            || !std::all_of(
                digest.begin(), digest.end(), [](const char value) {
                    return (value >= '0' && value <= '9')
                        || (value >= 'a' && value <= 'f');
                })) {
            throw std::logic_error(
                "native screen-batch result is inconsistent");
        }
        std::unordered_set<std::int64_t> observed_ids;
        std::int64_t duplicate_count = 0;
        std::int64_t negative_count = 0;
        std::int64_t screened_count = 0;
        for (std::size_t candidate = 0; candidate < candidate_count(); ++candidate) {
            if (!observed_ids.insert(candidate_ids[candidate]).second
                || statuses[candidate] < 0 || statuses[candidate] > 2) {
                throw std::logic_error(
                    "native screen-batch candidate identity is invalid");
            }
            if (statuses[candidate] == 1) {
                ++duplicate_count;
                if (duplicate_of[candidate] == candidate_ids[candidate]
                    || !observed_ids.contains(duplicate_of[candidate])) {
                    throw std::logic_error(
                        "native screen-batch duplicate identity is invalid");
                }
            } else {
                if (duplicate_of[candidate] != -1) {
                    throw std::logic_error(
                        "native screen-batch duplicate sentinel is invalid");
                }
                if (statuses[candidate] == 2) {
                    ++negative_count;
                } else {
                    ++screened_count;
                }
            }
        }
        if (duplicate_count != counters[2] || negative_count != counters[3]
            || screened_count != counters[4]) {
            throw std::logic_error(
                "native screen-batch counters are inconsistent");
        }
    }
};

struct CandidateRoundResultV2 final {
    ScreenBatchResultV2 screening;
    std::vector<std::int64_t> resolutions;
    std::vector<std::int64_t> sources;
    std::vector<std::int64_t> cache_journal;
    std::vector<std::int64_t> exact_candidate_ids;
    std::vector<std::int64_t> completion_order;
    native_kernels::ExactBatchOutput exact;
    std::array<std::int64_t, 10> counters{};
    std::array<double, 4> timings{};
    std::string digest;

    [[nodiscard]] std::size_t candidate_count() const noexcept {
        return screening.candidate_count();
    }

    void validate() const {
        screening.validate();
        const auto count = candidate_count();
        if (resolutions.size() != count || sources.size() != count
            || count > std::numeric_limits<std::size_t>::max() / 3
            || cache_journal.size() != count * 3
            || exact_candidate_ids.size() != exact.statuses.size()
            || exact.path_offsets.size() != exact.statuses.size() + 1
            || exact.reasons.size() != exact.statuses.size()
            || exact.statuses.size()
                > std::numeric_limits<std::size_t>::max() / 4
            || exact.metrics.size() != exact.statuses.size() * 4
            || exact.statuses.size()
                > std::numeric_limits<std::size_t>::max() / 3
            || exact.label_counters.size() != exact.statuses.size() * 3
            || exact.batch_counters.size() != 10
            || exact.path_offsets.empty() || exact.path_offsets.front() != 0
            || exact.path_offsets.back()
                != static_cast<std::int64_t>(exact.path_indices.size())
            || counters[0] != static_cast<std::int64_t>(count)
            || counters[1] < 0 || counters[2] < 0 || counters[3] < 0
            || counters[4] < 0 || counters[5] < 0 || counters[6] < 0
            || counters[7] < 0 || counters[8] < 0 || counters[9] != 0
            || counters[1] > counters[0] || counters[2] > counters[1]
            || counters[4] > counters[3] || counters[5] > counters[2]
            || counters[6]
                != static_cast<std::int64_t>(exact_candidate_ids.size())
            || counters[8] != screening.counters[2]
            || exact.batch_counters[0] != counters[6]
            || exact.batch_counters[1] != counters[6]
            || exact.batch_counters[1] < 0
            || exact.batch_counters[2] < 0
            || exact.batch_counters[3] < 0
            || exact.batch_counters[2] > exact.batch_counters[1]
            || exact.batch_counters[3]
                != exact.batch_counters[1] - exact.batch_counters[2]
            || std::any_of(
                timings.begin(), timings.end(), [](const double value) {
                    return !std::isfinite(value) || value < 0.0;
                })
            || digest.size() != 64
            || !std::all_of(
                digest.begin(), digest.end(), [](const char value) {
                    return (value >= '0' && value <= '9')
                        || (value >= 'a' && value <= 'f');
                })) {
            throw std::logic_error(
                "native candidate-round result is inconsistent");
        }
        for (std::size_t index = 0; index + 1 < exact.path_offsets.size(); ++index) {
            if (exact.path_offsets[index] < 0
                || exact.path_offsets[index] > exact.path_offsets[index + 1]
                || exact.path_offsets[index + 1]
                    > static_cast<std::int64_t>(exact.path_indices.size())) {
                throw std::logic_error(
                    "native candidate-round exact paths are inconsistent");
            }
        }
        std::unordered_map<std::int64_t, std::size_t> candidate_positions;
        candidate_positions.reserve(count);
        for (std::size_t candidate = 0; candidate < count; ++candidate) {
            candidate_positions.emplace(screening.candidate_ids[candidate], candidate);
        }
        std::unordered_set<std::int64_t> exact_id_set;
        exact_id_set.reserve(exact_candidate_ids.size());
        for (const auto candidate_id : exact_candidate_ids) {
            if (!candidate_positions.contains(candidate_id)
                || !exact_id_set.insert(candidate_id).second) {
                throw std::logic_error(
                    "native candidate-round exact identity is inconsistent");
            }
        }
        auto completion_id_set = std::unordered_set<std::int64_t>{};
        completion_id_set.reserve(completion_order.size());
        for (const auto candidate_id : completion_order) {
            if (!exact_id_set.contains(candidate_id)
                || !completion_id_set.insert(candidate_id).second) {
                throw std::logic_error(
                    "native candidate-round completion order is inconsistent");
            }
        }
        if (completion_id_set.size()
                != static_cast<std::size_t>(exact.batch_counters[2])
            || (exact.batch_counters[3] == 0
                && completion_id_set.size() != exact_id_set.size())) {
            throw std::logic_error(
                "native candidate-round completion order is inconsistent");
        }
        if (exact.completion_order.size() != completion_order.size()) {
            throw std::logic_error(
                "native candidate-round completion projection is inconsistent");
        }
        for (std::size_t index = 0; index < completion_order.size(); ++index) {
            const auto ordinal = exact.completion_order[index];
            if (ordinal < 0
                || static_cast<std::size_t>(ordinal) >= exact_candidate_ids.size()
                || completion_order[index]
                    != exact_candidate_ids[static_cast<std::size_t>(ordinal)]) {
                throw std::logic_error(
                    "native candidate-round completion projection is inconsistent");
            }
        }
        std::array<std::int64_t, 9> observed{};
        observed[0] = static_cast<std::int64_t>(count);
        for (std::size_t candidate = 0; candidate < count; ++candidate) {
            const auto resolution = resolutions[candidate];
            const auto status = screening.statuses[candidate];
            const auto passed = screening.codes[candidate * 16] == 1;
            const auto decision = cache_journal[candidate * 3 + 1];
            const auto ordinal = cache_journal[candidate * 3 + 2];
            if (resolution < 0 || resolution > 5
                || cache_journal[candidate * 3]
                    != screening.candidate_ids[candidate]) {
                throw std::logic_error(
                    "native candidate-round decision journal is inconsistent");
            }
            if (status == 1) {
                ++observed[8];
                const auto source_position = candidate_positions.find(
                    screening.duplicate_of[candidate]);
                if (sources[candidate] != screening.duplicate_of[candidate]
                    || source_position == candidate_positions.end()
                    || source_position->second >= candidate || decision != 6
                    || ordinal != -1) {
                    throw std::logic_error(
                        "native candidate-round duplicate journal is inconsistent");
                }
                const auto source_resolution = resolutions[source_position->second];
                const auto expected_resolution = source_resolution == 0
                    ? 0
                    : ((source_resolution == 1 || source_resolution == 2)
                           ? 5
                           : source_resolution);
                if (resolution != expected_resolution) {
                    throw std::logic_error(
                        "native candidate-round duplicate resolution is inconsistent");
                }
                continue;
            }
            if (sources[candidate] != -1 || resolution == 5) {
                throw std::logic_error(
                    "native candidate-round source is inconsistent");
            }
            if (!passed) {
                ++observed[3];
                if (status == 2) {
                    ++observed[4];
                }
                if (resolution != 0 || decision != 5 || ordinal != -1) {
                    throw std::logic_error(
                        "native candidate-round rejection is inconsistent");
                }
                continue;
            }
            ++observed[1];
            switch (resolution) {
            case 1:
                ++observed[2];
                ++observed[5];
                if (decision != 1 || ordinal != -1) {
                    throw std::logic_error(
                        "native candidate-round cache decision is inconsistent");
                }
                break;
            case 2:
                ++observed[2];
                ++observed[6];
                if (decision != 2 || ordinal < 0
                    || static_cast<std::size_t>(ordinal) >= exact_candidate_ids.size()
                    || exact_candidate_ids[static_cast<std::size_t>(ordinal)]
                        != screening.candidate_ids[candidate]) {
                    throw std::logic_error(
                        "native candidate-round exact ordinal is inconsistent");
                }
                break;
            case 3:
                if (decision != 4 || ordinal != -1) {
                    throw std::logic_error(
                        "native candidate-round unselected decision is inconsistent");
                }
                break;
            case 4:
                ++observed[2];
                ++observed[7];
                if (decision != 3 || ordinal != -1) {
                    throw std::logic_error(
                        "native candidate-round budget decision is inconsistent");
                }
                break;
            default:
                throw std::logic_error(
                    "native candidate-round accepted resolution is inconsistent");
            }
        }
        if (observed[0] != observed[1] + observed[3] + observed[8]
            || observed[2] != observed[5] + observed[6] + observed[7]
            || !std::equal(observed.begin(), observed.end(), counters.begin())) {
            throw std::logic_error(
                "native candidate-round decision counters are inconsistent");
        }
    }
};

[[nodiscard]] inline DynamicRemovalSelectionV2 select_dynamic_removal_v2(
    const std::int64_t customer_count,
    const std::int64_t stagnation_iterations,
    const std::int64_t iteration,
    const std::array<std::int64_t, 3>& thresholds,
    const std::array<double, 6>& fractions,
    const bool global_best_reset) {
    if (customer_count < 0 || stagnation_iterations < 0 || iteration < 0) {
        throw std::invalid_argument(
            "dynamic removal selection v2 values cannot be negative");
    }
    const auto medium_threshold = thresholds[0];
    const auto large_threshold = thresholds[1];
    const auto exploration_period = thresholds[2];
    if (medium_threshold < 0 || large_threshold <= medium_threshold
        || exploration_period <= 0) {
        throw std::invalid_argument("dynamic removal thresholds are invalid");
    }
    for (std::size_t index = 0; index < 3; ++index) {
        const auto minimum = fractions[index * 2];
        const auto maximum = fractions[index * 2 + 1];
        if (!std::isfinite(minimum) || !std::isfinite(maximum)
            || minimum <= 0.0 || maximum < minimum || maximum > 1.0) {
            throw std::invalid_argument(
                "dynamic removal fraction bounds are invalid");
        }
    }

    DynamicRemovalSelectionV2 selection;
    selection.stagnation_iterations = stagnation_iterations;
    selection.global_best_reset = global_best_reset ? 1 : 0;
    selection.iteration = iteration;
    if (stagnation_iterations >= large_threshold) {
        selection.tier = 2;
        selection.trigger = 2;
    } else if (stagnation_iterations >= medium_threshold) {
        selection.tier = 1;
        selection.trigger = 1;
    }
    if (iteration > 0 && iteration % exploration_period == 0
        && stagnation_iterations > medium_threshold && selection.tier < 2) {
        ++selection.tier;
        selection.trigger += 3;
    }
    if (customer_count > 1) {
        const auto upper_customer_bound = customer_count - 1;
        const auto tier_offset = static_cast<std::size_t>(selection.tier) * 2;
        selection.lower_bound = std::max<std::int64_t>(
            1,
            std::min<std::int64_t>(
                upper_customer_bound,
                static_cast<std::int64_t>(std::ceil(
                    static_cast<double>(customer_count)
                    * fractions[tier_offset]))));
        selection.upper_bound = std::max<std::int64_t>(
            selection.lower_bound,
            std::min<std::int64_t>(
                upper_customer_bound,
                static_cast<std::int64_t>(std::floor(
                    static_cast<double>(customer_count)
                    * fractions[tier_offset + 1]))));
        selection.requested_count = selection.lower_bound;
    } else {
        selection.trigger = 6;
    }
    return selection;
}

struct ProblemV2 final {
    std::vector<std::int64_t> node_kind;
    std::vector<double> demand;
    std::vector<double> ready_time;
    std::vector<double> due_date;
    std::vector<double> service_time;
    std::vector<double> distance;
    std::vector<std::uint8_t> reachable;
    std::array<double, 5> vehicle{};
    std::vector<std::int64_t> lexical_rank;
    std::vector<std::int64_t> node_name_offsets;
    std::vector<std::uint8_t> node_name_bytes;
    std::vector<std::int64_t> initial_route_offsets;
    std::vector<std::int64_t> initial_route_indices;

    [[nodiscard]] std::size_t node_count() const noexcept {
        return node_kind.size();
    }

    [[nodiscard]] std::size_t route_count() const noexcept {
        return initial_route_offsets.empty()
            ? 0U : initial_route_offsets.size() - 1U;
    }

    void validate() const {
        const auto nodes = node_count();
        if (nodes == 0 || demand.size() != nodes || ready_time.size() != nodes
            || due_date.size() != nodes || service_time.size() != nodes
            || lexical_rank.size() != nodes
            || distance.size() != checked_square(nodes)
            || reachable.size() != checked_square(nodes)) {
            throw std::invalid_argument(
                "native search problem arrays do not share one node shape");
        }
        if (node_name_offsets.size() != nodes + 1
            || node_name_offsets.front() != 0
            || node_name_offsets.back()
                != static_cast<std::int64_t>(node_name_bytes.size())) {
            throw std::invalid_argument(
                "native search node-name SoA does not span its byte buffer");
        }
        validate_offsets(
            node_name_offsets, node_name_bytes.size(), "node_name_offsets",
            true);
        for (std::size_t node = 0; node < nodes; ++node) {
            if (node_name_offsets[node] == node_name_offsets[node + 1]) {
                throw std::invalid_argument(
                    "native search node names must be non-empty");
            }
        }
        if (initial_route_offsets.size() < 2
            || initial_route_indices.empty()) {
            throw std::invalid_argument(
                "native search requires a non-empty warm start");
        }
        validate_offsets(
            initial_route_offsets, initial_route_indices.size(),
            "initial_route_offsets", false);
        std::unordered_set<std::int64_t> ranks;
        std::unordered_set<std::int64_t> expected_customers;
        std::unordered_set<std::int64_t> warm_customers;
        ranks.reserve(nodes);
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto kind = node_kind[node];
            if (kind == native_kernels::customer_kind) {
                expected_customers.insert(static_cast<std::int64_t>(node));
            } else if (kind != native_kernels::depot_kind
                && kind != native_kernels::station_kind) {
                throw std::invalid_argument(
                    "native search node kind is invalid");
            }
            const auto rank = lexical_rank[node];
            if (rank < 0 || rank >= static_cast<std::int64_t>(nodes)
                || !ranks.insert(rank).second) {
                throw std::invalid_argument(
                    "native search lexical rank must be a permutation");
            }
            if (!std::isfinite(demand[node]) || demand[node] < 0.0
                || !std::isfinite(ready_time[node])
                || !std::isfinite(due_date[node])
                || !std::isfinite(service_time[node])
                || service_time[node] < 0.0 || ready_time[node] > due_date[node]) {
                throw std::invalid_argument(
                    "native search node metrics are invalid");
            }
        }
        for (const auto value : vehicle) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument(
                    "native search vehicle metrics must be finite and positive");
            }
        }
        for (const auto value : distance) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "native search distance matrix is invalid");
            }
        }
        if (std::any_of(
                reachable.begin(), reachable.end(),
                [](std::uint8_t value) { return value > 1U; })) {
            throw std::invalid_argument(
                "native search reachability matrix must be binary");
        }
        for (const auto customer : initial_route_indices) {
            if (customer < 0 || customer >= static_cast<std::int64_t>(nodes)
                || node_kind[static_cast<std::size_t>(customer)]
                    != native_kernels::customer_kind
                || !warm_customers.insert(customer).second) {
                throw std::invalid_argument(
                    "native search warm start contains an invalid customer");
            }
        }
        if (warm_customers != expected_customers) {
            throw std::invalid_argument(
                "native search warm start must cover every customer exactly once");
        }
    }

private:
    [[nodiscard]] static std::size_t checked_square(std::size_t value) {
        if (value > 0
            && value > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("native search node square overflows");
        }
        return value * value;
    }

    static void validate_offsets(
        std::span<const std::int64_t> offsets,
        std::size_t terminal,
        std::string_view name,
        bool allow_empty_rows) {
        if (offsets.empty() || offsets.front() != 0
            || offsets.back() != static_cast<std::int64_t>(terminal)) {
            throw std::invalid_argument(std::string(name) + " boundary is invalid");
        }
        for (std::size_t index = 0; index + 1 < offsets.size(); ++index) {
            if (offsets[index] < 0
                || (allow_empty_rows
                    ? offsets[index] > offsets[index + 1]
                    : offsets[index] >= offsets[index + 1])) {
                throw std::invalid_argument(
                    std::string(name) + " is not monotone");
            }
        }
    }
};

struct ConfigV2 final {
    std::array<std::int64_t, 5> search_control{};
    // Relative duration is retained for audit; the absolute steady-clock
    // boundary is authoritative across local and same-host scheduler processes.
    std::array<double, 2> deadline{};
    std::array<std::int64_t, 13> protocol_control{};
    std::array<double, 2> protocol_options{};
    std::array<std::int64_t, 15> stage04_integer{};
    std::array<double, 15> stage04_float{};
    std::array<std::int64_t, 24> operator_integer{};
    std::array<double, 7> operator_float{};

    void validate() const {
        if (search_control[0] < 0 || search_control[1] <= 0
            || search_control[2] <= 0 || search_control[3] <= 0
            || search_control[4] < -1
            || !std::isfinite(deadline[0]) || deadline[0] <= 0.0
            || !std::isfinite(deadline[1]) || deadline[1] <= 0.0) {
            throw std::invalid_argument(
                "native search base control is invalid");
        }
        if ((protocol_control[0] != 0 && protocol_control[0] != 1)
            || protocol_control[1] <= 0 || protocol_control[2] <= 0
            || (protocol_control[3] != 1 && protocol_control[3] != 4)
            || protocol_control[4] < 10 || protocol_control[5] < 0) {
            throw std::invalid_argument(
                "native search Candidate Control configuration is invalid");
        }
        if (!(protocol_options[0] > 0.0 && protocol_options[0] <= 1.0)
            || !std::isfinite(protocol_options[1])
            || protocol_options[1] <= 0.0) {
            throw std::invalid_argument(
                "native search protocol options are invalid");
        }
        if (std::any_of(
                stage04_float.begin(), stage04_float.end(),
                [](double value) { return !std::isfinite(value); })
            || std::any_of(
                operator_float.begin(), operator_float.end(),
                [](double value) { return !std::isfinite(value); })) {
            throw std::invalid_argument(
                "native search floating configuration is non-finite");
        }
    }
};

struct RequestV2 final {
    ProblemV2 problem;
    ConfigV2 config;

    void validate() const {
        problem.validate();
        config.validate();
    }

    [[nodiscard]] std::string sha256() const {
        validate();
        std::string evidence("stage05.2-native-search-request-v2");
        append(evidence, problem.node_kind);
        append(evidence, problem.demand);
        append(evidence, problem.ready_time);
        append(evidence, problem.due_date);
        append(evidence, problem.service_time);
        append(evidence, problem.distance);
        append(evidence, problem.reachable);
        append(evidence, problem.vehicle);
        append(evidence, problem.lexical_rank);
        append(evidence, problem.node_name_offsets);
        append(evidence, problem.node_name_bytes);
        append(evidence, problem.initial_route_offsets);
        append(evidence, problem.initial_route_indices);
        append(evidence, config.search_control);
        // The relative budget is semantic input.  The absolute monotonic
        // boundary is transport/runtime state and must not make otherwise
        // identical local and host requests hash differently.
        append_scalar(evidence, config.deadline[0]);
        append(evidence, config.protocol_control);
        append(evidence, config.protocol_options);
        append(evidence, config.stage04_integer);
        append(evidence, config.stage04_float);
        append(evidence, config.operator_integer);
        append(evidence, config.operator_float);
        return native_protocol::native_sha256_hex(evidence);
    }

private:
    template <typename Container>
    static void append(std::string& output, const Container& values) {
        const auto count = static_cast<std::uint64_t>(values.size());
        append_scalar(output, count);
        if (!values.empty()) {
            const auto bytes = values.size() * sizeof(typename Container::value_type);
            output.append(
                reinterpret_cast<const char*>(values.data()), bytes);
        }
    }

    template <typename T>
    static void append_scalar(std::string& output, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        output.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
};

class PythonCompatibleFloatSumV2 final {
public:
    void add(double value) {
        const auto next = total_ + value;
        if (std::fabs(total_) >= std::fabs(value)) {
            compensation_ += (total_ - next) + value;
        } else {
            compensation_ += (value - next) + total_;
        }
        total_ = next;
    }

    [[nodiscard]] double value() const {
        if (compensation_ != 0.0 && std::isfinite(compensation_)) {
            return total_ + compensation_;
        }
        return total_;
    }

private:
    double total_ = 0.0;
    double compensation_ = 0.0;
};

struct InitialStateV2 final {
    native_kernels::ExactBatchOutput exact;
    std::array<std::int64_t, 2> objective_integer{};
    std::array<double, 2> objective_float{};
    std::array<std::int64_t, 4> accounting{};
    std::string request_sha256;

    [[nodiscard]] std::string sha256() const {
        if (request_sha256.size() != 64) {
            throw std::logic_error(
                "native initial search state lost its request identity");
        }
        std::string evidence("stage05.2-native-initial-search-state-v2");
        evidence.append(request_sha256);
        append(evidence, exact.path_offsets);
        append(evidence, exact.path_indices);
        append(evidence, exact.statuses);
        append(evidence, exact.reasons);
        append(evidence, exact.metrics);
        append(evidence, exact.label_counters);
        append(evidence, exact.batch_counters);
        append(evidence, exact.completion_order);
        append(evidence, objective_integer);
        append(evidence, objective_float);
        append(evidence, accounting);
        return native_protocol::native_sha256_hex(evidence);
    }

    void validate(const RequestV2& request) const {
        request.validate();
        if (request_sha256 != request.sha256()) {
            throw std::runtime_error(
                "native initial search state request identity is invalid");
        }
        const auto route_count = request.problem.route_count();
        if (exact.path_offsets.size() != route_count + 1
            || exact.path_offsets.front() != 0
            || exact.path_offsets.back()
                != static_cast<std::int64_t>(exact.path_indices.size())
            || exact.statuses.size() != route_count
            || exact.reasons.size() != route_count
            || exact.metrics.size() != route_count * 4
            || exact.label_counters.size() != route_count * 3
            || exact.batch_counters.size() != 10
            || exact.completion_order.size() != route_count) {
            throw std::runtime_error(
                "native initial search state typed schema is invalid");
        }
        std::vector<bool> completion_seen(route_count, false);
        for (const auto ordinal : exact.completion_order) {
            if (ordinal < 0 || static_cast<std::size_t>(ordinal) >= route_count
                || completion_seen[static_cast<std::size_t>(ordinal)]) {
                throw std::runtime_error(
                    "native initial search state completion order is invalid");
            }
            completion_seen[static_cast<std::size_t>(ordinal)] = true;
        }
        std::int64_t depot = -1;
        for (std::size_t node = 0; node < request.problem.node_count(); ++node) {
            if (request.problem.node_kind[node] == native_kernels::depot_kind) {
                if (depot >= 0) {
                    throw std::runtime_error(
                        "native initial search state problem has multiple depots");
                }
                depot = static_cast<std::int64_t>(node);
            }
        }
        if (depot < 0) {
            throw std::runtime_error(
                "native initial search state problem has no depot");
        }
        double total_distance = 0.0;
        double total_charging_time = 0.0;
        std::int64_t charging_count = 0;
        for (std::size_t route = 0; route < route_count; ++route) {
            const auto first = exact.path_offsets[route];
            const auto last = exact.path_offsets[route + 1];
            auto expected_customer = request.problem.initial_route_offsets[route];
            const auto expected_customer_end =
                request.problem.initial_route_offsets[route + 1];
            if (first < 0 || first >= last
                || last > static_cast<std::int64_t>(exact.path_indices.size())
                || exact.path_indices[static_cast<std::size_t>(first)] != depot
                || exact.path_indices[static_cast<std::size_t>(last - 1)]
                    != depot
                || exact.statuses[route] != native_kernels::feasible_status
                || exact.reasons[route] != native_kernels::no_failure_reason) {
                throw std::runtime_error(
                    "native initial search state route result is invalid");
            }
            for (auto position = first; position < last; ++position) {
                const auto node = exact.path_indices[
                    static_cast<std::size_t>(position)];
                if (node < 0
                    || static_cast<std::size_t>(node)
                        >= request.problem.node_count()) {
                    throw std::runtime_error(
                        "native initial search state path node is invalid");
                }
                charging_count += request.problem.node_kind[
                        static_cast<std::size_t>(node)]
                        == native_kernels::station_kind
                    ? 1 : 0;
                if (request.problem.node_kind[static_cast<std::size_t>(node)]
                    == native_kernels::customer_kind) {
                    if (expected_customer >= expected_customer_end
                        || request.problem.initial_route_indices[
                            static_cast<std::size_t>(expected_customer)]
                            != node) {
                        throw std::runtime_error(
                            "native initial search state path/customer order is invalid");
                    }
                    ++expected_customer;
                } else if (node == depot && position != first
                    && position != last - 1) {
                    throw std::runtime_error(
                        "native initial search state path contains an interior depot");
                }
            }
            if (expected_customer != expected_customer_end) {
                throw std::runtime_error(
                    "native initial search state path omits a customer");
            }
            for (std::size_t field = 0; field < 4; ++field) {
                const auto value = exact.metrics[route * 4 + field];
                if (!std::isfinite(value) || value < 0.0) {
                    throw std::runtime_error(
                        "native initial search state metric is invalid");
                }
            }
            total_distance += exact.metrics[route * 4];
            total_charging_time += exact.metrics[route * 4 + 3];
        }
        if (std::any_of(
                exact.label_counters.begin(), exact.label_counters.end(),
                [](std::int64_t value) { return value < 0; })) {
            throw std::runtime_error(
                "native initial search state label counter is invalid");
        }
        const auto& counters = exact.batch_counters;
        if (counters[0] != static_cast<std::int64_t>(route_count)
            || counters[1] != static_cast<std::int64_t>(route_count)
            || counters[2] != static_cast<std::int64_t>(route_count)
            || counters[3] != 0 || counters[4] != 1 || counters[8] != 1
            || counters[9] != request.config.search_control[2]
            || std::any_of(
                counters.begin() + 5, counters.begin() + 8,
                [](std::int64_t value) { return value < 0; })) {
            throw std::runtime_error(
                "native initial search state exact counters are invalid");
        }
        if (objective_integer[0] != static_cast<std::int64_t>(route_count)
            || objective_integer[1] != charging_count
            || objective_float[0] != total_distance
            || objective_float[1] != total_charging_time
            || accounting
                != std::array<std::int64_t, 4>{
                    static_cast<std::int64_t>(route_count),
                    static_cast<std::int64_t>(route_count), 0, 0}) {
            throw std::runtime_error(
                "native initial search state objective/accounting is invalid");
        }
    }

private:
    template <typename Container>
    static void append(std::string& output, const Container& values) {
        const auto count = static_cast<std::uint64_t>(values.size());
        output.append(reinterpret_cast<const char*>(&count), sizeof(count));
        if (!values.empty()) {
            output.append(
                reinterpret_cast<const char*>(values.data()),
                values.size() * sizeof(typename Container::value_type));
        }
    }
};

inline InitialStateV2 initialize_state(const RequestV2& request) {
    request.validate();
    const auto& problem = request.problem;
    const auto route_count = problem.route_count();
    const auto exact_budget = request.config.search_control[4];
    if (exact_budget >= 0
        && route_count > static_cast<std::size_t>(exact_budget)) {
        throw std::runtime_error(
            "native search warm start does not fit the exact-call budget");
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> stations;
    for (std::size_t node = 0; node < problem.node_count(); ++node) {
        if (problem.node_kind[node] == native_kernels::depot_kind) {
            if (depot >= 0) {
                throw std::invalid_argument(
                    "native search problem contains multiple depots");
            }
            depot = static_cast<std::int64_t>(node);
        } else if (problem.node_kind[node] == native_kernels::station_kind) {
            stations.push_back(static_cast<std::int64_t>(node));
        }
    }
    if (depot < 0) {
        throw std::invalid_argument(
            "native search problem does not contain a depot");
    }
    InitialStateV2 state;
    state.request_sha256 = request.sha256();
    const auto remaining_seconds = request.config.deadline[1]
        - std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    if (remaining_seconds <= 0.0) {
        throw std::runtime_error(
            "native search deadline expired before initial exact work");
    }
    state.exact = native_kernels::run_exact_charging_batch(
        problem.node_kind.data(), problem.ready_time.data(),
        problem.due_date.data(), problem.service_time.data(),
        problem.distance.data(), problem.vehicle.data(),
        problem.initial_route_offsets.data(),
        problem.initial_route_indices.data(), problem.node_count(), route_count,
        depot, stations, remaining_seconds,
        request.config.search_control[2]);
    if (state.exact.statuses.size() != route_count
        || state.exact.reasons.size() != route_count
        || state.exact.metrics.size() != route_count * 4
        || state.exact.label_counters.size() != route_count * 3
        || state.exact.path_offsets.size() != route_count + 1
        || state.exact.batch_counters.size() != 10) {
        throw std::logic_error(
            "native search initial exact state has an invalid schema");
    }
    if (std::any_of(
            state.exact.statuses.begin(), state.exact.statuses.end(),
            [](std::int64_t status) {
                return status != native_kernels::feasible_status;
            })) {
        throw std::runtime_error(
            "native search supplied warm start is not exact-feasible");
    }
    double total_distance = 0.0;
    double total_charging_time = 0.0;
    for (std::size_t route = 0; route < route_count; ++route) {
        total_distance += state.exact.metrics[route * 4];
        total_charging_time += state.exact.metrics[route * 4 + 3];
    }
    std::int64_t charging_count = 0;
    for (const auto node : state.exact.path_indices) {
        if (node < 0
            || static_cast<std::size_t>(node) >= problem.node_count()) {
            throw std::logic_error(
                "native search initial exact path contains an invalid node");
        }
        charging_count += problem.node_kind[static_cast<std::size_t>(node)]
                == native_kernels::station_kind
            ? 1 : 0;
    }
    state.objective_integer = {
        static_cast<std::int64_t>(route_count), charging_count};
    state.objective_float = {total_distance, total_charging_time};
    state.accounting = {
        static_cast<std::int64_t>(route_count),
        state.exact.batch_counters[2], state.exact.batch_counters[3], 0};
    if (std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count()
        >= request.config.deadline[1]) {
        throw std::runtime_error(
            "native search initial exact work completed at or after deadline");
    }
    state.validate(request);
    return state;
}

struct LaneStateV2 final {
    std::vector<std::int64_t> route_offsets;
    std::vector<std::int64_t> route_indices;
    native_kernels::ExactBatchOutput exact;
    std::array<std::int64_t, 2> objective_integer{};
    std::array<double, 2> objective_float{};

    [[nodiscard]] std::string sha256() const {
        std::string evidence("stage05.2-native-lane-state-v2");
        append(evidence, route_offsets);
        append(evidence, route_indices);
        append(evidence, exact.path_offsets);
        append(evidence, exact.path_indices);
        append(evidence, exact.statuses);
        append(evidence, exact.reasons);
        append(evidence, exact.metrics);
        append(evidence, exact.label_counters);
        append(evidence, exact.batch_counters);
        append(evidence, exact.completion_order);
        append(evidence, objective_integer);
        append(evidence, objective_float);
        return native_protocol::native_sha256_hex(evidence);
    }

    void validate_initial(const RequestV2& request) const {
        if (route_offsets != request.problem.initial_route_offsets
            || route_indices != request.problem.initial_route_indices) {
            throw std::runtime_error(
                "native initial lane routes do not match the owned request");
        }
        InitialStateV2 initial;
        initial.exact = exact;
        initial.objective_integer = objective_integer;
        initial.objective_float = objective_float;
        const auto route_count = static_cast<std::int64_t>(
            request.problem.route_count());
        initial.accounting = {route_count, route_count, 0, 0};
        initial.request_sha256 = request.sha256();
        initial.validate(request);
    }

    void validate_live(const ProblemV2& problem) const {
        problem.validate();
        validate_live_assuming_problem_valid(problem);
    }

    // The search engine freezes and validates one ProblemV2 during
    // initialization.  Re-validating its O(node_count^2) distance and
    // reachability buffers at every lane read would turn an invariant check
    // into the dominant solve cost.  This entry keeps the complete lane-level
    // validation while relying on that immutable initialization gate.
    void validate_live_assuming_problem_valid(const ProblemV2& problem) const {
        if (route_offsets.size() < 2 || route_offsets.front() != 0
            || route_offsets.back()
                != static_cast<std::int64_t>(route_indices.size())) {
            throw std::runtime_error(
                "native live lane route boundary is invalid");
        }
        const auto route_count = route_offsets.size() - 1;
        for (std::size_t route = 0; route < route_count; ++route) {
            if (route_offsets[route] < 0
                || route_offsets[route] >= route_offsets[route + 1]) {
                throw std::runtime_error(
                    "native live lane route offsets are not monotone");
            }
        }
        std::unordered_set<std::int64_t> expected_customers;
        std::unordered_set<std::int64_t> route_customers;
        std::int64_t depot = -1;
        for (std::size_t node = 0; node < problem.node_count(); ++node) {
            if (problem.node_kind[node] == native_kernels::customer_kind) {
                expected_customers.insert(static_cast<std::int64_t>(node));
            } else if (problem.node_kind[node] == native_kernels::depot_kind) {
                if (depot >= 0) {
                    throw std::runtime_error(
                        "native live lane problem has multiple depots");
                }
                depot = static_cast<std::int64_t>(node);
            }
        }
        for (const auto customer : route_indices) {
            if (customer < 0
                || static_cast<std::size_t>(customer) >= problem.node_count()
                || problem.node_kind[static_cast<std::size_t>(customer)]
                    != native_kernels::customer_kind
                || !route_customers.insert(customer).second) {
                throw std::runtime_error(
                    "native live lane contains an invalid customer");
            }
        }
        if (depot < 0 || route_customers != expected_customers) {
            throw std::runtime_error(
                "native live lane does not cover every customer exactly once");
        }
        if (exact.path_offsets.size() != route_count + 1
            || exact.path_offsets.front() != 0
            || exact.path_offsets.back()
                != static_cast<std::int64_t>(exact.path_indices.size())
            || exact.statuses.size() != route_count
            || exact.reasons.size() != route_count
            || exact.metrics.size() != route_count * 4
            || exact.label_counters.size() != route_count * 3
            || (!exact.batch_counters.empty()
                && exact.batch_counters.size() != 10)) {
            throw std::runtime_error(
                "native live lane exact result has an invalid schema");
        }
        PythonCompatibleFloatSumV2 total_distance;
        PythonCompatibleFloatSumV2 total_charging_time;
        double ordered_total_distance = 0.0;
        double ordered_total_charging_time = 0.0;
        std::int64_t charging_count = 0;
        for (std::size_t route = 0; route < route_count; ++route) {
            const auto first_path = exact.path_offsets[route];
            const auto end_path = exact.path_offsets[route + 1];
            auto expected = route_offsets[route];
            const auto expected_end = route_offsets[route + 1];
            if (first_path < 0 || first_path >= end_path
                || end_path > static_cast<std::int64_t>(exact.path_indices.size())
                || exact.path_indices[static_cast<std::size_t>(first_path)]
                    != depot
                || exact.path_indices[static_cast<std::size_t>(end_path - 1)]
                    != depot
                || exact.statuses[route] != native_kernels::feasible_status
                || exact.reasons[route] != native_kernels::no_failure_reason) {
                throw std::runtime_error(
                    "native live lane exact route is infeasible");
            }
            for (auto position = first_path; position < end_path; ++position) {
                const auto node = exact.path_indices[
                    static_cast<std::size_t>(position)];
                if (node < 0
                    || static_cast<std::size_t>(node) >= problem.node_count()) {
                    throw std::runtime_error(
                        "native live lane exact path node is invalid");
                }
                const auto kind = problem.node_kind[
                    static_cast<std::size_t>(node)];
                if (kind == native_kernels::customer_kind) {
                    if (expected >= expected_end
                        || route_indices[static_cast<std::size_t>(expected)]
                            != node) {
                        throw std::runtime_error(
                            "native live lane exact customer order is invalid");
                    }
                    ++expected;
                } else if (kind == native_kernels::station_kind) {
                    ++charging_count;
                } else if (node == depot && position != first_path
                    && position != end_path - 1) {
                    throw std::runtime_error(
                        "native live lane exact path has an interior depot");
                }
            }
            if (expected != expected_end) {
                throw std::runtime_error(
                    "native live lane exact path omits a customer");
            }
            for (std::size_t field = 0; field < 4; ++field) {
                const auto value = exact.metrics[route * 4 + field];
                if (!std::isfinite(value) || value < 0.0) {
                    throw std::runtime_error(
                        "native live lane exact metric is invalid");
                }
            }
            total_distance.add(exact.metrics[route * 4]);
            total_charging_time.add(exact.metrics[route * 4 + 3]);
            ordered_total_distance += exact.metrics[route * 4];
            ordered_total_charging_time += exact.metrics[route * 4 + 3];
        }
        const auto expected_distance = exact.batch_counters.empty()
            ? total_distance.value() : ordered_total_distance;
        const auto expected_charging_time = exact.batch_counters.empty()
            ? total_charging_time.value() : ordered_total_charging_time;
        if (objective_integer[0] != static_cast<std::int64_t>(route_count)
            || objective_integer[1] != charging_count
            || objective_float[0] != expected_distance
            || objective_float[1] != expected_charging_time
            || std::any_of(
                exact.label_counters.begin(), exact.label_counters.end(),
                [](std::int64_t value) { return value < 0; })) {
            throw std::runtime_error(
                "native live lane objective/accounting is invalid");
        }
    }

private:
    template <typename Container>
    static void append(std::string& output, const Container& values) {
        const auto count = static_cast<std::uint64_t>(values.size());
        output.append(reinterpret_cast<const char*>(&count), sizeof(count));
        if (!values.empty()) {
            output.append(
                reinterpret_cast<const char*>(values.data()),
                values.size() * sizeof(typename Container::value_type));
        }
    }
};

struct InitialFourLaneStateV2 final {
    // `constraint` is the existing engine's global/current constraint-guided
    // lane.  The other two Stage 2.3 lanes and global best are separately
    // owned from the first exact transaction onward.
    LaneStateV2 constraint;
    LaneStateV2 legacy;
    LaneStateV2 quality_shadow;
    LaneStateV2 global_best;
    std::array<std::int64_t, 4> accounting{};
    std::array<std::int64_t, 2> rng_seeds{};
    std::int64_t next_iteration = 0;
    std::vector<std::int64_t> node_kind;
    std::int64_t exact_batch_size = 0;
    std::string request_sha256;
    std::string initial_state_sha256;

    [[nodiscard]] std::string sha256() const {
        if (request_sha256.size() != 64 || initial_state_sha256.size() != 64) {
            throw std::logic_error(
                "native search state lost its request/initial identity");
        }
        std::string evidence("stage05.2-native-initial-four-lane-state-v2");
        evidence.append(request_sha256);
        evidence.append(initial_state_sha256);
        for (const auto* lane : {
                 &constraint, &legacy, &quality_shadow, &global_best}) {
            evidence.append(lane->sha256());
        }
        append(evidence, accounting);
        append(evidence, rng_seeds);
        append_scalar(evidence, next_iteration);
        append(evidence, node_kind);
        append_scalar(evidence, exact_batch_size);
        return native_protocol::native_sha256_hex(evidence);
    }

    void validate_initial(const RequestV2& request) const {
        request.validate();
        if (request_sha256 != request.sha256() || next_iteration != 0) {
            throw std::runtime_error(
                "native initial search state identity/iteration is invalid");
        }
        if (node_kind != request.problem.node_kind
            || exact_batch_size != request.config.search_control[2]) {
            throw std::runtime_error(
                "native initial search state problem/config projection is invalid");
        }
        constraint.validate_initial(request);
        legacy.validate_initial(request);
        quality_shadow.validate_initial(request);
        global_best.validate_initial(request);
        const auto constraint_sha256 = constraint.sha256();
        if (legacy.sha256() != constraint_sha256
            || quality_shadow.sha256() != constraint_sha256
            || global_best.sha256() != constraint_sha256) {
            throw std::runtime_error(
                "native initial lane states are not identical");
        }
        InitialStateV2 reconstructed;
        reconstructed.exact = constraint.exact;
        reconstructed.objective_integer = constraint.objective_integer;
        reconstructed.objective_float = constraint.objective_float;
        reconstructed.accounting = accounting;
        reconstructed.request_sha256 = request_sha256;
        reconstructed.validate(request);
        if (initial_state_sha256 != reconstructed.sha256()) {
            throw std::runtime_error(
                "native search state lost its initial-state identity");
        }
        const auto seed = request.config.search_control[0];
        if (rng_seeds != std::array<std::int64_t, 2>{
                seed, seed ^ std::int64_t{0x5EED23}}) {
            throw std::runtime_error(
                "native search state RNG seeds are invalid");
        }
    }

private:
    template <typename Container>
    static void append(std::string& output, const Container& values) {
        const auto count = static_cast<std::uint64_t>(values.size());
        append_scalar(output, count);
        if (!values.empty()) {
            output.append(
                reinterpret_cast<const char*>(values.data()),
                values.size() * sizeof(typename Container::value_type));
        }
    }

    template <typename T>
    static void append_scalar(std::string& output, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        output.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
};

inline InitialFourLaneStateV2 initialize_initial_four_lane_state(
    const RequestV2& request,
    const InitialStateV2& initial) {
    initial.validate(request);
    LaneStateV2 lane{
        request.problem.initial_route_offsets,
        request.problem.initial_route_indices,
        initial.exact,
        initial.objective_integer,
        initial.objective_float,
    };
    InitialFourLaneStateV2 state{
        lane,
        lane,
        lane,
        std::move(lane),
        initial.accounting,
        {
            request.config.search_control[0],
            request.config.search_control[0] ^ std::int64_t{0x5EED23},
        },
        0,
        request.problem.node_kind,
        request.config.search_control[2],
        initial.request_sha256,
        initial.sha256(),
    };
    state.validate_initial(request);
    return state;
}

inline InitialFourLaneStateV2 initialize_initial_four_lane_state(
    const RequestV2& request) {
    const auto initial = initialize_state(request);
    return initialize_initial_four_lane_state(request, initial);
}

inline std::array<std::int64_t, 8> receipt_counts(
    const RequestV2& request) {
    request.validate();
    return {
        static_cast<std::int64_t>(request.problem.node_count()),
        static_cast<std::int64_t>(request.problem.route_count()),
        static_cast<std::int64_t>(
            request.problem.initial_route_indices.size()),
        static_cast<std::int64_t>(request.problem.node_name_bytes.size()),
        static_cast<std::int64_t>(request.problem.distance.size()),
        request.config.search_control[1],
        request.config.protocol_control[2],
        request.config.search_control[3],
    };
}

inline std::vector<std::uint8_t> request_payload(
    const RequestV2& request,
    std::uint64_t request_id,
    native_protocol::KernelOperation operation =
        native_protocol::KernelOperation::search_request_receipt) {
    request.validate();
    if (operation != native_protocol::KernelOperation::search_request_receipt
        && operation
            != native_protocol::KernelOperation::search_initial_state) {
        throw std::invalid_argument(
            "native search request operation is invalid");
    }
    native_protocol::PayloadBuilder builder(
        operation, request_id);
    const auto& problem = request.problem;
    const auto nodes = problem.node_count();
    builder.add(native_protocol::NumericType::int64, problem.node_kind.data(),
        problem.node_kind.size(), problem.node_kind.size());
    builder.add(native_protocol::NumericType::float64, problem.demand.data(),
        problem.demand.size(), problem.demand.size());
    builder.add(native_protocol::NumericType::float64, problem.ready_time.data(),
        problem.ready_time.size(), problem.ready_time.size());
    builder.add(native_protocol::NumericType::float64, problem.due_date.data(),
        problem.due_date.size(), problem.due_date.size());
    builder.add(native_protocol::NumericType::float64, problem.service_time.data(),
        problem.service_time.size(), problem.service_time.size());
    builder.add(native_protocol::NumericType::float64, problem.distance.data(),
        problem.distance.size(), nodes, nodes);
    builder.add(native_protocol::NumericType::uint8, problem.reachable.data(),
        problem.reachable.size(), nodes, nodes);
    builder.add(native_protocol::NumericType::float64, problem.vehicle.data(),
        problem.vehicle.size(), problem.vehicle.size());
    builder.add(native_protocol::NumericType::int64, problem.lexical_rank.data(),
        problem.lexical_rank.size(), problem.lexical_rank.size());
    builder.add(native_protocol::NumericType::int64,
        problem.node_name_offsets.data(), problem.node_name_offsets.size(),
        problem.node_name_offsets.size());
    builder.add(native_protocol::NumericType::uint8,
        problem.node_name_bytes.data(), problem.node_name_bytes.size(),
        problem.node_name_bytes.size());
    builder.add(native_protocol::NumericType::int64,
        problem.initial_route_offsets.data(),
        problem.initial_route_offsets.size(),
        problem.initial_route_offsets.size());
    builder.add(native_protocol::NumericType::int64,
        problem.initial_route_indices.data(),
        problem.initial_route_indices.size(),
        problem.initial_route_indices.size());
    const auto& config = request.config;
    builder.add(native_protocol::NumericType::int64,
        config.search_control.data(), config.search_control.size(),
        config.search_control.size());
    builder.add(native_protocol::NumericType::float64,
        config.deadline.data(), config.deadline.size(), config.deadline.size());
    builder.add(native_protocol::NumericType::int64,
        config.protocol_control.data(), config.protocol_control.size(),
        config.protocol_control.size());
    builder.add(native_protocol::NumericType::float64,
        config.protocol_options.data(), config.protocol_options.size(),
        config.protocol_options.size());
    builder.add(native_protocol::NumericType::int64,
        config.stage04_integer.data(), config.stage04_integer.size(),
        config.stage04_integer.size());
    builder.add(native_protocol::NumericType::float64,
        config.stage04_float.data(), config.stage04_float.size(),
        config.stage04_float.size());
    builder.add(native_protocol::NumericType::int64,
        config.operator_integer.data(), config.operator_integer.size(),
        config.operator_integer.size());
    builder.add(native_protocol::NumericType::float64,
        config.operator_float.data(), config.operator_float.size(),
        config.operator_float.size());
    return builder.finish();
}

inline RequestV2 request_from_payload(
    const native_protocol::PayloadView& payload) {
    if ((payload.header().operation
            != native_protocol::KernelOperation::search_request_receipt
        && payload.header().operation
            != native_protocol::KernelOperation::search_initial_state)
        || payload.header().array_count != 21) {
        throw std::runtime_error(
            "native search request payload envelope is invalid");
    }
    const auto require_vector = [&payload](std::size_t index) {
        const auto& descriptor = payload.descriptor(index);
        if (descriptor.dimensions != 1
            || descriptor.shape[0] != descriptor.count
            || descriptor.shape[1] != 0) {
            throw std::runtime_error(
                "native search request vector descriptor is invalid");
        }
    };
    for (const auto index : {
             std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3},
             std::size_t{4}, std::size_t{7}, std::size_t{8}, std::size_t{9},
             std::size_t{10}, std::size_t{11}, std::size_t{12},
             std::size_t{13}, std::size_t{14}, std::size_t{15},
             std::size_t{16}, std::size_t{17}, std::size_t{18},
             std::size_t{19}, std::size_t{20}}) {
        require_vector(index);
    }
    const auto copy_vector = [&payload]<typename T>(
        std::size_t index,
        native_protocol::NumericType type) {
        const auto count = static_cast<std::size_t>(
            payload.descriptor(index).count);
        const auto* values = payload.data<T>(index, type);
        return std::vector<T>(values, values + count);
    };
    const auto copy_array = [&payload]<typename T, std::size_t Size>(
        std::size_t index,
        native_protocol::NumericType type) {
        if (payload.descriptor(index).count != Size
            || payload.descriptor(index).dimensions != 1) {
            throw std::runtime_error(
                "native search request fixed array is invalid");
        }
        std::array<T, Size> output{};
        const auto* values = payload.data<T>(index, type);
        std::copy(values, values + Size, output.begin());
        return output;
    };
    RequestV2 request;
    auto& problem = request.problem;
    problem.node_kind = copy_vector.template operator()<std::int64_t>(
        0, native_protocol::NumericType::int64);
    problem.demand = copy_vector.template operator()<double>(
        1, native_protocol::NumericType::float64);
    problem.ready_time = copy_vector.template operator()<double>(
        2, native_protocol::NumericType::float64);
    problem.due_date = copy_vector.template operator()<double>(
        3, native_protocol::NumericType::float64);
    problem.service_time = copy_vector.template operator()<double>(
        4, native_protocol::NumericType::float64);
    problem.distance = copy_vector.template operator()<double>(
        5, native_protocol::NumericType::float64);
    problem.reachable = copy_vector.template operator()<std::uint8_t>(
        6, native_protocol::NumericType::uint8);
    problem.vehicle = copy_array.template operator()<double, 5>(
        7, native_protocol::NumericType::float64);
    problem.lexical_rank = copy_vector.template operator()<std::int64_t>(
        8, native_protocol::NumericType::int64);
    problem.node_name_offsets = copy_vector.template operator()<std::int64_t>(
        9, native_protocol::NumericType::int64);
    problem.node_name_bytes = copy_vector.template operator()<std::uint8_t>(
        10, native_protocol::NumericType::uint8);
    problem.initial_route_offsets =
        copy_vector.template operator()<std::int64_t>(
            11, native_protocol::NumericType::int64);
    problem.initial_route_indices =
        copy_vector.template operator()<std::int64_t>(
            12, native_protocol::NumericType::int64);
    auto& config = request.config;
    config.search_control = copy_array.template operator()<std::int64_t, 5>(
        13, native_protocol::NumericType::int64);
    config.deadline = copy_array.template operator()<double, 2>(
        14, native_protocol::NumericType::float64);
    config.protocol_control = copy_array.template operator()<std::int64_t, 13>(
        15, native_protocol::NumericType::int64);
    config.protocol_options = copy_array.template operator()<double, 2>(
        16, native_protocol::NumericType::float64);
    config.stage04_integer = copy_array.template operator()<std::int64_t, 15>(
        17, native_protocol::NumericType::int64);
    config.stage04_float = copy_array.template operator()<double, 15>(
        18, native_protocol::NumericType::float64);
    config.operator_integer = copy_array.template operator()<std::int64_t, 24>(
        19, native_protocol::NumericType::int64);
    config.operator_float = copy_array.template operator()<double, 7>(
        20, native_protocol::NumericType::float64);
    const auto nodes = problem.node_count();
    if (payload.descriptor(5).dimensions != 2
        || payload.descriptor(5).shape[0] != nodes
        || payload.descriptor(5).shape[1] != nodes
        || payload.descriptor(6).dimensions != 2
        || payload.descriptor(6).shape[0] != nodes
        || payload.descriptor(6).shape[1] != nodes
        || payload.descriptor(14).count != 2
        || payload.descriptor(14).dimensions != 1) {
        throw std::runtime_error(
            "native search request matrix/control shape is invalid");
    }
    request.validate();
    return request;
}

}  // namespace evrptw::native_search
