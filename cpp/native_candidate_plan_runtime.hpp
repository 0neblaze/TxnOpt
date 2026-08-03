#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "native_solver_kernels.hpp"

namespace evrptw::native_candidate_plan {

struct RankingInput final {
    std::span<const std::int64_t> plan_offsets;
    std::span<const std::int64_t> route_offsets;
    std::span<const std::int64_t> route_indices;
    std::span<const double> route_distance_lower_bounds;
    std::span<const std::int64_t> current_route_offsets;
    std::span<const std::int64_t> current_route_indices;
    std::span<const std::int64_t> lexical_rank;
    std::span<const std::int64_t> attempted_flags;
    std::int64_t top_k = 0;
};

struct RankingResult final {
    std::vector<std::int64_t> ranked;
    std::vector<std::int64_t> selected;
    std::vector<std::int64_t> integer_metrics;
    std::vector<double> optimistic_distances;
};

struct PlanPreparationInput final {
    std::span<const std::int64_t> plan_offsets;
    std::span<const std::int64_t> route_offsets;
    std::span<const std::int64_t> route_indices;
    std::span<const std::int64_t> expected_customer_indices;
    std::span<const std::int64_t> node_kind;
    std::span<const std::int64_t> lexical_rank;
    std::span<const std::int64_t> complete_customer_indices;
    std::int64_t customer_kind = 0;
    bool allow_partial_customer_coverage = false;
};

struct PlanPreparationResult final {
    std::vector<std::int64_t> canonical_expected;
    std::vector<std::int64_t> coverage_eligible;
    std::vector<std::int64_t> unique_route_offsets;
    std::vector<std::int64_t> unique_route_indices;
    std::vector<std::int64_t> unique_row_by_route;
};

struct PlanDecisionInput final {
    std::span<const std::int64_t> plan_offsets;
    std::span<const std::int64_t> coverage_eligible;
    std::span<const std::int64_t> screening_passed;
    std::span<const std::int64_t> attempted_flags;
    std::int64_t current_route_count = 0;
};

struct PlanDecisionResult final {
    std::vector<std::int64_t> eligible;
    std::vector<std::int64_t> combined_attempted;
};

struct FeasibleOrderingInput final {
    std::span<const std::int64_t> plan_offsets;
    std::span<const std::int64_t> route_offsets;
    std::span<const std::int64_t> route_indices;
    std::span<const std::int64_t> objective_integer;
    std::span<const double> objective_float;
    std::span<const std::int64_t> lexical_rank;
    std::span<const std::int64_t> feasible_plan_ids;
};

inline void validate_offsets(
    std::span<const std::int64_t> offsets,
    std::size_t terminal,
    const char* name) {
    if (offsets.empty() || offsets.front() != 0
        || offsets.back() != static_cast<std::int64_t>(terminal)) {
        throw std::invalid_argument(std::string(name) + " boundary is invalid");
    }
    for (std::size_t index = 0; index + 1 < offsets.size(); ++index) {
        if (offsets[index] < 0 || offsets[index] > offsets[index + 1]) {
            throw std::invalid_argument(std::string(name) + " must be monotonic");
        }
    }
}

struct RouteSequenceHash final {
    [[nodiscard]] std::size_t operator()(
        const std::vector<std::int64_t>& route) const noexcept {
        std::size_t value = route.size();
        for (const auto node : route) {
            value ^= std::hash<std::int64_t>{}(node)
                + std::size_t{0x9e3779b9} + (value << 6U) + (value >> 2U);
        }
        return value;
    }
};

inline PlanPreparationResult prepare(const PlanPreparationInput& input) {
    if (input.plan_offsets.empty() || input.route_offsets.empty()
        || input.node_kind.empty() || input.lexical_rank.size() != input.node_kind.size()) {
        throw std::invalid_argument(
            "candidate-plan preparation shape/config is invalid");
    }
    const auto plan_count = input.plan_offsets.size() - 1;
    const auto route_count = input.route_offsets.size() - 1;
    if (plan_count == 0 || route_count == 0) {
        throw std::invalid_argument(
            "candidate-plan preparation requires a non-empty plan pool");
    }
    validate_offsets(input.plan_offsets, route_count, "plan_offsets");
    validate_offsets(input.route_offsets, input.route_indices.size(), "route_offsets");

    std::unordered_set<std::int64_t> expected;
    expected.reserve(input.expected_customer_indices.size());
    for (const auto customer : input.expected_customer_indices) {
        if (customer < 0
            || customer >= static_cast<std::int64_t>(input.node_kind.size())
            || input.node_kind[static_cast<std::size_t>(customer)] != input.customer_kind
            || !expected.insert(customer).second) {
            throw std::invalid_argument(
                "expected_customer_indices must contain unique customer nodes");
        }
    }
    if (expected.empty()) {
        throw std::invalid_argument("expected_customer_indices cannot be empty");
    }
    std::unordered_set<std::int64_t> complete(
        input.complete_customer_indices.begin(), input.complete_customer_indices.end());
    if (complete.size() != input.complete_customer_indices.size()) {
        throw std::invalid_argument(
            "complete_customer_indices must contain unique customer nodes");
    }
    for (const auto customer : input.complete_customer_indices) {
        if (customer < 0
            || customer >= static_cast<std::int64_t>(input.node_kind.size())
            || input.node_kind[static_cast<std::size_t>(customer)]
                != input.customer_kind) {
            throw std::invalid_argument(
                "complete_customer_indices must contain only customer nodes");
        }
    }
    if (!input.allow_partial_customer_coverage && expected != complete) {
        throw std::invalid_argument(
            "expected_customer_indices must attest the complete instance customer set");
    }

    PlanPreparationResult output;
    output.canonical_expected.assign(expected.begin(), expected.end());
    std::stable_sort(
        output.canonical_expected.begin(), output.canonical_expected.end(),
        [&](std::int64_t left, std::int64_t right) {
            return input.lexical_rank[static_cast<std::size_t>(left)]
                < input.lexical_rank[static_cast<std::size_t>(right)];
        });
    output.coverage_eligible.assign(plan_count, 1);
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        std::unordered_set<std::int64_t> observed;
        for (auto route = input.plan_offsets[plan];
             route < input.plan_offsets[plan + 1]; ++route) {
            for (auto cursor = input.route_offsets[static_cast<std::size_t>(route)];
                 cursor < input.route_offsets[static_cast<std::size_t>(route) + 1];
                 ++cursor) {
                const auto node = input.route_indices[static_cast<std::size_t>(cursor)];
                if (node < 0
                    || node >= static_cast<std::int64_t>(input.node_kind.size())
                    || input.node_kind[static_cast<std::size_t>(node)]
                        != input.customer_kind) {
                    throw std::invalid_argument(
                        "candidate plans must contain only customer nodes");
                }
                if (!observed.insert(node).second) {
                    output.coverage_eligible[plan] = 0;
                }
            }
        }
        if (observed != expected) {
            output.coverage_eligible[plan] = 0;
        }
    }

    output.unique_route_offsets.push_back(0);
    output.unique_row_by_route.resize(route_count);
    std::unordered_map<std::vector<std::int64_t>, std::size_t, RouteSequenceHash>
        unique_rows;
    unique_rows.reserve(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        std::vector<std::int64_t> sequence(
            input.route_indices.begin() + input.route_offsets[route],
            input.route_indices.begin() + input.route_offsets[route + 1]);
        const auto next_row = unique_rows.size();
        const auto [position, inserted] = unique_rows.emplace(sequence, next_row);
        if (inserted) {
            output.unique_route_indices.insert(
                output.unique_route_indices.end(), sequence.begin(), sequence.end());
            output.unique_route_offsets.push_back(
                static_cast<std::int64_t>(output.unique_route_indices.size()));
        }
        output.unique_row_by_route[route] =
            static_cast<std::int64_t>(position->second);
    }
    return output;
}

inline PlanDecisionResult decide(const PlanDecisionInput& input) {
    if (input.plan_offsets.empty() || input.current_route_count <= 0) {
        throw std::invalid_argument("candidate-plan decision shape/config is invalid");
    }
    const auto plan_count = input.plan_offsets.size() - 1;
    const auto route_count = static_cast<std::size_t>(input.plan_offsets.back());
    if (input.coverage_eligible.size() != plan_count
        || input.screening_passed.size() != route_count
        || input.attempted_flags.size() != plan_count) {
        throw std::invalid_argument("candidate-plan decision arrays do not align");
    }
    validate_offsets(input.plan_offsets, route_count, "plan_offsets");
    PlanDecisionResult output;
    output.eligible.assign(input.coverage_eligible.begin(), input.coverage_eligible.end());
    output.combined_attempted.resize(plan_count);
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        if (output.eligible[plan] != 0 && output.eligible[plan] != 1) {
            throw std::invalid_argument("coverage_eligible must contain zero or one");
        }
        if (input.attempted_flags[plan] != 0 && input.attempted_flags[plan] != 1) {
            throw std::invalid_argument("attempted_flags must contain zero or one");
        }
        for (auto route = input.plan_offsets[plan];
             route < input.plan_offsets[plan + 1]; ++route) {
            const auto screened = input.screening_passed[static_cast<std::size_t>(route)];
            if (screened != 0 && screened != 1) {
                throw std::invalid_argument(
                    "screening_passed must contain zero or one");
            }
            if (screened == 0) {
                output.eligible[plan] = 0;
            }
        }
        const auto vehicle_count = input.plan_offsets[plan + 1] - input.plan_offsets[plan];
        if (vehicle_count > input.current_route_count) {
            output.eligible[plan] = 0;
        }
        output.combined_attempted[plan] =
            output.eligible[plan] == 0 || input.attempted_flags[plan] != 0 ? 1 : 0;
    }
    return output;
}

inline std::vector<std::int64_t> order_feasible(
    const FeasibleOrderingInput& input) {
    if (input.plan_offsets.empty() || input.route_offsets.empty()
        || input.lexical_rank.empty()) {
        throw std::invalid_argument("feasible-plan ordering shape is invalid");
    }
    const auto plan_count = input.plan_offsets.size() - 1;
    const auto route_count = input.route_offsets.size() - 1;
    validate_offsets(input.plan_offsets, route_count, "plan_offsets");
    validate_offsets(input.route_offsets, input.route_indices.size(), "route_offsets");
    if (input.objective_integer.size() != plan_count * 2
        || input.objective_float.size() != plan_count * 2) {
        throw std::invalid_argument("feasible-plan objectives do not align");
    }
    std::unordered_set<std::int64_t> unique_plan_ids;
    for (const auto plan : input.feasible_plan_ids) {
        if (plan < 0 || plan >= static_cast<std::int64_t>(plan_count)
            || !unique_plan_ids.insert(plan).second) {
            throw std::invalid_argument(
                "feasible_plan_ids must identify unique candidate plans");
        }
        const auto offset = static_cast<std::size_t>(plan) * 2;
        if (input.objective_integer[offset] < 0
            || input.objective_integer[offset + 1] < 0
            || !std::isfinite(input.objective_float[offset])
            || !std::isfinite(input.objective_float[offset + 1])) {
            throw std::invalid_argument(
                "feasible plans require complete finite objective fields");
        }
    }
    const auto route_less = [&](std::int64_t left, std::int64_t right) {
        auto left_cursor = input.route_offsets[static_cast<std::size_t>(left)];
        auto right_cursor = input.route_offsets[static_cast<std::size_t>(right)];
        const auto left_end = input.route_offsets[static_cast<std::size_t>(left) + 1];
        const auto right_end = input.route_offsets[static_cast<std::size_t>(right) + 1];
        while (left_cursor < left_end && right_cursor < right_end) {
            const auto left_node = input.route_indices[
                static_cast<std::size_t>(left_cursor)];
            const auto right_node = input.route_indices[
                static_cast<std::size_t>(right_cursor)];
            if (left_node < 0 || right_node < 0
                || left_node >= static_cast<std::int64_t>(input.lexical_rank.size())
                || right_node >= static_cast<std::int64_t>(input.lexical_rank.size())) {
                throw std::invalid_argument(
                    "feasible candidate plan contains an unknown node");
            }
            const auto left_rank = input.lexical_rank[
                static_cast<std::size_t>(left_node)];
            const auto right_rank = input.lexical_rank[
                static_cast<std::size_t>(right_node)];
            if (left_rank != right_rank) {
                return left_rank < right_rank;
            }
            ++left_cursor;
            ++right_cursor;
        }
        return left_end - input.route_offsets[static_cast<std::size_t>(left)]
            < right_end - input.route_offsets[static_cast<std::size_t>(right)];
    };
    const auto round_objective = [](double value) {
        constexpr auto scale = 1'000'000'000.0;
        return std::nearbyint(value * scale) / scale;
    };
    const auto plan_less = [&](std::int64_t left, std::int64_t right) {
        const auto left_offset = static_cast<std::size_t>(left) * 2;
        const auto right_offset = static_cast<std::size_t>(right) * 2;
        const auto left_key = std::make_tuple(
            input.objective_integer[left_offset],
            round_objective(input.objective_float[left_offset]),
            round_objective(input.objective_float[left_offset + 1]),
            input.objective_integer[left_offset + 1]);
        const auto right_key = std::make_tuple(
            input.objective_integer[right_offset],
            round_objective(input.objective_float[right_offset]),
            round_objective(input.objective_float[right_offset + 1]),
            input.objective_integer[right_offset + 1]);
        if (left_key != right_key) {
            return left_key < right_key;
        }
        auto left_route = input.plan_offsets[static_cast<std::size_t>(left)];
        auto right_route = input.plan_offsets[static_cast<std::size_t>(right)];
        const auto left_end = input.plan_offsets[static_cast<std::size_t>(left) + 1];
        const auto right_end = input.plan_offsets[static_cast<std::size_t>(right) + 1];
        while (left_route < left_end && right_route < right_end) {
            if (route_less(left_route, right_route)) {
                return true;
            }
            if (route_less(right_route, left_route)) {
                return false;
            }
            ++left_route;
            ++right_route;
        }
        return left_end - input.plan_offsets[static_cast<std::size_t>(left)]
            < right_end - input.plan_offsets[static_cast<std::size_t>(right)];
    };
    std::vector<std::int64_t> output(
        input.feasible_plan_ids.begin(), input.feasible_plan_ids.end());
    std::stable_sort(output.begin(), output.end(), plan_less);
    return output;
}

inline RankingResult rank(const RankingInput& input) {
    if (input.plan_offsets.empty() || input.route_offsets.empty()
        || input.current_route_offsets.empty() || input.lexical_rank.empty()
        || input.top_k <= 0) {
        throw std::invalid_argument("candidate-plan ranking shape/config is invalid");
    }
    const auto plan_count = input.plan_offsets.size() - 1;
    const auto route_count = input.route_offsets.size() - 1;
    const auto current_count = input.current_route_offsets.size() - 1;
    if (input.route_distance_lower_bounds.size() != route_count
        || input.attempted_flags.size() != plan_count) {
        throw std::invalid_argument(
            "candidate-plan route metrics/flags do not align");
    }
    validate_offsets(input.plan_offsets, route_count, "plan_offsets");
    validate_offsets(
        input.route_offsets, input.route_indices.size(), "route_offsets");
    validate_offsets(
        input.current_route_offsets, input.current_route_indices.size(),
        "current_route_offsets");
    std::unordered_set<std::int64_t> lexical_values;
    lexical_values.reserve(input.lexical_rank.size());
    for (const auto rank : input.lexical_rank) {
        if (rank < 0
            || rank >= static_cast<std::int64_t>(input.lexical_rank.size())
            || !lexical_values.insert(rank).second) {
            throw std::invalid_argument("lexical_rank must be a permutation");
        }
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        const auto lower_bound = input.route_distance_lower_bounds[route];
        if (!std::isfinite(lower_bound) || lower_bound < 0.0) {
            throw std::invalid_argument(
                "route_distance_lower_bounds must be finite and non-negative");
        }
        for (auto cursor = input.route_offsets[route];
             cursor < input.route_offsets[route + 1]; ++cursor) {
            const auto node = input.route_indices[static_cast<std::size_t>(cursor)];
            if (node < 0
                || node >= static_cast<std::int64_t>(input.lexical_rank.size())) {
                throw std::invalid_argument(
                    "candidate plan contains an unknown node");
            }
        }
    }
    for (const auto attempted : input.attempted_flags) {
        if (attempted != 0 && attempted != 1) {
            throw std::invalid_argument(
                "attempted_flags must contain only zero or one");
        }
    }

    const auto route_equal = [](
        std::span<const std::int64_t> left,
        std::int64_t left_begin,
        std::int64_t left_end,
        std::span<const std::int64_t> right,
        std::int64_t right_begin,
        std::int64_t right_end) {
        return left_end - left_begin == right_end - right_begin
            && std::equal(
                left.begin() + left_begin, left.begin() + left_end,
                right.begin() + right_begin);
    };
    std::vector<std::int64_t> vehicle_counts(plan_count, 0);
    std::vector<std::int64_t> changed_counts(plan_count, 0);
    std::vector<double> optimistic_distances(plan_count, 0.0);
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        vehicle_counts[plan] =
            input.plan_offsets[plan + 1] - input.plan_offsets[plan];
        native_kernels::PythonFloatSum distance_sum;
        for (auto route = input.plan_offsets[plan];
             route < input.plan_offsets[plan + 1]; ++route) {
            distance_sum.add(
                input.route_distance_lower_bounds[
                    static_cast<std::size_t>(route)]);
            bool unchanged = false;
            for (std::size_t current = 0; current < current_count; ++current) {
                if (route_equal(
                        input.route_indices,
                        input.route_offsets[static_cast<std::size_t>(route)],
                        input.route_offsets[static_cast<std::size_t>(route) + 1],
                        input.current_route_indices,
                        input.current_route_offsets[current],
                        input.current_route_offsets[current + 1])) {
                    unchanged = true;
                    break;
                }
            }
            changed_counts[plan] += unchanged ? 0 : 1;
        }
        optimistic_distances[plan] = distance_sum.value();
    }
    const auto route_less = [&](std::int64_t left, std::int64_t right) {
        auto left_cursor = input.route_offsets[static_cast<std::size_t>(left)];
        auto right_cursor = input.route_offsets[static_cast<std::size_t>(right)];
        const auto left_end = input.route_offsets[static_cast<std::size_t>(left) + 1];
        const auto right_end = input.route_offsets[static_cast<std::size_t>(right) + 1];
        while (left_cursor < left_end && right_cursor < right_end) {
            const auto left_node = input.route_indices[
                static_cast<std::size_t>(left_cursor)];
            const auto right_node = input.route_indices[
                static_cast<std::size_t>(right_cursor)];
            const auto left_rank = input.lexical_rank[
                static_cast<std::size_t>(left_node)];
            const auto right_rank = input.lexical_rank[
                static_cast<std::size_t>(right_node)];
            if (left_rank != right_rank) {
                return left_rank < right_rank;
            }
            ++left_cursor;
            ++right_cursor;
        }
        return left_end - input.route_offsets[static_cast<std::size_t>(left)]
            < right_end - input.route_offsets[static_cast<std::size_t>(right)];
    };
    const auto plan_routes_less = [&](std::size_t left, std::size_t right) {
        auto left_route = input.plan_offsets[left];
        auto right_route = input.plan_offsets[right];
        while (left_route < input.plan_offsets[left + 1]
               && right_route < input.plan_offsets[right + 1]) {
            if (route_less(left_route, right_route)) {
                return true;
            }
            if (route_less(right_route, left_route)) {
                return false;
            }
            ++left_route;
            ++right_route;
        }
        return vehicle_counts[left] < vehicle_counts[right];
    };

    RankingResult output;
    output.ranked.resize(plan_count);
    std::iota(output.ranked.begin(), output.ranked.end(), 0);
    std::stable_sort(
        output.ranked.begin(), output.ranked.end(),
        [&](std::int64_t left_id, std::int64_t right_id) {
            const auto left = static_cast<std::size_t>(left_id);
            const auto right = static_cast<std::size_t>(right_id);
            if (vehicle_counts[left] != vehicle_counts[right]) {
                return vehicle_counts[left] < vehicle_counts[right];
            }
            if (optimistic_distances[left] != optimistic_distances[right]) {
                return optimistic_distances[left] < optimistic_distances[right];
            }
            if (changed_counts[left] != changed_counts[right]) {
                return changed_counts[left] < changed_counts[right];
            }
            if (plan_routes_less(left, right)) {
                return true;
            }
            if (plan_routes_less(right, left)) {
                return false;
            }
            return left < right;
        });
    output.selected.reserve(
        std::min<std::size_t>(plan_count, static_cast<std::size_t>(input.top_k)));
    for (const auto plan : output.ranked) {
        if (input.attempted_flags[static_cast<std::size_t>(plan)] == 0) {
            output.selected.push_back(plan);
            if (output.selected.size()
                == static_cast<std::size_t>(input.top_k)) {
                break;
            }
        }
    }
    output.integer_metrics.resize(plan_count * 2);
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        output.integer_metrics[plan * 2] = vehicle_counts[plan];
        output.integer_metrics[plan * 2 + 1] = changed_counts[plan];
    }
    output.optimistic_distances = std::move(optimistic_distances);
    return output;
}

}  // namespace evrptw::native_candidate_plan
