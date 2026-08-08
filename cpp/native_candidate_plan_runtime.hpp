#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
    bool allow_vehicle_increase = false;
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
    if (terminal
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())
        || offsets.empty() || offsets.front() != 0
        || offsets.back() != static_cast<std::int64_t>(terminal)) {
        throw std::invalid_argument(std::string(name) + " boundary is invalid");
    }
    for (std::size_t index = 0; index + 1 < offsets.size(); ++index) {
        if (offsets[index] < 0 || offsets[index] > offsets[index + 1]) {
            throw std::invalid_argument(std::string(name) + " must be monotonic");
        }
    }
}

inline void validate_nonempty_rows(
    std::span<const std::int64_t> offsets,
    std::size_t terminal,
    const char* name) {
    validate_offsets(offsets, terminal, name);
    for (std::size_t index = 0; index + 1 < offsets.size(); ++index) {
        if (offsets[index] == offsets[index + 1]) {
            throw std::invalid_argument(
                std::string(name) + " cannot contain an empty row");
        }
    }
}

inline void validate_lexical_rank(
    std::span<const std::int64_t> lexical_rank) {
    if (lexical_rank.empty()) {
        throw std::invalid_argument("lexical_rank cannot be empty");
    }
    std::unordered_set<std::int64_t> values;
    values.reserve(lexical_rank.size());
    for (const auto rank : lexical_rank) {
        if (rank < 0 || rank >= static_cast<std::int64_t>(lexical_rank.size())
            || !values.insert(rank).second) {
            throw std::invalid_argument("lexical_rank must be a permutation");
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

inline void validate_ranking_result(
    const RankingInput& input,
    const RankingResult& output) {
    const auto plan_count = input.plan_offsets.size() - 1;
    if (plan_count > std::numeric_limits<std::size_t>::max() / 2
        || output.ranked.size() != plan_count
        || output.integer_metrics.size() != plan_count * 2
        || output.optimistic_distances.size() != plan_count
        || output.selected.size()
            > std::min<std::size_t>(
                plan_count, static_cast<std::size_t>(input.top_k))) {
        throw std::logic_error("candidate-plan ranking output shape is invalid");
    }
    std::vector<std::int64_t> expected_selected;
    expected_selected.reserve(output.selected.size());
    std::vector<std::int64_t> seen(plan_count, 0);
    for (const auto plan : output.ranked) {
        if (plan < 0 || plan >= static_cast<std::int64_t>(plan_count)
            || seen[static_cast<std::size_t>(plan)] != 0) {
            throw std::logic_error(
                "candidate-plan ranking output is not a permutation");
        }
        seen[static_cast<std::size_t>(plan)] = 1;
        if (input.attempted_flags[static_cast<std::size_t>(plan)] == 0
            && expected_selected.size()
                < static_cast<std::size_t>(input.top_k)) {
            expected_selected.push_back(plan);
        }
    }
    if (output.selected != expected_selected) {
        throw std::logic_error(
            "candidate-plan ranking selected prefix is inconsistent");
    }
    const auto current_count = input.current_route_offsets.size() - 1;
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
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        const auto vehicle_count = output.integer_metrics[plan * 2];
        const auto changed_count = output.integer_metrics[plan * 2 + 1];
        const auto expected_vehicle_count =
            input.plan_offsets[plan + 1] - input.plan_offsets[plan];
        native_kernels::PythonFloatSum distance_sum;
        std::int64_t expected_changed_count = 0;
        for (auto route = input.plan_offsets[plan];
             route < input.plan_offsets[plan + 1]; ++route) {
            distance_sum.add(input.route_distance_lower_bounds[
                static_cast<std::size_t>(route)]);
            auto unchanged = false;
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
            expected_changed_count += unchanged ? 0 : 1;
        }
        if (vehicle_count != expected_vehicle_count || changed_count < 0
            || changed_count > vehicle_count
            || changed_count != expected_changed_count
            || !std::isfinite(output.optimistic_distances[plan])
            || output.optimistic_distances[plan] < 0.0
            || output.optimistic_distances[plan] != distance_sum.value()) {
            throw std::logic_error(
                "candidate-plan ranking metrics are inconsistent");
        }
    }
    const auto route_less = [&](std::int64_t left, std::int64_t right) {
        auto left_cursor = input.route_offsets[static_cast<std::size_t>(left)];
        auto right_cursor = input.route_offsets[static_cast<std::size_t>(right)];
        const auto left_end =
            input.route_offsets[static_cast<std::size_t>(left) + 1];
        const auto right_end =
            input.route_offsets[static_cast<std::size_t>(right) + 1];
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
        return output.integer_metrics[left * 2]
            < output.integer_metrics[right * 2];
    };
    std::vector<std::int64_t> expected_ranked(plan_count);
    std::iota(expected_ranked.begin(), expected_ranked.end(), 0);
    std::stable_sort(
        expected_ranked.begin(), expected_ranked.end(),
        [&](std::int64_t left_id, std::int64_t right_id) {
            const auto left = static_cast<std::size_t>(left_id);
            const auto right = static_cast<std::size_t>(right_id);
            if (output.integer_metrics[left * 2]
                != output.integer_metrics[right * 2]) {
                return output.integer_metrics[left * 2]
                    < output.integer_metrics[right * 2];
            }
            if (output.optimistic_distances[left]
                != output.optimistic_distances[right]) {
                return output.optimistic_distances[left]
                    < output.optimistic_distances[right];
            }
            if (output.integer_metrics[left * 2 + 1]
                != output.integer_metrics[right * 2 + 1]) {
                return output.integer_metrics[left * 2 + 1]
                    < output.integer_metrics[right * 2 + 1];
            }
            if (plan_routes_less(left, right)) {
                return true;
            }
            if (plan_routes_less(right, left)) {
                return false;
            }
            return left < right;
        });
    if (output.ranked != expected_ranked) {
        throw std::logic_error(
            "candidate-plan ranking order is inconsistent");
    }
}

inline void validate_preparation_result(
    const PlanPreparationInput& input,
    const PlanPreparationResult& output) {
    const auto plan_count = input.plan_offsets.size() - 1;
    const auto route_count = input.route_offsets.size() - 1;
    if (output.coverage_eligible.size() != plan_count
        || output.unique_row_by_route.size() != route_count
        || output.unique_route_offsets.empty()) {
        throw std::logic_error(
            "candidate-plan preparation output shape is invalid");
    }
    validate_nonempty_rows(
        output.unique_route_offsets,
        output.unique_route_indices.size(),
        "prepared unique_route_offsets");
    std::unordered_set<std::int64_t> canonical(
        output.canonical_expected.begin(), output.canonical_expected.end());
    std::unordered_set<std::int64_t> expected(
        input.expected_customer_indices.begin(),
        input.expected_customer_indices.end());
    if (canonical.size() != output.canonical_expected.size()
        || canonical != expected) {
        throw std::logic_error(
            "candidate-plan canonical expected customers are inconsistent");
    }
    for (std::size_t index = 1; index < output.canonical_expected.size(); ++index) {
        const auto previous = static_cast<std::size_t>(
            output.canonical_expected[index - 1]);
        const auto current = static_cast<std::size_t>(
            output.canonical_expected[index]);
        if (input.lexical_rank[previous] >= input.lexical_rank[current]) {
            throw std::logic_error(
                "candidate-plan canonical customer order is inconsistent");
        }
    }
    std::vector<std::int64_t> expected_coverage(plan_count, 1);
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        std::unordered_set<std::int64_t> observed;
        for (auto route = input.plan_offsets[plan];
             route < input.plan_offsets[plan + 1]; ++route) {
            for (auto cursor = input.route_offsets[static_cast<std::size_t>(route)];
                 cursor < input.route_offsets[static_cast<std::size_t>(route) + 1];
                 ++cursor) {
                if (!observed.insert(input.route_indices[
                        static_cast<std::size_t>(cursor)]).second) {
                    expected_coverage[plan] = 0;
                }
            }
        }
        if (observed != expected) {
            expected_coverage[plan] = 0;
        }
    }
    if (output.coverage_eligible != expected_coverage) {
        throw std::logic_error(
            "candidate-plan preparation coverage is inconsistent");
    }
    std::vector<std::int64_t> expected_unique_offsets{0};
    std::vector<std::int64_t> expected_unique_indices;
    std::vector<std::int64_t> expected_row_by_route(route_count);
    std::unordered_map<std::vector<std::int64_t>, std::size_t, RouteSequenceHash>
        expected_rows;
    expected_rows.reserve(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        std::vector<std::int64_t> sequence(
            input.route_indices.begin() + input.route_offsets[route],
            input.route_indices.begin() + input.route_offsets[route + 1]);
        const auto next_row = expected_rows.size();
        const auto [position, inserted] = expected_rows.emplace(sequence, next_row);
        if (inserted) {
            expected_unique_indices.insert(
                expected_unique_indices.end(), sequence.begin(), sequence.end());
            expected_unique_offsets.push_back(
                static_cast<std::int64_t>(expected_unique_indices.size()));
        }
        expected_row_by_route[route] =
            static_cast<std::int64_t>(position->second);
    }
    if (output.unique_route_offsets != expected_unique_offsets
        || output.unique_route_indices != expected_unique_indices
        || output.unique_row_by_route != expected_row_by_route) {
        throw std::logic_error(
            "candidate-plan unique-route projection is inconsistent");
    }
}

inline void validate_decision_result(
    const PlanDecisionInput& input,
    const PlanDecisionResult& output) {
    const auto plan_count = input.plan_offsets.size() - 1;
    if (output.eligible.size() != plan_count
        || output.combined_attempted.size() != plan_count) {
        throw std::logic_error(
            "candidate-plan decision output shape is invalid");
    }
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        auto expected_eligible = input.coverage_eligible[plan];
        if (expected_eligible != 0 && expected_eligible != 1) {
            throw std::logic_error(
                "candidate-plan decision input eligibility is invalid");
        }
        for (auto route = input.plan_offsets[plan];
             route < input.plan_offsets[plan + 1]; ++route) {
            if (input.screening_passed[static_cast<std::size_t>(route)] == 0) {
                expected_eligible = 0;
            }
        }
        const auto vehicle_count =
            input.plan_offsets[plan + 1] - input.plan_offsets[plan];
        if (!input.allow_vehicle_increase
            && vehicle_count > input.current_route_count) {
            expected_eligible = 0;
        }
        if ((output.eligible[plan] != 0 && output.eligible[plan] != 1)
            || (output.combined_attempted[plan] != 0
                && output.combined_attempted[plan] != 1)
            || output.eligible[plan] != expected_eligible
            || output.combined_attempted[plan]
                != (output.eligible[plan] == 0
                        || input.attempted_flags[plan] != 0
                    ? 1 : 0)) {
            throw std::logic_error(
                "candidate-plan decision output is inconsistent");
        }
    }
}

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
    validate_nonempty_rows(input.plan_offsets, route_count, "plan_offsets");
    validate_nonempty_rows(
        input.route_offsets, input.route_indices.size(), "route_offsets");
    validate_lexical_rank(input.lexical_rank);

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
    validate_preparation_result(input, output);
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
    validate_nonempty_rows(input.plan_offsets, route_count, "plan_offsets");
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
        if (!input.allow_vehicle_increase
            && vehicle_count > input.current_route_count) {
            output.eligible[plan] = 0;
        }
        output.combined_attempted[plan] =
            output.eligible[plan] == 0 || input.attempted_flags[plan] != 0 ? 1 : 0;
    }
    validate_decision_result(input, output);
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
    validate_nonempty_rows(input.plan_offsets, route_count, "plan_offsets");
    validate_nonempty_rows(
        input.route_offsets, input.route_indices.size(), "route_offsets");
    validate_lexical_rank(input.lexical_rank);
    if (plan_count > std::numeric_limits<std::size_t>::max() / 2
        || input.objective_integer.size() != plan_count * 2
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
        const auto vehicle_count = input.plan_offsets[static_cast<std::size_t>(plan) + 1]
            - input.plan_offsets[static_cast<std::size_t>(plan)];
        if (input.objective_integer[offset] != vehicle_count
            || input.objective_integer[offset + 1] < 0
            || !std::isfinite(input.objective_float[offset])
            || input.objective_float[offset] < 0.0
            || !std::isfinite(input.objective_float[offset + 1])
            || input.objective_float[offset + 1] < 0.0) {
            throw std::invalid_argument(
                "feasible plans require canonical objective fields aligned to routes");
        }
        for (auto route = input.plan_offsets[static_cast<std::size_t>(plan)];
             route < input.plan_offsets[static_cast<std::size_t>(plan) + 1];
             ++route) {
            for (auto cursor = input.route_offsets[static_cast<std::size_t>(route)];
                 cursor < input.route_offsets[static_cast<std::size_t>(route) + 1];
                 ++cursor) {
                const auto node = input.route_indices[
                    static_cast<std::size_t>(cursor)];
                if (node < 0
                    || node
                        >= static_cast<std::int64_t>(input.lexical_rank.size())) {
                    throw std::invalid_argument(
                        "feasible candidate plan contains an unknown node");
                }
            }
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
    const auto plan_less = [&](std::int64_t left, std::int64_t right) {
        const auto left_offset = static_cast<std::size_t>(left) * 2;
        const auto right_offset = static_cast<std::size_t>(right) * 2;
        const auto left_key = std::make_tuple(
            input.objective_integer[left_offset],
            input.objective_float[left_offset],
            input.objective_float[left_offset + 1],
            input.objective_integer[left_offset + 1]);
        const auto right_key = std::make_tuple(
            input.objective_integer[right_offset],
            input.objective_float[right_offset],
            input.objective_float[right_offset + 1],
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
    std::vector<std::int64_t> output(input.feasible_plan_ids.size());
    if (!output.empty()) {
        std::copy(
            input.feasible_plan_ids.begin(), input.feasible_plan_ids.end(),
            output.begin());
    }
    std::stable_sort(output.begin(), output.end(), plan_less);
    for (std::size_t index = 1; index < output.size(); ++index) {
        if (plan_less(output[index], output[index - 1])) {
            throw std::logic_error(
                "feasible candidate-plan ordering output is inconsistent");
        }
    }
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
    if (plan_count > std::numeric_limits<std::size_t>::max() / 2
        || input.route_distance_lower_bounds.size() != route_count
        || input.attempted_flags.size() != plan_count) {
        throw std::invalid_argument(
            "candidate-plan route metrics/flags do not align");
    }
    validate_nonempty_rows(input.plan_offsets, route_count, "plan_offsets");
    validate_nonempty_rows(
        input.route_offsets, input.route_indices.size(), "route_offsets");
    validate_nonempty_rows(
        input.current_route_offsets, input.current_route_indices.size(),
        "current_route_offsets");
    validate_lexical_rank(input.lexical_rank);
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
    for (const auto node : input.current_route_indices) {
        if (node < 0
            || node >= static_cast<std::int64_t>(input.lexical_rank.size())) {
            throw std::invalid_argument(
                "current candidate plan contains an unknown node");
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
    validate_ranking_result(input, output);
    return output;
}

}  // namespace evrptw::native_candidate_plan
