#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
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
