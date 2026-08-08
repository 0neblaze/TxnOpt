#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <list>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "formal_objective.hpp"
#include "native_sha256.hpp"
#include "native_candidate_plan_runtime.hpp"
#include "native_kernel_protocol.hpp"
#include "native_solver_kernels.hpp"

namespace evrptw::native_search {

class PythonRandomV2 final {
public:
    explicit PythonRandomV2(std::uint64_t seed) {
        std::vector<std::uint32_t> key;
        do {
            key.push_back(static_cast<std::uint32_t>(seed & 0xffffffffULL));
            seed >>= 32U;
        } while (seed != 0U);
        init_by_array(key);
    }

    [[nodiscard]] double random() {
        const auto upper = next_u32() >> 5U;
        const auto lower = next_u32() >> 6U;
        return (static_cast<double>(upper) * 67108864.0
                + static_cast<double>(lower))
            * (1.0 / 9007199254740992.0);
    }

    [[nodiscard]] std::uint64_t getrandbits(const std::uint32_t bits) {
        if (bits == 0U) {
            return 0U;
        }
        if (bits <= 32U) {
            return static_cast<std::uint64_t>(next_u32() >> (32U - bits));
        }
        if (bits > 64U) {
            throw std::invalid_argument(
                "native PythonRandom supports at most 64 bits");
        }
        const auto low = static_cast<std::uint64_t>(next_u32());
        const auto remaining = bits - 32U;
        const auto high = static_cast<std::uint64_t>(
            next_u32() >> (32U - remaining));
        return low | (high << 32U);
    }

    [[nodiscard]] std::uint64_t randbelow(const std::uint64_t upper_bound) {
        if (upper_bound == 0U) {
            throw std::invalid_argument(
                "PythonRandom randbelow bound must be positive");
        }
        std::uint32_t bits = 0U;
        for (auto value = upper_bound; value != 0U; value >>= 1U) {
            ++bits;
        }
        while (true) {
            const auto value = getrandbits(bits);
            if (value < upper_bound) {
                return value;
            }
        }
    }

    [[nodiscard]] std::vector<std::int64_t> sample_indices(
        const std::int64_t population_size,
        const std::int64_t sample_size) {
        if (population_size < 0 || sample_size < 0
            || sample_size > population_size) {
            throw std::invalid_argument("PythonRandom sample size is invalid");
        }
        std::int64_t set_size = 21;
        if (sample_size > 5) {
            auto power = std::int64_t{4};
            const auto target = sample_size * 3;
            while (power < target) {
                power *= 4;
            }
            set_size += power;
        }
        std::vector<std::int64_t> result;
        result.reserve(static_cast<std::size_t>(sample_size));
        if (population_size <= set_size) {
            std::vector<std::int64_t> pool(
                static_cast<std::size_t>(population_size));
            for (std::int64_t index = 0; index < population_size; ++index) {
                pool[static_cast<std::size_t>(index)] = index;
            }
            for (std::int64_t index = 0; index < sample_size; ++index) {
                const auto selected = static_cast<std::int64_t>(randbelow(
                    static_cast<std::uint64_t>(population_size - index)));
                result.push_back(pool[static_cast<std::size_t>(selected)]);
                pool[static_cast<std::size_t>(selected)] =
                    pool[static_cast<std::size_t>(population_size - index - 1)];
            }
            return result;
        }
        std::unordered_set<std::int64_t> selected_indices;
        for (std::int64_t index = 0; index < sample_size; ++index) {
            auto selected = static_cast<std::int64_t>(
                randbelow(static_cast<std::uint64_t>(population_size)));
            while (selected_indices.contains(selected)) {
                selected = static_cast<std::int64_t>(
                    randbelow(static_cast<std::uint64_t>(population_size)));
            }
            selected_indices.insert(selected);
            result.push_back(selected);
        }
        return result;
    }

    [[nodiscard]] std::size_t weighted_index(
        const std::vector<double>& weights) {
        if (weights.empty()) {
            throw std::invalid_argument(
                "PythonRandom weighted choice requires weights");
        }
        std::vector<double> cumulative;
        cumulative.reserve(weights.size());
        double total = 0.0;
        for (const auto weight : weights) {
            total += weight;
            cumulative.push_back(total);
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::invalid_argument(
                "PythonRandom total weight must be finite and positive");
        }
        const auto target = random() * total;
        return static_cast<std::size_t>(
            std::upper_bound(cumulative.begin(), cumulative.end() - 1, target)
            - cumulative.begin());
    }

    void shuffle(std::vector<std::int64_t>& values) {
        for (std::size_t index = values.size(); index > 1; --index) {
            const auto selected = static_cast<std::size_t>(randbelow(index));
            std::swap(values[index - 1], values[selected]);
        }
    }

private:
    static constexpr std::size_t state_size = 624;
    static constexpr std::size_t period = 397;
    std::array<std::uint32_t, state_size> state_{};
    std::size_t cursor_ = state_size;

    void init_genrand(const std::uint32_t seed) {
        state_[0] = seed;
        for (std::size_t index = 1; index < state_size; ++index) {
            state_[index] = 1812433253U
                    * (state_[index - 1] ^ (state_[index - 1] >> 30U))
                + static_cast<std::uint32_t>(index);
        }
        cursor_ = state_size;
    }

    void init_by_array(const std::vector<std::uint32_t>& key) {
        init_genrand(19650218U);
        auto state_index = std::size_t{1};
        auto key_index = std::size_t{0};
        auto rounds = std::max(state_size, key.size());
        for (; rounds != 0; --rounds) {
            state_[state_index] =
                (state_[state_index]
                 ^ ((state_[state_index - 1]
                     ^ (state_[state_index - 1] >> 30U))
                    * 1664525U))
                + key[key_index] + static_cast<std::uint32_t>(key_index);
            ++state_index;
            ++key_index;
            if (state_index >= state_size) {
                state_[0] = state_[state_size - 1];
                state_index = 1;
            }
            if (key_index >= key.size()) {
                key_index = 0;
            }
        }
        for (rounds = state_size - 1; rounds != 0; --rounds) {
            state_[state_index] =
                (state_[state_index]
                 ^ ((state_[state_index - 1]
                     ^ (state_[state_index - 1] >> 30U))
                    * 1566083941U))
                - static_cast<std::uint32_t>(state_index);
            ++state_index;
            if (state_index >= state_size) {
                state_[0] = state_[state_size - 1];
                state_index = 1;
            }
        }
        state_[0] = 0x80000000U;
    }

    [[nodiscard]] std::uint32_t next_u32() {
        if (cursor_ >= state_size) {
            twist();
        }
        auto value = state_[cursor_++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        value ^= value >> 18U;
        return value;
    }

    void twist() {
        constexpr auto upper_mask = std::uint32_t{0x80000000U};
        constexpr auto lower_mask = std::uint32_t{0x7fffffffU};
        constexpr auto matrix = std::uint32_t{0x9908b0dfU};
        for (std::size_t index = 0; index < state_size; ++index) {
            const auto value = (state_[index] & upper_mask)
                | (state_[(index + 1) % state_size] & lower_mask);
            state_[index] = state_[(index + period) % state_size]
                ^ (value >> 1U)
                ^ ((value & 1U) != 0U ? matrix : 0U);
        }
        cursor_ = 0;
    }
};

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

struct PlanBatchViewV2 final {
    std::span<const std::int64_t> plan_offsets;
    RouteBatchViewV2 routes;

    [[nodiscard]] std::size_t plan_count() const noexcept {
        return plan_offsets.empty() ? 0U : plan_offsets.size() - 1U;
    }

    void validate(const std::string_view name) const {
        routes.validate(std::string(name) + " routes");
        if (plan_offsets.empty() || plan_offsets.front() != 0
            || plan_offsets.back()
                != static_cast<std::int64_t>(routes.route_count())) {
            throw std::invalid_argument(
                std::string(name) + " plan boundary is invalid");
        }
        for (std::size_t plan = 0; plan < plan_count(); ++plan) {
            if (plan_offsets[plan] < 0
                || plan_offsets[plan] > plan_offsets[plan + 1]) {
                throw std::invalid_argument(
                    std::string(name) + " plan offsets are not monotone");
            }
        }
    }
};

class AttemptedPlanSetV2 final {
public:
    [[nodiscard]] std::vector<std::int64_t> lookup(
        const PlanBatchViewV2 plans) const {
        plans.validate("native attempted-plan");
        std::vector<std::int64_t> flags(plans.plan_count(), 0);
        for (std::size_t plan = 0; plan < plans.plan_count(); ++plan) {
            flags[plan] = attempted_.contains(plan_key(plans, plan)) ? 1 : 0;
        }
        return flags;
    }

    [[nodiscard]] std::vector<std::int64_t> begin_mark_many_atomic(
        const PlanBatchViewV2 plans,
        const std::span<const std::int64_t> plan_ids) {
        plans.validate("native attempted-plan");
        if (active_additions_.has_value()) {
            throw std::runtime_error(
                "native attempted-plan set already has an active batch");
        }
        std::unordered_set<std::int64_t> unique_ids;
        std::vector<std::string> additions;
        std::vector<std::int64_t> statuses(plan_ids.size(), 0);
        additions.reserve(plan_ids.size());
        for (std::size_t ordinal = 0; ordinal < plan_ids.size(); ++ordinal) {
            const auto plan_id = plan_ids[ordinal];
            if (plan_id < 0
                || static_cast<std::size_t>(plan_id) >= plans.plan_count()
                || !unique_ids.insert(plan_id).second) {
                throw std::invalid_argument(
                    "native attempted-plan IDs must be unique valid plan rows");
            }
            auto key = plan_key(plans, static_cast<std::size_t>(plan_id));
            if (attempted_.contains(key)
                || std::find(additions.begin(), additions.end(), key)
                    != additions.end()) {
                continue;
            }
            statuses[ordinal] = 1;
            additions.push_back(std::move(key));
        }
        std::vector<std::string> inserted;
        inserted.reserve(additions.size());
        try {
            for (const auto& key : additions) {
                attempted_.insert(key);
                inserted.push_back(key);
            }
            active_additions_ = std::move(additions);
        } catch (...) {
            for (const auto& key : inserted) {
                attempted_.erase(key);
            }
            throw;
        }
        return statuses;
    }

    [[nodiscard]] std::int64_t commit_mark_batch() {
        require_active("commit_mark_batch");
        active_additions_.reset();
        return size();
    }

    [[nodiscard]] std::int64_t rollback_mark_batch() {
        require_active("rollback_mark_batch");
        rollback_mark_batch_noexcept();
        return size();
    }

    [[nodiscard]] std::int64_t size() const noexcept {
        return static_cast<std::int64_t>(attempted_.size());
    }

    void reset_empty_noexcept() noexcept {
        active_additions_.reset();
        attempted_.clear();
    }

    [[nodiscard]] std::unordered_set<std::string> snapshot() const {
        return attempted_;
    }

    void restore(std::unordered_set<std::string> snapshot) noexcept {
        active_additions_.reset();
        attempted_ = std::move(snapshot);
    }

    void prepare_mark_commit() const {
        require_active("prepare_mark_commit");
    }

    void commit_mark_batch_noexcept() noexcept {
        active_additions_.reset();
    }

    void rollback_mark_batch_noexcept() noexcept {
        if (!active_additions_.has_value()) {
            std::terminate();
        }
        for (const auto& key : *active_additions_) {
            attempted_.erase(key);
        }
        active_additions_.reset();
    }

private:
    std::unordered_set<std::string> attempted_;
    std::optional<std::vector<std::string>> active_additions_;

    [[nodiscard]] static std::string plan_key(
        const PlanBatchViewV2 plans,
        const std::size_t plan) {
        std::string key;
        const auto append = [&key](const std::int64_t value) {
            const auto position = key.size();
            key.resize(position + sizeof(value));
            std::memcpy(key.data() + position, &value, sizeof(value));
        };
        const auto first_route = plans.plan_offsets[plan];
        const auto end_route = plans.plan_offsets[plan + 1];
        append(end_route - first_route);
        for (auto route = first_route; route < end_route; ++route) {
            const auto sequence = plans.routes.route(
                static_cast<std::size_t>(route));
            append(static_cast<std::int64_t>(sequence.size()));
            for (const auto node : sequence) {
                append(node);
            }
        }
        return key;
    }

    void require_active(const char* operation) const {
        if (!active_additions_.has_value()) {
            throw std::runtime_error(
                std::string("native attempted-plan ") + operation
                + " requires an active batch");
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

struct ThreeLaneRoundOutcomeStateV2 final {
    std::int64_t iteration = -1;
    std::int64_t legacy_operator_index = -1;
    std::int64_t legacy_candidate_feasible = 0;
    AcceptanceOutcomeV2 legacy_acceptance;

    void validate() const {
        const auto binary = [](const std::int64_t value) {
            return value == 0 || value == 1;
        };
        if (iteration < -1 || legacy_operator_index < -1
            || legacy_operator_index > 3
            || !binary(legacy_candidate_feasible)
            || !binary(legacy_acceptance.accepted)
            || !binary(legacy_acceptance.improved_global_best)
            || !binary(legacy_acceptance.vehicle_reduction)
            || (legacy_operator_index < 0
                && (legacy_candidate_feasible != 0
                    || legacy_acceptance.accepted != 0
                    || legacy_acceptance.improved_global_best != 0
                    || legacy_acceptance.vehicle_reduction != 0))
            || (legacy_acceptance.accepted != 0
                && legacy_candidate_feasible == 0)
            || (legacy_acceptance.improved_global_best != 0
                && legacy_acceptance.accepted == 0)
            || (legacy_acceptance.vehicle_reduction != 0
                && legacy_acceptance.accepted == 0)) {
            throw std::logic_error(
                "native three-lane round outcome is inconsistent");
        }
    }
};

struct CandidateRoundRequestV2 final {
    std::vector<std::int64_t> plan_offsets;
    std::vector<std::int64_t> route_offsets;
    std::vector<std::int64_t> route_indices;
    std::array<std::int64_t, 3> context{};
    double deadline_remaining = 0.0;
    std::int64_t batch_size = 0;
    std::vector<std::int64_t> expected_customers;
    std::vector<std::int64_t> ranking_route_offsets;
    std::vector<std::int64_t> ranking_route_indices;

    void validate() const {
        if (plan_offsets.size() < 2 || route_offsets.size() < 2
            || plan_offsets.front() != 0 || route_offsets.front() != 0
            || plan_offsets.back()
                != static_cast<std::int64_t>(route_offsets.size() - 1)
            || route_offsets.back()
                != static_cast<std::int64_t>(route_indices.size())) {
            throw std::invalid_argument(
                "full native plan transaction boundary is invalid");
        }
        if (ranking_route_offsets.empty()
                != ranking_route_indices.empty()
            || (!ranking_route_offsets.empty()
                && (ranking_route_offsets.size() < 2
                    || ranking_route_offsets.front() != 0
                    || ranking_route_offsets.back()
                        != static_cast<std::int64_t>(
                            ranking_route_indices.size())))) {
            throw std::invalid_argument(
                "full native ranking baseline boundary is invalid");
        }
        for (std::size_t plan = 0; plan + 1 < plan_offsets.size(); ++plan) {
            if (plan_offsets[plan] < 0
                || plan_offsets[plan] > plan_offsets[plan + 1]) {
                throw std::invalid_argument(
                    "full native plan offsets must be monotonic");
            }
        }
        for (std::size_t route = 0; route + 1 < route_offsets.size(); ++route) {
            if (route_offsets[route] < 0
                || route_offsets[route] > route_offsets[route + 1]) {
                throw std::invalid_argument(
                    "full native route offsets must be monotonic");
            }
        }
        for (std::size_t route = 0;
             route + 1 < ranking_route_offsets.size(); ++route) {
            if (ranking_route_offsets[route] < 0
                || ranking_route_offsets[route]
                    >= ranking_route_offsets[route + 1]) {
                throw std::invalid_argument(
                    "full native ranking route offsets must be monotonic and non-empty");
            }
        }
        if (context[0] < 0 || context[1] < 0 || context[2] < -1) {
            throw std::invalid_argument(
                "full native plan transaction context IDs are invalid");
        }
        if (!std::isfinite(deadline_remaining) || deadline_remaining <= 0.0) {
            throw std::invalid_argument(
                "full native plan transaction deadline must be finite and positive");
        }
        if (batch_size <= 0) {
            throw std::invalid_argument(
                "full native plan transaction batch size must be positive");
        }
    }
};

struct CandidateExactBatchV2 final {
    std::vector<std::int64_t> path_offsets{0};
    std::vector<std::int64_t> path_indices;
    std::vector<std::int64_t> statuses;
    std::vector<std::int64_t> reasons;
    std::vector<double> metrics;
    std::vector<std::int64_t> label_counters;

    void validate(const std::size_t route_count) const {
        if (path_offsets.size() != route_count + 1
            || path_offsets.front() != 0
            || path_offsets.back()
                != static_cast<std::int64_t>(path_indices.size())
            || statuses.size() != route_count
            || reasons.size() != route_count
            || metrics.size() != route_count * 4
            || label_counters.size() != route_count * 3) {
            throw std::logic_error(
                "full native staged exact result has an invalid typed shape");
        }
        for (std::size_t row = 0; row < route_count; ++row) {
            if (path_offsets[row] < 0
                || path_offsets[row] > path_offsets[row + 1]) {
                throw std::logic_error(
                    "full native staged exact offsets are not monotone");
            }
            if (statuses[row] < -1 || statuses[row] > 2
                || reasons[row] < -1) {
                throw std::logic_error(
                    "full native staged exact status/reason is invalid");
            }
            if (statuses[row] == -1
                && (reasons[row] != -1
                    || path_offsets[row] != path_offsets[row + 1]
                    || std::any_of(
                        metrics.begin()
                            + static_cast<std::ptrdiff_t>(row * 4),
                        metrics.begin()
                            + static_cast<std::ptrdiff_t>((row + 1) * 4),
                        [](const double value) { return value != 0.0; })
                    || std::any_of(
                        label_counters.begin()
                            + static_cast<std::ptrdiff_t>(row * 3),
                        label_counters.begin()
                            + static_cast<std::ptrdiff_t>((row + 1) * 3),
                        [](const std::int64_t value) {
                            return value != 0;
                        }))) {
                throw std::logic_error(
                    "full native staged exact sentinel is not canonical");
            }
        }
    }
};

struct CandidatePlanTransactionResultV2 final {
    std::vector<std::int64_t> selected;
    std::vector<std::int64_t> plan_offsets;
    std::vector<std::int64_t> route_offsets;
    std::vector<std::int64_t> route_indices;
    std::vector<std::int64_t> context;
    std::vector<std::int64_t> expected_customers;
    std::vector<std::int64_t> batch;
    std::vector<double> lower_bounds;
    std::vector<std::int64_t> ranked;
    std::vector<std::int64_t> statuses;
    std::vector<std::int64_t> feasible_order;
    std::vector<std::int64_t> exact_route_rows;
    std::vector<std::int64_t> objective_integer;
    std::vector<double> objective_float;
    std::vector<std::int64_t> route_resolutions;
    std::vector<std::int64_t> completion_order;
    std::vector<std::int64_t> counters;
    std::vector<std::int64_t> cache_statistics;
    std::vector<std::uint8_t> cache_hashes;
    std::size_t cache_hash_rows = 0;
    std::vector<std::int64_t> negative_offsets;
    std::vector<std::int64_t> negative_indices;
    std::vector<std::int64_t> negative_reasons;
    std::vector<std::int64_t> negative_statistics;
    std::vector<std::int64_t> budget_state;
    CandidateExactBatchV2 exact;
    std::string transaction_sha256;

    void validate_shape() const {
        if (plan_offsets.size() < 2 || route_offsets.size() < 2
            || plan_offsets.front() != 0 || route_offsets.front() != 0
            || plan_offsets.back()
                != static_cast<std::int64_t>(route_offsets.size() - 1)
            || route_offsets.back()
                != static_cast<std::int64_t>(route_indices.size())
            || transaction_sha256.size() != 64) {
            throw std::logic_error(
                "full native staged candidate round has an invalid identity");
        }
        const auto plan_count = plan_offsets.size() - 1;
        const auto route_count = route_offsets.size() - 1;
        for (std::size_t row = 0; row < plan_count; ++row) {
            if (plan_offsets[row] < 0
                || plan_offsets[row] >= plan_offsets[row + 1]) {
                throw std::logic_error(
                    "full native staged plan offsets are not monotone");
            }
        }
        for (std::size_t row = 0; row < route_count; ++row) {
            if (route_offsets[row] < 0
                || route_offsets[row] >= route_offsets[row + 1]) {
                throw std::logic_error(
                    "full native staged route offsets are not monotone");
            }
        }
        if (objective_integer.size() != plan_count * 2
            || objective_float.size() != plan_count * 2
            || statuses.size() != plan_count
            || cache_hashes.size() != cache_hash_rows * 32) {
            throw std::logic_error(
                "full native staged objective matrix has an invalid shape");
        }
        for (const auto plan : feasible_order) {
            if (plan < 0 || plan >= static_cast<std::int64_t>(plan_count)) {
                throw std::logic_error(
                    "full native staged feasible order has an invalid plan");
            }
        }
        for (const auto route : exact_route_rows) {
            if (route < 0 || route >= static_cast<std::int64_t>(route_count)) {
                throw std::logic_error(
                    "full native staged exact order has an invalid route");
            }
        }
        exact.validate(route_count);
    }
};

inline void append_candidate_evidence_u64_v2(
    std::string& evidence,
    const std::uint64_t value) {
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        evidence.push_back(
            static_cast<char>((value >> (byte * 8U)) & 0xffU));
    }
}

inline void append_candidate_evidence_i64_v2(
    std::string& evidence,
    const std::int64_t value) {
    append_candidate_evidence_u64_v2(
        evidence, static_cast<std::uint64_t>(value));
}

inline void append_candidate_evidence_f64_v2(
    std::string& evidence,
    const double value) {
    std::uint64_t bits = 0;
    if (std::isnan(value)) {
        bits = 0x7ff8000000000000ULL;
    } else {
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
    }
    append_candidate_evidence_u64_v2(evidence, bits);
}

template <typename T>
inline void append_candidate_evidence_values_v2(
    std::string& evidence,
    const std::span<const T> values) {
    append_candidate_evidence_u64_v2(
        evidence, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) {
        if constexpr (std::is_same_v<T, double>) {
            append_candidate_evidence_f64_v2(evidence, value);
        } else if constexpr (std::is_same_v<T, std::uint8_t>) {
            evidence.push_back(static_cast<char>(value));
        } else {
            append_candidate_evidence_i64_v2(
                evidence, static_cast<std::int64_t>(value));
        }
    }
}

template <typename T>
inline void append_candidate_evidence_vector_v2(
    std::string& evidence,
    const std::span<const T> values,
    const std::initializer_list<std::size_t> shape) {
    append_candidate_evidence_u64_v2(
        evidence, static_cast<std::uint64_t>(shape.size()));
    std::size_t expected_size = 1;
    for (const auto dimension : shape) {
        if (dimension != 0
            && expected_size
                > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(
                "native candidate evidence shape overflows");
        }
        expected_size *= dimension;
        append_candidate_evidence_u64_v2(
            evidence, static_cast<std::uint64_t>(dimension));
    }
    if (expected_size != values.size()) {
        throw std::logic_error(
            "native candidate evidence shape does not match its payload");
    }
    append_candidate_evidence_values_v2(evidence, values);
}

inline std::string candidate_plan_transaction_sha256_v2(
    const CandidatePlanTransactionResultV2& state) {
    const auto plan_count = state.plan_offsets.size() - 1;
    std::string evidence("stage05.2-native-solution-plan-transaction-v2");
    const auto append_i64 = [&evidence](
        const std::vector<std::int64_t>& values,
        const std::initializer_list<std::size_t> shape) {
        append_candidate_evidence_vector_v2<std::int64_t>(
            evidence, values, shape);
    };
    const auto append_f64 = [&evidence](
        const std::vector<double>& values,
        const std::initializer_list<std::size_t> shape) {
        append_candidate_evidence_vector_v2<double>(
            evidence, values, shape);
    };
    append_i64(state.selected, {state.selected.size()});
    append_i64(state.plan_offsets, {state.plan_offsets.size()});
    append_i64(state.route_offsets, {state.route_offsets.size()});
    append_i64(state.route_indices, {state.route_indices.size()});
    append_i64(state.context, {state.context.size()});
    append_i64(state.expected_customers, {state.expected_customers.size()});
    append_i64(state.batch, {state.batch.size()});
    append_f64(state.lower_bounds, {state.lower_bounds.size()});
    append_i64(state.ranked, {state.ranked.size()});
    append_i64(state.statuses, {state.statuses.size()});
    append_i64(state.objective_integer, {plan_count, 2});
    append_f64(state.objective_float, {plan_count, 2});
    append_i64(state.route_resolutions, {state.route_resolutions.size()});
    append_i64(state.exact_route_rows, {state.exact_route_rows.size()});
    append_i64(state.completion_order, {state.completion_order.size()});
    append_i64(state.counters, {state.counters.size()});
    append_i64(state.cache_statistics, {state.cache_statistics.size()});
    append_candidate_evidence_vector_v2<std::uint8_t>(
        evidence, state.cache_hashes, {state.cache_hash_rows, 32});
    append_i64(state.negative_offsets, {state.negative_offsets.size()});
    append_i64(state.negative_indices, {state.negative_indices.size()});
    append_i64(state.negative_reasons, {state.negative_reasons.size()});
    append_i64(
        state.negative_statistics, {state.negative_statistics.size()});
    append_i64(state.budget_state, {state.budget_state.size()});
    append_i64(state.feasible_order, {state.feasible_order.size()});
    for (std::size_t route = 0; route < state.exact.statuses.size(); ++route) {
        if (state.exact.statuses[route] < 0) {
            continue;
        }
        append_candidate_evidence_i64_v2(
            evidence, static_cast<std::int64_t>(route));
        append_candidate_evidence_i64_v2(
            evidence, state.exact.statuses[route]);
        append_candidate_evidence_i64_v2(
            evidence, state.exact.reasons[route]);
        append_candidate_evidence_values_v2<double>(
            evidence,
            {state.exact.metrics.data() + route * 4, 4});
        append_candidate_evidence_values_v2<std::int64_t>(
            evidence,
            {state.exact.label_counters.data() + route * 3, 3});
        const auto first = state.exact.path_offsets[route];
        const auto last = state.exact.path_offsets[route + 1];
        append_candidate_evidence_values_v2<std::int64_t>(
            evidence,
            {state.exact.path_indices.data() + first,
             static_cast<std::size_t>(last - first)});
    }
    return native_protocol::native_sha256_hex(evidence);
}

inline void validate_candidate_plan_transaction_v2(
    const CandidatePlanTransactionResultV2& state) {
    state.validate_shape();
    if (state.transaction_sha256
        != candidate_plan_transaction_sha256_v2(state)) {
        throw std::logic_error(
            "full native staged candidate round hash is invalid");
    }
}

struct CandidatePlanTransactionWireV2 final {
    static constexpr std::size_t integer_field_count = 27;
    static constexpr std::size_t double_field_count = 3;
    static constexpr std::size_t byte_field_count = 2;

    std::array<std::int64_t, integer_field_count + 1> integer_offsets{};
    std::vector<std::int64_t> integer_values;
    std::array<std::int64_t, double_field_count + 1> double_offsets{};
    std::vector<double> double_values;
    std::array<std::int64_t, byte_field_count + 1> byte_offsets{};
    std::vector<std::uint8_t> byte_values;

    void validate() const {
        const auto valid_offsets = [](const auto& offsets, const auto size) {
            return offsets.front() == 0 && std::ranges::is_sorted(offsets)
                && offsets.back() >= 0
                && static_cast<std::size_t>(offsets.back()) == size;
        };
        if (!valid_offsets(integer_offsets, integer_values.size())
            || !valid_offsets(double_offsets, double_values.size())
            || !valid_offsets(byte_offsets, byte_values.size())
            || byte_offsets[2] - byte_offsets[1] != 64) {
            throw std::logic_error(
                "native candidate-plan transaction wire layout is invalid");
        }
    }
};

inline CandidatePlanTransactionWireV2 encode_candidate_plan_transaction_v2(
    const CandidatePlanTransactionResultV2& result) {
    result.validate_shape();
    CandidatePlanTransactionWireV2 wire;
    std::size_t integer_field = 0;
    const auto append_integer = [
        &wire, &integer_field](const std::span<const std::int64_t> values) {
        if (wire.integer_values.size()
            > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()) - values.size()) {
            throw std::overflow_error(
                "native candidate-plan integer wire overflows");
        }
        wire.integer_values.insert(
            wire.integer_values.end(), values.begin(), values.end());
        wire.integer_offsets[++integer_field] = static_cast<std::int64_t>(
            wire.integer_values.size());
    };
    append_integer(result.selected);
    append_integer(result.plan_offsets);
    append_integer(result.route_offsets);
    append_integer(result.route_indices);
    append_integer(result.context);
    append_integer(result.expected_customers);
    append_integer(result.batch);
    append_integer(result.ranked);
    append_integer(result.statuses);
    append_integer(result.feasible_order);
    append_integer(result.exact_route_rows);
    append_integer(result.objective_integer);
    append_integer(result.route_resolutions);
    append_integer(result.completion_order);
    append_integer(result.counters);
    append_integer(result.cache_statistics);
    append_integer(result.negative_offsets);
    append_integer(result.negative_indices);
    append_integer(result.negative_reasons);
    append_integer(result.negative_statistics);
    append_integer(result.budget_state);
    append_integer(result.exact.path_offsets);
    append_integer(result.exact.path_indices);
    append_integer(result.exact.statuses);
    append_integer(result.exact.reasons);
    append_integer(result.exact.label_counters);
    if (result.cache_hash_rows > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "native candidate-plan cache hash row count overflows");
    }
    const std::array<std::int64_t, 1> cache_hash_rows{
        static_cast<std::int64_t>(result.cache_hash_rows)};
    append_integer(cache_hash_rows);
    if (integer_field != CandidatePlanTransactionWireV2::integer_field_count) {
        throw std::logic_error(
            "native candidate-plan integer wire field count is invalid");
    }

    std::size_t double_field = 0;
    const auto append_double = [
        &wire, &double_field](const std::span<const double> values) {
        if (wire.double_values.size()
            > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max()) - values.size()) {
            throw std::overflow_error(
                "native candidate-plan double wire overflows");
        }
        wire.double_values.insert(
            wire.double_values.end(), values.begin(), values.end());
        wire.double_offsets[++double_field] = static_cast<std::int64_t>(
            wire.double_values.size());
    };
    append_double(result.lower_bounds);
    append_double(result.objective_float);
    append_double(result.exact.metrics);
    if (double_field != CandidatePlanTransactionWireV2::double_field_count) {
        throw std::logic_error(
            "native candidate-plan double wire field count is invalid");
    }

    if (result.cache_hashes.size()
        > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max()) - 64) {
        throw std::overflow_error(
            "native candidate-plan byte wire overflows");
    }
    wire.byte_values.insert(
        wire.byte_values.end(), result.cache_hashes.begin(),
        result.cache_hashes.end());
    wire.byte_offsets[1] = static_cast<std::int64_t>(wire.byte_values.size());
    wire.byte_values.insert(
        wire.byte_values.end(), result.transaction_sha256.begin(),
        result.transaction_sha256.end());
    wire.byte_offsets[2] = static_cast<std::int64_t>(wire.byte_values.size());
    wire.validate();
    return wire;
}

inline CandidatePlanTransactionResultV2 decode_candidate_plan_transaction_v2(
    const CandidatePlanTransactionWireV2& wire) {
    wire.validate();
    const auto integer = [&wire](const std::size_t field) {
        const auto first = static_cast<std::size_t>(
            wire.integer_offsets[field]);
        const auto last = static_cast<std::size_t>(
            wire.integer_offsets[field + 1]);
        return std::vector<std::int64_t>(
            wire.integer_values.begin() + static_cast<std::ptrdiff_t>(first),
            wire.integer_values.begin() + static_cast<std::ptrdiff_t>(last));
    };
    const auto doubles = [&wire](const std::size_t field) {
        const auto first = static_cast<std::size_t>(wire.double_offsets[field]);
        const auto last = static_cast<std::size_t>(
            wire.double_offsets[field + 1]);
        return std::vector<double>(
            wire.double_values.begin() + static_cast<std::ptrdiff_t>(first),
            wire.double_values.begin() + static_cast<std::ptrdiff_t>(last));
    };
    const auto bytes = [&wire](const std::size_t field) {
        const auto first = static_cast<std::size_t>(wire.byte_offsets[field]);
        const auto last = static_cast<std::size_t>(wire.byte_offsets[field + 1]);
        return std::vector<std::uint8_t>(
            wire.byte_values.begin() + static_cast<std::ptrdiff_t>(first),
            wire.byte_values.begin() + static_cast<std::ptrdiff_t>(last));
    };
    const auto cache_hash_rows_field = integer(26);
    if (cache_hash_rows_field.size() != 1 || cache_hash_rows_field[0] < 0) {
        throw std::logic_error(
            "native candidate-plan cache hash row count is invalid");
    }
    const auto digest = bytes(1);
    CandidatePlanTransactionResultV2 result{
        integer(0),
        integer(1),
        integer(2),
        integer(3),
        integer(4),
        integer(5),
        integer(6),
        doubles(0),
        integer(7),
        integer(8),
        integer(9),
        integer(10),
        integer(11),
        doubles(1),
        integer(12),
        integer(13),
        integer(14),
        integer(15),
        bytes(0),
        static_cast<std::size_t>(cache_hash_rows_field[0]),
        integer(16),
        integer(17),
        integer(18),
        integer(19),
        integer(20),
        {integer(21), integer(22), integer(23), integer(24), doubles(2),
         integer(25)},
        std::string(digest.begin(), digest.end()),
    };
    result.validate_shape();
    return result;
}

#ifdef __linux__
inline std::vector<std::uint8_t> candidate_plan_transaction_wire_payload_v2(
    const CandidatePlanTransactionWireV2& wire,
    const std::uint64_t request_id,
    const std::span<const double> telemetry = {},
    const native_protocol::KernelOperation operation =
        native_protocol::KernelOperation::candidate_transaction_wire) {
    wire.validate();
    if (operation
            != native_protocol::KernelOperation::candidate_transaction_wire
        && operation
            != native_protocol::KernelOperation::candidate_transaction_execute) {
        throw std::invalid_argument(
            "native candidate-plan wire operation is invalid");
    }
    native_protocol::PayloadBuilder builder(
        operation, request_id);
    builder.add(
        native_protocol::NumericType::int64, wire.integer_offsets.data(),
        wire.integer_offsets.size(), wire.integer_offsets.size());
    builder.add(
        native_protocol::NumericType::int64, wire.integer_values.data(),
        wire.integer_values.size(), wire.integer_values.size());
    builder.add(
        native_protocol::NumericType::int64, wire.double_offsets.data(),
        wire.double_offsets.size(), wire.double_offsets.size());
    builder.add(
        native_protocol::NumericType::float64, wire.double_values.data(),
        wire.double_values.size(), wire.double_values.size());
    builder.add(
        native_protocol::NumericType::int64, wire.byte_offsets.data(),
        wire.byte_offsets.size(), wire.byte_offsets.size());
    builder.add(
        native_protocol::NumericType::uint8, wire.byte_values.data(),
        wire.byte_values.size(), wire.byte_values.size());
    if (!telemetry.empty()) {
        if (telemetry.size() != 7) {
            throw std::invalid_argument(
                "native candidate-plan transaction telemetry is invalid");
        }
        builder.add(
            native_protocol::NumericType::float64, telemetry.data(),
            telemetry.size(), telemetry.size());
    }
    return builder.finish();
}

inline CandidatePlanTransactionWireV2 candidate_plan_transaction_wire_from_payload_v2(
    const native_protocol::PayloadView& payload) {
    if ((payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_wire
        && payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_execute)
        || (payload.header().array_count != 6
            && payload.header().array_count != 7
            && payload.header().array_count != 10
            && payload.header().array_count != 11)
        || payload.descriptor(0).count
            != CandidatePlanTransactionWireV2::integer_field_count + 1
        || payload.descriptor(2).count
            != CandidatePlanTransactionWireV2::double_field_count + 1
        || payload.descriptor(4).count
            != CandidatePlanTransactionWireV2::byte_field_count + 1) {
        throw std::runtime_error(
            "native candidate-plan transaction payload schema is invalid");
    }
    const auto require_vector = [&payload](
        const std::size_t index,
        const native_protocol::NumericType type) {
        const auto& descriptor = payload.descriptor(index);
        if (descriptor.type != type || descriptor.dimensions != 1
            || descriptor.shape[0] != descriptor.count
            || descriptor.shape[1] != 0) {
            throw std::runtime_error(
                "native candidate-plan transaction payload array is invalid");
        }
    };
    require_vector(0, native_protocol::NumericType::int64);
    require_vector(1, native_protocol::NumericType::int64);
    require_vector(2, native_protocol::NumericType::int64);
    require_vector(3, native_protocol::NumericType::float64);
    require_vector(4, native_protocol::NumericType::int64);
    require_vector(5, native_protocol::NumericType::uint8);
    if (payload.header().array_count == 7
        || payload.header().array_count == 11) {
        const auto telemetry_index = payload.header().array_count - 1;
        require_vector(telemetry_index, native_protocol::NumericType::float64);
        if (payload.descriptor(telemetry_index).count != 7) {
            throw std::runtime_error(
                "native candidate-plan transaction telemetry shape is invalid");
        }
    }
    CandidatePlanTransactionWireV2 wire;
    const auto* integer_offsets = payload.data<std::int64_t>(
        0, native_protocol::NumericType::int64);
    std::copy(
        integer_offsets,
        integer_offsets + wire.integer_offsets.size(),
        wire.integer_offsets.begin());
    const auto* integer_values = payload.data<std::int64_t>(
        1, native_protocol::NumericType::int64);
    wire.integer_values.assign(
        integer_values,
        integer_values + payload.descriptor(1).count);
    const auto* double_offsets = payload.data<std::int64_t>(
        2, native_protocol::NumericType::int64);
    std::copy(
        double_offsets,
        double_offsets + wire.double_offsets.size(),
        wire.double_offsets.begin());
    const auto* double_values = payload.data<double>(
        3, native_protocol::NumericType::float64);
    wire.double_values.assign(
        double_values,
        double_values + payload.descriptor(3).count);
    const auto* byte_offsets = payload.data<std::int64_t>(
        4, native_protocol::NumericType::int64);
    std::copy(
        byte_offsets,
        byte_offsets + wire.byte_offsets.size(),
        wire.byte_offsets.begin());
    const auto* byte_values = payload.data<std::uint8_t>(
        5, native_protocol::NumericType::uint8);
    wire.byte_values.assign(
        byte_values,
        byte_values + payload.descriptor(5).count);
    wire.validate();
    return wire;
}

inline std::vector<std::uint8_t> candidate_session_token_payload_v2(
    const native_protocol::KernelOperation operation,
    const std::string_view token,
    const std::uint64_t request_id,
    const std::span<const double> telemetry = {}) {
    if ((operation != native_protocol::KernelOperation::candidate_session_close
         && operation != native_protocol::KernelOperation::candidate_session_open
         && operation
             != native_protocol::KernelOperation::candidate_transaction_commit
         && operation
             != native_protocol::KernelOperation::candidate_transaction_rollback
         && operation
             != native_protocol::KernelOperation::candidate_transaction_status)
        || token.size() != 64) {
        throw std::invalid_argument(
            "native candidate session token payload is invalid");
    }
    native_protocol::PayloadBuilder builder(operation, request_id);
    builder.add(
        native_protocol::NumericType::uint8, token.data(), token.size(),
        token.size());
    if (!telemetry.empty()) {
        if (telemetry.size() != 7) {
            throw std::invalid_argument(
                "native candidate session token telemetry is invalid");
        }
        builder.add(
            native_protocol::NumericType::float64, telemetry.data(),
            telemetry.size(), telemetry.size());
    }
    return builder.finish();
}

inline std::string candidate_session_token_from_payload_v2(
    const native_protocol::PayloadView& payload) {
    if ((payload.header().operation
            != native_protocol::KernelOperation::candidate_session_close
         && payload.header().operation
            != native_protocol::KernelOperation::candidate_session_open
         && payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_commit
         && payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_rollback
         && payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_status)
        || (payload.header().array_count != 1
            && payload.header().array_count != 2
            && (payload.header().operation
                    != native_protocol::KernelOperation::candidate_transaction_status
                || payload.header().array_count != 3))
        || payload.descriptor(0).type != native_protocol::NumericType::uint8
        || payload.descriptor(0).dimensions != 1
        || payload.descriptor(0).count != 64
        || payload.descriptor(0).shape[0] != 64
        || payload.descriptor(0).shape[1] != 0) {
        throw std::runtime_error(
            "native candidate session token schema is invalid");
    }
    if (payload.header().array_count == 2) {
        const auto& telemetry = payload.descriptor(1);
        if (telemetry.type != native_protocol::NumericType::float64
            || telemetry.dimensions != 1 || telemetry.count != 7
            || telemetry.shape[0] != 7 || telemetry.shape[1] != 0) {
            throw std::runtime_error(
                "native candidate session token telemetry schema is invalid");
        }
    } else if (payload.header().array_count == 3) {
        const auto& status = payload.descriptor(1);
        const auto& telemetry = payload.descriptor(2);
        if (status.type != native_protocol::NumericType::int64
            || status.dimensions != 1 || status.count != 1
            || status.shape[0] != 1 || status.shape[1] != 0
            || telemetry.type != native_protocol::NumericType::float64
            || telemetry.dimensions != 1 || telemetry.count != 7
            || telemetry.shape[0] != 7 || telemetry.shape[1] != 0) {
            throw std::runtime_error(
                "native candidate session status schema is invalid");
        }
    }
    const auto* token = payload.data<std::uint8_t>(
        0, native_protocol::NumericType::uint8);
    return std::string(
        reinterpret_cast<const char*>(token),
        static_cast<std::size_t>(payload.descriptor(0).count));
}

inline std::vector<std::uint8_t> candidate_session_status_payload_v2(
    const std::string_view token,
    const std::int64_t status,
    const std::uint64_t request_id,
    const std::span<const double> telemetry) {
    if (token.size() != 64 || status < 0 || status > 3
        || telemetry.size() != 7) {
        throw std::invalid_argument(
            "native candidate session status payload is invalid");
    }
    native_protocol::PayloadBuilder builder(
        native_protocol::KernelOperation::candidate_transaction_status,
        request_id);
    builder.add(
        native_protocol::NumericType::uint8, token.data(), token.size(),
        token.size());
    builder.add(native_protocol::NumericType::int64, &status, 1, 1);
    builder.add(
        native_protocol::NumericType::float64, telemetry.data(),
        telemetry.size(), telemetry.size());
    return builder.finish();
}

inline std::int64_t candidate_session_status_from_payload_v2(
    const native_protocol::PayloadView& payload) {
    static_cast<void>(candidate_session_token_from_payload_v2(payload));
    if (payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_status
        || payload.header().array_count != 3) {
        throw std::runtime_error(
            "native candidate session status receipt is invalid");
    }
    const auto status = *payload.data<std::int64_t>(
        1, native_protocol::NumericType::int64);
    if (status < 0 || status > 3) {
        throw std::runtime_error(
            "native candidate session status value is invalid");
    }
    return status;
}
#endif

struct ThreeLaneTerminationStateV2 final {
    std::int64_t reason = 0;
    std::int64_t exact_budget = 0;
    std::int64_t started = 0;
    std::int64_t completed = 0;
    std::int64_t interrupted = 0;
    std::int64_t completed_iterations = 0;
};

struct ThreeLaneLoopDecisionV2 final {
    ThreeLaneTerminationStateV2 termination;
    std::int64_t no_exact_rounds = 0;
    bool stop = false;
    bool exhaustion_applied = false;
};

class NegativeRouteCacheV2 final {
public:
    struct Entry final {
        std::string key;
        std::vector<std::int64_t> route;
        std::int64_t reason = 0;
    };

    struct LookupResult final {
        std::vector<std::int64_t> hit_flags;
        std::vector<std::int64_t> reasons;
        std::array<std::int64_t, 5> statistics{};
    };

    struct StoreSummary final {
        std::int64_t added = 0;
        std::int64_t evicted = 0;
        bool rollover = false;
    };

    struct Snapshot final {
        std::vector<Entry> entries;
        std::array<std::int64_t, 5> statistics{};
    };

    explicit NegativeRouteCacheV2(const std::int64_t capacity)
        : capacity_(capacity) {
        if (capacity_ <= 0) {
            throw std::invalid_argument(
                "native negative route-cache capacity must be positive");
        }
    }

    [[nodiscard]] LookupResult lookup_many(
        const RouteBatchViewV2 routes) const {
        routes.validate("native negative route-cache owned lookup");
        LookupResult result;
        result.hit_flags.resize(routes.route_count());
        result.reasons.resize(routes.route_count());
        for (std::size_t index = 0; index < routes.route_count(); ++index) {
            const auto found = find_entry(route_key(routes.route(index)));
            result.hit_flags[index] = found == entries_.end() ? 0 : 1;
            result.reasons[index] = found == entries_.end() ? 0 : found->reason;
        }
        result.statistics = statistics();
        return result;
    }

    [[nodiscard]] StoreSummary begin_store_many_atomic(
        const RouteBatchViewV2 routes,
        const std::span<const std::int64_t> reasons) {
        routes.validate("native negative route-cache owned store");
        if (active_batch_.has_value()) {
            throw std::runtime_error(
                "native negative route-cache already has an active batch");
        }
        if (reasons.size() != routes.route_count()) {
            throw std::invalid_argument(
                "native negative route-cache owned reasons do not align");
        }
        std::unordered_set<std::string> input_keys;
        std::vector<Entry> input;
        std::vector<Entry> additions;
        input.reserve(routes.route_count());
        additions.reserve(routes.route_count());
        for (std::size_t index = 0; index < routes.route_count(); ++index) {
            if (reasons[index] <= 0) {
                throw std::invalid_argument(
                    "native negative route-cache reason must be positive");
            }
            const auto route = routes.route(index);
            Entry entry{
                route_key(route),
                std::vector<std::int64_t>(route.begin(), route.end()),
                reasons[index],
            };
            if (!input_keys.insert(entry.key).second) {
                throw std::invalid_argument(
                    "native negative route-cache input routes must be unique");
            }
            const auto existing = find_entry(entry.key);
            if (existing != entries_.end() && existing->reason != entry.reason) {
                throw std::runtime_error(
                    "candidate negative cache reason changed during commit");
            }
            input.push_back(entry);
            if (existing == entries_.end()) {
                additions.push_back(std::move(entry));
            }
        }

        BatchJournal journal;
        journal.added_keys.reserve(additions.size());
        for (const auto& entry : additions) {
            journal.added_keys.push_back(entry.key);
        }
        journal.rollover = entries_.size() + additions.size()
            > static_cast<std::size_t>(capacity_);
        std::vector<Entry> next_entries;
        if (journal.rollover) {
            if (input.size() > static_cast<std::size_t>(capacity_)) {
                throw std::runtime_error(
                    "one candidate transaction exceeds the negative route-cache capacity");
            }
            journal.previous_entries = entries_;
            journal.evicted_count = static_cast<std::int64_t>(std::count_if(
                entries_.begin(), entries_.end(), [&](const Entry& entry) {
                    return !input_keys.contains(entry.key);
                }));
            next_entries = std::move(input);
        } else {
            next_entries = entries_;
            next_entries.insert(
                next_entries.end(), additions.begin(), additions.end());
        }
        active_batch_.emplace(std::move(journal));
        entries_.swap(next_entries);
        return active_summary();
    }

    [[nodiscard]] std::array<std::int64_t, 5> commit_store_batch() {
        require_active("commit_store_batch");
        commit_store_batch_noexcept();
        return statistics();
    }

    [[nodiscard]] std::array<std::int64_t, 5> rollback_store_batch() {
        require_active("rollback_store_batch");
        rollback_store_batch_noexcept();
        return statistics();
    }

    [[nodiscard]] Snapshot snapshot() const {
        return {entries_, statistics_};
    }

    void restore(Snapshot snapshot) noexcept {
        active_batch_.reset();
        entries_ = std::move(snapshot.entries);
        statistics_ = snapshot.statistics;
    }

    void reset_empty_noexcept() noexcept {
        active_batch_.reset();
        entries_.clear();
        statistics_.fill(0);
    }

    [[nodiscard]] std::array<std::int64_t, 5> statistics() const noexcept {
        auto output = statistics_;
        output[3] = static_cast<std::int64_t>(entries_.size());
        return output;
    }

    [[nodiscard]] std::array<std::int64_t, 5> projected_statistics() const noexcept {
        auto output = statistics();
        if (active_batch_.has_value()) {
            output[0] += static_cast<std::int64_t>(active_batch_->added_keys.size());
            output[1] += active_batch_->evicted_count;
            output[2] += active_batch_->rollover ? 1 : 0;
            output[4] = std::max(
                output[4], static_cast<std::int64_t>(entries_.size()));
        }
        return output;
    }

    void prepare_store_commit() const {
        require_active("prepare_store_commit");
    }

    void commit_store_batch_noexcept() noexcept {
        if (!active_batch_.has_value()) {
            std::terminate();
        }
        statistics_[0] += static_cast<std::int64_t>(
            active_batch_->added_keys.size());
        statistics_[1] += active_batch_->evicted_count;
        statistics_[2] += active_batch_->rollover ? 1 : 0;
        statistics_[3] = static_cast<std::int64_t>(entries_.size());
        statistics_[4] = std::max(
            statistics_[4], static_cast<std::int64_t>(entries_.size()));
        active_batch_.reset();
    }

    void rollback_store_batch_noexcept() noexcept {
        if (!active_batch_.has_value()) {
            std::terminate();
        }
        if (active_batch_->rollover) {
            entries_ = std::move(active_batch_->previous_entries);
        } else {
            const auto& added = active_batch_->added_keys;
            std::erase_if(entries_, [&](const Entry& entry) {
                return std::find(added.begin(), added.end(), entry.key)
                    != added.end();
            });
        }
        statistics_[3] = static_cast<std::int64_t>(entries_.size());
        active_batch_.reset();
    }

private:
    struct BatchJournal final {
        std::vector<std::string> added_keys;
        std::vector<Entry> previous_entries;
        std::int64_t evicted_count = 0;
        bool rollover = false;
    };

    std::int64_t capacity_;
    std::vector<Entry> entries_;
    // stores, evictions, rollovers, current entries, peak entries
    std::array<std::int64_t, 5> statistics_{};
    std::optional<BatchJournal> active_batch_;

    [[nodiscard]] static std::string route_key(
        const std::span<const std::int64_t> route) {
        if (route.empty()
            || route.size()
                > std::numeric_limits<std::size_t>::max()
                    / sizeof(std::int64_t) - 1) {
            throw std::invalid_argument(
                "native negative route-cache key size is invalid");
        }
        std::string key;
        key.resize((route.size() + 1) * sizeof(std::int64_t));
        const auto length = static_cast<std::int64_t>(route.size());
        std::memcpy(key.data(), &length, sizeof(length));
        if (!route.empty()) {
            std::memcpy(
                key.data() + sizeof(length), route.data(),
                route.size() * sizeof(std::int64_t));
        }
        return key;
    }

    [[nodiscard]] std::vector<Entry>::const_iterator find_entry(
        const std::string& key) const {
        return std::find_if(
            entries_.begin(), entries_.end(),
            [&](const Entry& entry) { return entry.key == key; });
    }

    void require_active(const char* operation) const {
        if (!active_batch_.has_value()) {
            throw std::runtime_error(
                std::string("native negative route-cache ") + operation
                + " requires an active batch");
        }
    }

    [[nodiscard]] StoreSummary active_summary() const noexcept {
        return {
            static_cast<std::int64_t>(active_batch_->added_keys.size()),
            active_batch_->evicted_count,
            active_batch_->rollover,
        };
    }
};

class ExactRouteCacheV2 final {
public:
    struct ExactPayload final {
        std::vector<std::int64_t> path;
        std::int64_t status = -1;
        std::int64_t reason = -1;
        std::array<double, 4> metrics{};
        std::array<std::int64_t, 3> label_counters{};
        bool operator==(const ExactPayload&) const = default;
    };

    struct Entry final {
        std::string key;
        std::vector<std::int64_t> route;
        std::array<std::uint8_t, 32> semantic_hash{};
        std::int64_t entry_bytes = 0;
        std::optional<ExactPayload> exact_payload;
    };

    struct HashLookupResult final {
        std::vector<std::int64_t> hit_flags;
        std::vector<std::uint8_t> semantic_hashes;
        std::array<std::int64_t, 11> statistics{};
    };

    struct ExactLookupResult final {
        std::vector<std::int64_t> hit_flags;
        std::vector<std::int64_t> path_offsets;
        std::vector<std::int64_t> path_indices;
        std::vector<std::int64_t> statuses;
        std::vector<std::int64_t> reasons;
        std::vector<double> metrics;
        std::vector<std::int64_t> label_counters;
        std::vector<std::uint8_t> semantic_hashes;
        std::array<std::int64_t, 11> statistics{};
    };

    struct StoreSummary final {
        std::vector<std::int64_t> statuses;
        std::vector<std::int64_t> eviction_counts;
        std::array<std::int64_t, 11> statistics{};
    };

    struct Snapshot final {
        std::list<Entry> entries;
        std::unordered_set<std::string> seen_keys;
        std::array<std::int64_t, 11> statistics{};
    };

    ExactRouteCacheV2(
        const std::int64_t max_entries,
        const std::int64_t max_memory_bytes)
        : max_entries_(max_entries), max_memory_bytes_(max_memory_bytes) {
        if (max_entries_ <= 0 || max_memory_bytes_ <= 0) {
            throw std::invalid_argument(
                "native route-cache limits must be positive");
        }
    }

    [[nodiscard]] HashLookupResult lookup_many(const RouteBatchViewV2 routes) {
        routes.validate("native route-cache lookup");
        require_no_active_batch("lookup");
        HashLookupResult result;
        result.hit_flags.assign(routes.route_count(), 0);
        result.semantic_hashes.assign(routes.route_count() * 32, 0);
        for (std::size_t index = 0; index < routes.route_count(); ++index) {
            ++statistics_[0];
            const auto key = route_key(routes.route(index));
            record_seen(key, "lookup");
            const auto found = find_entry(key);
            if (found == entries_.end()) {
                ++statistics_[2];
                continue;
            }
            ++statistics_[1];
            result.hit_flags[index] = 1;
            std::copy(
                found->semantic_hash.begin(), found->semantic_hash.end(),
                result.semantic_hashes.begin()
                    + static_cast<std::ptrdiff_t>(index * 32));
            record_protocol_move(found);
            entries_.splice(entries_.end(), entries_, found);
        }
        result.statistics = statistics_;
        return result;
    }

    [[nodiscard]] StoreSummary begin_store_many_atomic(
        const RouteBatchViewV2 routes,
        const std::span<const std::array<std::uint8_t, 32>> hashes,
        const std::span<const std::int64_t> entry_bytes) {
        routes.validate("native route-cache owned store");
        require_no_active_batch("begin_store_many_atomic_owned");
        if (hashes.size() != routes.route_count()
            || entry_bytes.size() != routes.route_count()) {
            throw std::invalid_argument(
                "native route-cache owned store values do not align");
        }
        BatchJournal journal;
        journal.statistics_before = statistics_;
        journal.protocol_operation_count_before = protocol_operation_count();
        journal.active = true;
        journal.inserted_keys.reserve(routes.route_count());
        journal.evicted_index_nodes.reserve(entries_.size());
        index_.reserve(
            static_cast<std::size_t>(max_entries_) + routes.route_count());
        std::vector<std::int64_t> statuses(routes.route_count(), 0);
        std::vector<std::int64_t> eviction_counts(routes.route_count(), 0);
        try {
            for (std::size_t index = 0; index < routes.route_count(); ++index) {
                if (entry_bytes[index] <= 0) {
                    throw std::invalid_argument(
                        "native route-cache entry bytes must be positive");
                }
                const auto route = routes.route(index);
                const auto key = route_key(route);
                const auto existing = find_entry(key);
                if (existing != entries_.end()) {
                    if (existing->semantic_hash != hashes[index]) {
                        throw std::runtime_error(
                            "atomic native route-cache semantic conflict");
                    }
                    statuses[index] = 1;
                    continue;
                }
                if (entry_bytes[index] > max_memory_bytes_) {
                    ++statistics_[5];
                    statuses[index] = 2;
                    continue;
                }
                journal.inserted_keys.push_back(key);
                const std::unordered_set<std::string> inserted(
                    journal.inserted_keys.begin(), journal.inserted_keys.end());
                while (!entries_.empty()
                       && (statistics_[6] >= max_entries_
                           || statistics_[8] + entry_bytes[index]
                                > max_memory_bytes_)) {
                    auto evicted = entries_.begin();
                    const auto evicted_key = evicted->key;
                    const auto evicted_bytes = evicted->entry_bytes;
                    const auto retain_eviction = !inserted.contains(evicted_key);
                    if (retain_eviction) {
                        record_protocol_eviction(evicted_key, next_key(evicted));
                    } else {
                        cancel_protocol_insertion(evicted_key);
                    }
                    auto index_node = index_.extract(evicted_key);
                    if (index_node.empty()) {
                        throw std::logic_error(
                            "native route-cache eviction lost its index node");
                    }
                    --statistics_[6];
                    statistics_[8] -= evicted_bytes;
                    ++statistics_[4];
                    ++eviction_counts[index];
                    if (retain_eviction) {
                        journal.evicted_index_nodes.push_back(
                            std::move(index_node));
                        journal.evicted_entries.splice(
                            journal.evicted_entries.end(), entries_, evicted);
                    } else {
                        entries_.erase(evicted);
                    }
                }
                Entry stored;
                stored.key = key;
                stored.route.assign(route.begin(), route.end());
                stored.semantic_hash = hashes[index];
                stored.entry_bytes = entry_bytes[index];
                entries_.push_back(std::move(stored));
                auto stored_entry = std::prev(entries_.end());
                try {
                    const auto [_, inserted_entry] = index_.emplace(
                        stored_entry->key, stored_entry);
                    if (!inserted_entry) {
                        throw std::logic_error(
                            "native route-cache insertion duplicated an index key");
                    }
                } catch (...) {
                    entries_.erase(stored_entry);
                    throw;
                }
                record_protocol_insertion(stored_entry->key);
                ++statistics_[3];
                ++statistics_[6];
                statistics_[8] += entry_bytes[index];
                statistics_[7] = std::max(statistics_[7], statistics_[6]);
                statistics_[9] = std::max(statistics_[9], statistics_[8]);
            }
            StoreSummary result{
                std::move(statuses), std::move(eviction_counts), statistics_};
            active_batch_.emplace(std::move(journal));
            return result;
        } catch (...) {
            rollback_journal(journal);
            throw;
        }
    }

    [[nodiscard]] StoreSummary begin_store_exact_many_atomic(
        const RouteBatchViewV2 routes,
        const native_kernels::ExactBatchOutput& exact,
        const std::span<const std::array<std::uint8_t, 32>> semantic_hashes,
        const std::span<const std::int64_t> entry_bytes) {
        routes.validate("native exact route-cache owned store");
        const auto count = routes.route_count();
        if (exact.path_offsets.size() != count + 1
            || exact.statuses.size() != count || exact.reasons.size() != count
            || exact.metrics.size() != count * 4
            || exact.label_counters.size() != count * 3
            || exact.path_offsets.front() != 0
            || exact.path_offsets.back()
                != static_cast<std::int64_t>(exact.path_indices.size())) {
            throw std::invalid_argument(
                "native exact route-cache owned payload values do not align");
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (exact.path_offsets[index] < 0
                || exact.path_offsets[index] > exact.path_offsets[index + 1]) {
                throw std::invalid_argument(
                    "native exact route-cache owned path offsets must be monotonic");
            }
        }
        auto summary = begin_store_many_atomic(
            routes, semantic_hashes, entry_bytes);
        try {
            const std::unordered_set<std::string> inserted(
                active_batch_->inserted_keys.begin(),
                active_batch_->inserted_keys.end());
            for (std::size_t index = 0; index < routes.route_count(); ++index) {
                if (exact.statuses[index] < 0 || exact.statuses[index] > 2
                    || exact.reasons[index] < 0) {
                    throw std::invalid_argument(
                        "native exact route-cache status/reason is invalid");
                }
                const auto key = route_key(routes.route(index));
                const auto entry = find_entry(key);
                if (entry == entries_.end()) {
                    continue;
                }
                ExactPayload payload;
                payload.path.assign(
                    exact.path_indices.begin() + exact.path_offsets[index],
                    exact.path_indices.begin() + exact.path_offsets[index + 1]);
                payload.status = exact.statuses[index];
                payload.reason = exact.reasons[index];
                std::copy_n(
                    exact.metrics.begin()
                        + static_cast<std::ptrdiff_t>(index * 4),
                    4, payload.metrics.begin());
                std::copy_n(
                    exact.label_counters.begin()
                        + static_cast<std::ptrdiff_t>(index * 3),
                    3, payload.label_counters.begin());
                if (entry->exact_payload.has_value()) {
                    if (entry->exact_payload != payload) {
                        throw std::runtime_error(
                            "atomic native route-cache exact payload conflict");
                    }
                } else if (inserted.contains(key)) {
                    entry->exact_payload = std::move(payload);
                } else {
                    throw std::runtime_error(
                        "existing native route-cache entry lacks exact payload");
                }
            }
        } catch (...) {
            static_cast<void>(rollback_store_batch());
            throw;
        }
        return summary;
    }

    [[nodiscard]] ExactLookupResult lookup_exact_many(
        const RouteBatchViewV2 routes) {
        routes.validate("native route-cache owned lookup");
        require_no_active_batch("lookup_exact_many_owned");
        ExactLookupResult result;
        result.hit_flags.assign(routes.route_count(), 0);
        result.statuses.assign(routes.route_count(), -1);
        result.reasons.assign(routes.route_count(), -1);
        result.metrics.assign(routes.route_count() * 4, 0.0);
        result.label_counters.assign(routes.route_count() * 3, 0);
        result.semantic_hashes.assign(routes.route_count() * 32, 0);
        result.path_offsets.push_back(0);
        for (std::size_t index = 0; index < routes.route_count(); ++index) {
            ++statistics_[0];
            const auto key = route_key(routes.route(index));
            record_seen(key, "exact lookup");
            const auto found = find_entry(key);
            if (found == entries_.end()) {
                ++statistics_[2];
                result.path_offsets.push_back(
                    static_cast<std::int64_t>(result.path_indices.size()));
                continue;
            }
            if (!found->exact_payload.has_value()) {
                throw std::runtime_error(
                    "native route-cache hit lacks typed exact payload");
            }
            ++statistics_[1];
            result.hit_flags[index] = 1;
            const auto& payload = *found->exact_payload;
            result.statuses[index] = payload.status;
            result.reasons[index] = payload.reason;
            std::copy(
                payload.metrics.begin(), payload.metrics.end(),
                result.metrics.begin() + static_cast<std::ptrdiff_t>(index * 4));
            std::copy(
                payload.label_counters.begin(), payload.label_counters.end(),
                result.label_counters.begin()
                    + static_cast<std::ptrdiff_t>(index * 3));
            std::copy(
                found->semantic_hash.begin(), found->semantic_hash.end(),
                result.semantic_hashes.begin()
                    + static_cast<std::ptrdiff_t>(index * 32));
            result.path_indices.insert(
                result.path_indices.end(), payload.path.begin(), payload.path.end());
            result.path_offsets.push_back(
                static_cast<std::int64_t>(result.path_indices.size()));
            record_protocol_move(found);
            entries_.splice(entries_.end(), entries_, found);
        }
        result.statistics = statistics_;
        return result;
    }

    void begin_protocol_transaction() {
        require_no_active_batch("begin_protocol_transaction");
        if (protocol_snapshot_.has_value()) {
            throw std::runtime_error(
                "native route-cache protocol transaction is already active");
        }
        ProtocolSnapshot snapshot;
        snapshot.statistics = statistics_;
        protocol_snapshot_.emplace(std::move(snapshot));
    }

    [[nodiscard]] std::array<std::int64_t, 11> commit_protocol_transaction() {
        require_no_active_batch("commit_protocol_transaction");
        require_protocol_snapshot("commit_protocol_transaction");
        commit_protocol_transaction_noexcept();
        return statistics_;
    }

    [[nodiscard]] std::array<std::int64_t, 11> rollback_protocol_transaction() {
        require_no_active_batch("rollback_protocol_transaction");
        require_protocol_snapshot("rollback_protocol_transaction");
        prepare_protocol_rollback();
        rollback_protocol_transaction_noexcept();
        return statistics_;
    }

    void inject_protocol_journal_failure_once() {
        require_protocol_snapshot("inject_protocol_journal_failure_once");
        protocol_journal_failure_injection_ = true;
    }

    [[nodiscard]] std::array<std::int64_t, 11> commit_store_batch() {
        require_active_batch("commit_store_batch");
        prepare_store_commit();
        commit_store_batch_noexcept();
        return statistics_;
    }

    [[nodiscard]] std::array<std::int64_t, 11> rollback_store_batch() {
        require_active_batch("rollback_store_batch");
        auto journal = std::move(*active_batch_);
        active_batch_.reset();
        rollback_journal(journal);
        return statistics_;
    }

    [[nodiscard]] Snapshot snapshot() const {
        return {entries_, seen_keys_, statistics_};
    }

    void restore(Snapshot snapshot) noexcept {
        active_batch_.reset();
        protocol_snapshot_.reset();
        entries_ = std::move(snapshot.entries);
        index_.clear();
        for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator) {
            index_.emplace(iterator->key, iterator);
        }
        seen_keys_ = std::move(snapshot.seen_keys);
        statistics_ = snapshot.statistics;
        protocol_journal_failure_injection_ = false;
    }

    void reset_empty_noexcept() noexcept {
        active_batch_.reset();
        protocol_snapshot_.reset();
        entries_.clear();
        index_.clear();
        seen_keys_.clear();
        statistics_.fill(0);
        protocol_journal_failure_injection_ = false;
    }

    [[nodiscard]] const std::list<Entry>& entries() const noexcept {
        return entries_;
    }

    [[nodiscard]] const std::array<std::int64_t, 11>& statistics() const noexcept {
        return statistics_;
    }

    void prepare_protocol_commit() const {
        require_no_active_batch("prepare_protocol_commit");
        require_protocol_snapshot("prepare_protocol_commit");
    }

    void prepare_protocol_rollback() const {
        require_no_active_batch("prepare_protocol_rollback");
        require_protocol_snapshot("prepare_protocol_rollback");
        for (const auto& operation : protocol_snapshot_->operations) {
            if (operation.kind != ProtocolOperationKind::eviction) {
                continue;
            }
            const auto retained_entry = std::find_if(
                protocol_snapshot_->retained_entries.begin(),
                protocol_snapshot_->retained_entries.end(),
                [&](const Entry& entry) { return entry.key == operation.key; });
            const auto retained_node = std::find_if(
                protocol_snapshot_->retained_index_nodes.begin(),
                protocol_snapshot_->retained_index_nodes.end(),
                [&](const auto& node) {
                    return !node.empty() && node.key() == operation.key;
                });
            if (retained_entry == protocol_snapshot_->retained_entries.end()
                || retained_node
                    == protocol_snapshot_->retained_index_nodes.end()) {
                throw std::logic_error(
                    "native route-cache protocol rollback journal is incomplete");
            }
        }
    }

    void commit_protocol_transaction_noexcept() noexcept {
        protocol_snapshot_.reset();
    }

    void rollback_protocol_transaction_noexcept() noexcept {
        rollback_protocol_operations_noexcept(*protocol_snapshot_);
        statistics_ = protocol_snapshot_->statistics;
        protocol_snapshot_.reset();
    }

    void prepare_store_commit() {
        require_active_batch("prepare_store_commit");
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->retained_index_nodes.reserve(
                protocol_snapshot_->retained_index_nodes.size()
                + active_batch_->evicted_index_nodes.size());
        }
    }

    void commit_store_batch_noexcept() noexcept {
        if (!active_batch_.has_value()) {
            std::terminate();
        }
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->retained_entries.splice(
                protocol_snapshot_->retained_entries.end(),
                active_batch_->evicted_entries);
            for (auto& node : active_batch_->evicted_index_nodes) {
                protocol_snapshot_->retained_index_nodes.push_back(std::move(node));
            }
        }
        active_batch_.reset();
    }

private:
    using Index = std::unordered_map<std::string, std::list<Entry>::iterator>;

    struct BatchJournal final {
        std::vector<std::string> inserted_keys;
        std::list<Entry> evicted_entries;
        std::vector<Index::node_type> evicted_index_nodes;
        std::array<std::int64_t, 11> statistics_before{};
        std::size_t protocol_operation_count_before = 0;
        bool active = false;
    };

    enum class ProtocolOperationKind { seen, move, insertion, eviction };

    struct ProtocolOperation final {
        ProtocolOperationKind kind;
        std::string key;
        std::optional<std::string> next_key;
    };

    struct ProtocolSnapshot final {
        std::array<std::int64_t, 11> statistics{};
        std::vector<ProtocolOperation> operations;
        std::list<Entry> retained_entries;
        std::vector<Index::node_type> retained_index_nodes;
    };

    std::int64_t max_entries_;
    std::int64_t max_memory_bytes_;
    std::list<Entry> entries_;
    Index index_;
    std::unordered_set<std::string> seen_keys_;
    std::array<std::int64_t, 11> statistics_{};
    std::optional<BatchJournal> active_batch_;
    std::optional<ProtocolSnapshot> protocol_snapshot_;
    bool protocol_journal_failure_injection_ = false;

    [[nodiscard]] static std::string route_key(
        const std::span<const std::int64_t> route) {
        std::string key;
        key.resize((route.size() + 1) * sizeof(std::int64_t));
        const auto length = static_cast<std::int64_t>(route.size());
        std::memcpy(key.data(), &length, sizeof(length));
        if (!route.empty()) {
            std::memcpy(
                key.data() + sizeof(length), route.data(),
                route.size() * sizeof(std::int64_t));
        }
        return key;
    }

    [[nodiscard]] std::list<Entry>::iterator find_entry(
        const std::string& key) {
        const auto found = index_.find(key);
        return found == index_.end() ? entries_.end() : found->second;
    }

    [[nodiscard]] std::optional<std::string> next_key(
        const std::list<Entry>::iterator current) const {
        const auto following = std::next(current);
        return following == entries_.end()
            ? std::nullopt
            : std::optional<std::string>(following->key);
    }

    void record_seen(const std::string& key, const std::string_view operation) {
        if (seen_keys_.contains(key)) {
            statistics_[10] = static_cast<std::int64_t>(seen_keys_.size());
            return;
        }
        record_protocol_seen(key);
        const auto [_, inserted] = seen_keys_.insert(key);
        if (!inserted) {
            throw std::logic_error(
                "native route-cache seen identity changed during "
                + std::string(operation));
        }
        statistics_[10] = static_cast<std::int64_t>(seen_keys_.size());
    }

    [[nodiscard]] std::size_t protocol_operation_count() const noexcept {
        return protocol_snapshot_.has_value()
            ? protocol_snapshot_->operations.size()
            : 0;
    }

    void record_protocol_seen(const std::string& key) {
        if (!protocol_snapshot_.has_value()) {
            return;
        }
        if (protocol_journal_failure_injection_) {
            protocol_journal_failure_injection_ = false;
            throw std::runtime_error(
                "injected native route-cache protocol journal failure");
        }
        protocol_snapshot_->operations.push_back(
            {ProtocolOperationKind::seen, key, std::nullopt});
    }

    void record_protocol_move(const std::list<Entry>::iterator entry) {
        if (protocol_snapshot_.has_value() && std::next(entry) != entries_.end()) {
            protocol_snapshot_->operations.push_back(
                {ProtocolOperationKind::move, entry->key, next_key(entry)});
        }
    }

    void record_protocol_insertion(const std::string& key) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.push_back(
                {ProtocolOperationKind::insertion, key, std::nullopt});
        }
    }

    void record_protocol_eviction(
        const std::string& key,
        std::optional<std::string> following_key) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.push_back(
                {ProtocolOperationKind::eviction, key, std::move(following_key)});
        }
    }

    void cancel_protocol_insertion(const std::string& key) {
        if (!protocol_snapshot_.has_value()) {
            return;
        }
        auto& operations = protocol_snapshot_->operations;
        const auto found = std::find_if(
            operations.rbegin(), operations.rend(),
            [&](const ProtocolOperation& operation) {
                return operation.kind == ProtocolOperationKind::insertion
                    && operation.key == key;
            });
        if (found == operations.rend()) {
            throw std::logic_error(
                "native route-cache transient eviction lost its insertion journal");
        }
        operations.erase(std::next(found).base());
    }

    void require_no_active_batch(const char* operation) const {
        if (active_batch_.has_value()) {
            throw std::runtime_error(
                std::string("native route-cache ") + operation
                + " is forbidden while a write batch is active");
        }
    }

    void require_active_batch(const char* operation) const {
        if (!active_batch_.has_value() || !active_batch_->active) {
            throw std::runtime_error(
                std::string("native route-cache ") + operation
                + " requires an active write batch");
        }
    }

    void require_protocol_snapshot(const char* operation) const {
        if (!protocol_snapshot_.has_value()) {
            throw std::runtime_error(
                std::string("native route-cache ") + operation
                + " requires an active protocol transaction");
        }
    }

    void rollback_journal(BatchJournal& journal) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.resize(
                journal.protocol_operation_count_before);
        }
        for (const auto& key : journal.inserted_keys) {
            const auto found = index_.find(key);
            if (found == index_.end()) {
                continue;
            }
            entries_.erase(found->second);
            index_.erase(found);
        }
        entries_.splice(entries_.begin(), journal.evicted_entries);
        for (auto& node : journal.evicted_index_nodes) {
            const auto restored = index_.insert(std::move(node));
            if (!restored.inserted) {
                throw std::logic_error(
                    "native route-cache rollback could not restore an index node");
            }
        }
        statistics_ = journal.statistics_before;
    }

    void rollback_protocol_operations_noexcept(
        ProtocolSnapshot& snapshot) noexcept {
        for (auto operation = snapshot.operations.rbegin();
             operation != snapshot.operations.rend(); ++operation) {
            if (operation->kind == ProtocolOperationKind::seen) {
                seen_keys_.erase(operation->key);
                continue;
            }
            if (operation->kind == ProtocolOperationKind::insertion) {
                const auto found = find_entry(operation->key);
                if (found == entries_.end()) {
                    std::terminate();
                }
                index_.erase(operation->key);
                entries_.erase(found);
                continue;
            }
            if (operation->kind == ProtocolOperationKind::eviction) {
                if (index_.contains(operation->key)) {
                    std::terminate();
                }
                auto position = entries_.end();
                if (operation->next_key.has_value()) {
                    const auto following = index_.find(*operation->next_key);
                    if (following == index_.end()) {
                        std::terminate();
                    }
                    position = following->second;
                }
                const auto retained_entry = std::find_if(
                    snapshot.retained_entries.begin(),
                    snapshot.retained_entries.end(),
                    [&](const Entry& entry) {
                        return entry.key == operation->key;
                    });
                const auto retained_node = std::find_if(
                    snapshot.retained_index_nodes.begin(),
                    snapshot.retained_index_nodes.end(),
                    [&](const auto& node) {
                        return !node.empty() && node.key() == operation->key;
                    });
                if (retained_entry == snapshot.retained_entries.end()
                    || retained_node == snapshot.retained_index_nodes.end()) {
                    std::terminate();
                }
                entries_.splice(
                    position, snapshot.retained_entries, retained_entry);
                const auto restored = index_.insert(std::move(*retained_node));
                if (!restored.inserted) {
                    std::terminate();
                }
                continue;
            }
            const auto found = find_entry(operation->key);
            if (found == entries_.end()) {
                std::terminate();
            }
            auto position = entries_.end();
            if (operation->next_key.has_value()) {
                const auto following = index_.find(*operation->next_key);
                if (following == index_.end()) {
                    std::terminate();
                }
                position = following->second;
            }
            entries_.splice(position, entries_, found);
        }
    }
};

class ThreeLaneLoopControllerV2 final {
public:
    ThreeLaneLoopControllerV2(
        const std::int64_t max_iterations,
        const bool fixed_work,
        const std::int64_t exhaustion_rounds,
        const std::int64_t min_iterations_before_exhaustion,
        const std::int64_t started_before_bootstrap,
        const ThreeLaneTerminationStateV2& bootstrap,
        const bool bootstrap_iteration_completed,
        const bool bootstrap_exhaustion_eligible)
        : max_iterations_(max_iterations),
          fixed_work_(fixed_work),
          exhaustion_rounds_(exhaustion_rounds),
          min_iterations_before_exhaustion_(
              min_iterations_before_exhaustion),
          previous_started_(bootstrap.started),
          no_exact_rounds_(
              bootstrap_exhaustion_eligible
                  && bootstrap.started == started_before_bootstrap
              ? 1
              : 0),
          effective_iterations_(bootstrap_iteration_completed ? 1 : 0) {
        if (max_iterations_ <= 0 || exhaustion_rounds_ < 0
            || min_iterations_before_exhaustion_ < 0
            || started_before_bootstrap < 0 || bootstrap.started < 0
            || bootstrap.started < started_before_bootstrap
            || bootstrap.reason < 0 || bootstrap.reason > 3) {
            throw std::invalid_argument(
                "native three-lane loop bootstrap state is invalid");
        }
    }

    [[nodiscard]] ThreeLaneLoopDecisionV2 observe_followup(
        const std::int64_t iteration,
        ThreeLaneTerminationStateV2 termination,
        const bool iteration_completed,
        const bool exhaustion_eligible) {
        if (iteration <= 0 || iteration >= max_iterations_
            || iteration != next_iteration_ || termination.reason < 0
            || termination.reason > 3 || termination.started < previous_started_
            || termination.completed_iterations < 0) {
            throw std::invalid_argument(
                "native three-lane follow-up state is invalid");
        }
        if (iteration_completed) {
            ++effective_iterations_;
        }
        if (exhaustion_eligible) {
            no_exact_rounds_ = termination.started == previous_started_
                ? no_exact_rounds_ + 1
                : 0;
        }
        previous_started_ = termination.started;
        ++next_iteration_;
        auto exhaustion_applied = false;
        if (iteration_completed && exhaustion_eligible && fixed_work_
            && effective_iterations_ >= min_iterations_before_exhaustion_
            && no_exact_rounds_ >= exhaustion_rounds_) {
            termination.reason = 3;
            exhaustion_applied = true;
        }
        return {
            termination,
            no_exact_rounds_,
            termination.reason != 0,
            exhaustion_applied,
        };
    }

    [[nodiscard]] std::int64_t next_iteration() const noexcept {
        return next_iteration_;
    }

private:
    std::int64_t max_iterations_ = 0;
    bool fixed_work_ = false;
    std::int64_t exhaustion_rounds_ = 0;
    std::int64_t min_iterations_before_exhaustion_ = 0;
    std::int64_t previous_started_ = 0;
    std::int64_t no_exact_rounds_ = 0;
    std::int64_t effective_iterations_ = 0;
    std::int64_t next_iteration_ = 1;
};

class SearchBudgetStateV2 final {
public:
    struct RoundReservation final {
        std::int64_t requested = 0;
        std::int64_t granted = 0;
        std::int64_t remaining = 0;
    };

    struct ExactReservation final {
        std::int64_t requested = 0;
        std::int64_t granted = 0;
    };

    struct Snapshot final {
        bool round_active = false;
        std::int64_t lane_id = -1;
        std::int64_t iteration = -1;
        std::int64_t round_used = 0;
        std::int64_t started = 0;
        std::int64_t completed = 0;
        std::int64_t interrupted = 0;
    };

    SearchBudgetStateV2(
        const std::int64_t exact_budget,
        const std::int64_t round_budget)
        : exact_budget_(exact_budget), round_budget_(round_budget) {
        if (exact_budget_ == 0 || exact_budget_ < -1 || round_budget_ <= 0) {
            throw std::invalid_argument("native budget limits are invalid");
        }
    }

    void begin_round(
        const std::int64_t lane_id,
        const std::int64_t iteration) {
        validate_round_identity(lane_id, iteration);
        if (round_active_ && lane_id_ == lane_id && iteration_ == iteration) {
            return;
        }
        round_active_ = true;
        lane_id_ = lane_id;
        iteration_ = iteration;
        round_used_ = 0;
    }

    void begin_shared_iteration_round(
        const std::int64_t semantic_lane_id,
        const std::int64_t iteration) {
        validate_round_identity(semantic_lane_id, iteration);
        if (round_active_ && iteration_ == iteration) {
            lane_id_ = semantic_lane_id;
            return;
        }
        begin_round(semantic_lane_id, iteration);
    }

    void finish_round() noexcept {
        round_active_ = false;
        lane_id_ = -1;
        iteration_ = -1;
        round_used_ = 0;
    }

    [[nodiscard]] RoundReservation reserve_round(
        const std::int64_t requested,
        const bool atomic) {
        const auto reservation = plan_round_reservation(requested, atomic);
        if (reservation.granted > 0
            && round_used_ > std::numeric_limits<std::int64_t>::max()
                    - reservation.granted) {
            throw std::overflow_error("native round budget counter overflow");
        }
        round_used_ += reservation.granted;
        return reservation;
    }

    [[nodiscard]] ExactReservation reserve_exact(
        const std::int64_t requested) {
        if (requested <= 0) {
            throw std::invalid_argument(
                "exact-call reservation must be positive");
        }
        const auto granted = exact_budget_ < 0
            ? requested
            : std::min(
                requested,
                std::max<std::int64_t>(0, exact_budget_ - started_));
        if (granted > 0
            && started_ > std::numeric_limits<std::int64_t>::max() - granted) {
            throw std::overflow_error("native exact budget counter overflow");
        }
        started_ += granted;
        return {requested, granted};
    }

    void settle_exact(
        const std::int64_t completed,
        const std::int64_t interrupted) {
        validate_exact_settlement(completed, interrupted);
        completed_ += completed;
        interrupted_ += interrupted;
    }

    [[nodiscard]] std::int64_t exact_remaining() const noexcept {
        return exact_budget_ < 0
            ? -1
            : std::max<std::int64_t>(0, exact_budget_ - started_);
    }

    [[nodiscard]] std::int64_t exact_budget() const noexcept {
        return exact_budget_;
    }

    [[nodiscard]] std::int64_t round_remaining() const noexcept {
        return round_active_
            ? std::max<std::int64_t>(0, round_budget_ - round_used_)
            : round_budget_;
    }

    [[nodiscard]] bool budget_reached() const noexcept {
        return exact_budget_ >= 0 && started_ >= exact_budget_;
    }

    [[nodiscard]] Snapshot snapshot() const noexcept {
        return {
            round_active_, lane_id_, iteration_, round_used_,
            started_, completed_, interrupted_,
        };
    }

    void restore(const Snapshot& snapshot) noexcept {
        round_active_ = snapshot.round_active;
        lane_id_ = snapshot.lane_id;
        iteration_ = snapshot.iteration;
        round_used_ = snapshot.round_used;
        started_ = snapshot.started;
        completed_ = snapshot.completed;
        interrupted_ = snapshot.interrupted;
    }

    [[nodiscard]] std::array<std::int64_t, 9> values() const noexcept {
        return values_after_settlement(0, 0);
    }

    [[nodiscard]] std::array<std::int64_t, 9> values_after_settlement(
        const std::int64_t completed,
        const std::int64_t interrupted) const noexcept {
        return {
            round_active_ ? 1 : 0,
            lane_id_,
            iteration_,
            round_used_,
            round_remaining(),
            started_,
            completed_ + completed,
            interrupted_ + interrupted,
            exact_budget_ >= 0 && started_ >= exact_budget_ ? 1 : 0,
        };
    }

    [[nodiscard]] Snapshot validated_snapshot_from_values(
        const std::span<const std::int64_t> values) const {
        if (values.size() != 9 || (values[0] != 0 && values[0] != 1)
            || values[3] < 0 || values[5] < 0 || values[6] < 0
            || values[7] < 0 || values[6] > values[5]
            || values[7] > values[5] - values[6]
            || (values[0] == 1 && (values[1] < 0 || values[2] < 0))) {
            throw std::invalid_argument(
                "native budget snapshot values are invalid");
        }
        const auto active = values[0] == 1;
        const auto round_used = active ? values[3] : 0;
        if (round_used > round_budget_
            || (exact_budget_ >= 0 && values[5] > exact_budget_)) {
            throw std::invalid_argument(
                "native budget snapshot exceeds configured limits");
        }
        return {
            active,
            active ? values[1] : -1,
            active ? values[2] : -1,
            round_used,
            values[5],
            values[6],
            values[7],
        };
    }

    void rollback_preserving_exact(
        const Snapshot& snapshot,
        const bool restore_round) noexcept {
        const auto started_delta = started_ - snapshot.started;
        const auto completed_delta = completed_ - snapshot.completed;
        const auto interrupted_delta = interrupted_ - snapshot.interrupted;
        if (started_delta < 0 || completed_delta < 0 || interrupted_delta < 0
            || completed_delta + interrupted_delta != started_delta) {
            std::terminate();
        }
        if (restore_round || started_delta == 0) {
            round_active_ = snapshot.round_active;
            lane_id_ = snapshot.lane_id;
            iteration_ = snapshot.iteration;
            round_used_ = snapshot.round_used;
        }
        started_ = snapshot.started + started_delta;
        completed_ = snapshot.completed + completed_delta;
        interrupted_ = snapshot.interrupted + interrupted_delta;
    }

private:
    std::int64_t exact_budget_ = 0;
    std::int64_t round_budget_ = 0;
    bool round_active_ = false;
    std::int64_t lane_id_ = -1;
    std::int64_t iteration_ = -1;
    std::int64_t round_used_ = 0;
    std::int64_t started_ = 0;
    std::int64_t completed_ = 0;
    std::int64_t interrupted_ = 0;

    static void validate_round_identity(
        const std::int64_t lane_id,
        const std::int64_t iteration) {
        if (lane_id < 0 || iteration < 0) {
            throw std::invalid_argument(
                "native round identity must be non-negative");
        }
    }

    [[nodiscard]] RoundReservation plan_round_reservation(
        const std::int64_t requested,
        const bool atomic) const noexcept {
        if (requested <= 0) {
            return {requested, 0, round_remaining()};
        }
        auto granted = requested;
        if (round_active_) {
            granted = atomic
                ? (requested <= round_remaining() ? requested : 0)
                : std::min(requested, round_remaining());
        }
        const auto remaining = round_active_
            ? std::max<std::int64_t>(0, round_remaining() - granted)
            : round_budget_;
        return {requested, granted, remaining};
    }

    void validate_exact_settlement(
        const std::int64_t completed,
        const std::int64_t interrupted) const {
        const auto unsettled = started_ - completed_ - interrupted_;
        if (completed < 0 || interrupted < 0 || unsettled < 0
            || completed > unsettled
            || interrupted > unsettled - completed) {
            throw std::runtime_error("invalid exact-call settlement");
        }
    }
};

struct Stage04BoundaryStateV2 final {
    std::array<std::int64_t, 4> statuses{};
    std::array<double, 8> old_new_weights{};
    std::array<std::int64_t, 4> calls_at_boundary{};
    std::array<double, 4> rewards_at_boundary{};
    std::array<std::int64_t, 7> control_status{};
    double reheat_floor = 0.0;

    void validate() const {
        if (std::ranges::any_of(
                statuses,
                [](const std::int64_t value) {
                    return value < -1 || value > 1;
                })
            || std::ranges::any_of(
                old_new_weights,
                [](const double value) {
                    return !std::isfinite(value) || value < 0.0;
                })
            || std::ranges::any_of(
                calls_at_boundary,
                [](const std::int64_t value) { return value < 0; })
            || std::ranges::any_of(
                rewards_at_boundary,
                [](const double value) {
                    return !std::isfinite(value) || value < 0.0;
                })
            || !std::isfinite(reheat_floor) || reheat_floor < 0.0) {
            throw std::logic_error(
                "native Stage 4 boundary state is inconsistent");
        }
    }
};

struct Stage04InitializationStateV2 final {
    double temperature = 0.0;
    std::vector<double> positive_deltas;
    std::vector<std::int64_t> plan_offsets;
    std::vector<std::int64_t> route_offsets;
    std::vector<std::int64_t> route_indices;

    void validate() const {
        const auto offsets_valid = [](const auto& offsets) {
            return !offsets.empty() && offsets.front() == 0
                && std::ranges::is_sorted(offsets)
                && offsets.back() >= 0;
        };
        if (!std::isfinite(temperature) || temperature <= 0.0
            || std::ranges::any_of(
                positive_deltas,
                [](const double value) {
                    return !std::isfinite(value) || value <= 0.0;
                })
            || !offsets_valid(plan_offsets)
            || !offsets_valid(route_offsets)
            || static_cast<std::size_t>(plan_offsets.back())
                != route_offsets.size() - 1
            || static_cast<std::size_t>(route_offsets.back())
                != route_indices.size()) {
            throw std::logic_error(
                "native Stage 4 initialization state is inconsistent");
        }
    }
};

struct FullStage04StateV2 final {
    std::array<double, 20> weights{};
    std::array<double, 20> segment_rewards{};
    std::array<std::int64_t, 20> segment_calls{};
    std::array<std::array<std::int64_t, 8>, 20> totals{};

    void validate() const {
        for (std::size_t operation = 0; operation < weights.size(); ++operation) {
            if (!std::isfinite(weights[operation]) || weights[operation] < 0.0
                || !std::isfinite(segment_rewards[operation])
                || segment_rewards[operation] < 0.0
                || segment_calls[operation] < 0
                || std::any_of(
                    totals[operation].begin(), totals[operation].end(),
                    [](const std::int64_t value) { return value < 0; })) {
                throw std::logic_error(
                    "native full Stage 4 state is inconsistent");
            }
        }
    }
};

inline void accumulate_stage04_operator_outcome_v2(
    std::array<std::int64_t, 8>& totals,
    double& segment_reward,
    std::int64_t& segment_calls,
    const std::array<double, 7>& rewards,
    const bool accepted,
    const std::int64_t comparison,
    const bool is_global_best,
    const bool vehicle_reduction,
    const bool adaptive) noexcept {
    ++totals[0];
    auto reward = rewards[0];
    if (accepted) {
        ++totals[1];
        if (comparison < 0) {
            ++totals[2];
        } else if (comparison == 0) {
            ++totals[3];
        } else {
            ++totals[4];
        }
        reward = is_global_best
            ? (vehicle_reduction ? rewards[6] : rewards[5])
            : comparison < 0
            ? (vehicle_reduction ? rewards[4] : rewards[3])
            : comparison == 0 ? rewards[2] : rewards[1];
        if (is_global_best) {
            ++totals[6];
        }
        if (vehicle_reduction) {
            ++totals[7];
        }
    } else {
        ++totals[5];
    }
    if (adaptive) {
        segment_reward += reward;
        ++segment_calls;
    }
}

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

struct ProblemV2;

[[nodiscard]] RepairResultV2 repair_candidate_routes_v2(
    const ProblemV2& problem,
    RouteBatchViewV2 partial_routes,
    std::span<const std::int64_t> removed_customers,
    double epsilon,
    std::int64_t route_change_limit,
    bool allow_new_routes);

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

struct ExactProblemViewV2 final {
    std::span<const std::int64_t> node_kind;
    std::span<const double> ready_time;
    std::span<const double> due_date;
    std::span<const double> service_time;
    std::span<const double> distance;
    std::span<const double> vehicle;

    void validate() const {
        const auto nodes = node_kind.size();
        if (nodes == 0 || ready_time.size() != nodes
            || due_date.size() != nodes || service_time.size() != nodes
            || nodes > std::numeric_limits<std::size_t>::max() / nodes
            || distance.size() != nodes * nodes || vehicle.size() != 5) {
            throw std::invalid_argument(
                "native exact problem arrays do not share one shape");
        }
        if (!std::isfinite(vehicle[0]) || vehicle[0] < 0.0
            || !std::isfinite(vehicle[2]) || vehicle[2] < 0.0
            || !std::isfinite(vehicle[3]) || vehicle[3] < 0.0
            || !std::isfinite(vehicle[4]) || vehicle[4] <= 0.0) {
            throw std::invalid_argument(
                "native exact vehicle parameters are invalid");
        }
        std::int64_t depot_count = 0;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto kind = node_kind[node];
            if (kind == native_kernels::depot_kind) {
                ++depot_count;
            } else if (kind != native_kernels::customer_kind
                && kind != native_kernels::station_kind) {
                throw std::invalid_argument(
                    "native exact node kind is invalid");
            }
            if (!std::isfinite(ready_time[node])
                || !std::isfinite(due_date[node])
                || !std::isfinite(service_time[node])) {
                throw std::invalid_argument(
                    "native exact node metadata is non-finite");
            }
        }
        if (depot_count != 1) {
            throw std::invalid_argument(
                "native exact problem requires exactly one depot");
        }
        if (std::any_of(distance.begin(), distance.end(), [](const double value) {
                return !std::isfinite(value) || value < 0.0;
            })) {
            throw std::invalid_argument(
                "native exact distance matrix is invalid");
        }
    }
};

[[nodiscard]] inline native_kernels::ExactBatchOutput
run_local_exact_batch_v2(
    const ExactProblemViewV2& problem,
    const RouteBatchViewV2 routes,
    const double deadline_remaining,
    const std::int64_t batch_size) {
    problem.validate();
    routes.validate("native exact routes");
    if (batch_size <= 0 || std::isnan(deadline_remaining)) {
        throw std::invalid_argument(
            "native exact deadline/batch control is invalid");
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> stations;
    for (std::size_t node = 0; node < problem.node_kind.size(); ++node) {
        if (problem.node_kind[node] == native_kernels::depot_kind) {
            depot = static_cast<std::int64_t>(node);
        } else if (problem.node_kind[node] == native_kernels::station_kind) {
            stations.push_back(static_cast<std::int64_t>(node));
        }
    }
    for (std::size_t row = 0; row < routes.route_count(); ++row) {
        std::unordered_set<std::int64_t> seen;
        seen.reserve(routes.route(row).size());
        for (const auto node : routes.route(row)) {
            if (node < 0
                || static_cast<std::size_t>(node) >= problem.node_kind.size()
                || problem.node_kind[static_cast<std::size_t>(node)]
                    != native_kernels::customer_kind
                || !seen.insert(node).second) {
                throw std::invalid_argument(
                    "native exact route contains an invalid customer");
            }
        }
    }
    auto output = native_kernels::run_exact_charging_batch(
        problem.node_kind.data(), problem.ready_time.data(),
        problem.due_date.data(), problem.service_time.data(),
        problem.distance.data(), problem.vehicle.data(), routes.offsets.data(),
        routes.indices.data(), problem.node_kind.size(), routes.route_count(),
        depot, stations, deadline_remaining, batch_size);
    native_kernels::validate_exact_batch_output(
        output, problem.node_kind.data(), routes.offsets.data(),
        routes.indices.data(), problem.node_kind.size(), routes.route_count(),
        routes.indices.size(), depot, batch_size);
    return output;
}

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

struct CandidateRoundPlanInputV2 final {
    const ScreenBatchResultV2& screening;
    RouteBatchViewV2 routes;
    std::span<const std::int64_t> lexical_rank;
    std::span<const std::int64_t> cache_hit_flags;
    std::int64_t top_k = 0;
    std::int64_t exact_budget = 0;

    void validate() const {
        screening.validate();
        routes.validate("native candidate-round routes");
        const auto count = screening.candidate_count();
        if (routes.route_count() != count || cache_hit_flags.size() != count
            || lexical_rank.empty() || top_k <= 0 || exact_budget < 0) {
            throw std::invalid_argument(
                "native candidate-round plan input shape/control is invalid");
        }
        std::unordered_set<std::int64_t> lexical_values;
        lexical_values.reserve(lexical_rank.size());
        for (const auto value : lexical_rank) {
            if (value < 0
                || static_cast<std::size_t>(value) >= lexical_rank.size()
                || !lexical_values.insert(value).second) {
                throw std::invalid_argument(
                    "native candidate-round lexical rank is not a permutation");
            }
        }
        for (std::size_t candidate = 0; candidate < count; ++candidate) {
            if (routes.offsets[candidate] == routes.offsets[candidate + 1]) {
                throw std::invalid_argument(
                    "native candidate-round route cannot be empty");
            }
            if (cache_hit_flags[candidate] != 0
                && cache_hit_flags[candidate] != 1) {
                throw std::invalid_argument(
                    "native candidate-round cache flag is invalid");
            }
        }
        for (const auto node : routes.indices) {
            if (node < 0
                || static_cast<std::size_t>(node) >= lexical_rank.size()) {
                throw std::invalid_argument(
                    "native candidate-round route node is out of range");
            }
        }
    }
};

struct CandidateRoundPlanV2 final {
    std::vector<std::int64_t> resolutions;
    std::vector<std::int64_t> sources;
    std::vector<std::int64_t> cache_journal;
    std::vector<std::int64_t> exact_candidate_ids;
    std::vector<std::int64_t> exact_route_offsets;
    std::vector<std::int64_t> exact_route_indices;
    std::array<std::int64_t, 10> counters{};
};

[[nodiscard]] inline CandidateRoundPlanV2 plan_candidate_round_v2(
    const CandidateRoundPlanInputV2& input) {
    input.validate();
    const auto count = input.screening.candidate_count();
    const auto& ids = input.screening.candidate_ids;
    const auto& statuses = input.screening.statuses;
    const auto& duplicates = input.screening.duplicate_of;
    const auto& codes = input.screening.codes;
    const auto& metrics = input.screening.metrics;

    struct RankableCandidate final {
        std::size_t index;
        double distance_lower_bound;
    };
    std::vector<RankableCandidate> rankable;
    rankable.reserve(count);
    std::int64_t rejected_count = 0;
    std::int64_t negative_hit_count = 0;
    std::int64_t duplicate_count = 0;
    for (std::size_t index = 0; index < count; ++index) {
        if (statuses[index] == 1) {
            ++duplicate_count;
            continue;
        }
        if (codes[index * 16] == 0) {
            ++rejected_count;
            if (statuses[index] == 2) {
                ++negative_hit_count;
            }
            continue;
        }
        rankable.push_back({index, metrics[index * 15 + 3]});
    }
    const auto route_less = [&](const std::size_t left,
                                const std::size_t right) {
        auto left_cursor = input.routes.offsets[left];
        auto right_cursor = input.routes.offsets[right];
        const auto left_end = input.routes.offsets[left + 1];
        const auto right_end = input.routes.offsets[right + 1];
        while (left_cursor < left_end && right_cursor < right_end) {
            const auto left_rank = input.lexical_rank[static_cast<std::size_t>(
                input.routes.indices[static_cast<std::size_t>(left_cursor)])];
            const auto right_rank = input.lexical_rank[static_cast<std::size_t>(
                input.routes.indices[static_cast<std::size_t>(right_cursor)])];
            if (left_rank != right_rank) {
                return left_rank < right_rank;
            }
            ++left_cursor;
            ++right_cursor;
        }
        return (left_end - input.routes.offsets[left])
            < (right_end - input.routes.offsets[right]);
    };
    std::stable_sort(
        rankable.begin(), rankable.end(),
        [&](const RankableCandidate& left, const RankableCandidate& right) {
            if (left.distance_lower_bound != right.distance_lower_bound) {
                return left.distance_lower_bound < right.distance_lower_bound;
            }
            if (route_less(left.index, right.index)) {
                return true;
            }
            if (route_less(right.index, left.index)) {
                return false;
            }
            return ids[left.index] < ids[right.index];
        });
    const auto selected_count = std::min<std::size_t>(
        rankable.size(), static_cast<std::size_t>(input.top_k));

    CandidateRoundPlanV2 plan;
    plan.resolutions.assign(count, 3);
    plan.sources.assign(count, -1);
    plan.cache_journal.assign(count * 3, 0);
    for (std::size_t index = 0; index < count; ++index) {
        plan.cache_journal[index * 3] = ids[index];
        plan.cache_journal[index * 3 + 2] = -1;
        if (statuses[index] == 1) {
            plan.resolutions[index] = 5;
            plan.sources[index] = duplicates[index];
            plan.cache_journal[index * 3 + 1] = 6;
        } else if (codes[index * 16] == 0) {
            plan.resolutions[index] = 0;
            plan.cache_journal[index * 3 + 1] = 5;
        } else {
            plan.cache_journal[index * 3 + 1] = 4;
        }
    }

    std::vector<std::size_t> exact_indices;
    std::int64_t cache_hit_count = 0;
    for (std::size_t rank = 0; rank < selected_count; ++rank) {
        const auto index = rankable[rank].index;
        if (input.cache_hit_flags[index] == 1) {
            plan.resolutions[index] = 1;
            plan.cache_journal[index * 3 + 1] = 1;
            ++cache_hit_count;
        } else {
            exact_indices.push_back(index);
        }
    }
    std::int64_t budget_skip_count = 0;
    if (exact_indices.size() > static_cast<std::size_t>(input.exact_budget)) {
        budget_skip_count = static_cast<std::int64_t>(exact_indices.size());
        for (const auto index : exact_indices) {
            plan.resolutions[index] = 4;
            plan.cache_journal[index * 3 + 1] = 3;
        }
        exact_indices.clear();
    }

    plan.exact_route_offsets.push_back(0);
    plan.exact_candidate_ids.reserve(exact_indices.size());
    for (std::size_t ordinal = 0; ordinal < exact_indices.size(); ++ordinal) {
        const auto index = exact_indices[ordinal];
        plan.resolutions[index] = 2;
        plan.cache_journal[index * 3 + 1] = 2;
        plan.cache_journal[index * 3 + 2] =
            static_cast<std::int64_t>(ordinal);
        plan.exact_candidate_ids.push_back(ids[index]);
        const auto route = input.routes.route(index);
        plan.exact_route_indices.insert(
            plan.exact_route_indices.end(), route.begin(), route.end());
        plan.exact_route_offsets.push_back(
            static_cast<std::int64_t>(plan.exact_route_indices.size()));
    }

    std::unordered_map<std::int64_t, std::size_t> candidate_positions;
    candidate_positions.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        candidate_positions.emplace(ids[index], index);
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (statuses[index] != 1) {
            continue;
        }
        const auto source = candidate_positions.find(duplicates[index]);
        if (source == candidate_positions.end() || source->second >= index) {
            throw std::logic_error(
                "native candidate-round duplicate source identity was lost");
        }
        const auto source_resolution = plan.resolutions[source->second];
        plan.resolutions[index] = source_resolution == 0
            ? 0
            : ((source_resolution == 1 || source_resolution == 2)
                   ? 5
                   : source_resolution);
    }

    plan.counters = {
        static_cast<std::int64_t>(count),
        static_cast<std::int64_t>(rankable.size()),
        static_cast<std::int64_t>(selected_count),
        rejected_count,
        negative_hit_count,
        cache_hit_count,
        static_cast<std::int64_t>(plan.exact_candidate_ids.size()),
        budget_skip_count,
        duplicate_count,
        0,
    };
    return plan;
}

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
            || protocol_control[4] < 10 || protocol_control[5] < 0
            || protocol_control[6] < 0 || protocol_control[7] <= 0
            || protocol_control[8] <= 0) {
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
    InitialStateV2 state;
    state.request_sha256 = request.sha256();
    const auto remaining_seconds = request.config.deadline[1]
        - std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    if (remaining_seconds <= 0.0) {
        throw std::runtime_error(
            "native search deadline expired before initial exact work");
    }
    state.exact = run_local_exact_batch_v2(
        {
            problem.node_kind,
            problem.ready_time,
            problem.due_date,
            problem.service_time,
            problem.distance,
            problem.vehicle,
        },
        {problem.initial_route_offsets, problem.initial_route_indices},
        remaining_seconds, request.config.search_control[2]);
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

inline std::vector<std::array<std::uint8_t, 32>>
exact_semantic_hashes_v2(
    const RouteBatchViewV2 routes,
    const native_kernels::ExactBatchOutput& exact) {
    routes.validate("native exact semantic hash");
    const auto count = routes.route_count();
    if (exact.path_offsets.size() != count + 1
        || exact.path_offsets.front() != 0
        || exact.path_offsets.back()
            != static_cast<std::int64_t>(exact.path_indices.size())
        || exact.statuses.size() != count || exact.reasons.size() != count
        || exact.metrics.size() != count * 4
        || exact.label_counters.size() != count * 3) {
        throw std::logic_error(
            "native exact semantic hash shapes do not reconcile");
    }
    std::vector<std::array<std::uint8_t, 32>> hashes(count);
    for (std::size_t route = 0; route < count; ++route) {
        if (exact.path_offsets[route] < 0
            || exact.path_offsets[route] > exact.path_offsets[route + 1]) {
            throw std::logic_error(
                "native exact semantic path offsets are not monotone");
        }
        std::string evidence("stage05.2-native-route-result-v2");
        append_candidate_evidence_values_v2<std::int64_t>(
            evidence, routes.route(route));
        append_candidate_evidence_values_v2<std::int64_t>(
            evidence, {exact.statuses.data() + route, 1});
        append_candidate_evidence_values_v2<std::int64_t>(
            evidence, {exact.reasons.data() + route, 1});
        append_candidate_evidence_values_v2<double>(
            evidence, {exact.metrics.data() + route * 4, 4});
        append_candidate_evidence_values_v2<std::int64_t>(
            evidence, {exact.label_counters.data() + route * 3, 3});
        const auto first = exact.path_offsets[route];
        const auto last = exact.path_offsets[route + 1];
        append_candidate_evidence_values_v2<std::int64_t>(
            evidence,
            {exact.path_indices.data() + first,
             static_cast<std::size_t>(last - first)});
        hashes[route] = native_protocol::native_sha256_digest(evidence);
    }
    return hashes;
}

inline void append_exact_json_hex_escape_v2(
    std::string& output,
    const std::uint16_t value) {
    constexpr char hexadecimal[] = "0123456789abcdef";
    output += "\\u";
    output.push_back(hexadecimal[(value >> 12U) & 0xFU]);
    output.push_back(hexadecimal[(value >> 8U) & 0xFU]);
    output.push_back(hexadecimal[(value >> 4U) & 0xFU]);
    output.push_back(hexadecimal[value & 0xFU]);
}

inline void append_exact_json_string_v2(
    std::string& output,
    const std::string_view value) {
    output.push_back('"');
    for (std::size_t offset = 0; offset < value.size();) {
        const auto first = static_cast<unsigned char>(value[offset]);
        if (first < 0x80U) {
            ++offset;
            switch (first) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (first < 0x20U) {
                    append_exact_json_hex_escape_v2(
                        output, static_cast<std::uint16_t>(first));
                } else {
                    output.push_back(static_cast<char>(first));
                }
                break;
            }
            continue;
        }
        std::uint32_t codepoint = 0;
        std::size_t width = 0;
        if ((first & 0xE0U) == 0xC0U) {
            codepoint = first & 0x1FU;
            width = 2;
        } else if ((first & 0xF0U) == 0xE0U) {
            codepoint = first & 0x0FU;
            width = 3;
        } else if ((first & 0xF8U) == 0xF0U) {
            codepoint = first & 0x07U;
            width = 4;
        } else {
            throw std::invalid_argument(
                "native node name is not valid UTF-8");
        }
        if (offset + width > value.size()) {
            throw std::invalid_argument(
                "native node name is truncated UTF-8");
        }
        for (std::size_t continuation = 1; continuation < width;
             ++continuation) {
            const auto byte = static_cast<unsigned char>(
                value[offset + continuation]);
            if ((byte & 0xC0U) != 0x80U) {
                throw std::invalid_argument(
                    "native node name is not valid UTF-8");
            }
            codepoint = (codepoint << 6U) | (byte & 0x3FU);
        }
        const auto minimum = width == 2 ? 0x80U
            : width == 3 ? 0x800U : 0x10000U;
        if (codepoint < minimum || codepoint > 0x10FFFFU
            || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            throw std::invalid_argument(
                "native node name is non-canonical UTF-8");
        }
        offset += width;
        if (codepoint <= 0xFFFFU) {
            append_exact_json_hex_escape_v2(
                output, static_cast<std::uint16_t>(codepoint));
        } else {
            const auto adjusted = codepoint - 0x10000U;
            append_exact_json_hex_escape_v2(
                output,
                static_cast<std::uint16_t>(0xD800U + (adjusted >> 10U)));
            append_exact_json_hex_escape_v2(
                output,
                static_cast<std::uint16_t>(
                    0xDC00U + (adjusted & 0x3FFU)));
        }
    }
    output.push_back('"');
}

inline void append_exact_json_integer_v2(
    std::string& output,
    const std::int64_t value) {
    char buffer[32];
    const auto converted = std::to_chars(
        std::begin(buffer), std::end(buffer), value);
    if (converted.ec != std::errc{}) {
        throw std::runtime_error("native JSON integer conversion failed");
    }
    output.append(buffer, converted.ptr);
}

inline void append_exact_json_float_v2(
    std::string& output,
    const double value) {
    if (std::isnan(value)) {
        output += "NaN";
        return;
    }
    if (std::isinf(value)) {
        output += std::signbit(value) ? "-Infinity" : "Infinity";
        return;
    }
    char buffer[64];
    const auto converted = std::to_chars(
        std::begin(buffer), std::end(buffer), value,
        std::chars_format::general);
    if (converted.ec != std::errc{}) {
        throw std::runtime_error("native JSON float conversion failed");
    }
    const auto first = output.size();
    output.append(buffer, converted.ptr);
    if (output.find_first_of(".eE", first) == std::string::npos) {
        output += ".0";
    }
}

inline std::string node_name_v2(
    const ProblemV2& problem,
    const std::int64_t node) {
    if (node < 0
        || static_cast<std::size_t>(node) >= problem.node_count()) {
        throw std::logic_error(
            "native exact path references an unknown node name");
    }
    const auto first = problem.node_name_offsets[static_cast<std::size_t>(node)];
    const auto last = problem.node_name_offsets[
        static_cast<std::size_t>(node) + 1];
    return std::string(
        reinterpret_cast<const char*>(problem.node_name_bytes.data() + first),
        static_cast<std::size_t>(last - first));
}

inline std::vector<std::int64_t> exact_entry_bytes_v2(
    const ProblemV2& problem,
    const native_kernels::ExactBatchOutput& exact) {
    if (exact.path_offsets.empty()) {
        throw std::logic_error(
            "native exact cache size requires path offsets");
    }
    const auto count = exact.path_offsets.size() - 1;
    if (exact.path_offsets.front() != 0
        || exact.path_offsets.back()
            != static_cast<std::int64_t>(exact.path_indices.size())
        || exact.statuses.size() != count || exact.reasons.size() != count
        || exact.metrics.size() != count * 4
        || exact.label_counters.size() != count * 3) {
        throw std::logic_error(
            "native exact cache size shapes do not reconcile");
    }
    std::vector<std::int64_t> output(count);
    for (std::size_t route = 0; route < count; ++route) {
        const bool feasible =
            exact.statuses[route] == native_kernels::feasible_status;
        std::string payload;
        payload.reserve(256);
        payload += "{\"charged_energy\":";
        append_exact_json_float_v2(payload, exact.metrics[route * 4 + 2]);
        payload += ",\"charging_time\":";
        append_exact_json_float_v2(payload, exact.metrics[route * 4 + 3]);
        payload += ",\"distance\":";
        append_exact_json_float_v2(payload, exact.metrics[route * 4]);
        payload += ",\"failure_reason\":";
        if (feasible) {
            append_exact_json_string_v2(payload, "");
        } else if (exact.reasons[route] == 1) {
            append_exact_json_string_v2(
                payload,
                "no feasible station-insertion pattern for fixed customer order");
        } else {
            throw std::logic_error(
                "native cache cannot serialize an interrupted exact result");
        }
        payload += ",\"feasible\":";
        payload += feasible ? "true" : "false";
        payload += ",\"labels_expanded\":";
        append_exact_json_integer_v2(
            payload, exact.label_counters[route * 3 + 1]);
        payload += ",\"labels_generated\":";
        append_exact_json_integer_v2(
            payload, exact.label_counters[route * 3]);
        payload += ",\"labels_pruned\":";
        append_exact_json_integer_v2(
            payload, exact.label_counters[route * 3 + 2]);
        payload += ",\"route\":[";
        for (auto path = exact.path_offsets[route];
             path < exact.path_offsets[route + 1]; ++path) {
            if (path != exact.path_offsets[route]) {
                payload.push_back(',');
            }
            append_exact_json_string_v2(
                payload,
                node_name_v2(problem, exact.path_indices[path]));
        }
        payload += "],\"total_energy\":";
        append_exact_json_float_v2(payload, exact.metrics[route * 4 + 1]);
        payload.push_back('}');
        output[route] = static_cast<std::int64_t>(128 + payload.size());
    }
    return output;
}

class CandidateTransactionStateV2 final {
public:
    struct Snapshot final {
        ExactRouteCacheV2::Snapshot exact_cache;
        NegativeRouteCacheV2::Snapshot negative_cache;
        SearchBudgetStateV2::Snapshot budget;
        std::unordered_set<std::string> attempted_plans;
        std::vector<std::int64_t> active_route_offsets;
        std::vector<std::int64_t> active_route_indices;
        native_kernels::ExactBatchOutput active_exact;
        std::array<std::int64_t, 2> active_objective_integer{};
        std::array<double, 2> active_objective_float{};
    };

    CandidateTransactionStateV2(
        const RequestV2& request,
        const InitialStateV2& initial)
        : exact_cache_(
              request.config.protocol_control[7],
              request.config.protocol_control[8]),
          negative_cache_(std::max<std::int64_t>(
              1, request.config.protocol_control[6])),
          budget_(
              request.config.search_control[4],
              request.config.protocol_control[2]),
          active_route_offsets_(request.problem.initial_route_offsets),
          active_route_indices_(request.problem.initial_route_indices),
          active_exact_(initial.exact),
          active_objective_integer_(initial.objective_integer),
          active_objective_float_(initial.objective_float) {
        initial.validate(request);
        const RouteBatchViewV2 warm_routes{
            active_route_offsets_, active_route_indices_};
        const auto budget_before = budget_.snapshot();
        try {
            const auto before = exact_cache_.lookup_exact_many(warm_routes);
            if (std::any_of(
                    before.hit_flags.begin(), before.hit_flags.end(),
                    [](const std::int64_t value) { return value != 0; })) {
                throw std::logic_error(
                    "native transaction warm cache was not empty");
            }
            const auto route_count = static_cast<std::int64_t>(
                warm_routes.route_count());
            const auto reservation = budget_.reserve_exact(route_count);
            if (reservation.granted != route_count) {
                throw std::runtime_error(
                    "native transaction warm start exceeds exact budget");
            }
            const auto hashes = exact_semantic_hashes_v2(
                warm_routes, initial.exact);
            const auto entry_bytes = exact_entry_bytes_v2(
                request.problem, initial.exact);
            static_cast<void>(exact_cache_.begin_store_exact_many_atomic(
                warm_routes, initial.exact, hashes, entry_bytes));
            exact_cache_.prepare_store_commit();
            exact_cache_.commit_store_batch_noexcept();
            budget_.settle_exact(route_count, 0);
            const auto after = exact_cache_.lookup_exact_many(warm_routes);
            if (std::any_of(
                    after.hit_flags.begin(), after.hit_flags.end(),
                    [](const std::int64_t value) { return value != 1; })) {
                throw std::logic_error(
                    "native transaction warm cache did not retain exact work");
            }
            const std::array<std::int64_t, 2> plan_offsets{0, route_count};
            const std::array<std::int64_t, 1> plan_ids{0};
            static_cast<void>(attempted_plans_.begin_mark_many_atomic(
                {plan_offsets, warm_routes}, plan_ids));
            attempted_plans_.prepare_mark_commit();
            attempted_plans_.commit_mark_batch_noexcept();
        } catch (...) {
            exact_cache_.reset_empty_noexcept();
            negative_cache_.reset_empty_noexcept();
            attempted_plans_.reset_empty_noexcept();
            budget_.restore(budget_before);
            throw;
        }
    }

    CandidateTransactionStateV2(const CandidateTransactionStateV2&) = delete;
    CandidateTransactionStateV2& operator=(
        const CandidateTransactionStateV2&) = delete;

    [[nodiscard]] ExactRouteCacheV2& exact_cache() noexcept {
        return exact_cache_;
    }
    [[nodiscard]] NegativeRouteCacheV2& negative_cache() noexcept {
        return negative_cache_;
    }
    [[nodiscard]] SearchBudgetStateV2& budget() noexcept { return budget_; }
    [[nodiscard]] AttemptedPlanSetV2& attempted_plans() noexcept {
        return attempted_plans_;
    }
    [[nodiscard]] const std::vector<std::int64_t>& active_route_offsets()
        const noexcept {
        return active_route_offsets_;
    }
    [[nodiscard]] const std::vector<std::int64_t>& active_route_indices()
        const noexcept {
        return active_route_indices_;
    }
    [[nodiscard]] const native_kernels::ExactBatchOutput& active_exact()
        const noexcept {
        return active_exact_;
    }
    [[nodiscard]] const std::array<std::int64_t, 2>&
    active_objective_integer() const noexcept {
        return active_objective_integer_;
    }
    [[nodiscard]] const std::array<double, 2>& active_objective_float()
        const noexcept {
        return active_objective_float_;
    }

    [[nodiscard]] Snapshot snapshot() const {
        return {
            exact_cache_.snapshot(),
            negative_cache_.snapshot(),
            budget_.snapshot(),
            attempted_plans_.snapshot(),
            active_route_offsets_,
            active_route_indices_,
            active_exact_,
            active_objective_integer_,
            active_objective_float_,
        };
    }

    void restore(Snapshot snapshot) noexcept {
        exact_cache_.restore(std::move(snapshot.exact_cache));
        negative_cache_.restore(std::move(snapshot.negative_cache));
        budget_.restore(snapshot.budget);
        attempted_plans_.restore(std::move(snapshot.attempted_plans));
        active_route_offsets_ = std::move(snapshot.active_route_offsets);
        active_route_indices_ = std::move(snapshot.active_route_indices);
        active_exact_ = std::move(snapshot.active_exact);
        active_objective_integer_ = snapshot.active_objective_integer;
        active_objective_float_ = snapshot.active_objective_float;
    }

    void restore_preserving_exact_budget(Snapshot snapshot) noexcept {
        const auto current_budget = budget_.snapshot();
        exact_cache_.restore(std::move(snapshot.exact_cache));
        negative_cache_.restore(std::move(snapshot.negative_cache));
        attempted_plans_.restore(std::move(snapshot.attempted_plans));
        budget_.restore(current_budget);
        budget_.rollback_preserving_exact(snapshot.budget, true);
        active_route_offsets_ = std::move(snapshot.active_route_offsets);
        active_route_indices_ = std::move(snapshot.active_route_indices);
        active_exact_ = std::move(snapshot.active_exact);
        active_objective_integer_ = snapshot.active_objective_integer;
        active_objective_float_ = snapshot.active_objective_float;
    }

    void commit_active_plan(
        std::vector<std::int64_t> route_offsets,
        std::vector<std::int64_t> route_indices,
        native_kernels::ExactBatchOutput exact,
        const std::array<std::int64_t, 2> objective_integer,
        const std::array<double, 2> objective_float) {
        const RouteBatchViewV2 routes{route_offsets, route_indices};
        routes.validate("native active candidate plan");
        if (exact.path_offsets.size() != routes.route_count() + 1
            || exact.statuses.size() != routes.route_count()
            || std::any_of(
                exact.statuses.begin(), exact.statuses.end(),
                [](const std::int64_t status) {
                    return status != native_kernels::feasible_status;
                })
            || objective_integer[0]
                != static_cast<std::int64_t>(routes.route_count())) {
            throw std::invalid_argument(
                "native active candidate plan is not exact-feasible");
        }
        active_route_offsets_ = std::move(route_offsets);
        active_route_indices_ = std::move(route_indices);
        active_exact_ = std::move(exact);
        active_objective_integer_ = objective_integer;
        active_objective_float_ = objective_float;
    }

private:
    ExactRouteCacheV2 exact_cache_;
    NegativeRouteCacheV2 negative_cache_;
    SearchBudgetStateV2 budget_;
    AttemptedPlanSetV2 attempted_plans_;
    std::vector<std::int64_t> active_route_offsets_;
    std::vector<std::int64_t> active_route_indices_;
    native_kernels::ExactBatchOutput active_exact_;
    std::array<std::int64_t, 2> active_objective_integer_{};
    std::array<double, 2> active_objective_float_{};
};

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
    void validate_live_assuming_problem_valid(
        const ProblemV2& problem,
        const bool require_complete_customer_coverage = true) const {
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
        if (depot < 0
            || (require_complete_customer_coverage
                && route_customers != expected_customers)) {
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

struct ExactRouteStateViewV2 final {
    std::span<const std::int64_t> path_offsets;
    std::span<const std::int64_t> path_indices;
    std::span<const std::int64_t> statuses;
    std::span<const std::int64_t> reasons;
    std::span<const double> metrics;
    std::span<const std::int64_t> label_counters;

    void validate(const std::size_t route_count) const {
        if (path_offsets.size() != route_count + 1
            || path_offsets.empty() || path_offsets.front() != 0
            || path_offsets.back()
                != static_cast<std::int64_t>(path_indices.size())
            || statuses.size() != route_count || reasons.size() != route_count
            || route_count > std::numeric_limits<std::size_t>::max() / 4
            || metrics.size() != route_count * 4
            || route_count > std::numeric_limits<std::size_t>::max() / 3
            || label_counters.size() != route_count * 3) {
            throw std::invalid_argument(
                "native exact route-state view has an invalid shape");
        }
        for (std::size_t route = 0; route < route_count; ++route) {
            if (path_offsets[route] < 0
                || path_offsets[route] > path_offsets[route + 1]) {
                throw std::invalid_argument(
                    "native exact route-state offsets are not monotone");
            }
        }
    }
};

[[nodiscard]] inline LaneStateV2 slice_lane_state_v2(
    const RouteBatchViewV2 all_routes,
    const ExactRouteStateViewV2 all_exact,
    const std::size_t first_route,
    const std::size_t end_route,
    const std::array<std::int64_t, 2>& objective_integer,
    const std::array<double, 2>& objective_float) {
    all_routes.validate("native candidate lane routes");
    const auto total_routes = all_routes.route_count();
    all_exact.validate(total_routes);
    if (first_route >= end_route || end_route > total_routes) {
        throw std::invalid_argument(
            "native candidate lane route slice is invalid");
    }
    const auto first_index = all_routes.offsets[first_route];
    const auto end_index = all_routes.offsets[end_route];
    const auto first_path = all_exact.path_offsets[first_route];
    const auto end_path = all_exact.path_offsets[end_route];
    const auto selected_routes = end_route - first_route;

    LaneStateV2 state;
    state.route_offsets.reserve(selected_routes + 1);
    for (std::size_t route = first_route; route <= end_route; ++route) {
        state.route_offsets.push_back(all_routes.offsets[route] - first_index);
    }
    state.route_indices.assign(
        all_routes.indices.begin() + first_index,
        all_routes.indices.begin() + end_index);
    state.exact.path_offsets.reserve(selected_routes + 1);
    for (std::size_t route = first_route; route <= end_route; ++route) {
        state.exact.path_offsets.push_back(
            all_exact.path_offsets[route] - first_path);
    }
    state.exact.path_indices.assign(
        all_exact.path_indices.begin() + first_path,
        all_exact.path_indices.begin() + end_path);
    state.exact.statuses.assign(
        all_exact.statuses.begin()
            + static_cast<std::ptrdiff_t>(first_route),
        all_exact.statuses.begin() + static_cast<std::ptrdiff_t>(end_route));
    state.exact.reasons.assign(
        all_exact.reasons.begin() + static_cast<std::ptrdiff_t>(first_route),
        all_exact.reasons.begin() + static_cast<std::ptrdiff_t>(end_route));
    state.exact.metrics.assign(
        all_exact.metrics.begin()
            + static_cast<std::ptrdiff_t>(first_route * 4),
        all_exact.metrics.begin() + static_cast<std::ptrdiff_t>(end_route * 4));
    state.exact.label_counters.assign(
        all_exact.label_counters.begin()
            + static_cast<std::ptrdiff_t>(first_route * 3),
        all_exact.label_counters.begin()
            + static_cast<std::ptrdiff_t>(end_route * 3));
    state.objective_integer = objective_integer;
    state.objective_float = objective_float;
    return state;
}

using LaneObjectiveKeyV2 = evrptw::formal_objective::Key;

[[nodiscard]] inline LaneObjectiveKeyV2 lane_objective_key_v2(
    const LaneStateV2& state) {
    return evrptw::formal_objective::key(
        state.objective_integer[0], state.objective_float[0],
        state.objective_float[1], state.objective_integer[1]);
}

[[nodiscard]] inline std::int64_t compare_lane_objective_v2(
    const LaneStateV2& left,
    const LaneStateV2& right) {
    const auto left_key = lane_objective_key_v2(left);
    const auto right_key = lane_objective_key_v2(right);
    return left_key < right_key ? -1 : left_key == right_key ? 0 : 1;
}

[[nodiscard]] inline AcceptanceOutcomeV2 decide_lane_acceptance_v2(
    const LaneStateV2& current,
    const LaneStateV2& candidate,
    const LaneStateV2& global_best,
    const double temperature,
    const double random_draw) {
    if (!std::isfinite(temperature) || temperature <= 0.0
        || !std::isfinite(random_draw) || random_draw < 0.0
        || random_draw > 1.0) {
        throw std::invalid_argument(
            "native lane acceptance inputs are invalid");
    }
    const auto current_key = lane_objective_key_v2(current);
    const auto candidate_key = lane_objective_key_v2(candidate);
    const auto best_key = lane_objective_key_v2(global_best);
    const auto current_vehicles = std::get<0>(current_key);
    const auto candidate_vehicles = std::get<0>(candidate_key);
    auto accepted = false;
    if (candidate_vehicles < current_vehicles) {
        accepted = true;
    } else if (candidate_vehicles == current_vehicles
        && candidate_key <= current_key) {
        accepted = true;
    } else if (candidate_vehicles == current_vehicles
        && std::get<1>(candidate_key) != std::get<1>(current_key)) {
        const auto distance_delta = candidate.objective_float[0]
            - current.objective_float[0];
        accepted = random_draw < std::exp(-distance_delta / temperature);
    }
    return {
        accepted ? 1 : 0,
        accepted && candidate_key < best_key ? 1 : 0,
        accepted && candidate_vehicles < current_vehicles ? 1 : 0,
    };
}

struct ThreeLaneSemanticStateV2 final {
    std::array<LaneStateV2, 4> lanes;
    ThreeLaneTerminationStateV2 termination;
    FullStage04StateV2 stage04;
    ThreeLaneRoundOutcomeStateV2 round;
    std::optional<Stage04InitializationStateV2> stage04_initialization;
    std::optional<Stage04BoundaryStateV2> stage04_boundary;

    void validate(const ProblemV2& problem) const {
        problem.validate();
        validate_assuming_problem_valid(problem);
    }

    void validate_assuming_problem_valid(const ProblemV2& problem) const {
        for (const auto& lane : lanes) {
            lane.validate_live_assuming_problem_valid(problem);
        }
        stage04.validate();
        round.validate();
        if (stage04_initialization.has_value()) {
            stage04_initialization->validate();
        }
        if (stage04_boundary.has_value()) {
            stage04_boundary->validate();
        }
        if (termination.reason < 0 || termination.reason > 3
            || termination.exact_budget < -1 || termination.started < 0
            || termination.completed < 0 || termination.interrupted < 0
            || termination.completed + termination.interrupted
                > termination.started
            || termination.completed_iterations < 0) {
            throw std::logic_error(
                "native three-lane semantic state is inconsistent");
        }
    }
};

class ThreeLaneSemanticTranscriptV2 final {
public:
    void append(
        ThreeLaneSemanticStateV2 state,
        const ProblemV2& problem) {
        state.validate_assuming_problem_valid(problem);
        const auto expected_iteration = static_cast<std::int64_t>(
            states_.size());
        if (state.round.iteration != expected_iteration) {
            throw std::logic_error(
                "native three-lane semantic transcript is not contiguous");
        }
        states_.push_back(std::move(state));
    }

    void finalize(
        const ThreeLaneTerminationStateV2& termination,
        const ProblemV2& problem) {
        if (states_.empty()) {
            throw std::logic_error(
                "native three-lane semantic transcript is empty");
        }
        states_.back().termination = termination;
        states_.back().validate_assuming_problem_valid(problem);
    }

    [[nodiscard]] std::span<const ThreeLaneSemanticStateV2> completed_prefix(
        const std::int64_t completed_iterations,
        const ProblemV2& problem) const {
        if (completed_iterations < 0
            || static_cast<std::size_t>(completed_iterations) > states_.size()) {
            throw std::logic_error(
                "native three-lane completed prefix is out of range");
        }
        for (std::int64_t iteration = 0;
             iteration < completed_iterations; ++iteration) {
            const auto& state = states_[static_cast<std::size_t>(iteration)];
            state.validate_assuming_problem_valid(problem);
            if (state.round.iteration != iteration
                || state.termination.completed_iterations < iteration + 1) {
                throw std::logic_error(
                    "native three-lane completed prefix lost an iteration");
            }
        }
        return {states_.data(), static_cast<std::size_t>(completed_iterations)};
    }

    [[nodiscard]] bool empty() const noexcept {
        return states_.empty();
    }

private:
    std::vector<ThreeLaneSemanticStateV2> states_;
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
            != native_protocol::KernelOperation::search_initial_state
        && operation
            != native_protocol::KernelOperation::candidate_session_open) {
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
            != native_protocol::KernelOperation::search_initial_state
        && payload.header().operation
            != native_protocol::KernelOperation::candidate_session_open)
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
