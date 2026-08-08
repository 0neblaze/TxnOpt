#pragma once

#include <chrono>
#include <optional>
#include <stdexcept>
#include <utility>

#include "native_search_core.hpp"

namespace evrptw::native_search {

enum class CandidateTransactionDeadlinePhaseV2 : std::uint8_t {
    before_transaction,
    budget_skip,
    before_exact,
    exact_work,
    candidate_commit,
    transaction_return,
    remote_worker,
};

class CandidateTransactionDeadlineV2 final : public std::runtime_error {
public:
    CandidateTransactionDeadlineV2(
        const CandidateTransactionDeadlinePhaseV2 phase,
        std::string message)
        : std::runtime_error(std::move(message)), phase_(phase) {}

    [[nodiscard]] CandidateTransactionDeadlinePhaseV2 phase() const noexcept {
        return phase_;
    }

private:
    CandidateTransactionDeadlinePhaseV2 phase_;
};

class CandidateTransactionKernelsV2 {
public:
    virtual ~CandidateTransactionKernelsV2() = default;

    [[nodiscard]] virtual std::vector<native_kernels::ScreenOutput> screen(
        const ProblemV2& problem,
        RouteBatchViewV2 routes,
        double epsilon) = 0;

    [[nodiscard]] virtual native_kernels::ExactBatchOutput exact(
        const ProblemV2& problem,
        RouteBatchViewV2 routes,
        double deadline_remaining,
        std::int64_t batch_size) = 0;
};

class LocalCandidateTransactionKernelsV2 final
    : public CandidateTransactionKernelsV2 {
public:
    [[nodiscard]] std::vector<native_kernels::ScreenOutput> screen(
        const ProblemV2& problem,
        const RouteBatchViewV2 routes,
        const double epsilon) override {
        problem.validate();
        routes.validate("local candidate transaction screening");
        std::int64_t depot = -1;
        std::vector<std::int64_t> recharge_nodes;
        for (std::size_t node = 0; node < problem.node_count(); ++node) {
            if (problem.node_kind[node] == native_kernels::depot_kind) {
                depot = static_cast<std::int64_t>(node);
                recharge_nodes.push_back(depot);
            } else if (
                problem.node_kind[node] == native_kernels::station_kind) {
                recharge_nodes.push_back(static_cast<std::int64_t>(node));
            }
        }
        if (depot < 0 || !std::isfinite(epsilon) || epsilon <= 0.0) {
            throw std::invalid_argument(
                "local candidate transaction screening control is invalid");
        }
        const std::array<double, 4> options{1.0, epsilon, 0.0, 0.0};
        const std::array<double, 6> incremental{};
        std::vector<native_kernels::ScreenOutput> output(
            routes.route_count());
        for (std::size_t route = 0; route < routes.route_count(); ++route) {
            const auto sequence = routes.route(route);
            output[route] = native_kernels::run_screen_route(
                problem.node_kind.data(), problem.demand.data(),
                problem.ready_time.data(), problem.due_date.data(),
                problem.service_time.data(), problem.distance.data(),
                problem.reachable.data(), problem.vehicle.data(),
                sequence.data(), sequence.size(), problem.node_count(), depot,
                recharge_nodes, options.data(), incremental.data());
        }
        return output;
    }

    [[nodiscard]] native_kernels::ExactBatchOutput exact(
        const ProblemV2& problem,
        const RouteBatchViewV2 routes,
        const double deadline_remaining,
        const std::int64_t batch_size) override {
        return run_local_exact_batch_v2(
            {
                problem.node_kind,
                problem.ready_time,
                problem.due_date,
                problem.service_time,
                problem.distance,
                problem.vehicle,
            },
            routes, deadline_remaining, batch_size);
    }
};

struct CandidateTransactionOptionsV2 final {
    bool suppress_attempted_plan_journal = false;
    bool suppress_round_budget = false;
    bool suppress_screening_negative_cache = false;
    bool allow_partial_customer_coverage = false;
    bool defer_commit = false;
    bool allow_vehicle_increase = false;
};

struct CandidatePlanExecutionTraceV2 final {
    std::int64_t plan_id = -1;
    std::vector<std::int64_t> cache_hit_flags;
    std::vector<std::int64_t> missing_local_rows;
    std::vector<std::int64_t> exact_batch_sizes;
    std::array<std::int64_t, 3> budget_reservation{};
    native_kernels::ExactBatchOutput exact;
    std::vector<std::int64_t> cache_store_statuses;
    std::vector<std::int64_t> cache_eviction_counts;
};

inline void validate_candidate_plan_execution_trace_v2(
    const CandidatePlanExecutionTraceV2& plan) {
    if (std::any_of(
            plan.cache_hit_flags.begin(), plan.cache_hit_flags.end(),
            [](const auto value) { return value != 0 && value != 1; })
        || std::any_of(
            plan.exact_batch_sizes.begin(), plan.exact_batch_sizes.end(),
            [](const auto value) { return value <= 0; })) {
        throw std::logic_error("native candidate plan trace flags are invalid");
    }
    std::int64_t exact_rows = 0;
    for (const auto batch_size : plan.exact_batch_sizes) {
        if (exact_rows > std::numeric_limits<std::int64_t>::max() - batch_size) {
            throw std::overflow_error(
                "native candidate plan trace exact row count overflows");
        }
        exact_rows += batch_size;
    }
    const auto exact_row_count = static_cast<std::size_t>(exact_rows);
    if (plan.budget_reservation[0] != exact_rows
        || plan.budget_reservation[1] != exact_rows
        || plan.budget_reservation[2] != exact_rows
        || plan.missing_local_rows.size() != exact_row_count
        || plan.exact.statuses.size() != exact_row_count
        || plan.exact.reasons.size() != exact_row_count
        || plan.exact.metrics.size() != exact_row_count * 4
        || plan.exact.label_counters.size() != exact_row_count * 3
        || plan.exact.batch_counters.size()
            != plan.exact_batch_sizes.size() * 10
        || plan.exact.completion_order.size() != exact_row_count
        || plan.cache_store_statuses.size() != exact_row_count
        || plan.cache_eviction_counts.size() != exact_row_count) {
        throw std::logic_error("native candidate plan trace shapes are invalid");
    }
    if (exact_row_count == 0) {
        if (!plan.exact.path_offsets.empty()
            || !plan.exact.path_indices.empty()) {
            throw std::logic_error(
                "native candidate empty exact trace has path data");
        }
        return;
    }
    if (plan.exact.path_offsets.size() != exact_row_count + 1
        || plan.exact.path_offsets.front() != 0
        || !std::is_sorted(
            plan.exact.path_offsets.begin(), plan.exact.path_offsets.end())
        || plan.exact.path_offsets.back() < 0
        || static_cast<std::size_t>(plan.exact.path_offsets.back())
            != plan.exact.path_indices.size()) {
        throw std::logic_error(
            "native candidate plan trace path offsets are invalid");
    }
    std::vector<std::int64_t> completion = plan.exact.completion_order;
    std::sort(completion.begin(), completion.end());
    for (std::size_t row = 0; row < completion.size(); ++row) {
        if (completion[row] != static_cast<std::int64_t>(row)) {
            throw std::logic_error(
                "native candidate plan trace completion order is invalid");
        }
    }
    for (std::size_t batch = 0;
         batch < plan.exact_batch_sizes.size(); ++batch) {
        const auto offset = batch * 10;
        if (plan.exact.batch_counters[offset + 2]
                != plan.exact_batch_sizes[batch]
            || plan.exact.batch_counters[offset + 3] != 0) {
            throw std::logic_error(
                "native candidate plan trace batch counters are invalid");
        }
    }
}

inline native_kernels::ExactBatchOutput slice_candidate_exact_batch_v2(
    const CandidatePlanExecutionTraceV2& plan,
    const std::size_t batch_index) {
    validate_candidate_plan_execution_trace_v2(plan);
    if (batch_index >= plan.exact_batch_sizes.size()) {
        throw std::out_of_range("native candidate exact batch index is invalid");
    }
    const auto first_row = std::accumulate(
        plan.exact_batch_sizes.begin(),
        plan.exact_batch_sizes.begin() + static_cast<std::ptrdiff_t>(batch_index),
        std::int64_t{0});
    const auto row_count = plan.exact_batch_sizes[batch_index];
    const auto last_row = first_row + row_count;
    if (first_row < 0 || row_count <= 0
        || static_cast<std::size_t>(last_row) > plan.exact.statuses.size()
        || plan.exact.path_offsets.size() != plan.exact.statuses.size() + 1
        || plan.exact.reasons.size() != plan.exact.statuses.size()
        || plan.exact.metrics.size() != plan.exact.statuses.size() * 4
        || plan.exact.label_counters.size() != plan.exact.statuses.size() * 3
        || plan.exact.batch_counters.size()
            != plan.exact_batch_sizes.size() * 10) {
        throw std::logic_error("native candidate exact batch trace is invalid");
    }
    native_kernels::ExactBatchOutput output;
    output.path_offsets.push_back(0);
    const auto path_first = plan.exact.path_offsets[static_cast<std::size_t>(first_row)];
    const auto path_last = plan.exact.path_offsets[static_cast<std::size_t>(last_row)];
    output.path_indices.assign(
        plan.exact.path_indices.begin() + path_first,
        plan.exact.path_indices.begin() + path_last);
    for (auto row = first_row; row < last_row; ++row) {
        output.path_offsets.push_back(
            plan.exact.path_offsets[static_cast<std::size_t>(row + 1)]
            - path_first);
    }
    output.statuses.assign(
        plan.exact.statuses.begin() + first_row,
        plan.exact.statuses.begin() + last_row);
    output.reasons.assign(
        plan.exact.reasons.begin() + first_row,
        plan.exact.reasons.begin() + last_row);
    output.metrics.assign(
        plan.exact.metrics.begin() + first_row * 4,
        plan.exact.metrics.begin() + last_row * 4);
    output.label_counters.assign(
        plan.exact.label_counters.begin() + first_row * 3,
        plan.exact.label_counters.begin() + last_row * 3);
    output.batch_counters.assign(
        plan.exact.batch_counters.begin()
            + static_cast<std::ptrdiff_t>(batch_index * 10),
        plan.exact.batch_counters.begin()
            + static_cast<std::ptrdiff_t>((batch_index + 1) * 10));
    for (const auto ordinal : plan.exact.completion_order) {
        if (ordinal >= first_row && ordinal < last_row) {
            output.completion_order.push_back(ordinal - first_row);
        }
    }
    return output;
}

struct CandidateTransactionTraceV2 final {
    std::vector<std::int64_t> canonical_expected;
    std::vector<std::int64_t> coverage_eligible;
    std::vector<std::int64_t> unique_route_offsets;
    std::vector<std::int64_t> unique_route_indices;
    std::vector<std::int64_t> unique_row_by_route;
    std::vector<std::int64_t> negative_hit_flags;
    std::vector<std::int64_t> negative_hit_reasons;
    std::vector<std::int64_t> physical_screen_rows;
    std::vector<native_kernels::ScreenOutput> screen_outputs;
    std::vector<std::int64_t> screening_passed;
    std::vector<std::int64_t> attempted_flags;
    std::vector<std::int64_t> eligible_flags;
    std::vector<std::int64_t> combined_attempted_flags;
    std::vector<std::int64_t> ranked;
    std::vector<std::int64_t> selected;
    std::vector<std::int64_t> ranking_integer;
    std::vector<double> ranking_float;
    std::vector<CandidatePlanExecutionTraceV2> plans;
    std::vector<std::int64_t> completed_plan_ids;
    bool exact_protocol_active = false;
    bool negative_store_active = false;
    bool attempted_mark_active = false;
    double screening_seconds = 0.0;
};

struct CandidateTransactionExecutionV2 final {
    CandidatePlanTransactionResultV2 result;
    CandidateTransactionTraceV2 trace;
};

struct CandidateTransactionTraceWireV2 final {
    std::vector<std::int64_t> integer_offsets{0};
    std::vector<std::int64_t> integer_values;
    std::vector<std::int64_t> double_offsets{0};
    std::vector<double> double_values;

    void validate() const {
        const auto validate_offsets = [](const auto& offsets,
                                         const std::size_t value_count,
                                         const std::string_view name) {
            if (offsets.empty() || offsets.front() != 0
                || offsets.back() != static_cast<std::int64_t>(value_count)
                || !std::is_sorted(offsets.begin(), offsets.end())) {
                throw std::logic_error(
                    "native candidate trace wire " + std::string(name)
                    + " offsets are invalid");
            }
        };
        validate_offsets(integer_offsets, integer_values.size(), "integer");
        validate_offsets(double_offsets, double_values.size(), "double");
    }
};

inline CandidateTransactionTraceWireV2 encode_candidate_transaction_trace_v2(
    const CandidateTransactionTraceV2& trace) {
    CandidateTransactionTraceWireV2 wire;
    const auto add_integer = [&wire](const auto& values) {
        wire.integer_values.insert(
            wire.integer_values.end(), values.begin(), values.end());
        wire.integer_offsets.push_back(
            static_cast<std::int64_t>(wire.integer_values.size()));
    };
    const auto add_double = [&wire](const auto& values) {
        wire.double_values.insert(
            wire.double_values.end(), values.begin(), values.end());
        wire.double_offsets.push_back(
            static_cast<std::int64_t>(wire.double_values.size()));
    };
    add_integer(trace.canonical_expected);
    add_integer(trace.coverage_eligible);
    add_integer(trace.unique_route_offsets);
    add_integer(trace.unique_route_indices);
    add_integer(trace.unique_row_by_route);
    add_integer(trace.negative_hit_flags);
    add_integer(trace.negative_hit_reasons);
    add_integer(trace.physical_screen_rows);
    std::vector<std::int64_t> screen_codes;
    std::vector<double> screen_metrics;
    for (const auto& screen : trace.screen_outputs) {
        screen_codes.insert(
            screen_codes.end(), screen.codes.begin(), screen.codes.end());
        screen_codes.push_back(screen.reachability_queries);
        screen_metrics.insert(
            screen_metrics.end(), screen.metrics.begin(), screen.metrics.end());
    }
    add_integer(screen_codes);
    add_integer(trace.screening_passed);
    add_integer(trace.attempted_flags);
    add_integer(trace.eligible_flags);
    add_integer(trace.combined_attempted_flags);
    add_integer(trace.ranked);
    add_integer(trace.selected);
    add_integer(trace.ranking_integer);
    add_integer(trace.completed_plan_ids);
    const std::array<std::int64_t, 3> active_flags{
        trace.exact_protocol_active ? 1 : 0,
        trace.negative_store_active ? 1 : 0,
        trace.attempted_mark_active ? 1 : 0,
    };
    add_integer(active_flags);
    std::vector<std::int64_t> plan_ids;
    plan_ids.reserve(trace.plans.size());
    for (const auto& plan : trace.plans) {
        plan_ids.push_back(plan.plan_id);
    }
    add_integer(plan_ids);
    add_double(trace.ranking_float);
    add_double(screen_metrics);
    const std::array<double, 1> screening_seconds{trace.screening_seconds};
    add_double(screening_seconds);
    for (const auto& plan : trace.plans) {
        validate_candidate_plan_execution_trace_v2(plan);
        add_integer(plan.cache_hit_flags);
        add_integer(plan.missing_local_rows);
        add_integer(plan.exact_batch_sizes);
        add_integer(plan.budget_reservation);
        add_integer(plan.exact.path_offsets);
        add_integer(plan.exact.path_indices);
        add_integer(plan.exact.statuses);
        add_integer(plan.exact.reasons);
        add_integer(plan.exact.label_counters);
        add_integer(plan.exact.batch_counters);
        add_integer(plan.exact.completion_order);
        add_integer(plan.cache_store_statuses);
        add_integer(plan.cache_eviction_counts);
        add_double(plan.exact.metrics);
    }
    wire.validate();
    return wire;
}

inline CandidateTransactionTraceV2 decode_candidate_transaction_trace_v2(
    const CandidateTransactionTraceWireV2& wire) {
    wire.validate();
    if (wire.integer_offsets.size() < 20 || wire.double_offsets.size() < 4) {
        throw std::runtime_error(
            "native candidate trace wire is missing fixed fields");
    }
    const auto integer_field = [&wire](const std::size_t field) {
        return std::span<const std::int64_t>(
            wire.integer_values.data() + wire.integer_offsets.at(field),
            static_cast<std::size_t>(
                wire.integer_offsets.at(field + 1)
                - wire.integer_offsets.at(field)));
    };
    const auto double_field = [&wire](const std::size_t field) {
        return std::span<const double>(
            wire.double_values.data() + wire.double_offsets.at(field),
            static_cast<std::size_t>(
                wire.double_offsets.at(field + 1)
                - wire.double_offsets.at(field)));
    };
    const auto copy_integer = [&integer_field](const std::size_t field) {
        const auto values = integer_field(field);
        return std::vector<std::int64_t>(values.begin(), values.end());
    };
    const auto copy_double = [&double_field](const std::size_t field) {
        const auto values = double_field(field);
        return std::vector<double>(values.begin(), values.end());
    };
    const auto plan_ids = integer_field(18);
    if (wire.integer_offsets.size() != 20 + plan_ids.size() * 13
        || wire.double_offsets.size() != 4 + plan_ids.size()) {
        throw std::runtime_error(
            "native candidate trace wire plan field count is invalid");
    }
    CandidateTransactionTraceV2 trace;
    trace.canonical_expected = copy_integer(0);
    trace.coverage_eligible = copy_integer(1);
    trace.unique_route_offsets = copy_integer(2);
    trace.unique_route_indices = copy_integer(3);
    trace.unique_row_by_route = copy_integer(4);
    trace.negative_hit_flags = copy_integer(5);
    trace.negative_hit_reasons = copy_integer(6);
    trace.physical_screen_rows = copy_integer(7);
    const auto screen_codes = integer_field(8);
    const auto screen_metrics = double_field(1);
    if (screen_codes.size() != trace.negative_hit_flags.size() * 17
        || screen_metrics.size() != trace.negative_hit_flags.size() * 15) {
        throw std::runtime_error(
            "native candidate trace wire screening shape is invalid");
    }
    trace.screen_outputs.resize(trace.negative_hit_flags.size());
    for (std::size_t row = 0; row < trace.screen_outputs.size(); ++row) {
        std::copy_n(
            screen_codes.begin() + static_cast<std::ptrdiff_t>(row * 17), 16,
            trace.screen_outputs[row].codes.begin());
        trace.screen_outputs[row].reachability_queries =
            screen_codes[row * 17 + 16];
        std::copy_n(
            screen_metrics.begin() + static_cast<std::ptrdiff_t>(row * 15), 15,
            trace.screen_outputs[row].metrics.begin());
    }
    trace.screening_passed = copy_integer(9);
    trace.attempted_flags = copy_integer(10);
    trace.eligible_flags = copy_integer(11);
    trace.combined_attempted_flags = copy_integer(12);
    trace.ranked = copy_integer(13);
    trace.selected = copy_integer(14);
    trace.ranking_integer = copy_integer(15);
    trace.completed_plan_ids = copy_integer(16);
    const auto flags = integer_field(17);
    const auto screening_seconds = double_field(2);
    if (flags.size() != 3 || screening_seconds.size() != 1
        || std::any_of(flags.begin(), flags.end(), [](const auto value) {
            return value != 0 && value != 1;
        })) {
        throw std::runtime_error(
            "native candidate trace wire control fields are invalid");
    }
    trace.exact_protocol_active = flags[0] != 0;
    trace.negative_store_active = flags[1] != 0;
    trace.attempted_mark_active = flags[2] != 0;
    trace.screening_seconds = screening_seconds[0];
    trace.ranking_float = copy_double(0);
    trace.plans.reserve(plan_ids.size());
    for (std::size_t plan = 0; plan < plan_ids.size(); ++plan) {
        const auto base = 19 + plan * 13;
        CandidatePlanExecutionTraceV2 decoded;
        decoded.plan_id = plan_ids[plan];
        decoded.cache_hit_flags = copy_integer(base);
        decoded.missing_local_rows = copy_integer(base + 1);
        decoded.exact_batch_sizes = copy_integer(base + 2);
        const auto reservation = integer_field(base + 3);
        if (reservation.size() != decoded.budget_reservation.size()) {
            throw std::runtime_error(
                "native candidate trace wire budget reservation is invalid");
        }
        std::copy(
            reservation.begin(), reservation.end(),
            decoded.budget_reservation.begin());
        decoded.exact.path_offsets = copy_integer(base + 4);
        decoded.exact.path_indices = copy_integer(base + 5);
        decoded.exact.statuses = copy_integer(base + 6);
        decoded.exact.reasons = copy_integer(base + 7);
        decoded.exact.label_counters = copy_integer(base + 8);
        decoded.exact.batch_counters = copy_integer(base + 9);
        decoded.exact.completion_order = copy_integer(base + 10);
        decoded.cache_store_statuses = copy_integer(base + 11);
        decoded.cache_eviction_counts = copy_integer(base + 12);
        decoded.exact.metrics = copy_double(3 + plan);
        validate_candidate_plan_execution_trace_v2(decoded);
        trace.plans.push_back(std::move(decoded));
    }
    return trace;
}

#ifdef __linux__
inline std::vector<std::uint8_t> candidate_transaction_execution_payload_v2(
    const CandidateTransactionExecutionV2& execution,
    const std::uint64_t request_id,
    const std::span<const double> telemetry) {
    if (telemetry.size() != 7) {
        throw std::invalid_argument(
            "native candidate execution telemetry is invalid");
    }
    const auto result_wire = encode_candidate_plan_transaction_v2(
        execution.result);
    const auto trace_wire = encode_candidate_transaction_trace_v2(
        execution.trace);
    native_protocol::PayloadBuilder builder(
        native_protocol::KernelOperation::candidate_transaction_execute,
        request_id);
    builder.add(native_protocol::NumericType::int64,
        result_wire.integer_offsets.data(), result_wire.integer_offsets.size(),
        result_wire.integer_offsets.size());
    builder.add(native_protocol::NumericType::int64,
        result_wire.integer_values.data(), result_wire.integer_values.size(),
        result_wire.integer_values.size());
    builder.add(native_protocol::NumericType::int64,
        result_wire.double_offsets.data(), result_wire.double_offsets.size(),
        result_wire.double_offsets.size());
    builder.add(native_protocol::NumericType::float64,
        result_wire.double_values.data(), result_wire.double_values.size(),
        result_wire.double_values.size());
    builder.add(native_protocol::NumericType::int64,
        result_wire.byte_offsets.data(), result_wire.byte_offsets.size(),
        result_wire.byte_offsets.size());
    builder.add(native_protocol::NumericType::uint8,
        result_wire.byte_values.data(), result_wire.byte_values.size(),
        result_wire.byte_values.size());
    builder.add(native_protocol::NumericType::int64,
        trace_wire.integer_offsets.data(), trace_wire.integer_offsets.size(),
        trace_wire.integer_offsets.size());
    builder.add(native_protocol::NumericType::int64,
        trace_wire.integer_values.data(), trace_wire.integer_values.size(),
        trace_wire.integer_values.size());
    builder.add(native_protocol::NumericType::int64,
        trace_wire.double_offsets.data(), trace_wire.double_offsets.size(),
        trace_wire.double_offsets.size());
    builder.add(native_protocol::NumericType::float64,
        trace_wire.double_values.data(), trace_wire.double_values.size(),
        trace_wire.double_values.size());
    builder.add(native_protocol::NumericType::float64,
        telemetry.data(), telemetry.size(), telemetry.size());
    return builder.finish();
}

inline CandidateTransactionExecutionV2
candidate_transaction_execution_from_payload_v2(
    const native_protocol::PayloadView& payload) {
    if (payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_execute
        || payload.header().array_count != 11) {
        throw std::runtime_error(
            "native candidate execution response schema is invalid");
    }
    const auto result_wire = candidate_plan_transaction_wire_from_payload_v2(
        payload);
    CandidateTransactionTraceWireV2 trace_wire;
    const auto copy_integer = [&payload](const std::size_t index) {
        const auto* values = payload.data<std::int64_t>(
            index, native_protocol::NumericType::int64);
        return std::vector<std::int64_t>(
            values, values + payload.descriptor(index).count);
    };
    const auto copy_double = [&payload](const std::size_t index) {
        const auto* values = payload.data<double>(
            index, native_protocol::NumericType::float64);
        return std::vector<double>(
            values, values + payload.descriptor(index).count);
    };
    trace_wire.integer_offsets = copy_integer(6);
    trace_wire.integer_values = copy_integer(7);
    trace_wire.double_offsets = copy_integer(8);
    trace_wire.double_values = copy_double(9);
    CandidateTransactionExecutionV2 execution{
        decode_candidate_plan_transaction_v2(result_wire),
        decode_candidate_transaction_trace_v2(trace_wire),
    };
    validate_candidate_plan_transaction_v2(execution.result);
    return execution;
}
#endif

struct CandidateTransactionExecuteRequestV2 final {
    std::string token;
    CandidateRoundRequestV2 round;
    CandidateTransactionOptionsV2 options;
};

inline std::vector<std::uint8_t> candidate_transaction_execute_payload_v2(
    const std::string_view token,
    const CandidateRoundRequestV2& round,
    const CandidateTransactionOptionsV2 options,
    const std::uint64_t request_id) {
    round.validate();
    if (token.size() != 64) {
        throw std::invalid_argument(
            "native candidate transaction session token is invalid");
    }
    native_protocol::PayloadBuilder builder(
        native_protocol::KernelOperation::candidate_transaction_execute,
        request_id);
    builder.add(
        native_protocol::NumericType::uint8,
        token.data(), token.size(), token.size());
    builder.add(
        native_protocol::NumericType::int64,
        round.plan_offsets.data(), round.plan_offsets.size(),
        round.plan_offsets.size());
    builder.add(
        native_protocol::NumericType::int64,
        round.route_offsets.data(), round.route_offsets.size(),
        round.route_offsets.size());
    builder.add(
        native_protocol::NumericType::int64,
        round.route_indices.data(), round.route_indices.size(),
        round.route_indices.size());
    builder.add(
        native_protocol::NumericType::int64,
        round.context.data(), round.context.size(), round.context.size());
    const auto deadline_absolute =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count()
        + round.deadline_remaining;
    builder.add(
        native_protocol::NumericType::float64,
        &deadline_absolute, 1, 1);
    builder.add(
        native_protocol::NumericType::int64,
        &round.batch_size, 1, 1);
    builder.add(
        native_protocol::NumericType::int64,
        round.expected_customers.data(), round.expected_customers.size(),
        round.expected_customers.size());
    const std::array<std::int64_t, 6> flags{
        options.suppress_attempted_plan_journal ? 1 : 0,
        options.suppress_round_budget ? 1 : 0,
        options.suppress_screening_negative_cache ? 1 : 0,
        options.allow_partial_customer_coverage ? 1 : 0,
        options.defer_commit ? 1 : 0,
        options.allow_vehicle_increase ? 1 : 0,
    };
    builder.add(
        native_protocol::NumericType::int64,
        flags.data(), flags.size(), flags.size());
    builder.add(
        native_protocol::NumericType::int64,
        round.ranking_route_offsets.data(),
        round.ranking_route_offsets.size(),
        round.ranking_route_offsets.size());
    builder.add(
        native_protocol::NumericType::int64,
        round.ranking_route_indices.data(),
        round.ranking_route_indices.size(),
        round.ranking_route_indices.size());
    return builder.finish();
}

inline CandidateTransactionExecuteRequestV2
candidate_transaction_execute_from_payload_v2(
    const native_protocol::PayloadView& payload) {
    if (payload.header().operation
            != native_protocol::KernelOperation::candidate_transaction_execute
        || payload.header().array_count != 11) {
        throw std::runtime_error(
            "native candidate transaction execute payload schema is invalid");
    }
    const auto require_vector = [&payload](
        const std::size_t index,
        const native_protocol::NumericType type) {
        const auto& descriptor = payload.descriptor(index);
        if (descriptor.type != type || descriptor.dimensions != 1
            || descriptor.shape[0] != descriptor.count
            || descriptor.shape[1] != 0) {
            throw std::runtime_error(
                "native candidate transaction execute descriptor is invalid");
        }
    };
    for (const auto index : {
             std::size_t{0}, std::size_t{1}, std::size_t{2},
             std::size_t{3}, std::size_t{4}, std::size_t{6},
             std::size_t{7}, std::size_t{8}, std::size_t{9},
             std::size_t{10}}) {
        require_vector(index,
            index == 0 ? native_protocol::NumericType::uint8
                       : native_protocol::NumericType::int64);
    }
    require_vector(5, native_protocol::NumericType::float64);
    if (payload.descriptor(0).count != 64
        || payload.descriptor(4).count != 3
        || payload.descriptor(5).count != 1
        || payload.descriptor(6).count != 1
        || payload.descriptor(8).count != 6) {
        throw std::runtime_error(
            "native candidate transaction execute dimensions are invalid");
    }
    const auto copy_i64 = [&payload](const std::size_t index) {
        const auto count = static_cast<std::size_t>(
            payload.descriptor(index).count);
        const auto* values = payload.data<std::int64_t>(
            index, native_protocol::NumericType::int64);
        return std::vector<std::int64_t>(values, values + count);
    };
    CandidateTransactionExecuteRequestV2 request;
    const auto* token = payload.data<std::uint8_t>(
        0, native_protocol::NumericType::uint8);
    request.token.assign(
        reinterpret_cast<const char*>(token),
        static_cast<std::size_t>(payload.descriptor(0).count));
    request.round.plan_offsets = copy_i64(1);
    request.round.route_offsets = copy_i64(2);
    request.round.route_indices = copy_i64(3);
    const auto* context = payload.data<std::int64_t>(
        4, native_protocol::NumericType::int64);
    std::copy(context, context + 3, request.round.context.begin());
    const auto deadline_absolute = payload.data<double>(
        5, native_protocol::NumericType::float64)[0];
    request.round.deadline_remaining = deadline_absolute
        - std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    request.round.batch_size = payload.data<std::int64_t>(
        6, native_protocol::NumericType::int64)[0];
    request.round.expected_customers = copy_i64(7);
    const auto* flags = payload.data<std::int64_t>(
        8, native_protocol::NumericType::int64);
    if (payload.descriptor(8).count != 6
        || std::any_of(flags, flags + 6, [](const std::int64_t value) {
            return value != 0 && value != 1;
        })) {
        throw std::runtime_error(
            "native candidate transaction execute flags are invalid");
    }
    request.options = {
        flags[0] != 0,
        flags[1] != 0,
        flags[2] != 0,
        flags[3] != 0,
        flags[4] != 0,
        flags[5] != 0,
    };
    request.round.ranking_route_offsets = copy_i64(9);
    request.round.ranking_route_indices = copy_i64(10);
    request.round.validate();
    return request;
}

class CandidateTransactionExecutorV2 final {
public:
    CandidateTransactionExecutorV2(
        const RequestV2& request,
        CandidateTransactionStateV2& state,
        CandidateTransactionKernelsV2& kernels)
        : CandidateTransactionExecutorV2(
              request,
              state.exact_cache(),
              state.negative_cache(),
              state.budget(),
              state.attempted_plans(),
              state.active_route_offsets(),
              state.active_route_indices(),
              kernels,
              request.config.protocol_control[1],
              request.config.protocol_options[1]) {}

    CandidateTransactionExecutorV2(
        const RequestV2& request,
        ExactRouteCacheV2& exact_cache,
        NegativeRouteCacheV2& negative_cache,
        SearchBudgetStateV2& budget,
        AttemptedPlanSetV2& attempted_plans,
        const std::span<const std::int64_t> active_route_offsets,
        const std::span<const std::int64_t> active_route_indices,
        CandidateTransactionKernelsV2& kernels,
        const std::int64_t proposal_top_k = -1,
        const double screening_epsilon = -1.0,
        const std::span<const std::optional<LaneStateV2>> incumbent_lanes = {})
        : problem_(request.problem),
          exact_cache_(exact_cache),
          negative_cache_(negative_cache),
          budget_(budget),
          attempted_plans_(attempted_plans),
          active_route_offsets_(active_route_offsets),
          active_route_indices_(active_route_indices),
          incumbent_lanes_(incumbent_lanes),
          kernels_(kernels),
          proposal_top_k_(
              proposal_top_k < 0
                  ? request.config.protocol_control[1]
                  : proposal_top_k),
          screening_epsilon_(
              screening_epsilon < 0.0
                  ? request.config.protocol_options[1]
                  : screening_epsilon) {
        request.validate();
        RouteBatchViewV2{
            active_route_offsets_, active_route_indices_}.validate(
                "native candidate transaction active routes");
    }

    [[nodiscard]] CandidatePlanTransactionResultV2 execute(
        CandidateRoundRequestV2 round,
        const CandidateTransactionOptionsV2 options = {}) {
        return execute_with_trace(std::move(round), options).result;
    }

    [[nodiscard]] CandidateTransactionExecutionV2 execute_with_trace(
        CandidateRoundRequestV2 round,
        const CandidateTransactionOptionsV2 options = {}) {
        round.validate();
        const auto started = std::chrono::steady_clock::now();
        auto exact_before = exact_cache_.snapshot();
        auto negative_before = negative_cache_.snapshot();
        const auto budget_before = budget_.snapshot();
        auto attempted_before = attempted_plans_.snapshot();
        try {
            return execute_owned(std::move(round), options, started);
        } catch (...) {
            exact_cache_.restore(std::move(exact_before));
            negative_cache_.restore(std::move(negative_before));
            attempted_plans_.restore(std::move(attempted_before));
            budget_.rollback_preserving_exact(budget_before, true);
            throw;
        }
    }

    CandidateTransactionExecutorV2(
        const ProblemV2& problem,
        ExactRouteCacheV2& exact_cache,
        NegativeRouteCacheV2& negative_cache,
        SearchBudgetStateV2& budget,
        AttemptedPlanSetV2& attempted_plans,
        const std::span<const std::int64_t> active_route_offsets,
        const std::span<const std::int64_t> active_route_indices,
        CandidateTransactionKernelsV2& kernels,
        const std::int64_t proposal_top_k,
        const double screening_epsilon,
        const std::span<const std::optional<LaneStateV2>> incumbent_lanes = {})
        : problem_(problem),
          exact_cache_(exact_cache),
          negative_cache_(negative_cache),
          budget_(budget),
          attempted_plans_(attempted_plans),
          active_route_offsets_(active_route_offsets),
          active_route_indices_(active_route_indices),
          incumbent_lanes_(incumbent_lanes),
          kernels_(kernels),
          proposal_top_k_(proposal_top_k),
          screening_epsilon_(screening_epsilon) {
        problem_.validate();
        if (proposal_top_k_ <= 0 || !std::isfinite(screening_epsilon_)
            || screening_epsilon_ <= 0.0) {
            throw std::invalid_argument(
                "native candidate transaction configuration is invalid");
        }
        RouteBatchViewV2{
            active_route_offsets_, active_route_indices_}.validate(
                "native candidate transaction active routes");
    }

private:
    using ExactPayload = ExactRouteCacheV2::ExactPayload;

    const ProblemV2& problem_;
    ExactRouteCacheV2& exact_cache_;
    NegativeRouteCacheV2& negative_cache_;
    SearchBudgetStateV2& budget_;
    AttemptedPlanSetV2& attempted_plans_;
    std::span<const std::int64_t> active_route_offsets_;
    std::span<const std::int64_t> active_route_indices_;
    std::span<const std::optional<LaneStateV2>> incumbent_lanes_;
    CandidateTransactionKernelsV2& kernels_;
    std::int64_t proposal_top_k_ = 0;
    double screening_epsilon_ = 0.0;

    [[nodiscard]] std::optional<ExactPayload> incumbent_payload(
        const std::span<const std::int64_t> sequence) const {
        const RouteBatchViewV2 active_routes{
            active_route_offsets_, active_route_indices_};
        bool unchanged = false;
        for (std::size_t row = 0; row < active_routes.route_count(); ++row) {
            const auto active = active_routes.route(row);
            if (active.size() == sequence.size()
                && std::equal(
                    active.begin(), active.end(), sequence.begin())) {
                unchanged = true;
                break;
            }
        }
        if (!unchanged) {
            return std::nullopt;
        }
        for (const auto& optional_lane : incumbent_lanes_) {
            if (!optional_lane.has_value()) {
                continue;
            }
            const auto& lane = *optional_lane;
            const RouteBatchViewV2 routes{
                lane.route_offsets, lane.route_indices};
            for (std::size_t row = 0; row < routes.route_count(); ++row) {
                const auto incumbent = routes.route(row);
                if (incumbent.size() != sequence.size()
                    || !std::equal(
                        incumbent.begin(), incumbent.end(), sequence.begin())) {
                    continue;
                }
                ExactPayload payload;
                payload.path.assign(
                    lane.exact.path_indices.begin()
                        + lane.exact.path_offsets[row],
                    lane.exact.path_indices.begin()
                        + lane.exact.path_offsets[row + 1]);
                payload.status = lane.exact.statuses[row];
                payload.reason = lane.exact.reasons[row];
                std::copy_n(
                    lane.exact.metrics.begin()
                        + static_cast<std::ptrdiff_t>(row * 4),
                    4, payload.metrics.begin());
                std::copy_n(
                    lane.exact.label_counters.begin()
                        + static_cast<std::ptrdiff_t>(row * 3),
                    3, payload.label_counters.begin());
                return payload;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] static double elapsed_seconds(
        const std::chrono::steady_clock::time_point started) {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }

    static void require_before_deadline(
        const std::chrono::steady_clock::time_point started,
        const double limit,
        const CandidateTransactionDeadlinePhaseV2 phase,
        const std::string_view phase_name) {
        if (elapsed_seconds(started) >= limit) {
            throw CandidateTransactionDeadlineV2(
                phase,
                "native candidate transaction reached deadline during "
                + std::string(phase_name));
        }
    }

    [[nodiscard]] std::vector<std::int64_t> all_customers() const {
        std::vector<std::int64_t> output;
        for (std::size_t node = 0; node < problem_.node_count(); ++node) {
            if (problem_.node_kind[node]
                == native_kernels::customer_kind) {
                output.push_back(static_cast<std::int64_t>(node));
            }
        }
        return output;
    }

    [[nodiscard]] CandidateTransactionExecutionV2 execute_owned(
        CandidateRoundRequestV2 round,
        const CandidateTransactionOptionsV2 options,
        const std::chrono::steady_clock::time_point started) {
        const auto plan_count = round.plan_offsets.size() - 1;
        const auto route_count = round.route_offsets.size() - 1;
        const PlanBatchViewV2 plans{
            round.plan_offsets,
            {round.route_offsets, round.route_indices},
        };
        const auto customers = all_customers();
        const auto prepared = native_candidate_plan::prepare({
            round.plan_offsets,
            round.route_offsets,
            round.route_indices,
            round.expected_customers,
            problem_.node_kind,
            problem_.lexical_rank,
            customers,
            native_kernels::customer_kind,
            options.allow_partial_customer_coverage,
        });
        CandidateTransactionTraceV2 trace;
        trace.canonical_expected = prepared.canonical_expected;
        trace.coverage_eligible = prepared.coverage_eligible;
        trace.unique_route_offsets = prepared.unique_route_offsets;
        trace.unique_route_indices = prepared.unique_route_indices;
        trace.unique_row_by_route = prepared.unique_row_by_route;
        if (!options.suppress_round_budget) {
            if (round.context[2] < 0) {
                throw std::invalid_argument(
                    "native candidate transaction round budget requires an iteration");
            }
            budget_.begin_shared_iteration_round(
                round.context[0], round.context[2]);
        }
        const auto attempted = options.suppress_attempted_plan_journal
            ? std::vector<std::int64_t>(plan_count, 0)
            : attempted_plans_.lookup(plans);
        trace.attempted_flags = attempted;

        const RouteBatchViewV2 unique_routes{
            prepared.unique_route_offsets, prepared.unique_route_indices};
        std::vector<std::int64_t> negative_hits(unique_routes.route_count(), 0);
        std::vector<std::int64_t> negative_reasons(
            unique_routes.route_count(), 0);
        if (!options.suppress_screening_negative_cache) {
            auto lookup = negative_cache_.lookup_many(unique_routes);
            negative_hits = std::move(lookup.hit_flags);
            negative_reasons = std::move(lookup.reasons);
        }
        trace.negative_hit_flags = negative_hits;
        trace.negative_hit_reasons = negative_reasons;
        std::vector<std::int64_t> screen_offsets{0};
        std::vector<std::int64_t> screen_indices;
        std::vector<std::size_t> screen_rows;
        for (std::size_t row = 0; row < unique_routes.route_count(); ++row) {
            if (negative_hits[row] != 0) {
                continue;
            }
            const auto sequence = unique_routes.route(row);
            screen_indices.insert(
                screen_indices.end(), sequence.begin(), sequence.end());
            screen_offsets.push_back(
                static_cast<std::int64_t>(screen_indices.size()));
            screen_rows.push_back(row);
        }
        trace.physical_screen_rows.reserve(screen_rows.size());
        std::transform(
            screen_rows.begin(), screen_rows.end(),
            std::back_inserter(trace.physical_screen_rows),
            [](const std::size_t row) {
                return static_cast<std::int64_t>(row);
            });
        const auto screening_started = std::chrono::steady_clock::now();
        auto screened = kernels_.screen(
            problem_, {screen_offsets, screen_indices}, screening_epsilon_);
        trace.screening_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - screening_started).count();
        if (screened.size() != screen_rows.size()) {
            throw std::logic_error(
                "native candidate screening output count is invalid");
        }
        std::vector<native_kernels::ScreenOutput> screen_outputs(
            unique_routes.route_count());
        for (std::size_t ordinal = 0; ordinal < screen_rows.size(); ++ordinal) {
            screen_outputs[screen_rows[ordinal]] = std::move(screened[ordinal]);
        }
        std::int64_t negative_hit_count = 0;
        std::vector<std::int64_t> rejected_offsets{0};
        std::vector<std::int64_t> rejected_indices;
        std::vector<std::int64_t> rejected_reasons;
        for (std::size_t row = 0; row < unique_routes.route_count(); ++row) {
            if (negative_hits[row] != 0) {
                screen_outputs[row].codes[0] = 0;
                screen_outputs[row].codes[1] = negative_reasons[row];
                ++negative_hit_count;
                continue;
            }
            if (screen_outputs[row].codes[0] == 0) {
                const auto sequence = unique_routes.route(row);
                rejected_indices.insert(
                    rejected_indices.end(), sequence.begin(), sequence.end());
                rejected_offsets.push_back(
                    static_cast<std::int64_t>(rejected_indices.size()));
                rejected_reasons.push_back(screen_outputs[row].codes[1]);
            }
        }
        trace.screen_outputs = screen_outputs;
        bool negative_store_active = false;
        if (!options.suppress_screening_negative_cache
            && !rejected_reasons.empty()) {
            static_cast<void>(
                negative_cache_.begin_store_many_atomic(
                    {rejected_offsets, rejected_indices}, rejected_reasons));
            negative_store_active = true;
        }

        std::vector<double> lower_bounds(route_count, 0.0);
        std::vector<std::int64_t> screening_passed(route_count, 0);
        for (std::size_t route = 0; route < route_count; ++route) {
            const auto row = static_cast<std::size_t>(
                prepared.unique_row_by_route[route]);
            lower_bounds[route] = screen_outputs[row].metrics[3];
            screening_passed[route] =
                screen_outputs[row].codes[0] == 1 ? 1 : 0;
        }
        trace.screening_passed = screening_passed;
        const auto decision = native_candidate_plan::decide({
            round.plan_offsets,
            prepared.coverage_eligible,
            screening_passed,
            attempted,
            static_cast<std::int64_t>(
                active_route_offsets_.size() - 1),
            options.allow_vehicle_increase,
        });
        const auto ranking = native_candidate_plan::rank({
            round.plan_offsets,
            round.route_offsets,
            round.route_indices,
            lower_bounds,
            active_route_offsets_,
            active_route_indices_,
            problem_.lexical_rank,
            decision.combined_attempted,
            proposal_top_k_,
        });
        trace.eligible_flags = decision.eligible;
        trace.combined_attempted_flags = decision.combined_attempted;
        trace.ranked = ranking.ranked;
        trace.selected = ranking.selected;
        trace.ranking_integer = ranking.integer_metrics;
        trace.ranking_float = ranking.optimistic_distances;

        std::vector<std::int64_t> statuses(plan_count, 2);
        for (std::size_t plan = 0; plan < plan_count; ++plan) {
            statuses[plan] = decision.eligible[plan] == 0 ? 0
                : attempted[plan] != 0 ? 1 : 2;
        }
        std::vector<std::int64_t> objective_integer(plan_count * 2, -1);
        std::vector<double> objective_float(
            plan_count * 2, std::numeric_limits<double>::quiet_NaN());
        std::vector<std::int64_t> route_resolutions(route_count, 0);
        std::vector<std::optional<ExactPayload>> route_payloads(route_count);
        std::vector<std::int64_t> exact_route_rows;
        std::vector<std::int64_t> completion_order;
        std::vector<std::int64_t> feasible_plans;
        std::vector<std::int64_t> completed_plans;
        std::int64_t feasible_count = 0;
        std::int64_t infeasible_count = 0;
        std::int64_t budget_skip_count = 0;

        exact_cache_.begin_protocol_transaction();
        for (const auto selected_plan : ranking.selected) {
            CandidatePlanExecutionTraceV2 plan_trace;
            plan_trace.plan_id = selected_plan;
            const auto plan_budget_before = budget_.snapshot();
            const auto plan = static_cast<std::size_t>(selected_plan);
            const auto first_route = static_cast<std::size_t>(
                round.plan_offsets[plan]);
            const auto last_route = static_cast<std::size_t>(
                round.plan_offsets[plan + 1]);
            std::vector<std::int64_t> local_offsets{0};
            std::vector<std::int64_t> local_indices;
            for (auto route = first_route; route < last_route; ++route) {
                const auto sequence = plans.routes.route(route);
                local_indices.insert(
                    local_indices.end(), sequence.begin(), sequence.end());
                local_offsets.push_back(
                    static_cast<std::int64_t>(local_indices.size()));
            }
            const RouteBatchViewV2 local_routes{local_offsets, local_indices};
            std::vector<ExactPayload> payloads(last_route - first_route);
            std::vector<std::size_t> cache_rows;
            std::vector<std::int64_t> cache_offsets{0};
            std::vector<std::int64_t> cache_indices;
            for (std::size_t local = 0; local < payloads.size(); ++local) {
                if (auto precomputed = incumbent_payload(
                        local_routes.route(local)); precomputed.has_value()) {
                    payloads[local] = std::move(*precomputed);
                    route_resolutions[first_route + local] = 3;
                    continue;
                }
                cache_rows.push_back(local);
                const auto sequence = local_routes.route(local);
                cache_indices.insert(
                    cache_indices.end(), sequence.begin(), sequence.end());
                cache_offsets.push_back(
                    static_cast<std::int64_t>(cache_indices.size()));
            }
            ExactRouteCacheV2::ExactLookupResult cached;
            if (!cache_rows.empty()) {
                cached = exact_cache_.lookup_exact_many(
                    {cache_offsets, cache_indices});
            }
            plan_trace.cache_hit_flags = cached.hit_flags;
            for (std::size_t row = 0; row < cache_rows.size(); ++row) {
                const auto local = cache_rows[row];
                if (cached.hit_flags[row] == 1) {
                    auto& payload = payloads[local];
                    payload.path.assign(
                        cached.path_indices.begin() + cached.path_offsets[row],
                        cached.path_indices.begin()
                            + cached.path_offsets[row + 1]);
                    payload.status = cached.statuses[row];
                    payload.reason = cached.reasons[row];
                    std::copy_n(
                        cached.metrics.begin()
                            + static_cast<std::ptrdiff_t>(row * 4),
                        4, payload.metrics.begin());
                    std::copy_n(
                        cached.label_counters.begin()
                            + static_cast<std::ptrdiff_t>(row * 3),
                        3, payload.label_counters.begin());
                    route_resolutions[first_route + local] = 1;
                }
            }
            struct PendingExactStore final {
                std::vector<std::int64_t> offsets;
                std::vector<std::int64_t> indices;
                native_kernels::ExactBatchOutput exact;
                std::vector<std::array<std::uint8_t, 32>> hashes;
                std::vector<std::int64_t> bytes;
            };
            std::vector<PendingExactStore> pending_exact_stores;
            bool plan_budget_skipped = false;
            const auto execute_missing_group = [
                &, this](const std::vector<std::size_t>& missing_rows) {
                const auto requested_exact = static_cast<std::int64_t>(
                    missing_rows.size());
                if (requested_exact == 0) {
                    return;
                }
                const auto exact_remaining = budget_.exact_remaining();
                const auto round_remaining = options.suppress_round_budget
                    ? requested_exact
                    : budget_.round_remaining();
                const auto available_exact = std::min<std::int64_t>(
                    round_remaining,
                    exact_remaining < 0 ? requested_exact : exact_remaining);
                if (requested_exact > available_exact) {
                    plan_budget_skipped = true;
                    require_before_deadline(
                        started, round.deadline_remaining,
                        CandidateTransactionDeadlinePhaseV2::budget_skip,
                        "budget skip");
                    return;
                }
                std::vector<std::int64_t> missing_offsets{0};
                std::vector<std::int64_t> missing_indices;
                for (const auto local : missing_rows) {
                    const auto sequence = local_routes.route(local);
                    missing_indices.insert(
                        missing_indices.end(), sequence.begin(), sequence.end());
                    missing_offsets.push_back(
                        static_cast<std::int64_t>(missing_indices.size()));
                }
                const auto round_reservation = options.suppress_round_budget
                    ? SearchBudgetStateV2::RoundReservation{
                        requested_exact, requested_exact, round_remaining}
                    : budget_.reserve_round(requested_exact, true);
                const auto exact_reservation =
                    budget_.reserve_exact(requested_exact);
                if (round_reservation.granted != requested_exact
                    || exact_reservation.granted != requested_exact) {
                    throw std::logic_error(
                        "native candidate budget changed during reservation");
                }
                plan_trace.budget_reservation[0] += requested_exact;
                plan_trace.budget_reservation[1] += round_reservation.granted;
                plan_trace.budget_reservation[2] += exact_reservation.granted;
                const auto exact_remaining_seconds = round.deadline_remaining
                    - elapsed_seconds(started);
                if (exact_remaining_seconds <= 0.0) {
                    budget_.restore(plan_budget_before);
                    throw CandidateTransactionDeadlineV2(
                        CandidateTransactionDeadlinePhaseV2::before_exact,
                        "native candidate deadline expired before exact work");
                }
                native_kernels::ExactBatchOutput exact;
                bool exact_settled = false;
                try {
                    exact = kernels_.exact(
                        problem_,
                        {missing_offsets, missing_indices},
                        exact_remaining_seconds,
                        round.batch_size);
                    if (exact.batch_counters.size() != 10) {
                        throw std::logic_error(
                            "native candidate exact counters are missing");
                    }
                    const auto completed = exact.batch_counters[2];
                    const auto interrupted = exact.batch_counters[3];
                    if (elapsed_seconds(started) >= round.deadline_remaining) {
                        budget_.settle_exact(0, requested_exact);
                    } else {
                        budget_.settle_exact(completed, interrupted);
                    }
                    exact_settled = true;
                } catch (...) {
                    if (!exact_settled) {
                        budget_.settle_exact(0, requested_exact);
                    }
                    throw;
                }
                const auto completed = exact.batch_counters[2];
                const auto interrupted = exact.batch_counters[3];
                if (completed != requested_exact || interrupted != 0
                    || elapsed_seconds(started) >= round.deadline_remaining) {
                    throw CandidateTransactionDeadlineV2(
                        CandidateTransactionDeadlinePhaseV2::exact_work,
                        "native candidate exact work reached deadline");
                }
                const auto aggregate_row_base = plan_trace.exact.statuses.size();
                if (plan_trace.exact.path_offsets.empty()) {
                    plan_trace.exact.path_offsets.push_back(0);
                }
                const auto path_base = plan_trace.exact.path_offsets.back();
                for (std::size_t row = 1; row < exact.path_offsets.size(); ++row) {
                    plan_trace.exact.path_offsets.push_back(
                        path_base + exact.path_offsets[row]);
                }
                plan_trace.exact.path_indices.insert(
                    plan_trace.exact.path_indices.end(),
                    exact.path_indices.begin(), exact.path_indices.end());
                plan_trace.exact.statuses.insert(
                    plan_trace.exact.statuses.end(),
                    exact.statuses.begin(), exact.statuses.end());
                plan_trace.exact.reasons.insert(
                    plan_trace.exact.reasons.end(),
                    exact.reasons.begin(), exact.reasons.end());
                plan_trace.exact.metrics.insert(
                    plan_trace.exact.metrics.end(),
                    exact.metrics.begin(), exact.metrics.end());
                plan_trace.exact.label_counters.insert(
                    plan_trace.exact.label_counters.end(),
                    exact.label_counters.begin(), exact.label_counters.end());
                plan_trace.exact.batch_counters.insert(
                    plan_trace.exact.batch_counters.end(),
                    exact.batch_counters.begin(), exact.batch_counters.end());
                for (const auto ordinal : exact.completion_order) {
                    plan_trace.exact.completion_order.push_back(
                        static_cast<std::int64_t>(aggregate_row_base) + ordinal);
                }
                plan_trace.exact_batch_sizes.push_back(requested_exact);
                for (std::size_t exact_row = 0;
                     exact_row < missing_rows.size(); ++exact_row) {
                    const auto local = missing_rows[exact_row];
                    auto& payload = payloads[local];
                    payload.path.assign(
                        exact.path_indices.begin() + exact.path_offsets[exact_row],
                        exact.path_indices.begin()
                            + exact.path_offsets[exact_row + 1]);
                    payload.status = exact.statuses[exact_row];
                    payload.reason = exact.reasons[exact_row];
                    std::copy_n(
                        exact.metrics.begin()
                            + static_cast<std::ptrdiff_t>(exact_row * 4),
                        4, payload.metrics.begin());
                    std::copy_n(
                        exact.label_counters.begin()
                            + static_cast<std::ptrdiff_t>(exact_row * 3),
                        3, payload.label_counters.begin());
                    route_resolutions[first_route + local] = 2;
                    exact_route_rows.push_back(
                        static_cast<std::int64_t>(first_route + local));
                    plan_trace.missing_local_rows.push_back(
                        static_cast<std::int64_t>(local));
                }
                for (const auto exact_ordinal : exact.completion_order) {
                    if (exact_ordinal < 0
                        || static_cast<std::size_t>(exact_ordinal)
                            >= missing_rows.size()) {
                        throw std::logic_error(
                            "native candidate completion order is invalid");
                    }
                    completion_order.push_back(static_cast<std::int64_t>(
                        first_route
                        + missing_rows[static_cast<std::size_t>(
                            exact_ordinal)]));
                }
                const auto hashes = exact_semantic_hashes_v2(
                    {missing_offsets, missing_indices}, exact);
                const auto bytes = exact_entry_bytes_v2(
                    problem_, exact);
                pending_exact_stores.push_back({
                    std::move(missing_offsets), std::move(missing_indices),
                    std::move(exact), std::move(hashes), std::move(bytes)});
            };
            std::vector<std::size_t> pending_missing;
            for (std::size_t row = 0; row < cache_rows.size(); ++row) {
                if (cached.hit_flags[row] == 1) {
                    execute_missing_group(pending_missing);
                    pending_missing.clear();
                } else {
                    pending_missing.push_back(cache_rows[row]);
                }
            }
            execute_missing_group(pending_missing);

            // Python's internal repair/probe paths commit exact prefixes even
            // when a later miss group exhausts the round budget.  Those paths
            // suppress the public attempted-plan journal or allow partial
            // customer coverage.  Public complete-candidate transactions keep
            // the v2 atomic rollback contract.
            const auto commit_legacy_exact_prefix =
                options.suppress_attempted_plan_journal
                || options.allow_partial_customer_coverage;
            if (!plan_budget_skipped || commit_legacy_exact_prefix) {
                for (auto& pending_store : pending_exact_stores) {
                    const auto store = exact_cache_.begin_store_exact_many_atomic(
                        {pending_store.offsets, pending_store.indices},
                        pending_store.exact, pending_store.hashes,
                        pending_store.bytes);
                    plan_trace.cache_store_statuses.insert(
                        plan_trace.cache_store_statuses.end(),
                        store.statuses.begin(), store.statuses.end());
                    plan_trace.cache_eviction_counts.insert(
                        plan_trace.cache_eviction_counts.end(),
                        store.eviction_counts.begin(),
                        store.eviction_counts.end());
                    exact_cache_.prepare_store_commit();
                    exact_cache_.commit_store_batch_noexcept();
                }
            }

            if (plan_budget_skipped) {
                if (!commit_legacy_exact_prefix) {
                    plan_trace.cache_store_statuses.assign(
                        plan_trace.exact.statuses.size(), 0);
                    plan_trace.cache_eviction_counts.assign(
                        plan_trace.exact.statuses.size(), 0);
                }
                ++budget_skip_count;
                statuses[plan] = 3;
                trace.plans.push_back(std::move(plan_trace));
                require_before_deadline(started, round.deadline_remaining,
                    CandidateTransactionDeadlinePhaseV2::budget_skip,
                    "budget skip after exact groups");
                continue;
            }

            bool feasible = true;
            PythonCompatibleFloatSumV2 distance;
            PythonCompatibleFloatSumV2 charging_time;
            std::int64_t charging_count = 0;
            for (std::size_t local = 0; local < payloads.size(); ++local) {
                const auto& payload = payloads[local];
                feasible = feasible
                    && payload.status == native_kernels::feasible_status;
                if (payload.status == native_kernels::feasible_status) {
                    distance.add(payload.metrics[0]);
                    charging_time.add(payload.metrics[3]);
                    for (const auto node : payload.path) {
                        charging_count += problem_.node_kind[
                                static_cast<std::size_t>(node)]
                                == native_kernels::station_kind
                            ? 1 : 0;
                    }
                }
                route_payloads[first_route + local] = payload;
            }
            if (feasible) {
                statuses[plan] = 5;
                ++feasible_count;
                feasible_plans.push_back(selected_plan);
                objective_integer[plan * 2] = static_cast<std::int64_t>(
                    last_route - first_route);
                objective_integer[plan * 2 + 1] = charging_count;
                objective_float[plan * 2] = distance.value();
                objective_float[plan * 2 + 1] = charging_time.value();
            } else {
                statuses[plan] = 4;
                ++infeasible_count;
            }
            completed_plans.push_back(selected_plan);
            trace.plans.push_back(std::move(plan_trace));
            require_before_deadline(
                started, round.deadline_remaining,
                CandidateTransactionDeadlinePhaseV2::candidate_commit,
                "candidate commit");
        }
        require_before_deadline(
            started, round.deadline_remaining,
            CandidateTransactionDeadlinePhaseV2::transaction_return,
            "transaction return");

        bool attempted_active = false;
        if (!completed_plans.empty()
            && !options.suppress_attempted_plan_journal) {
            static_cast<void>(
                attempted_plans_.begin_mark_many_atomic(
                    plans, completed_plans));
            attempted_active = true;
        }
        feasible_plans = native_candidate_plan::order_feasible({
            round.plan_offsets,
            round.route_offsets,
            round.route_indices,
            objective_integer,
            objective_float,
            problem_.lexical_rank,
            feasible_plans,
        });

        CandidateExactBatchV2 transaction_exact;
        transaction_exact.statuses.assign(route_count, -1);
        transaction_exact.reasons.assign(route_count, -1);
        transaction_exact.metrics.assign(route_count * 4, 0.0);
        transaction_exact.label_counters.assign(route_count * 3, 0);
        for (std::size_t route = 0; route < route_count; ++route) {
            if (!route_payloads[route].has_value()) {
                transaction_exact.path_offsets.push_back(
                    static_cast<std::int64_t>(
                        transaction_exact.path_indices.size()));
                continue;
            }
            const auto& payload = *route_payloads[route];
            transaction_exact.path_indices.insert(
                transaction_exact.path_indices.end(),
                payload.path.begin(), payload.path.end());
            transaction_exact.path_offsets.push_back(
                static_cast<std::int64_t>(
                    transaction_exact.path_indices.size()));
            transaction_exact.statuses[route] = payload.status;
            transaction_exact.reasons[route] = payload.reason;
            std::copy(
                payload.metrics.begin(), payload.metrics.end(),
                transaction_exact.metrics.begin()
                    + static_cast<std::ptrdiff_t>(route * 4));
            std::copy(
                payload.label_counters.begin(), payload.label_counters.end(),
                transaction_exact.label_counters.begin()
                    + static_cast<std::ptrdiff_t>(route * 3));
        }

        const auto cache_snapshot = exact_cache_.snapshot();
        const auto negative_snapshot = negative_cache_.snapshot();
        CandidatePlanTransactionResultV2 result;
        result.selected = ranking.selected;
        result.plan_offsets = std::move(round.plan_offsets);
        result.route_offsets = std::move(round.route_offsets);
        result.route_indices = std::move(round.route_indices);
        result.context.assign(round.context.begin(), round.context.end());
        result.expected_customers = prepared.canonical_expected;
        result.batch = {round.batch_size};
        result.lower_bounds = std::move(lower_bounds);
        result.ranked = ranking.ranked;
        result.statuses = std::move(statuses);
        result.feasible_order = std::move(feasible_plans);
        result.exact_route_rows = std::move(exact_route_rows);
        result.objective_integer = std::move(objective_integer);
        result.objective_float = std::move(objective_float);
        result.route_resolutions = std::move(route_resolutions);
        result.completion_order = std::move(completion_order);
        result.counters = {
            static_cast<std::int64_t>(plan_count),
            static_cast<std::int64_t>(ranking.selected.size()),
            feasible_count,
            infeasible_count,
            budget_skip_count,
            static_cast<std::int64_t>(result.exact_route_rows.size()),
            attempted_plans_.size(),
            negative_hit_count,
        };
        result.cache_statistics.assign(
            cache_snapshot.statistics.begin(),
            cache_snapshot.statistics.end());
        result.cache_hash_rows = cache_snapshot.entries.size();
        for (const auto& entry : cache_snapshot.entries) {
            result.cache_hashes.insert(
                result.cache_hashes.end(),
                entry.semantic_hash.begin(), entry.semantic_hash.end());
        }
        result.negative_offsets = {0};
        for (const auto& entry : negative_snapshot.entries) {
            result.negative_indices.insert(
                result.negative_indices.end(),
                entry.route.begin(), entry.route.end());
            result.negative_offsets.push_back(
                static_cast<std::int64_t>(result.negative_indices.size()));
            result.negative_reasons.push_back(entry.reason);
        }
        auto negative_statistics =
            negative_cache_.projected_statistics();
        result.negative_statistics.assign(
            negative_statistics.begin(), negative_statistics.end());
        const auto budget_values = budget_.values();
        result.budget_state.assign(budget_values.begin(), budget_values.end());
        result.exact = std::move(transaction_exact);
        result.transaction_sha256 = candidate_plan_transaction_sha256_v2(result);
        validate_candidate_plan_transaction_v2(result);

        exact_cache_.prepare_protocol_commit();
        if (negative_store_active) {
            negative_cache_.prepare_store_commit();
        }
        if (attempted_active) {
            attempted_plans_.prepare_mark_commit();
        }
        trace.exact_protocol_active = true;
        trace.negative_store_active = negative_store_active;
        trace.attempted_mark_active = attempted_active;
        if (!options.defer_commit) {
            exact_cache_.commit_protocol_transaction_noexcept();
            if (negative_store_active) {
                negative_cache_.commit_store_batch_noexcept();
            }
            if (attempted_active) {
                attempted_plans_.commit_mark_batch_noexcept();
            }
            trace.exact_protocol_active = false;
            trace.negative_store_active = false;
            trace.attempted_mark_active = false;
        }
        trace.completed_plan_ids = completed_plans;
        return {std::move(result), std::move(trace)};
    }
};

}  // namespace evrptw::native_search
