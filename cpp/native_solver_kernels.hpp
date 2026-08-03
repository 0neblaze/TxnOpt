#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace evrptw::native_kernels {
class PythonFloatSum {
public:
    void add(double value) {
        // Match CPython 3.13's float-specialized sum() exactly.  Stage 5.2
        // compares native and Python raw evidence bit-for-bit, so ordinary
        // left-to-right += accumulation is not semantically equivalent.
        const auto next = total_ + value;
        if (std::fabs(total_) >= std::fabs(value)) {
            compensation_ += (total_ - next) + value;
        } else {
            compensation_ += (value - next) + total_;
        }
        total_ = next;
    }

    [[nodiscard]] double value() const {
        auto result = total_;
        if (compensation_ != 0.0 && std::isfinite(compensation_)) {
            result += compensation_;
        }
        return result;
    }

private:
    double total_ = 0.0;
    double compensation_ = 0.0;
};
constexpr double exact_epsilon = 1e-9;
constexpr std::int64_t depot_kind = 0;
constexpr std::int64_t customer_kind = 1;
constexpr std::int64_t station_kind = 2;
constexpr std::int64_t feasible_status = 0;
constexpr std::int64_t infeasible_status = 1;
constexpr std::int64_t interrupted_status = 2;
constexpr std::int64_t no_failure_reason = 0;
constexpr std::int64_t no_feasible_pattern_reason = 1;
constexpr std::int64_t deadline_reason = 2;

struct ExactLabel {
    std::int64_t progress;
    std::int64_t node;
    double elapsed_time;
    double battery;
    double distance;
    double total_energy;
    double charged_energy;
    double charging_time;
    std::int64_t parent;
    bool live;
};

struct ExactQueueEntry {
    double distance;
    double elapsed_time;
    std::int64_t negative_progress;
    std::int64_t serial;
    std::size_t label_index;
};

struct ExactQueueLater {
    bool operator()(const ExactQueueEntry& left, const ExactQueueEntry& right) const {
        return std::tie(left.distance, left.elapsed_time, left.negative_progress, left.serial)
            > std::tie(right.distance, right.elapsed_time, right.negative_progress, right.serial);
    }
};

struct ExactSearchState {
    std::vector<std::int64_t> order;
    std::vector<ExactLabel> labels;
    std::vector<std::vector<std::size_t>> state_labels;
    std::priority_queue<ExactQueueEntry, std::vector<ExactQueueEntry>, ExactQueueLater> queue;
    std::optional<ExactLabel> best;
    std::int64_t generated = 1;
    std::int64_t expanded = 0;
    std::int64_t pruned = 0;
    std::int64_t serial = 1;
    bool completed = false;
    bool interrupted = false;
};

struct ExactRequest {
    std::size_t route;
    std::size_t label_index;
    std::int64_t destination;
    std::int64_t progress;
};

struct ExactBatchOutput {
    std::vector<std::int64_t> path_offsets;
    std::vector<std::int64_t> path_indices;
    std::vector<std::int64_t> statuses;
    std::vector<std::int64_t> reasons;
    std::vector<double> metrics;
    std::vector<std::int64_t> label_counters;
    std::vector<std::int64_t> batch_counters;
};

bool exact_dominates(const ExactLabel& left, const ExactLabel& right) {
    const bool no_worse = left.elapsed_time <= right.elapsed_time + exact_epsilon
        && left.battery + exact_epsilon >= right.battery
        && left.distance <= right.distance + exact_epsilon;
    const bool strictly_better = left.elapsed_time < right.elapsed_time - exact_epsilon
        || left.battery > right.battery + exact_epsilon
        || left.distance < right.distance - exact_epsilon;
    return no_worse && strictly_better;
}

bool exact_better_terminal(const ExactLabel& candidate, const ExactLabel& incumbent) {
    return std::tie(candidate.distance, candidate.elapsed_time)
        < std::tie(incumbent.distance, incumbent.elapsed_time);
}

std::optional<std::size_t> pop_live_exact_label(ExactSearchState& state) {
    while (!state.queue.empty()) {
        const auto entry = state.queue.top();
        state.queue.pop();
        if (state.labels[entry.label_index].live) {
            return entry.label_index;
        }
    }
    return std::nullopt;
}

std::vector<std::int64_t> reconstruct_exact_path(
    const ExactSearchState& state,
    const ExactLabel& terminal) {
    std::vector<std::int64_t> reversed;
    reversed.push_back(terminal.node);
    auto parent = terminal.parent;
    while (parent >= 0) {
        const auto& label = state.labels[static_cast<std::size_t>(parent)];
        reversed.push_back(label.node);
        parent = label.parent;
    }
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

ExactBatchOutput run_exact_charging_batch(
    const std::int64_t* node_kinds,
    const double* ready,
    const double* due,
    const double* service,
    const double* distances,
    const double* vehicle,
    const std::int64_t* order_offsets,
    const std::int64_t* order_indices,
    std::size_t node_count,
    std::size_t route_count,
    std::int64_t depot,
    const std::vector<std::int64_t>& stations,
    double deadline_remaining,
    std::int64_t batch_size) {
    const auto started = std::chrono::steady_clock::now();
    ExactBatchOutput output;
    output.path_offsets.assign(route_count + 1, 0);
    output.statuses.assign(route_count, interrupted_status);
    output.reasons.assign(route_count, deadline_reason);
    output.metrics.assign(route_count * 4, 0.0);
    output.label_counters.assign(route_count * 3, 0);
    output.batch_counters.assign(10, 0);
    output.batch_counters[0] = static_cast<std::int64_t>(route_count);
    output.batch_counters[1] = static_cast<std::int64_t>(route_count);
    output.batch_counters[4] = route_count == 0 ? 0 : 1;
    output.batch_counters[8] = route_count == 0 ? 0 : 1;
    output.batch_counters[9] = batch_size;

    const auto deadline_expired = [&]() {
        ++output.batch_counters[7];
        if (std::isinf(deadline_remaining) && deadline_remaining > 0.0) {
            return false;
        }
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return elapsed >= std::max(0.0, deadline_remaining);
    };
    if (deadline_expired()) {
        output.batch_counters[3] = static_cast<std::int64_t>(route_count);
        return output;
    }

    std::vector<ExactSearchState> states(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        if (deadline_expired()) {
            output.batch_counters[3] = static_cast<std::int64_t>(route_count);
            return output;
        }
        auto& state = states[route];
        const auto begin = order_offsets[route];
        const auto end = order_offsets[route + 1];
        if (end > begin) {
            state.order.assign(order_indices + begin, order_indices + end);
        }
        state.state_labels.resize((state.order.size() + 1) * node_count);
        state.labels.push_back(ExactLabel{
            0,
            depot,
            std::max(0.0, ready[depot]),
            vehicle[0],
            0.0,
            0.0,
            0.0,
            0.0,
            -1,
            true,
        });
        state.state_labels[static_cast<std::size_t>(depot)].push_back(0);
        state.queue.push(ExactQueueEntry{0.0, std::max(0.0, ready[depot]), 0, 0, 0});
    }

    bool deadline_hit = deadline_expired();
    while (!deadline_hit) {
        std::vector<ExactRequest> requests;
        bool progressed = false;
        for (std::size_t route = 0; route < route_count; ++route) {
            auto& state = states[route];
            if (state.completed) {
                continue;
            }
            const auto live_index = pop_live_exact_label(state);
            if (!live_index.has_value()) {
                state.completed = true;
                continue;
            }
            const auto& label = state.labels[*live_index];
            if (state.best.has_value()
                && label.distance >= state.best->distance - exact_epsilon) {
                ++state.pruned;
                progressed = true;
                continue;
            }
            ++state.expanded;
            progressed = true;
            if (label.progress < static_cast<std::int64_t>(state.order.size())) {
                requests.push_back(ExactRequest{
                    route,
                    *live_index,
                    state.order[static_cast<std::size_t>(label.progress)],
                    label.progress + 1,
                });
            } else {
                requests.push_back(ExactRequest{route, *live_index, depot, label.progress});
            }
            for (const auto station : stations) {
                if (station != label.node) {
                    requests.push_back(ExactRequest{
                        route,
                        *live_index,
                        station,
                        label.progress,
                    });
                }
            }
        }
        deadline_hit = deadline_expired();
        if (deadline_hit) {
            break;
        }
        if (requests.empty()) {
            if (!progressed) {
                break;
            }
            continue;
        }

        for (std::size_t offset = 0; offset < requests.size();
             offset += static_cast<std::size_t>(batch_size)) {
            deadline_hit = deadline_expired();
            if (deadline_hit) {
                break;
            }
            const auto chunk_end = std::min(
                requests.size(), offset + static_cast<std::size_t>(batch_size));
            ++output.batch_counters[5];
            output.batch_counters[6] += static_cast<std::int64_t>(chunk_end - offset);
            for (std::size_t request_index = offset; request_index < chunk_end; ++request_index) {
                const auto& request = requests[request_index];
                auto& state = states[request.route];
                const auto& label = state.labels[request.label_index];
                const auto destination = request.destination;
                ++state.generated;
                const auto leg_distance = distances[
                    static_cast<std::size_t>(label.node) * node_count
                    + static_cast<std::size_t>(destination)];
                const auto energy = leg_distance * vehicle[2];
                if (energy > label.battery + exact_epsilon) {
                    ++state.pruned;
                    continue;
                }
                auto battery = std::max(0.0, label.battery - energy);
                auto elapsed = std::max(
                    label.elapsed_time + leg_distance / vehicle[4], ready[destination]);
                if (elapsed > due[destination] + exact_epsilon) {
                    ++state.pruned;
                    continue;
                }
                double charged = 0.0;
                double charging_time = 0.0;
                if (node_kinds[destination] == customer_kind) {
                    elapsed += service[destination];
                } else if (node_kinds[destination] == station_kind) {
                    charged = vehicle[0] - battery;
                    charging_time = charged * vehicle[3];
                    elapsed += charging_time;
                    if (elapsed > due[destination] + exact_epsilon) {
                        ++state.pruned;
                        continue;
                    }
                    battery = vehicle[0];
                }
                ExactLabel candidate{
                    request.progress,
                    destination,
                    elapsed,
                    battery,
                    label.distance + leg_distance,
                    label.total_energy + energy,
                    label.charged_energy + charged,
                    label.charging_time + charging_time,
                    static_cast<std::int64_t>(request.label_index),
                    true,
                };
                if (request.progress == static_cast<std::int64_t>(state.order.size())
                    && node_kinds[destination] == depot_kind) {
                    if (!state.best.has_value()
                        || exact_better_terminal(candidate, *state.best)) {
                        state.best = candidate;
                    }
                    continue;
                }

                const auto group_index = static_cast<std::size_t>(request.progress) * node_count
                    + static_cast<std::size_t>(destination);
                auto& current = state.state_labels[group_index];
                bool dominated = false;
                for (const auto existing_index : current) {
                    if (exact_dominates(state.labels[existing_index], candidate)) {
                        dominated = true;
                        break;
                    }
                }
                if (dominated) {
                    ++state.pruned;
                    continue;
                }
                std::vector<std::size_t> survivors;
                survivors.reserve(current.size() + 1);
                for (const auto existing_index : current) {
                    if (exact_dominates(candidate, state.labels[existing_index])) {
                        state.labels[existing_index].live = false;
                        ++state.pruned;
                    } else {
                        survivors.push_back(existing_index);
                    }
                }
                const auto candidate_index = state.labels.size();
                state.labels.push_back(candidate);
                survivors.push_back(candidate_index);
                current = std::move(survivors);
                ++state.serial;
                state.queue.push(ExactQueueEntry{
                    candidate.distance,
                    candidate.elapsed_time,
                    -candidate.progress,
                    state.serial,
                    candidate_index,
                });
            }
        }
    }

    for (std::size_t route = 0; route < route_count; ++route) {
        auto& state = states[route];
        if (!state.completed && deadline_hit) {
            state.interrupted = true;
        } else if (!state.completed) {
            state.completed = true;
        }
        output.label_counters[route * 3] = state.generated;
        output.label_counters[route * 3 + 1] = state.expanded;
        output.label_counters[route * 3 + 2] = state.pruned;
        if (state.interrupted) {
            continue;
        }
        ++output.batch_counters[2];
        if (!state.best.has_value()) {
            output.statuses[route] = infeasible_status;
            output.reasons[route] = no_feasible_pattern_reason;
            output.metrics[route * 4] = std::numeric_limits<double>::infinity();
            continue;
        }
        output.statuses[route] = feasible_status;
        output.reasons[route] = no_failure_reason;
        output.metrics[route * 4] = state.best->distance;
        output.metrics[route * 4 + 1] = state.best->total_energy;
        output.metrics[route * 4 + 2] = state.best->charged_energy;
        output.metrics[route * 4 + 3] = state.best->charging_time;
        const auto path = reconstruct_exact_path(state, *state.best);
        output.path_indices.insert(output.path_indices.end(), path.begin(), path.end());
        output.path_offsets[route + 1] = static_cast<std::int64_t>(output.path_indices.size());
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (output.path_offsets[route + 1] == 0) {
            output.path_offsets[route + 1] = output.path_offsets[route];
        }
    }
    output.batch_counters[3] = static_cast<std::int64_t>(route_count) - output.batch_counters[2];
    return output;
}
constexpr std::int64_t screen_reason_none = 0;
constexpr std::int64_t screen_reason_structure = 1;
constexpr std::int64_t screen_reason_capacity = 2;
constexpr std::int64_t screen_reason_forward = 3;
constexpr std::int64_t screen_reason_backward = 4;
constexpr std::int64_t screen_reason_slack = 5;
constexpr std::int64_t screen_reason_energy = 6;
constexpr std::int64_t screen_reason_structural_energy = 7;
constexpr std::int64_t screen_reason_legacy_time = 8;
constexpr std::int64_t screen_reason_legacy_energy = 9;
constexpr std::int64_t check_structure = 1;
constexpr std::int64_t check_capacity = 2;
constexpr std::int64_t check_forward = 3;
constexpr std::int64_t check_backward = 4;
constexpr std::int64_t check_slack = 5;
constexpr std::int64_t check_distance = 6;
constexpr std::int64_t check_energy = 7;
constexpr std::int64_t check_structural_energy = 8;
constexpr std::int64_t check_fail = 0;
constexpr std::int64_t check_pass = 1;
constexpr std::int64_t check_recorded = 2;

struct ScreenOutput {
    std::vector<std::int64_t> codes = std::vector<std::int64_t>(16, 0);
    std::vector<double> metrics = std::vector<double>(15, 0.0);
    std::int64_t reachability_queries = 0;
};

void append_screen_event(
    ScreenOutput& output,
    std::int64_t check,
    std::int64_t status,
    double value) {
    const auto count = static_cast<std::size_t>(output.codes[7]);
    if (count >= 8) {
        throw std::logic_error("screening emitted more than eight canonical check events");
    }
    output.codes[8 + count] = check * 10 + status;
    output.metrics[7 + count] = value;
    output.codes[7] += 1;
}

void reject_screen(
    ScreenOutput& output,
    std::int64_t reason,
    std::int64_t failed_check,
    double value) {
    output.codes[0] = 0;
    output.codes[1] = reason;
    output.codes[2] = failed_check;
    append_screen_event(output, failed_check, check_fail, value);
}

ScreenOutput run_screen_route(
    const std::int64_t* kinds,
    const double* demands,
    const double* ready,
    const double* due,
    const double* service,
    const double* distances,
    const std::uint8_t* reachable,
    const double* vehicle,
    const std::int64_t* route,
    std::size_t route_size,
    std::size_t node_count,
    std::int64_t depot,
    const std::vector<std::int64_t>& recharge_nodes,
    const double* options,
    const double* incremental) {
    ScreenOutput output;
    const bool full = options[0] >= 0.5;
    const auto epsilon = options[1];
    const bool has_reference = options[3] >= 0.5;
    const bool use_incremental = incremental[0] >= 0.5;
    output.codes[4] = 0;
    output.codes[5] = full ? 1 : 0;
    output.codes[6] = use_incremental ? 1 : 0;
    output.codes[3] = full ? 0 : 1;
    output.metrics[4] = has_reference
        ? 0.0
        : std::numeric_limits<double>::quiet_NaN();

    bool known_sequence = true;
    bool customer_sequence = true;
    bool duplicate_free = true;
    std::vector<bool> seen(node_count, false);
    for (std::size_t position = 0; position < route_size; ++position) {
        const auto node = route[position];
        if (node < 0 || static_cast<std::size_t>(node) >= node_count) {
            known_sequence = false;
            customer_sequence = false;
            continue;
        }
        if (kinds[node] != customer_kind) {
            customer_sequence = false;
        }
        if (seen[static_cast<std::size_t>(node)]) {
            duplicate_free = false;
        }
        seen[static_cast<std::size_t>(node)] = true;
    }

    double distance_lower_bound = 0.0;
    if (use_incremental) {
        distance_lower_bound = incremental[1];
    } else if (known_sequence) {
        PythonFloatSum distance_sum;
        auto origin = depot;
        for (std::size_t position = 0; position < route_size; ++position) {
            const auto destination = route[position];
            distance_sum.add(distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)]);
            origin = destination;
        }
        distance_sum.add(distances[
            static_cast<std::size_t>(origin) * node_count
            + static_cast<std::size_t>(depot)]);
        distance_lower_bound = distance_sum.value();
    }
    output.metrics[3] = distance_lower_bound;
    if (has_reference) {
        output.metrics[4] = distance_lower_bound - options[2];
    }
    if (!full) {
        output.metrics[3] = 0.0;
        output.metrics[4] = std::numeric_limits<double>::quiet_NaN();
    }
    if (!known_sequence || !customer_sequence) {
        output.codes[3] = 0;
        output.metrics[0] = 0.0;
        output.metrics[1] = 0.0;
        if (full) {
            reject_screen(output, screen_reason_structure, check_structure, 0.0);
        } else {
            output.codes[1] = screen_reason_structure;
            output.codes[2] = check_structure;
        }
        return output;
    }

    PythonFloatSum demand_sum;
    for (std::size_t position = 0; position < route_size; ++position) {
        demand_sum.add(demands[route[position]]);
    }
    const auto demand = demand_sum.value();
    output.metrics[0] = demand;
    if (demand > vehicle[1] + epsilon) {
        output.codes[3] = 0;
        if (full) {
            reject_screen(output, screen_reason_capacity, check_capacity, demand);
        } else {
            output.codes[1] = screen_reason_capacity;
            output.codes[2] = check_capacity;
        }
        return output;
    }

    auto current_time = std::max(0.0, ready[depot]);
    if (!full) {
        auto origin = depot;
        for (std::size_t position = 0; position <= route_size; ++position) {
            const auto destination = position < route_size ? route[position] : depot;
            current_time += distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] / vehicle[4];
            current_time = std::max(current_time, ready[destination]);
            if (kinds[destination] == customer_kind) {
                if (current_time > due[destination] + epsilon) {
                    output.codes[1] = screen_reason_legacy_time;
                    output.codes[2] = check_forward;
                    output.metrics[1] = current_time;
                    return output;
                }
                current_time += service[destination];
            }
            origin = destination;
        }
        output.metrics[1] = current_time;
        origin = depot;
        for (std::size_t position = 0; position <= route_size; ++position) {
            const auto destination = position < route_size ? route[position] : depot;
            ++output.reachability_queries;
            if (reachable[
                    static_cast<std::size_t>(origin) * node_count
                    + static_cast<std::size_t>(destination)] == 0U) {
                output.codes[1] = screen_reason_legacy_energy;
                output.codes[2] = check_energy;
                return output;
            }
            origin = destination;
        }
        output.codes[0] = 1;
        output.codes[3] = 1;
        output.codes[4] = 1;
        output.metrics[6] = 1.0;
        return output;
    }

    output.codes[3] = 1;
    output.codes[4] = 1;
    output.metrics[6] = 1.0;
    append_screen_event(
        output, check_structure, check_pass, duplicate_free ? 1.0 : 0.0);
    if (!duplicate_free) {
        reject_screen(output, screen_reason_structure, check_structure, 0.0);
        return output;
    }
    append_screen_event(output, check_capacity, check_pass, demand);

    auto min_slack = std::numeric_limits<double>::infinity();
    std::vector<double> earliest(node_count, 0.0);
    std::vector<bool> has_earliest(node_count, false);
    if (use_incremental) {
        current_time = incremental[3];
        min_slack = incremental[2];
        if (incremental[4] < 0.5) {
            output.metrics[1] = current_time;
            output.metrics[2] = min_slack;
            reject_screen(output, screen_reason_forward, check_forward, min_slack);
            return output;
        }
    } else {
        auto origin = depot;
        for (std::size_t position = 0; position <= route_size; ++position) {
            const auto destination = position < route_size ? route[position] : depot;
            current_time += distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] / vehicle[4];
            current_time = std::max(current_time, ready[destination]);
            if (kinds[destination] == customer_kind) {
                earliest[static_cast<std::size_t>(destination)] = current_time;
                has_earliest[static_cast<std::size_t>(destination)] = true;
                const auto slack = due[destination] - current_time;
                min_slack = std::min(min_slack, slack);
                if (slack < -epsilon) {
                    output.metrics[1] = current_time;
                    output.metrics[2] = slack;
                    reject_screen(output, screen_reason_forward, check_forward, slack);
                    return output;
                }
                current_time += service[destination];
            } else if (kinds[destination] == depot_kind) {
                const auto slack = due[destination] - current_time;
                min_slack = std::min(min_slack, slack);
                if (slack < -epsilon) {
                    output.metrics[1] = current_time;
                    output.metrics[2] = slack;
                    reject_screen(output, screen_reason_forward, check_forward, slack);
                    return output;
                }
            }
            origin = destination;
        }
    }
    append_screen_event(output, check_forward, check_pass, current_time);

    if (use_incremental) {
        if (incremental[5] < 0.5) {
            output.metrics[1] = current_time;
            output.metrics[2] = min_slack;
            reject_screen(output, screen_reason_backward, check_backward, min_slack);
            return output;
        }
    } else {
        auto latest_departure = due[depot];
        std::vector<double> latest(node_count, 0.0);
        std::vector<bool> has_latest(node_count, false);
        for (std::size_t reverse = route_size + 1; reverse-- > 0;) {
            const auto origin = reverse == 0 ? depot : route[reverse - 1];
            const auto destination = reverse < route_size ? route[reverse] : depot;
            double latest_arrival = 0.0;
            if (kinds[destination] == customer_kind) {
                latest_arrival = std::min(
                    due[destination], latest_departure - service[destination]);
                latest[static_cast<std::size_t>(destination)] = latest_arrival;
                has_latest[static_cast<std::size_t>(destination)] = true;
            } else {
                latest_arrival = std::min(due[destination], latest_departure);
            }
            latest_departure = latest_arrival - distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] / vehicle[4];
        }
        for (std::size_t node = 0; node < node_count; ++node) {
            if (has_earliest[node] && has_latest[node]) {
                const auto slack = latest[node] - earliest[node];
                min_slack = std::min(min_slack, slack);
                if (slack < -epsilon) {
                    output.metrics[1] = current_time;
                    output.metrics[2] = slack;
                    reject_screen(output, screen_reason_backward, check_backward, slack);
                    return output;
                }
            }
        }
    }
    const auto reported_slack = std::isfinite(min_slack) ? min_slack : 0.0;
    append_screen_event(output, check_backward, check_pass, reported_slack);
    if (min_slack < -epsilon) {
        output.metrics[1] = current_time;
        output.metrics[2] = min_slack;
        reject_screen(output, screen_reason_slack, check_slack, min_slack);
        return output;
    }
    append_screen_event(output, check_slack, check_pass, reported_slack);
    append_screen_event(output, check_distance, check_recorded, distance_lower_bound);

    auto origin = depot;
    for (std::size_t position = 0; position <= route_size; ++position) {
        const auto destination = position < route_size ? route[position] : depot;
        ++output.reachability_queries;
        if (reachable[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] == 0U) {
            output.codes[3] = 0;
            output.codes[4] = 0;
            output.metrics[1] = current_time;
            output.metrics[2] = reported_slack;
            output.metrics[6] = 0.0;
            reject_screen(output, screen_reason_energy, check_energy, 0.0);
            return output;
        }
        origin = destination;
    }
    output.codes[3] = 1;
    output.codes[4] = 1;
    append_screen_event(output, check_energy, check_pass, 1.0);

    double structural_energy = 0.0;
    for (std::size_t position = 0; position < route_size; ++position) {
        const auto customer = route[position];
        auto to_customer = std::numeric_limits<double>::infinity();
        auto from_customer = std::numeric_limits<double>::infinity();
        for (const auto recharge : recharge_nodes) {
            to_customer = std::min(
                to_customer,
                distances[static_cast<std::size_t>(recharge) * node_count
                    + static_cast<std::size_t>(customer)]);
            from_customer = std::min(
                from_customer,
                distances[static_cast<std::size_t>(customer) * node_count
                    + static_cast<std::size_t>(recharge)]);
        }
        structural_energy = std::max(
            structural_energy, (to_customer + from_customer) * vehicle[2]);
    }
    output.metrics[5] = structural_energy;
    if (structural_energy > vehicle[0] + epsilon) {
        output.metrics[1] = current_time;
        output.metrics[2] = reported_slack;
        reject_screen(
            output,
            screen_reason_structural_energy,
            check_structural_energy,
            structural_energy);
        return output;
    }
    append_screen_event(output, check_structural_energy, check_pass, structural_energy);
    output.codes[0] = 1;
    output.codes[1] = screen_reason_none;
    output.codes[2] = 0;
    output.metrics[1] = current_time;
    output.metrics[2] = reported_slack;
    output.metrics[6] = 1.0;
    return output;
}

}  // namespace evrptw::native_kernels

