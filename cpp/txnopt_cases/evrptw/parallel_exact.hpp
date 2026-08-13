#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "../../txnopt_core/concurrency.hpp"
#include "exact_kernels.hpp"

namespace txnopt::cases::evrptw::native {

inline std::int64_t exact_elapsed_nanoseconds(
    const std::chrono::steady_clock::duration value) noexcept {
    const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
        value).count();
    return count <= 0 ? 0 : static_cast<std::int64_t>(count);
}

template <typename RunBatch>
ExactBatchOutput run_exact_charging_parallel(
    txnopt::native::NativeWorkPool& pool,
    const std::int64_t* offsets,
    const std::int64_t* indices,
    const std::size_t route_count,
    const double deadline_remaining,
    RunBatch&& run_batch) {
    if (offsets == nullptr || route_count == 0
        || offsets[0] != 0 || offsets[route_count] < 0) {
        throw std::invalid_argument(
            "native parallel exact route input is invalid");
    }
    if (pool.thread_count() <= 0) {
        throw std::invalid_argument(
            "native parallel exact work pool is disabled");
    }
    const auto started = std::chrono::steady_clock::now();

    const auto chunk_count = std::min<std::size_t>(
        route_count, static_cast<std::size_t>(pool.thread_count()));
    const auto base_chunk_size = route_count / chunk_count;
    const auto oversized_chunk_count = route_count % chunk_count;
    std::vector<ExactBatchOutput> chunk_outputs(
        chunk_count);
    std::vector<std::int64_t> chunk_workers(chunk_count, -1);
    std::vector<std::int64_t> chunk_started(chunk_count, -1);
    std::vector<std::int64_t> chunk_completed(chunk_count, -1);
    const auto chunk_bounds = [base_chunk_size, oversized_chunk_count](
                                  const std::size_t chunk) {
        const auto first = chunk * base_chunk_size
            + std::min(chunk, oversized_chunk_count);
        const auto size = base_chunk_size
            + (chunk < oversized_chunk_count ? 1U : 0U);
        return std::pair{first, first + size};
    };
    const auto submitted = exact_elapsed_nanoseconds(
        std::chrono::steady_clock::now() - started);
    pool.parallel_for(chunk_count, [&](const std::size_t chunk) {
        const auto [first, last] = chunk_bounds(chunk);
        chunk_workers[chunk] = txnopt::native::NativeWorkPool::current_worker_index();
        chunk_started[chunk] = exact_elapsed_nanoseconds(
            std::chrono::steady_clock::now() - started);
        std::vector<std::int64_t> chunk_offsets{0};
        std::vector<std::int64_t> chunk_indices;
        for (auto route = first; route < last; ++route) {
            const auto route_first = offsets[route];
            const auto route_last = offsets[route + 1];
            if (route_first < 0 || route_last < route_first) {
                throw std::runtime_error(
                    "native parallel exact route offsets are invalid");
            }
            chunk_indices.insert(
                chunk_indices.end(), indices + route_first,
                indices + route_last);
            chunk_offsets.push_back(
                static_cast<std::int64_t>(chunk_indices.size()));
        }
        const auto remaining = deadline_remaining
            - std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
        chunk_outputs[chunk] = run_batch(
            chunk_offsets.data(), chunk_indices.data(), last - first,
            remaining);
        chunk_completed[chunk] = exact_elapsed_nanoseconds(
            std::chrono::steady_clock::now() - started);
    });
    const auto pool_receipts = pool.consume_task_receipts();
    if (pool_receipts.size() != chunk_count) {
        throw std::runtime_error(
            "native parallel exact work-pool receipt count is invalid");
    }

    ExactBatchOutput output;
    output.path_offsets.push_back(0);
    output.statuses.reserve(route_count);
    output.reasons.reserve(route_count);
    output.metrics.reserve(route_count * 4);
    output.label_counters.reserve(route_count * 3);
    output.batch_counters.assign(10, 0);
    output.batch_counters[0] = static_cast<std::int64_t>(route_count);
    output.batch_counters[1] = static_cast<std::int64_t>(route_count);
    output.batch_counters[4] = 1;
    output.batch_counters[8] = 1;
    output.completion_rounds.assign(route_count, -1);
    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const auto [first, last] = chunk_bounds(chunk);
        const auto& child = chunk_outputs[chunk];
        const auto child_route_count = child.statuses.size();
        if (child_route_count != last - first
            || child.completion_rounds.size() != child_route_count
            || chunk_workers[chunk] < 0
            || chunk_started[chunk] < submitted
            || chunk_completed[chunk] < chunk_started[chunk]) {
            throw std::runtime_error(
                "native parallel exact child receipt is invalid");
        }
        const auto path_base = output.path_offsets.back();
        for (std::size_t row = 1; row < child.path_offsets.size(); ++row) {
            output.path_offsets.push_back(path_base + child.path_offsets[row]);
        }
        output.path_indices.insert(
            output.path_indices.end(), child.path_indices.begin(),
            child.path_indices.end());
        output.statuses.insert(
            output.statuses.end(), child.statuses.begin(),
            child.statuses.end());
        output.reasons.insert(
            output.reasons.end(), child.reasons.begin(),
            child.reasons.end());
        output.metrics.insert(
            output.metrics.end(), child.metrics.begin(), child.metrics.end());
        output.label_counters.insert(
            output.label_counters.end(), child.label_counters.begin(),
            child.label_counters.end());
        output.batch_counters[2] += child.batch_counters[2];
        output.batch_counters[3] += child.batch_counters[3];
        output.batch_counters[5] += child.batch_counters[5];
        output.batch_counters[6] += child.batch_counters[6];
        output.batch_counters[7] += child.batch_counters[7];
        std::copy(
            child.completion_rounds.begin(), child.completion_rounds.end(),
            output.completion_rounds.begin()
                + static_cast<std::ptrdiff_t>(first));
        output.physical_task_receipts.insert(
            output.physical_task_receipts.end(),
            {
                static_cast<std::int64_t>(chunk),
                chunk_workers[chunk],
                static_cast<std::int64_t>(first),
                static_cast<std::int64_t>(last),
                submitted,
                chunk_started[chunk],
                chunk_completed[chunk],
            });
    }
    output.batch_counters[9] = chunk_outputs.front().batch_counters[9];
    std::vector<std::size_t> routes_by_completion;
    routes_by_completion.reserve(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        if (output.completion_rounds[route] >= 0) {
            routes_by_completion.push_back(route);
        }
    }
    std::stable_sort(
        routes_by_completion.begin(), routes_by_completion.end(),
        [&](const auto left, const auto right) {
            return std::tie(output.completion_rounds[left], left)
                < std::tie(output.completion_rounds[right], right);
        });
    for (const auto route : routes_by_completion) {
        output.completion_order.push_back(static_cast<std::int64_t>(route));
    }

    std::vector<std::size_t> chunks_by_completion(chunk_count);
    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
        chunks_by_completion[chunk] = chunk;
    }
    std::stable_sort(
        chunks_by_completion.begin(), chunks_by_completion.end(),
        [&](const auto left, const auto right) {
            return std::tie(chunk_completed[left], left)
                < std::tie(chunk_completed[right], right);
        });
    for (const auto chunk : chunks_by_completion) {
        const auto [first, last] = chunk_bounds(chunk);
        static_cast<void>(last);
        for (const auto child_ordinal : chunk_outputs[chunk].completion_order) {
            output.physical_completion_order.push_back(
                static_cast<std::int64_t>(first) + child_ordinal);
        }
    }
    return output;
}

}  // namespace txnopt::cases::evrptw::native
