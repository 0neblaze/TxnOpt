#include "native_concurrency.hpp"
#include "native_exact_parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

int main() {
    constexpr std::int64_t compute_threads = 24;
    constexpr std::size_t request_threads = 6;
    constexpr std::size_t request_count = 600;
    constexpr std::size_t work_per_request = 64;

    NativeWorkPool work_pool(compute_threads);
    NativeRequestQueue requests;
    std::vector<std::atomic<std::size_t>> completions(request_count);
    for (auto& completion : completions) {
        completion.store(0, std::memory_order_relaxed);
    }
    std::atomic<std::size_t> completed_requests{0};
    std::vector<std::thread> consumers;
    consumers.reserve(request_threads);
    for (std::size_t worker = 0; worker < request_threads; ++worker) {
        consumers.emplace_back([&]() {
            while (const auto request = requests.take()) {
                const auto request_id = static_cast<std::size_t>(
                    request->descriptor);
                work_pool.parallel_for(work_per_request, [&](std::size_t) {
                    completions[request_id].fetch_add(
                        1, std::memory_order_relaxed);
                });
                completed_requests.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (std::size_t request = 0; request < request_count; ++request) {
        while (!requests.submit(static_cast<int>(request))) {
            std::this_thread::yield();
        }
    }
    requests.stop();
    for (auto& consumer : consumers) {
        consumer.join();
    }
    if (work_pool.thread_count() != compute_threads
        || completed_requests.load(std::memory_order_relaxed) != request_count) {
        std::cerr << "native concurrency topology/count mismatch\n";
        return 1;
    }
    for (const auto& completion : completions) {
        if (completion.load(std::memory_order_relaxed) != work_per_request) {
            std::cerr << "native concurrency request isolation mismatch\n";
            return 1;
        }
    }
    const auto statistics = work_pool.statistics();
    for (const auto& receipt : statistics.task_receipts) {
        if (receipt.first_index >= receipt.last_index) {
            std::cerr << "native concurrency emitted an empty task chunk\n";
            return 1;
        }
    }
    bool propagated = false;
    try {
        work_pool.parallel_for(16, [](std::size_t index) {
            if (index == 7) {
                throw std::runtime_error("injected worker failure");
            }
        });
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    if (!propagated) {
        std::cerr << "native concurrency worker failure was swallowed\n";
        return 1;
    }

    NativeWorkPool balanced_pool(4);
    balanced_pool.parallel_for(5, [](std::size_t) {});
    auto balanced_statistics = balanced_pool.statistics();
    if (balanced_statistics.completed_tasks != 4
        || balanced_statistics.task_receipts.size() != 4) {
        std::cerr << "native concurrency did not use every available worker chunk\n";
        return 1;
    }
    std::sort(
        balanced_statistics.task_receipts.begin(),
        balanced_statistics.task_receipts.end(),
        [](const auto& first, const auto& second) {
            return first.first_index < second.first_index;
        });
    std::size_t covered = 0;
    for (const auto& receipt : balanced_statistics.task_receipts) {
        if (receipt.first_index != covered
            || receipt.first_index >= receipt.last_index) {
            std::cerr << "native concurrency balanced chunk order is invalid\n";
            return 1;
        }
        covered = receipt.last_index;
    }
    if (covered != 5) {
        std::cerr << "native concurrency balanced chunks do not cover the input\n";
        return 1;
    }

    std::vector<std::int64_t> exact_offsets(26, 0);
    const auto exact = evrptw::native_parallel::run_exact_charging_parallel(
        work_pool,
        exact_offsets.data(),
        nullptr,
        25,
        30.0,
        [](const std::int64_t*,
           const std::int64_t*,
           const std::size_t route_count,
           const double) {
            evrptw::native_kernels::ExactBatchOutput output;
            output.path_offsets.assign(route_count + 1, 0);
            output.statuses.assign(route_count, 0);
            output.reasons.assign(route_count, 0);
            output.metrics.assign(route_count * 4, 0.0);
            output.label_counters.assign(route_count * 3, 0);
            output.batch_counters.assign(10, 0);
            output.completion_rounds.assign(route_count, 0);
            for (std::size_t index = 0; index < route_count; ++index) {
                output.completion_order.push_back(
                    static_cast<std::int64_t>(index));
            }
            return output;
        });
    if (exact.physical_task_receipts.size() != 24 * 7) {
        std::cerr << "native exact N+1 batch did not use every worker chunk\n";
        return 1;
    }

    constexpr std::size_t streamed_requests = 1'400;
    constexpr std::size_t streamed_work = 64;
    constexpr std::size_t streamed_chunks_per_request = 24;
    std::atomic<std::size_t> streamed_receipts{0};
    std::atomic<bool> invalid_streamed_receipt{false};
    NativeWorkPool streamed_pool(
        compute_threads,
        0,
        [&](const NativeWorkPool::TaskReceipt& receipt) {
            if (receipt.first_index >= receipt.last_index
                || receipt.submitted_nanoseconds > receipt.started_nanoseconds
                || receipt.started_nanoseconds > receipt.completed_nanoseconds) {
                invalid_streamed_receipt.store(true, std::memory_order_relaxed);
            }
            streamed_receipts.fetch_add(1, std::memory_order_relaxed);
        },
        []() {});
    for (std::size_t request = 0; request < streamed_requests; ++request) {
        streamed_pool.parallel_for(streamed_work, [](std::size_t) {});
    }
    streamed_pool.wait_until_idle();
    const auto streamed_statistics = streamed_pool.statistics();
    const auto expected_streamed_receipts =
        streamed_requests * streamed_chunks_per_request;
    if (invalid_streamed_receipt.load(std::memory_order_relaxed)
        || streamed_receipts.load(std::memory_order_relaxed)
            != expected_streamed_receipts
        || streamed_statistics.completed_tasks != expected_streamed_receipts
        || streamed_statistics.task_receipt_dropped_count != 0
        || !streamed_statistics.task_receipts.empty()) {
        std::cerr << "native concurrency streamed task-receipt mismatch\n";
        return 1;
    }

    std::atomic<bool> sink_failed{false};
    NativeWorkPool failing_sink_pool(
        2,
        0,
        [&](const NativeWorkPool::TaskReceipt&) {
            sink_failed.store(true, std::memory_order_relaxed);
            throw std::runtime_error("injected task-receipt sink failure");
        },
        []() {});
    propagated = false;
    try {
        failing_sink_pool.parallel_for(4, [](std::size_t) {});
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    if (!propagated || !sink_failed.load(std::memory_order_relaxed)) {
        std::cerr << "native concurrency receipt failure was swallowed\n";
        return 1;
    }

    std::atomic<bool> flush_failed{false};
    NativeWorkPool failing_flush_pool(
        2,
        0,
        [](const NativeWorkPool::TaskReceipt&) {},
        [&]() {
            flush_failed.store(true, std::memory_order_relaxed);
            throw std::runtime_error("injected task-receipt flush failure");
        });
    propagated = false;
    try {
        failing_flush_pool.parallel_for(4, [](std::size_t) {});
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    if (!propagated || !flush_failed.load(std::memory_order_relaxed)) {
        std::cerr << "native concurrency receipt flush failure was swallowed\n";
        return 1;
    }
    std::cout << "stage052_native_concurrency_tsan=PASS\n";
    return 0;
}
