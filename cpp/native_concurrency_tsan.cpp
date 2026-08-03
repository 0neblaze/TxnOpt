#include "native_concurrency.hpp"

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
    std::cout << "stage052_native_concurrency_tsan=PASS\n";
    return 0;
}
