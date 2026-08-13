#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "concurrency.hpp"

int main() {
    constexpr std::size_t item_count = 64;
    constexpr std::size_t rounds = 20'000;
    txnopt::native::NativeWorkPool pool(4, 8);
    for (std::size_t round = 0; round < rounds; ++round) {
        std::vector<std::uint64_t> values(item_count, 0);
        std::atomic<std::size_t> nested_count{0};
        pool.parallel_for(item_count, [&](const std::size_t index) {
            values[index] = static_cast<std::uint64_t>((round + 1) * (index + 1));
            if (index == 0) {
                pool.parallel_for(32, [&](const std::size_t) {
                    nested_count.fetch_add(1, std::memory_order_relaxed);
                });
            }
        });
        pool.wait_until_idle();
        const auto receipts = pool.consume_task_receipts();
        if (receipts.empty()) {
            throw std::runtime_error("parallel work-pool emitted no task receipts");
        }
        if (nested_count.load(std::memory_order_relaxed) != 32) {
            throw std::runtime_error("nested work-pool execution was incomplete");
        }
        for (std::size_t index = 0; index < item_count; ++index) {
            const auto expected = static_cast<std::uint64_t>((round + 1) * (index + 1));
            if (values[index] != expected) {
                throw std::runtime_error("parallel work-pool result mismatch");
            }
        }
        const auto statistics = pool.statistics();
        if (statistics.pending_tasks != 0 || statistics.active_tasks != 0
            || !statistics.task_receipts.empty()
            || statistics.completed_tasks == 0 || statistics.rejected_count != 0) {
            throw std::runtime_error("parallel work-pool accounting mismatch");
        }
    }
    return 0;
}
