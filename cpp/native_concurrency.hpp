#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

class NativeWorkPool final {
public:
    explicit NativeWorkPool(std::int64_t thread_count) {
        if (thread_count < 0) {
            throw std::invalid_argument(
                "native work-pool thread count must be non-negative");
        }
        threads_.reserve(static_cast<std::size_t>(thread_count));
        try {
            for (std::int64_t index = 0; index < thread_count; ++index) {
                threads_.emplace_back([this]() { worker_loop(); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            for (auto& thread : threads_) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
            throw;
        }
    }

    NativeWorkPool(const NativeWorkPool&) = delete;
    NativeWorkPool& operator=(const NativeWorkPool&) = delete;

    ~NativeWorkPool() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::int64_t thread_count() const noexcept {
        return static_cast<std::int64_t>(threads_.size());
    }

    std::size_t active_task_count() const noexcept {
        return active_tasks_.load(std::memory_order_relaxed);
    }

    std::size_t peak_active_task_count() const noexcept {
        return peak_active_tasks_.load(std::memory_order_relaxed);
    }

    template <typename Function>
    void parallel_for(std::size_t count, Function function) {
        if (count == 0) {
            return;
        }
        if (threads_.empty()) {
            throw std::runtime_error(
                "disabled native work pool cannot execute local work");
        }
        struct Completion final {
            std::mutex mutex;
            std::condition_variable ready;
            std::size_t remaining;
            std::exception_ptr error;
        };
        auto completion = std::make_shared<Completion>();
        completion->remaining = count;
        {
            std::lock_guard lock(mutex_);
            if (stopping_) {
                throw std::runtime_error("native work pool is stopping");
            }
            for (std::size_t index = 0; index < count; ++index) {
                tasks_.emplace_back(
                    [completion, function, index]() mutable noexcept {
                        try {
                            function(index);
                        } catch (...) {
                            std::lock_guard completion_lock(completion->mutex);
                            if (!completion->error) {
                                completion->error = std::current_exception();
                            }
                        }
                        {
                            std::lock_guard completion_lock(completion->mutex);
                            --completion->remaining;
                        }
                        completion->ready.notify_one();
                    });
            }
        }
        ready_.notify_all();
        std::unique_lock completion_lock(completion->mutex);
        completion->ready.wait(
            completion_lock,
            [&]() { return completion->remaining == 0; });
        if (completion->error) {
            std::rethrow_exception(completion->error);
        }
    }

private:
    void worker_loop() noexcept {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&]() { return stopping_ || !tasks_.empty(); });
                if (tasks_.empty()) {
                    if (stopping_) {
                        return;
                    }
                    continue;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            const auto active = active_tasks_.fetch_add(
                1, std::memory_order_relaxed) + 1;
            auto peak = peak_active_tasks_.load(std::memory_order_relaxed);
            while (active > peak
                   && !peak_active_tasks_.compare_exchange_weak(
                       peak, active, std::memory_order_relaxed)) {
            }
            task();
            active_tasks_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread> threads_;
    std::atomic<std::size_t> active_tasks_{0};
    std::atomic<std::size_t> peak_active_tasks_{0};
    bool stopping_ = false;
};

struct NativeQueuedRequest final {
    int descriptor;
    std::chrono::steady_clock::time_point submitted_at;
    std::size_t queue_depth_on_submit;
};

class NativeRequestQueue final {
public:
    static constexpr std::size_t maximum_pending_requests = 64;

    NativeRequestQueue() = default;
    NativeRequestQueue(const NativeRequestQueue&) = delete;
    NativeRequestQueue& operator=(const NativeRequestQueue&) = delete;

    bool submit(int descriptor) {
        {
            std::lock_guard lock(mutex_);
            if (stopping_) {
                throw std::runtime_error("native request queue is stopping");
            }
            if (descriptors_.size() >= maximum_pending_requests) {
                return false;
            }
            descriptors_.push_back(NativeQueuedRequest{
                descriptor,
                std::chrono::steady_clock::now(),
                descriptors_.size() + 1,
            });
        }
        ready_.notify_one();
        return true;
    }

    std::optional<NativeQueuedRequest> take() {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&]() { return stopping_ || !descriptors_.empty(); });
        if (descriptors_.empty()) {
            return std::nullopt;
        }
        auto descriptor = descriptors_.front();
        descriptors_.pop_front();
        return descriptor;
    }

    void stop() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<NativeQueuedRequest> descriptors_;
    bool stopping_ = false;
};
