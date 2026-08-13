#pragma once

#include <algorithm>
#include <array>
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
#include <type_traits>
#include <utility>
#include <vector>

namespace txnopt::native {

class NativeWorkPool final {
public:
    static constexpr std::size_t latency_histogram_bins = 32;
    static constexpr std::size_t maximum_task_receipts = 65'536;

    struct TaskReceipt final {
        std::uint64_t task_sequence = 0;
        std::int64_t worker_index = -1;
        std::size_t first_index = 0;
        std::size_t last_index = 0;
        std::uint64_t submitted_nanoseconds = 0;
        std::uint64_t started_nanoseconds = 0;
        std::uint64_t completed_nanoseconds = 0;
    };

    using TaskReceiptSink = std::function<void(const TaskReceipt&)>;
    using TaskReceiptFlush = std::function<void()>;

    struct Statistics final {
        std::size_t pending_tasks = 0;
        std::size_t active_tasks = 0;
        std::size_t peak_pending_tasks = 0;
        std::size_t peak_active_tasks = 0;
        std::size_t queue_full_count = 0;
        std::size_t rejected_count = 0;
        std::size_t completed_tasks = 0;
        double total_wait_seconds = 0.0;
        double maximum_wait_seconds = 0.0;
        double total_service_seconds = 0.0;
        double maximum_service_seconds = 0.0;
        std::array<std::size_t, latency_histogram_bins> wait_histogram{};
        std::array<std::size_t, latency_histogram_bins> service_histogram{};
        std::size_t task_receipt_dropped_count = 0;
        std::vector<TaskReceipt> task_receipts;
    };

    explicit NativeWorkPool(
        std::int64_t thread_count,
        std::size_t maximum_pending_tasks = 0,
        TaskReceiptSink task_receipt_sink = {},
        TaskReceiptFlush task_receipt_flush = {})
        : task_receipt_sink_(std::move(task_receipt_sink)),
          task_receipt_flush_(std::move(task_receipt_flush)) {
        if (static_cast<bool>(task_receipt_sink_)
            != static_cast<bool>(task_receipt_flush_)) {
            throw std::invalid_argument(
                "native work-pool task receipt sink/flush must be paired");
        }
        if (thread_count < 0) {
            throw std::invalid_argument(
                "native work-pool thread count must be non-negative");
        }
        const auto thread_count_size = static_cast<std::size_t>(thread_count);
        maximum_pending_tasks_ = maximum_pending_tasks != 0
            ? maximum_pending_tasks
            : std::max<std::size_t>(1, thread_count_size * 2);
        threads_.reserve(thread_count_size);
        try {
            for (std::int64_t index = 0; index < thread_count; ++index) {
                threads_.emplace_back([this, index]() { worker_loop(index); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            space_.notify_all();
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
        space_.notify_all();
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::int64_t thread_count() const noexcept {
        return static_cast<std::int64_t>(threads_.size());
    }

    static std::int64_t current_worker_index() noexcept {
        return current_worker_index_;
    }

    std::size_t maximum_pending_task_count() const noexcept {
        return maximum_pending_tasks_;
    }

    std::size_t pending_task_count() const noexcept {
        std::lock_guard lock(mutex_);
        return tasks_.size();
    }

    std::size_t active_task_count() const noexcept {
        return active_tasks_.load(std::memory_order_relaxed);
    }

    std::size_t peak_active_task_count() const noexcept {
        return peak_active_tasks_.load(std::memory_order_relaxed);
    }

    void wait_until_idle() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [&]() {
            return tasks_.empty()
                && active_tasks_.load(std::memory_order_acquire) == 0;
        });
        lock.unlock();
        flush_task_receipts();
        rethrow_task_receipt_error();
    }

    [[nodiscard]] Statistics statistics() const noexcept {
        Statistics output;
        {
            std::lock_guard lock(mutex_);
            output.pending_tasks = tasks_.size();
            output.task_receipts.assign(
                task_receipts_.begin(), task_receipts_.end());
        }
        output.task_receipt_dropped_count = task_receipt_dropped_count_.load(
            std::memory_order_relaxed);
        output.active_tasks = active_tasks_.load(std::memory_order_relaxed);
        output.peak_pending_tasks = peak_pending_tasks_.load(
            std::memory_order_relaxed);
        output.peak_active_tasks = peak_active_tasks_.load(
            std::memory_order_relaxed);
        output.queue_full_count = queue_full_count_.load(
            std::memory_order_relaxed);
        output.rejected_count = rejected_count_.load(
            std::memory_order_relaxed);
        output.completed_tasks = completed_tasks_.load(
            std::memory_order_relaxed);
        output.total_wait_seconds = nanoseconds_to_seconds(
            total_wait_nanoseconds_.load(std::memory_order_relaxed));
        output.maximum_wait_seconds = nanoseconds_to_seconds(
            maximum_wait_nanoseconds_.load(std::memory_order_relaxed));
        output.total_service_seconds = nanoseconds_to_seconds(
            total_service_nanoseconds_.load(std::memory_order_relaxed));
        output.maximum_service_seconds = nanoseconds_to_seconds(
            maximum_service_nanoseconds_.load(std::memory_order_relaxed));
        for (std::size_t index = 0; index < latency_histogram_bins; ++index) {
            output.wait_histogram[index] = wait_histogram_[index].load(
                std::memory_order_relaxed);
            output.service_histogram[index] = service_histogram_[index].load(
                std::memory_order_relaxed);
        }
        return output;
    }

    template <typename Function>
    void parallel_for(std::size_t count, Function function) {
        rethrow_task_receipt_error();
        if (count == 0) {
            flush_task_receipts();
            return;
        }
        if (threads_.empty()) {
            rejected_count_.fetch_add(1, std::memory_order_relaxed);
            throw std::runtime_error(
                "disabled native work pool cannot execute local work");
        }

        // A worker that calls back into the same pool must not wait for a
        // queue slot: all workers could otherwise be waiting on one another.
        // This is also the safe path for nested candidate screening.
        if (current_pool_ == this) {
            execute_inline(count, std::forward<Function>(function));
            return;
        }

        struct Completion final {
            std::mutex mutex;
            std::condition_variable ready;
            std::size_t remaining = 0;
            std::exception_ptr error;
        };
        const auto chunk_count = std::min<std::size_t>(
            count, threads_.size());
        const auto base_chunk_size = count / chunk_count;
        const auto remainder = count % chunk_count;
        auto completion = std::make_shared<Completion>();
        completion->remaining = chunk_count;
        auto shared_function = std::make_shared<std::decay_t<Function>>(
            std::forward<Function>(function));
        std::vector<Task> tasks;
        tasks.reserve(chunk_count);
        std::size_t next_index = 0;
        for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
            const auto first = next_index;
            const auto size = base_chunk_size + (chunk < remainder ? 1 : 0);
            const auto last = first + size;
            next_index = last;
            Task task;
            task.submitted_at = std::chrono::steady_clock::now();
            task.task_sequence = next_task_sequence_.fetch_add(
                1, std::memory_order_relaxed);
            task.first_index = first;
            task.last_index = last;
            task.function = [completion, shared_function, first, last]()
                mutable noexcept {
                std::exception_ptr error;
                for (auto index = first; index < last; ++index) {
                    try {
                        (*shared_function)(index);
                    } catch (...) {
                        if (!error) {
                            error = std::current_exception();
                        }
                    }
                }
                if (error) {
                    std::lock_guard completion_lock(completion->mutex);
                    if (!completion->error) {
                        completion->error = error;
                    }
                }
            };
            task.completion = [completion](std::exception_ptr error) noexcept {
                {
                    std::lock_guard completion_lock(completion->mutex);
                    if (error && !completion->error) {
                        completion->error = std::move(error);
                    }
                    --completion->remaining;
                }
                completion->ready.notify_one();
            };
            tasks.push_back(std::move(task));
        }
        if (!enqueue_all(std::move(tasks))) {
            {
                std::lock_guard completion_lock(completion->mutex);
                completion->remaining = 0;
                completion->error = std::make_exception_ptr(
                    std::runtime_error("native work pool is stopping"));
            }
            completion->ready.notify_one();
        }
        ready_.notify_all();
        std::unique_lock completion_lock(completion->mutex);
        completion->ready.wait(
            completion_lock,
            [&]() { return completion->remaining == 0; });
        auto error = completion->error;
        completion_lock.unlock();
        try {
            flush_task_receipts();
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
        }
        if (!error) {
            error = task_receipt_error();
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    struct Task final {
        std::function<void()> function;
        std::function<void(std::exception_ptr)> completion;
        std::chrono::steady_clock::time_point submitted_at;
        std::uint64_t task_sequence = 0;
        std::size_t first_index = 0;
        std::size_t last_index = 0;
    };

    static double nanoseconds_to_seconds(std::uint64_t value) noexcept {
        return static_cast<double>(value) / 1.0e9;
    }

    static std::uint64_t elapsed_nanoseconds(
        const std::chrono::steady_clock::duration duration) noexcept {
        const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
            duration).count();
        if (count <= 0) {
            return 0;
        }
        return static_cast<std::uint64_t>(count);
    }

    std::uint64_t timestamp_nanoseconds(
        const std::chrono::steady_clock::time_point value) const noexcept {
        return elapsed_nanoseconds(value - epoch_);
    }

    static std::size_t histogram_bucket(
        const std::uint64_t nanoseconds) noexcept {
        auto micros = nanoseconds / 1'000;
        std::size_t bucket = 0;
        while (micros > 1 && bucket + 1 < latency_histogram_bins) {
            micros >>= 1;
            ++bucket;
        }
        return bucket;
    }

    static void update_max(
        std::atomic<std::uint64_t>& target,
        const std::uint64_t value) noexcept {
        auto current = target.load(std::memory_order_relaxed);
        while (value > current
               && !target.compare_exchange_weak(
                   current, value, std::memory_order_relaxed)) {
        }
    }

    void record_wait(const std::uint64_t nanoseconds) noexcept {
        total_wait_nanoseconds_.fetch_add(
            nanoseconds, std::memory_order_relaxed);
        update_max(maximum_wait_nanoseconds_, nanoseconds);
        wait_histogram_[histogram_bucket(nanoseconds)].fetch_add(
            1, std::memory_order_relaxed);
    }

    void record_service(const std::uint64_t nanoseconds) noexcept {
        total_service_nanoseconds_.fetch_add(
            nanoseconds, std::memory_order_relaxed);
        update_max(maximum_service_nanoseconds_, nanoseconds);
        service_histogram_[histogram_bucket(nanoseconds)].fetch_add(
            1, std::memory_order_relaxed);
        completed_tasks_.fetch_add(1, std::memory_order_relaxed);
    }

    std::exception_ptr record_task_receipt(TaskReceipt receipt) noexcept {
        if (task_receipt_sink_) {
            try {
                task_receipt_sink_(receipt);
            } catch (...) {
                task_receipt_dropped_count_.fetch_add(
                    1, std::memory_order_relaxed);
                const auto error = std::current_exception();
                remember_task_receipt_error(error);
                return error;
            }
            return {};
        }
        try {
            std::lock_guard lock(mutex_);
            if (task_receipts_.size() == maximum_task_receipts) {
                throw std::runtime_error(
                    "native work-pool task receipt capacity exceeded");
            }
            task_receipts_.push_back(std::move(receipt));
        } catch (...) {
            task_receipt_dropped_count_.fetch_add(
                1, std::memory_order_relaxed);
            const auto error = std::current_exception();
            remember_task_receipt_error(error);
            return error;
        }
        return {};
    }

    void remember_task_receipt_error(std::exception_ptr error) noexcept {
        if (!error) {
            return;
        }
        try {
            std::lock_guard lock(task_receipt_error_mutex_);
            if (!task_receipt_error_) {
                task_receipt_error_ = std::move(error);
            }
        } catch (...) {
            std::terminate();
        }
    }

    [[nodiscard]] std::exception_ptr task_receipt_error() const noexcept {
        try {
            std::lock_guard lock(task_receipt_error_mutex_);
            return task_receipt_error_;
        } catch (...) {
            std::terminate();
        }
    }

    void rethrow_task_receipt_error() const {
        if (const auto error = task_receipt_error()) {
            std::rethrow_exception(error);
        }
    }

    void flush_task_receipts() {
        if (!task_receipt_flush_) {
            return;
        }
        try {
            task_receipt_flush_();
        } catch (...) {
            const auto error = std::current_exception();
            remember_task_receipt_error(error);
            std::rethrow_exception(error);
        }
    }

    void update_peak_pending(const std::size_t pending) noexcept {
        auto peak = peak_pending_tasks_.load(std::memory_order_relaxed);
        while (pending > peak
               && !peak_pending_tasks_.compare_exchange_weak(
                   peak, pending, std::memory_order_relaxed)) {
        }
    }

    bool enqueue_all(std::vector<Task> tasks) {
        if (tasks.empty() || tasks.size() > maximum_pending_tasks_) {
            throw std::invalid_argument(
                "native work-pool batch exceeds the bounded queue");
        }
        std::unique_lock lock(mutex_);
        while (tasks_.size() + tasks.size() > maximum_pending_tasks_
               && !stopping_) {
            queue_full_count_.fetch_add(1, std::memory_order_relaxed);
            space_.wait(lock);
        }
        if (stopping_) {
            rejected_count_.fetch_add(tasks.size(), std::memory_order_relaxed);
            return false;
        }
        for (auto& task : tasks) {
            tasks_.push_back(std::move(task));
        }
        update_peak_pending(tasks_.size());
        return true;
    }

    template <typename Function>
    void execute_inline(const std::size_t count, Function function) {
        std::exception_ptr error;
        const auto submitted = std::chrono::steady_clock::now();
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < count; ++index) {
            try {
                function(index);
            } catch (...) {
                if (!error) {
                    error = std::current_exception();
                }
            }
        }
        const auto completed = std::chrono::steady_clock::now();
        record_service(elapsed_nanoseconds(completed - started));
        const auto receipt_error = record_task_receipt(TaskReceipt{
            next_task_sequence_.fetch_add(1, std::memory_order_relaxed),
            current_worker_index_,
            0,
            count,
            timestamp_nanoseconds(submitted),
            timestamp_nanoseconds(started),
            timestamp_nanoseconds(completed),
        });
        if (error) {
            std::rethrow_exception(error);
        }
        if (receipt_error) {
            std::rethrow_exception(receipt_error);
        }
        flush_task_receipts();
    }

    void worker_loop(const std::int64_t worker_index) noexcept {
        current_pool_ = this;
        current_worker_index_ = worker_index;
        while (true) {
            Task task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&]() { return stopping_ || !tasks_.empty(); });
                if (tasks_.empty()) {
                    if (stopping_) {
                        current_pool_ = nullptr;
                        current_worker_index_ = -1;
                        return;
                    }
                    continue;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
                space_.notify_one();
            }
            const auto now = std::chrono::steady_clock::now();
            record_wait(elapsed_nanoseconds(now - task.submitted_at));
            const auto active = active_tasks_.fetch_add(
                1, std::memory_order_relaxed) + 1;
            auto peak = peak_active_tasks_.load(std::memory_order_relaxed);
            while (active > peak
                   && !peak_active_tasks_.compare_exchange_weak(
                       peak, active, std::memory_order_relaxed)) {
            }
            const auto started = std::chrono::steady_clock::now();
            task.function();
            const auto completed = std::chrono::steady_clock::now();
            record_service(elapsed_nanoseconds(completed - started));
            const auto receipt_error = record_task_receipt(TaskReceipt{
                task.task_sequence,
                worker_index,
                task.first_index,
                task.last_index,
                timestamp_nanoseconds(task.submitted_at),
                timestamp_nanoseconds(started),
                timestamp_nanoseconds(completed),
            });
            active_tasks_.fetch_sub(1, std::memory_order_release);
            idle_.notify_all();
            task.completion(receipt_error);
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable space_;
    std::condition_variable idle_;
    std::deque<Task> tasks_;
    std::deque<TaskReceipt> task_receipts_;
    TaskReceiptSink task_receipt_sink_;
    TaskReceiptFlush task_receipt_flush_;
    mutable std::mutex task_receipt_error_mutex_;
    std::exception_ptr task_receipt_error_;
    std::vector<std::thread> threads_;
    const std::chrono::steady_clock::time_point epoch_ =
        std::chrono::steady_clock::now();
    std::size_t maximum_pending_tasks_ = 1;
    std::atomic<std::size_t> active_tasks_{0};
    std::atomic<std::size_t> peak_active_tasks_{0};
    std::atomic<std::size_t> peak_pending_tasks_{0};
    std::atomic<std::size_t> queue_full_count_{0};
    std::atomic<std::size_t> rejected_count_{0};
    std::atomic<std::size_t> completed_tasks_{0};
    std::atomic<std::uint64_t> total_wait_nanoseconds_{0};
    std::atomic<std::uint64_t> maximum_wait_nanoseconds_{0};
    std::atomic<std::uint64_t> total_service_nanoseconds_{0};
    std::atomic<std::uint64_t> maximum_service_nanoseconds_{0};
    std::atomic<std::uint64_t> next_task_sequence_{0};
    std::array<std::atomic<std::size_t>, latency_histogram_bins> wait_histogram_{};
    std::array<std::atomic<std::size_t>, latency_histogram_bins> service_histogram_{};
    std::atomic<std::size_t> task_receipt_dropped_count_{0};
    bool stopping_ = false;

    inline static thread_local NativeWorkPool* current_pool_ = nullptr;
    inline static thread_local std::int64_t current_worker_index_ = -1;
};

struct NativeQueuedRequest final {
    int descriptor;
    std::chrono::steady_clock::time_point submitted_at;
    std::size_t queue_depth_on_submit;
};

class NativeRequestQueue final {
public:
    static constexpr std::size_t maximum_pending_requests = 64;

    struct Statistics final {
        std::size_t pending_requests = 0;
        std::size_t peak_pending_requests = 0;
        std::size_t queue_full_count = 0;
        std::size_t rejected_count = 0;
        std::size_t completed_requests = 0;
        double total_wait_seconds = 0.0;
        double maximum_wait_seconds = 0.0;
        double total_service_seconds = 0.0;
        double maximum_service_seconds = 0.0;
        std::array<std::size_t, NativeWorkPool::latency_histogram_bins>
            wait_histogram{};
        std::array<std::size_t, NativeWorkPool::latency_histogram_bins>
            service_histogram{};
    };

    NativeRequestQueue() = default;
    NativeRequestQueue(const NativeRequestQueue&) = delete;
    NativeRequestQueue& operator=(const NativeRequestQueue&) = delete;

    bool submit(int descriptor) {
        {
            std::lock_guard lock(mutex_);
            if (stopping_) {
                ++rejected_count_;
                throw std::runtime_error("native request queue is stopping");
            }
            if (descriptors_.size() >= maximum_pending_requests) {
                ++queue_full_count_;
                ++rejected_count_;
                return false;
            }
            descriptors_.push_back(NativeQueuedRequest{
                descriptor,
                std::chrono::steady_clock::now(),
                descriptors_.size() + 1,
            });
            peak_pending_requests_ = std::max(
                peak_pending_requests_, descriptors_.size());
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
        const auto wait_nanoseconds = elapsed_nanoseconds(
            std::chrono::steady_clock::now() - descriptor.submitted_at);
        total_wait_nanoseconds_ += wait_nanoseconds;
        maximum_wait_nanoseconds_ = std::max(
            maximum_wait_nanoseconds_, wait_nanoseconds);
        ++wait_histogram_[histogram_bucket(wait_nanoseconds)];
        return descriptor;
    }

    void record_service(
        const std::chrono::steady_clock::duration duration) noexcept {
        const auto service_nanoseconds = elapsed_nanoseconds(duration);
        std::lock_guard lock(mutex_);
        total_service_nanoseconds_ += service_nanoseconds;
        maximum_service_nanoseconds_ = std::max(
            maximum_service_nanoseconds_, service_nanoseconds);
        ++service_histogram_[histogram_bucket(service_nanoseconds)];
        ++completed_requests_;
    }

    void stop() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
    }

    [[nodiscard]] Statistics statistics() const noexcept {
        std::lock_guard lock(mutex_);
        Statistics output;
        output.pending_requests = descriptors_.size();
        output.peak_pending_requests = peak_pending_requests_;
        output.queue_full_count = queue_full_count_;
        output.rejected_count = rejected_count_;
        output.completed_requests = completed_requests_;
        output.total_wait_seconds = nanoseconds_to_seconds(
            total_wait_nanoseconds_);
        output.maximum_wait_seconds = nanoseconds_to_seconds(
            maximum_wait_nanoseconds_);
        output.total_service_seconds = nanoseconds_to_seconds(
            total_service_nanoseconds_);
        output.maximum_service_seconds = nanoseconds_to_seconds(
            maximum_service_nanoseconds_);
        output.wait_histogram = wait_histogram_;
        output.service_histogram = service_histogram_;
        return output;
    }

private:
    static double nanoseconds_to_seconds(std::uint64_t value) noexcept {
        return static_cast<double>(value) / 1.0e9;
    }

    static std::uint64_t elapsed_nanoseconds(
        const std::chrono::steady_clock::duration duration) noexcept {
        const auto count = std::chrono::duration_cast<std::chrono::nanoseconds>(
            duration).count();
        return count <= 0 ? 0U : static_cast<std::uint64_t>(count);
    }

    static std::size_t histogram_bucket(
        const std::uint64_t nanoseconds) noexcept {
        auto micros = nanoseconds / 1'000;
        std::size_t bucket = 0;
        while (
            micros > 1
            && bucket + 1 < NativeWorkPool::latency_histogram_bins) {
            micros >>= 1;
            ++bucket;
        }
        return bucket;
    }

    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<NativeQueuedRequest> descriptors_;
    std::size_t peak_pending_requests_ = 0;
    std::size_t queue_full_count_ = 0;
    std::size_t rejected_count_ = 0;
    std::size_t completed_requests_ = 0;
    std::uint64_t total_wait_nanoseconds_ = 0;
    std::uint64_t maximum_wait_nanoseconds_ = 0;
    std::uint64_t total_service_nanoseconds_ = 0;
    std::uint64_t maximum_service_nanoseconds_ = 0;
    std::array<std::size_t, NativeWorkPool::latency_histogram_bins>
        wait_histogram_{};
    std::array<std::size_t, NativeWorkPool::latency_histogram_bins>
        service_histogram_{};
    bool stopping_ = false;
};

}  // namespace txnopt::native
