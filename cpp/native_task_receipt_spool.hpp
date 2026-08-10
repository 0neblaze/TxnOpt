#pragma once

#ifdef __linux__

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "native_concurrency.hpp"
#include "native_sha256.hpp"

namespace evrptw::native_telemetry {

struct TaskReceiptFileDescriptor final {
    std::string path;
    std::string sha256;
    std::size_t bytes = 0;
    std::size_t count = 0;
    std::size_t receipt_batch_capacity = 0;
    std::size_t queue_bound_batches = 0;
    std::size_t peak_queued_batches = 0;
    std::size_t submitted_batches = 0;
    std::size_t completed_batches = 0;
    std::size_t dropped_count = 0;
    double producer_wait_seconds = 0.0;
    double writer_wall_seconds = 0.0;
    double writer_cpu_seconds = 0.0;
    double serialization_seconds = 0.0;
    double write_seconds = 0.0;
    double file_fsync_seconds = 0.0;
    double atomic_publish_seconds = 0.0;
    double parent_fsync_seconds = 0.0;
};

class TaskReceiptSpool final {
public:
    static constexpr std::size_t receipt_batch_capacity = 4'096;
    static constexpr std::size_t queue_bound_batches = 1;

    explicit TaskReceiptSpool(std::filesystem::path target)
        : target_(std::move(target)),
          parent_(target_.parent_path().empty()
              ? std::filesystem::path(".")
              : target_.parent_path()),
          temporary_(target_.string() + ".tmp-" + std::to_string(::getpid())) {
        const auto filename = target_.filename().string();
        if (filename.empty()
            || filename.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
                != std::string::npos) {
            throw std::invalid_argument(
                "native task-receipt filename is invalid");
        }
        if (!std::filesystem::is_directory(parent_)
            || std::filesystem::exists(target_)
            || std::filesystem::exists(temporary_)) {
            throw std::runtime_error(
                "native task-receipt target is unavailable");
        }
        descriptor_ = ::open(
            temporary_.c_str(),
            O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
            0600);
        if (descriptor_ < 0) {
            throw std::runtime_error(
                "native task-receipt temporary could not be opened");
        }
        producer_batch_.reserve(receipt_batch_capacity);
        writer_ = std::thread([this]() noexcept { writer_loop(); });
    }

    TaskReceiptSpool(const TaskReceiptSpool&) = delete;
    TaskReceiptSpool& operator=(const TaskReceiptSpool&) = delete;

    ~TaskReceiptSpool() noexcept {
        {
            std::lock_guard lock(mutex_);
            aborting_ = true;
            stopping_ = true;
        }
        ready_.notify_all();
        space_.notify_all();
        drained_.notify_all();
        if (writer_.joinable()) {
            writer_.join();
        }
        if (descriptor_ >= 0) {
            ::close(descriptor_);
            descriptor_ = -1;
        }
        if (!published_) {
            std::error_code ignored;
            std::filesystem::remove(temporary_, ignored);
        }
    }

    void record(const NativeWorkPool::TaskReceipt& receipt) {
        std::unique_lock lock(mutex_);
        rethrow_writer_error_locked();
        if (stopping_ || finalized_) {
            throw std::runtime_error(
                "native task-receipt spool is unavailable");
        }
        producer_batch_.push_back(receipt);
        if (producer_batch_.size() == receipt_batch_capacity) {
            submit_producer_batch_locked(lock);
        }
    }

    void flush() {
        std::unique_lock lock(mutex_);
        rethrow_writer_error_locked();
        if (finalized_) {
            return;
        }
        if (stopping_) {
            throw std::runtime_error(
                "native task-receipt spool is stopping");
        }
        if (!producer_batch_.empty()) {
            submit_producer_batch_locked(lock);
        }
        const auto target_batch = submitted_batches_;
        drained_.wait(lock, [&]() {
            return completed_batches_ >= target_batch
                || writer_error_ || aborting_;
        });
        rethrow_writer_error_locked();
        if (aborting_) {
            throw std::runtime_error("native task-receipt spool aborted");
        }
    }

    TaskReceiptFileDescriptor finalize(
        const std::size_t completed_tasks,
        const std::size_t dropped_count) {
        {
            std::lock_guard lock(mutex_);
            if (finalized_) {
                if (!final_descriptor_.has_value()) {
                    throw std::logic_error(
                        "native task-receipt descriptor is unavailable");
                }
                return *final_descriptor_;
            }
        }
        flush();
        {
            std::lock_guard lock(mutex_);
            if (dropped_count != 0 || receipt_count_ != completed_tasks) {
                throw std::runtime_error(
                    "native task-receipt spool counters are invalid");
            }
            final_completed_tasks_ = completed_tasks;
            final_dropped_count_ = dropped_count;
            stopping_ = true;
        }
        ready_.notify_all();
        writer_.join();
        {
            std::lock_guard lock(mutex_);
            rethrow_writer_error_locked();
        }
        const auto publish_started = std::chrono::steady_clock::now();
        publish_no_replace();
        atomic_publish_seconds_ = elapsed_seconds(publish_started);
        const auto parent_fsync_started = std::chrono::steady_clock::now();
        fsync_parent();
        parent_fsync_seconds_ = elapsed_seconds(parent_fsync_started);
        published_ = true;
        finalized_ = true;
        final_descriptor_ = TaskReceiptFileDescriptor{
            target_.filename().string(),
            native_protocol::native_sha256_digest_hex(digest_.finalize()),
            total_bytes_,
            receipt_count_,
            receipt_batch_capacity,
            queue_bound_batches,
            peak_queued_batches_,
            submitted_batches_,
            completed_batches_,
            final_dropped_count_,
            nanoseconds_to_seconds(producer_wait_nanoseconds_),
            nanoseconds_to_seconds(writer_wall_nanoseconds_),
            nanoseconds_to_seconds(writer_cpu_nanoseconds_),
            nanoseconds_to_seconds(serialization_nanoseconds_),
            nanoseconds_to_seconds(write_nanoseconds_),
            file_fsync_seconds_,
            atomic_publish_seconds_,
            parent_fsync_seconds_,
        };
        return *final_descriptor_;
    }

private:
    using ReceiptBatch = std::vector<NativeWorkPool::TaskReceipt>;

    static std::uint64_t elapsed_nanoseconds(
        const std::chrono::steady_clock::time_point started) noexcept {
        const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        return value > 0 ? static_cast<std::uint64_t>(value) : 0;
    }

    static double elapsed_seconds(
        const std::chrono::steady_clock::time_point started) noexcept {
        return static_cast<double>(elapsed_nanoseconds(started)) / 1.0e9;
    }

    static double nanoseconds_to_seconds(const std::uint64_t value) noexcept {
        return static_cast<double>(value) / 1.0e9;
    }

    static std::uint64_t current_thread_cpu_nanoseconds() {
        timespec value{};
        if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) {
            throw std::runtime_error(
                "native task-receipt writer CPU clock is unavailable");
        }
        return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL
            + static_cast<std::uint64_t>(value.tv_nsec);
    }

    void submit_producer_batch_locked(std::unique_lock<std::mutex>& lock) {
        const auto wait_started = std::chrono::steady_clock::now();
        space_.wait(lock, [&]() {
            return !queued_batch_.has_value() || writer_error_ || aborting_;
        });
        producer_wait_nanoseconds_ += elapsed_nanoseconds(wait_started);
        rethrow_writer_error_locked();
        if (aborting_) {
            throw std::runtime_error("native task-receipt spool aborted");
        }
        // Another concurrent flusher may have submitted this producer batch
        // while the condition-variable wait temporarily released the mutex.
        if (producer_batch_.empty()) {
            return;
        }
        queued_batch_.emplace(std::move(producer_batch_));
        producer_batch_.clear();
        producer_batch_.reserve(receipt_batch_capacity);
        ++submitted_batches_;
        peak_queued_batches_ = 1;
        ready_.notify_one();
    }

    void writer_loop() noexcept {
        try {
            if (::pthread_setname_np(::pthread_self(), "s52-task-write") != 0) {
                throw std::runtime_error(
                    "native task-receipt writer name could not be set");
            }
            const auto cpu_started = current_thread_cpu_nanoseconds();
            while (true) {
                ReceiptBatch batch;
                std::size_t batch_ordinal = 0;
                {
                    std::unique_lock lock(mutex_);
                    ready_.wait(lock, [&]() {
                        return queued_batch_.has_value() || stopping_;
                    });
                    if (!queued_batch_.has_value()) {
                        if (stopping_) {
                            break;
                        }
                        continue;
                    }
                    batch = std::move(*queued_batch_);
                    queued_batch_.reset();
                    batch_ordinal = completed_batches_;
                    space_.notify_all();
                }
                const auto wall_started = std::chrono::steady_clock::now();
                write_batch(batch, batch_ordinal);
                writer_wall_nanoseconds_ += elapsed_nanoseconds(wall_started);
                {
                    std::lock_guard lock(mutex_);
                    ++completed_batches_;
                }
                drained_.notify_all();
            }
            bool should_finalize = false;
            {
                std::lock_guard lock(mutex_);
                should_finalize = !aborting_;
            }
            if (should_finalize) {
                const auto wall_started = std::chrono::steady_clock::now();
                write_trailer();
                const auto fsync_started = std::chrono::steady_clock::now();
                if (::fsync(descriptor_) != 0) {
                    throw std::runtime_error(
                        "native task-receipt file fsync failed");
                }
                file_fsync_seconds_ = elapsed_seconds(fsync_started);
                if (::close(descriptor_) != 0) {
                    descriptor_ = -1;
                    throw std::runtime_error(
                        "native task-receipt file close failed");
                }
                descriptor_ = -1;
                writer_wall_nanoseconds_ += elapsed_nanoseconds(wall_started);
            }
            const auto cpu_finished = current_thread_cpu_nanoseconds();
            writer_cpu_nanoseconds_ = cpu_finished >= cpu_started
                ? cpu_finished - cpu_started
                : 0;
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!writer_error_) {
                writer_error_ = std::current_exception();
            }
            aborting_ = true;
        }
        ready_.notify_all();
        space_.notify_all();
        drained_.notify_all();
    }

    void write_batch(const ReceiptBatch& batch, const std::size_t ordinal) {
        if (batch.empty()) {
            throw std::logic_error("native task-receipt batch is empty");
        }
        native_protocol::NativeSha256 batch_digest;
        const auto serialization_started = std::chrono::steady_clock::now();
        std::string encoded;
        encoded.reserve(batch.size() * 224);
        for (const auto& receipt : batch) {
            const auto row = std::string("{\"kind\":\"task\",\"task_sequence\":")
                + std::to_string(receipt.task_sequence)
                + ",\"worker_index\":" + std::to_string(receipt.worker_index)
                + ",\"first_index\":" + std::to_string(receipt.first_index)
                + ",\"last_index\":" + std::to_string(receipt.last_index)
                + ",\"submitted_nanoseconds\":"
                + std::to_string(receipt.submitted_nanoseconds)
                + ",\"started_nanoseconds\":"
                + std::to_string(receipt.started_nanoseconds)
                + ",\"completed_nanoseconds\":"
                + std::to_string(receipt.completed_nanoseconds) + "}\n";
            batch_digest.update(
                reinterpret_cast<const std::uint8_t*>(row.data()), row.size());
            encoded.append(row);
        }
        encoded.append(
            std::string("{\"kind\":\"batch\",\"batch_ordinal\":")
            + std::to_string(ordinal)
            + ",\"row_count\":" + std::to_string(batch.size())
            + ",\"first_task_sequence\":"
            + std::to_string(batch.front().task_sequence)
            + ",\"last_task_sequence\":"
            + std::to_string(batch.back().task_sequence)
            + ",\"sha256\":\""
            + native_protocol::native_sha256_digest_hex(batch_digest.finalize())
            + "\"}\n");
        serialization_nanoseconds_ += elapsed_nanoseconds(serialization_started);
        write_bytes(encoded);
        receipt_count_ += batch.size();
    }

    void write_trailer() {
        const auto serialization_started = std::chrono::steady_clock::now();
        const auto trailer = std::string(
            "{\"kind\":\"trailer\",\"schema_version\":"
            "\"stage05.2-native-work-task-receipts-v3\"")
            + ",\"storage_model\":\"bounded_async_fifo_stream\""
            + ",\"receipt_batch_capacity\":"
            + std::to_string(receipt_batch_capacity)
            + ",\"queue_bound_batches\":"
            + std::to_string(queue_bound_batches)
            + ",\"submitted_batches\":"
            + std::to_string(submitted_batches_)
            + ",\"completed_batches\":"
            + std::to_string(completed_batches_)
            + ",\"task_receipt_dropped_count\":"
            + std::to_string(final_dropped_count_)
            + ",\"completed_tasks\":"
            + std::to_string(final_completed_tasks_)
            + ",\"receipt_count\":" + std::to_string(receipt_count_) + "}\n";
        serialization_nanoseconds_ += elapsed_nanoseconds(serialization_started);
        write_bytes(trailer);
    }

    void write_bytes(const std::string& bytes) {
        const auto write_started = std::chrono::steady_clock::now();
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto written = ::write(
                descriptor_, bytes.data() + offset, bytes.size() - offset);
            if (written < 0 && errno == EINTR) {
                continue;
            }
            if (written <= 0) {
                throw std::runtime_error("native task-receipt write failed");
            }
            offset += static_cast<std::size_t>(written);
        }
        write_nanoseconds_ += elapsed_nanoseconds(write_started);
        digest_.update(
            reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
        total_bytes_ += bytes.size();
    }

    void publish_no_replace() {
#ifdef SYS_renameat2
        if (::syscall(
                SYS_renameat2,
                AT_FDCWD,
                temporary_.c_str(),
                AT_FDCWD,
                target_.c_str(),
                1U) == 0) {
            return;
        }
        if (errno != ENOSYS && errno != EINVAL) {
            throw std::runtime_error(
                "native task-receipt no-replace publish failed");
        }
#endif
        if (::link(temporary_.c_str(), target_.c_str()) != 0) {
            throw std::runtime_error(
                "native task-receipt no-replace link failed");
        }
        if (::unlink(temporary_.c_str()) != 0) {
            throw std::runtime_error(
                "native task-receipt temporary unlink failed");
        }
    }

    void fsync_parent() {
        const auto parent_descriptor = ::open(
            parent_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (parent_descriptor < 0) {
            throw std::runtime_error(
                "native task-receipt parent could not be opened");
        }
        const auto fsync_result = ::fsync(parent_descriptor);
        const auto close_result = ::close(parent_descriptor);
        if (fsync_result != 0 || close_result != 0) {
            throw std::runtime_error(
                "native task-receipt parent fsync failed");
        }
    }

    void rethrow_writer_error_locked() const {
        if (writer_error_) {
            std::rethrow_exception(writer_error_);
        }
    }

    std::filesystem::path target_;
    std::filesystem::path parent_;
    std::filesystem::path temporary_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable space_;
    std::condition_variable drained_;
    ReceiptBatch producer_batch_;
    std::optional<ReceiptBatch> queued_batch_;
    std::thread writer_;
    native_protocol::NativeSha256 digest_;
    std::exception_ptr writer_error_;
    std::optional<TaskReceiptFileDescriptor> final_descriptor_;
    std::size_t total_bytes_ = 0;
    std::size_t receipt_count_ = 0;
    std::size_t submitted_batches_ = 0;
    std::size_t completed_batches_ = 0;
    std::size_t peak_queued_batches_ = 0;
    std::size_t final_completed_tasks_ = 0;
    std::size_t final_dropped_count_ = 0;
    std::uint64_t producer_wait_nanoseconds_ = 0;
    std::uint64_t writer_wall_nanoseconds_ = 0;
    std::uint64_t writer_cpu_nanoseconds_ = 0;
    std::uint64_t serialization_nanoseconds_ = 0;
    std::uint64_t write_nanoseconds_ = 0;
    double file_fsync_seconds_ = 0.0;
    double atomic_publish_seconds_ = 0.0;
    double parent_fsync_seconds_ = 0.0;
    int descriptor_ = -1;
    bool stopping_ = false;
    bool aborting_ = false;
    bool finalized_ = false;
    bool published_ = false;
};

}  // namespace evrptw::native_telemetry

#endif  // __linux__
