#ifdef __linux__

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/socket.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <pthread.h>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifndef EVRPTW_BUILD_PERFORMANCE_PROFILE
#define EVRPTW_BUILD_PERFORMANCE_PROFILE "unknown"
#endif
#ifndef EVRPTW_BUILD_COMPILER_ID
#define EVRPTW_BUILD_COMPILER_ID "unknown"
#endif
#ifndef EVRPTW_BUILD_COMPILER_VERSION
#define EVRPTW_BUILD_COMPILER_VERSION "unknown"
#endif
#ifndef EVRPTW_BUILD_INTERPROCEDURAL_OPTIMIZATION
#define EVRPTW_BUILD_INTERPROCEDURAL_OPTIMIZATION 0
#endif
#ifndef EVRPTW_BUILD_HOST_NATIVE
#define EVRPTW_BUILD_HOST_NATIVE 0
#endif

#include "native_concurrency.hpp"
#include "native_exact_parallel.hpp"
#include "native_candidate_transaction_executor.hpp"
#include "native_kernel_protocol.hpp"
#include "native_search_core.hpp"
#include "native_sha256.hpp"
#include "native_solver_kernels.hpp"
#include "native_task_receipt_spool.hpp"

namespace protocol = evrptw::native_protocol;
namespace kernels = evrptw::native_kernels;

namespace {

struct ProcessIoCounters {
    std::uint64_t read_bytes{};
    std::uint64_t write_bytes{};
};

ProcessIoCounters read_terminal_process_io() {
    std::ifstream stream("/proc/self/io");
    if (!stream) {
        throw std::runtime_error("native scheduler terminal I/O is unavailable");
    }
    ProcessIoCounters counters;
    bool read_seen = false;
    bool write_seen = false;
    std::string key;
    std::uint64_t value = 0;
    while (stream >> key >> value) {
        if (key == "read_bytes:") {
            counters.read_bytes = value;
            read_seen = true;
        } else if (key == "write_bytes:") {
            counters.write_bytes = value;
            write_seen = true;
        }
    }
    if (!stream.eof() || !read_seen || !write_seen) {
        throw std::runtime_error("native scheduler terminal I/O is invalid");
    }
    return counters;
}

std::atomic<std::uint64_t> segment_counter{0};
std::string scheduler_run_nonce;
std::string production_fault;
std::atomic<bool> production_fault_consumed{false};

void set_thread_name(const char* name) noexcept {
#ifdef __linux__
    ::pthread_setname_np(::pthread_self(), name);
#else
    static_cast<void>(name);
#endif
}

void apply_cpu_affinity(const std::string_view specification) {
    if (specification.empty()) {
        throw std::invalid_argument(
            "native scheduler CPU affinity cannot be empty");
    }
    cpu_set_t requested;
    CPU_ZERO(&requested);
    std::size_t start = 0;
    std::size_t count = 0;
    while (start <= specification.size()) {
        const auto comma = specification.find(',', start);
        const auto token = specification.substr(
            start, comma == std::string_view::npos
                ? specification.size() - start
                : comma - start);
        if (token.empty()) {
            throw std::invalid_argument(
                "native scheduler CPU affinity contains an empty CPU id");
        }
        std::size_t consumed = 0;
        const auto cpu = std::stoll(std::string(token), &consumed, 10);
        if (consumed != token.size() || cpu < 0 || cpu >= CPU_SETSIZE) {
            throw std::invalid_argument(
                "native scheduler CPU affinity contains an invalid CPU id");
        }
        const auto cpu_id = static_cast<int>(cpu);
        if (CPU_ISSET(cpu_id, &requested)) {
            throw std::invalid_argument(
                "native scheduler CPU affinity contains a duplicate CPU id");
        }
        CPU_SET(cpu_id, &requested);
        ++count;
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    if (count == 0) {
        throw std::invalid_argument(
            "native scheduler CPU affinity must contain at least one CPU");
    }
    if (::sched_setaffinity(0, sizeof(requested), &requested) != 0) {
        throw std::system_error(
            errno, std::generic_category(),
            "native scheduler sched_setaffinity failed");
    }
    cpu_set_t actual;
    CPU_ZERO(&actual);
    if (::sched_getaffinity(0, sizeof(actual), &actual) != 0
        || CPU_COUNT(&actual) == 0) {
        throw std::system_error(
            errno == 0 ? EINVAL : errno, std::generic_category(),
            "native scheduler sched_getaffinity failed after apply");
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &requested) != CPU_ISSET(cpu, &actual)) {
            throw std::runtime_error(
                "native scheduler CPU affinity was not applied exactly");
        }
    }
}

class CandidateSessionRegistry final {
public:
    struct Session final {
        static constexpr std::int64_t transaction_idle = 0;
        static constexpr std::int64_t transaction_pending = 1;
        static constexpr std::int64_t transaction_committed = 2;
        static constexpr std::int64_t transaction_rolled_back = 3;

        pid_t owner_pid = 0;
        evrptw::native_search::RequestV2 request;
        evrptw::native_search::InitialStateV2 initial_state;
        std::unique_ptr<evrptw::native_search::CandidateTransactionStateV2>
            transaction_state;
        std::optional<evrptw::native_search::SearchBudgetStateV2::Snapshot>
            pending_budget_snapshot;
        bool pending_exact_protocol = false;
        bool pending_negative_store = false;
        bool pending_attempted_mark = false;
        std::int64_t transaction_resolution = transaction_idle;
        std::mutex mutex;

        void commit_pending() {
            if (!pending_budget_snapshot.has_value() || !transaction_state) {
                throw std::runtime_error(
                    "native candidate session commit has no pending transaction; "
                    "resolution=" + std::to_string(transaction_resolution));
            }
            transaction_state->exact_cache()
                .commit_protocol_transaction_noexcept();
            if (pending_negative_store) {
                transaction_state->negative_cache()
                    .commit_store_batch_noexcept();
            }
            if (pending_attempted_mark) {
                transaction_state->attempted_plans()
                    .commit_mark_batch_noexcept();
            }
            clear_pending();
            transaction_resolution = transaction_committed;
        }

        void rollback_pending() {
            if (!pending_budget_snapshot.has_value() || !transaction_state) {
                throw std::runtime_error(
                    "native candidate session rollback has no pending transaction; "
                    "resolution=" + std::to_string(transaction_resolution));
            }
            if (pending_attempted_mark) {
                transaction_state->attempted_plans()
                    .rollback_mark_batch_noexcept();
            }
            if (pending_negative_store) {
                transaction_state->negative_cache()
                    .rollback_store_batch_noexcept();
            }
            if (pending_exact_protocol) {
                transaction_state->exact_cache()
                    .rollback_protocol_transaction_noexcept();
            }
            transaction_state->budget().rollback_preserving_exact(
                *pending_budget_snapshot, true);
            clear_pending();
            transaction_resolution = transaction_rolled_back;
        }

        void clear_pending() noexcept {
            pending_budget_snapshot.reset();
            pending_exact_protocol = false;
            pending_negative_store = false;
            pending_attempted_mark = false;
        }
    };

    struct OpenResult final {
        std::string token;
        std::shared_ptr<Session> session;
    };

    [[nodiscard]] OpenResult open(
        const pid_t owner_pid,
        evrptw::native_search::RequestV2 request) {
        if (owner_pid <= 0) {
            throw std::invalid_argument(
                "native candidate session owner is invalid");
        }
        auto session = std::make_shared<Session>();
        session->owner_pid = owner_pid;
        session->initial_state = evrptw::native_search::initialize_state(request);
        session->request = std::move(request);
        session->transaction_state = std::make_unique<
            evrptw::native_search::CandidateTransactionStateV2>(
                session->request, session->initial_state);
        const auto ordinal = next_id_.fetch_add(1, std::memory_order_relaxed);
        std::string identity("stage05.2-native-candidate-session-v2");
        identity.append(scheduler_run_nonce);
        identity.append(
            reinterpret_cast<const char*>(&owner_pid), sizeof(owner_pid));
        identity.append(
            reinterpret_cast<const char*>(&ordinal), sizeof(ordinal));
        identity.append(session->request.sha256());
        const auto token = protocol::native_sha256_hex(identity);
        std::lock_guard lock(mutex_);
        if (!sessions_.emplace(token, session).second) {
            throw std::runtime_error(
                "native candidate session identity collision");
        }
        return {token, std::move(session)};
    }

    void close(const pid_t owner_pid, const std::string_view token) {
        std::lock_guard lock(mutex_);
        const auto found = sessions_.find(std::string(token));
        if (found == sessions_.end() || found->second->owner_pid != owner_pid) {
            throw std::runtime_error(
                "native candidate session close identity mismatch");
        }
        // Retain the session until after the guard unlocks its mutex.  Erasing
        // the map's last shared_ptr while the guard still referenced the
        // session-owned mutex destroyed the mutex before unlock.
        auto session = found->second;
        std::lock_guard session_lock(session->mutex);
        if (session->pending_budget_snapshot.has_value()) {
            session->rollback_pending();
        }
        sessions_.erase(found);
    }

    [[nodiscard]] std::shared_ptr<Session> lookup(
        const pid_t owner_pid,
        const std::string_view token) {
        std::lock_guard lock(mutex_);
        const auto found = sessions_.find(std::string(token));
        if (found == sessions_.end() || found->second->owner_pid != owner_pid) {
            throw std::runtime_error(
                "native candidate session execute identity mismatch");
        }
        return found->second;
    }

    void rollback_open_noexcept(
        const pid_t owner_pid,
        const std::string_view token) noexcept {
        std::lock_guard lock(mutex_);
        const auto found = sessions_.find(std::string(token));
        if (found != sessions_.end() && found->second->owner_pid == owner_pid) {
            sessions_.erase(found);
        }
    }

    [[nodiscard]] bool rollback_pending_noexcept(
        const pid_t owner_pid,
        const std::string_view token) noexcept {
        try {
            auto session = lookup(owner_pid, token);
            std::lock_guard session_lock(session->mutex);
            if (session->pending_budget_snapshot.has_value()) {
                session->rollback_pending();
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    void commit_pending(
        const pid_t owner_pid,
        const std::string_view token) {
        auto session = lookup(owner_pid, token);
        std::lock_guard session_lock(session->mutex);
        session->commit_pending();
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
    std::atomic<std::uint64_t> next_id_{1};
};

class SchedulerCandidateTransactionKernelsV2 final
    : public evrptw::native_search::CandidateTransactionKernelsV2 {
public:
    explicit SchedulerCandidateTransactionKernelsV2(NativeWorkPool& pool)
        : pool_(pool) {}

    [[nodiscard]] std::vector<kernels::ScreenOutput> screen(
        const evrptw::native_search::ProblemV2& problem,
        const evrptw::native_search::RouteBatchViewV2 routes,
        const double epsilon) override {
        routes.validate("scheduler candidate transaction screening");
        std::vector<kernels::ScreenOutput> output(routes.route_count());
        pool_.parallel_for(routes.route_count(), [&](const std::size_t row) {
            const auto route = routes.route(row);
            const std::array<std::int64_t, 2> offsets{
                0, static_cast<std::int64_t>(route.size())};
            evrptw::native_search::LocalCandidateTransactionKernelsV2 local;
            auto screened = local.screen(
                problem, {offsets, route}, epsilon);
            if (screened.size() != 1) {
                throw std::logic_error(
                    "scheduler candidate screening task lost its result");
            }
            output[row] = std::move(screened.front());
        });
        return output;
    }

    [[nodiscard]] kernels::ExactBatchOutput exact(
        const evrptw::native_search::ProblemV2& problem,
        const evrptw::native_search::RouteBatchViewV2 routes,
        const double deadline_remaining,
        const std::int64_t batch_size) override {
        routes.validate("scheduler candidate transaction exact");
        const evrptw::native_search::ExactProblemViewV2 exact_problem{
            problem.node_kind,
            problem.ready_time,
            problem.due_date,
            problem.service_time,
            problem.distance,
            problem.vehicle,
        };
        const auto run_batch = [&](const std::int64_t* offsets,
                                   const std::int64_t* indices,
                                   const std::size_t route_count,
                                   const double remaining) {
            return evrptw::native_search::run_local_exact_batch_v2(
                exact_problem,
                {
                    std::span<const std::int64_t>(offsets, route_count + 1),
                    std::span<const std::int64_t>(
                        indices,
                        static_cast<std::size_t>(offsets[route_count])),
                },
                remaining, batch_size);
        };
        auto output = evrptw::native_parallel::run_exact_charging_parallel(
            pool_, routes.offsets.data(), routes.indices.data(),
            routes.route_count(), deadline_remaining, run_batch);
        const auto depot = std::find(
            problem.node_kind.begin(), problem.node_kind.end(),
            kernels::depot_kind);
        if (depot == problem.node_kind.end()) {
            throw std::logic_error(
                "scheduler candidate exact depot is unavailable");
        }
        kernels::validate_exact_batch_output(
            output, problem.node_kind.data(), routes.offsets.data(),
            routes.indices.data(), problem.node_count(), routes.route_count(),
            routes.indices.size(),
            static_cast<std::int64_t>(
                std::distance(problem.node_kind.begin(), depot)),
            batch_size);
        return output;
    }

private:
    NativeWorkPool& pool_;
};

struct SchedulerConcurrencySnapshot final {
    std::size_t peak_active_requests;
    std::size_t peak_distinct_client_pids;
};

class SchedulerConcurrencyTelemetry final {
public:
    void begin(pid_t peer_pid) {
        std::lock_guard lock(mutex_);
        ++active_requests_;
        ++active_by_pid_[peer_pid];
        peak_active_requests_ = std::max(
            peak_active_requests_, active_requests_);
        peak_distinct_client_pids_ = std::max(
            peak_distinct_client_pids_, active_by_pid_.size());
    }

    void end(pid_t peer_pid) noexcept {
        std::lock_guard lock(mutex_);
        const auto found = active_by_pid_.find(peer_pid);
        if (found == active_by_pid_.end() || active_requests_ == 0) {
            std::terminate();
        }
        --active_requests_;
        if (--found->second == 0) {
            active_by_pid_.erase(found);
        }
    }

    SchedulerConcurrencySnapshot snapshot() const {
        std::lock_guard lock(mutex_);
        return {peak_active_requests_, peak_distinct_client_pids_};
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<pid_t, std::size_t> active_by_pid_;
    std::size_t active_requests_ = 0;
    std::size_t peak_active_requests_ = 0;
    std::size_t peak_distinct_client_pids_ = 0;
};

class ActiveRequestGuard final {
public:
    ActiveRequestGuard(
        SchedulerConcurrencyTelemetry& telemetry, pid_t peer_pid)
        : telemetry_(telemetry), peer_pid_(peer_pid) {
        telemetry_.begin(peer_pid_);
    }

    ~ActiveRequestGuard() noexcept { telemetry_.end(peer_pid_); }

    ActiveRequestGuard(const ActiveRequestGuard&) = delete;
    ActiveRequestGuard& operator=(const ActiveRequestGuard&) = delete;

private:
    SchedulerConcurrencyTelemetry& telemetry_;
    pid_t peer_pid_;
};

bool read_exact(int descriptor, void* output, std::size_t size) {
    auto* cursor = static_cast<std::uint8_t*>(output);
    while (size > 0) {
        const auto received = ::recv(descriptor, cursor, size, 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            return false;
        }
        cursor += received;
        size -= static_cast<std::size_t>(received);
    }
    return true;
}

void send_exact(int descriptor, const void* input, std::size_t size) {
    const auto* cursor = static_cast<const std::uint8_t*>(input);
    while (size > 0) {
        const auto sent = ::send(descriptor, cursor, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            throw std::runtime_error("native scheduler could not send a control frame");
        }
        cursor += sent;
        size -= static_cast<std::size_t>(sent);
    }
}

void validate_control(const protocol::ControlFrame& frame) {
    if (frame.magic != protocol::kernel_magic
        || frame.version != protocol::kernel_protocol_version) {
        throw std::runtime_error("native scheduler control identity mismatch");
    }
}

std::vector<std::uint8_t> run_exact(
    const protocol::PayloadView& input,
    double queue_wait_seconds,
    std::size_t queue_depth,
    NativeWorkPool& pool) {
    if (input.header().array_count != 10) {
        throw std::runtime_error("native scheduler exact request shape is invalid");
    }
    const auto& kind_shape = input.descriptor(0);
    const auto& offset_shape = input.descriptor(6);
    const auto& index_shape = input.descriptor(7);
    const auto node_count = static_cast<std::size_t>(kind_shape.count);
    const auto one_dimensional = [&](const std::size_t index,
                                     const protocol::NumericType type,
                                     const std::uint64_t count) {
        const auto& item = input.descriptor(index);
        return item.type == type && item.dimensions == 1
            && item.count == count && item.shape[0] == count
            && item.shape[1] == 0;
    };
    const auto two_dimensional = [&](const std::size_t index,
                                     const protocol::NumericType type,
                                     const std::uint64_t first,
                                     const std::uint64_t second) {
        const auto& item = input.descriptor(index);
        return item.type == type && item.dimensions == 2
            && item.count == first * second && item.shape[0] == first
            && item.shape[1] == second;
    };
    if (node_count == 0
        || !one_dimensional(0, protocol::NumericType::int64, node_count)
        || !one_dimensional(1, protocol::NumericType::float64, node_count)
        || !one_dimensional(2, protocol::NumericType::float64, node_count)
        || !one_dimensional(3, protocol::NumericType::float64, node_count)
        || !two_dimensional(
            4, protocol::NumericType::float64, node_count, node_count)
        || !one_dimensional(5, protocol::NumericType::float64, 5)
        || offset_shape.count < 1
        || !one_dimensional(
            6, protocol::NumericType::int64, offset_shape.count)
        || !one_dimensional(
            7, protocol::NumericType::int64, index_shape.count)
        || !one_dimensional(8, protocol::NumericType::float64, 1)
        || !one_dimensional(9, protocol::NumericType::int64, 1)) {
        throw std::runtime_error("native scheduler exact dimensions are invalid");
    }
    const auto route_count = static_cast<std::size_t>(offset_shape.count - 1);
    const auto* kinds = input.data<std::int64_t>(0, protocol::NumericType::int64);
    const auto* ready = input.data<double>(1, protocol::NumericType::float64);
    const auto* due = input.data<double>(2, protocol::NumericType::float64);
    const auto* service = input.data<double>(3, protocol::NumericType::float64);
    const auto* distances = input.data<double>(4, protocol::NumericType::float64);
    const auto* vehicle = input.data<double>(5, protocol::NumericType::float64);
    const auto* offsets = input.data<std::int64_t>(6, protocol::NumericType::int64);
    const auto* indices = input.data<std::int64_t>(7, protocol::NumericType::int64);
    if (offsets[0] != 0
        || offsets[route_count] != static_cast<std::int64_t>(index_shape.count)) {
        throw std::runtime_error("native scheduler exact offsets are invalid");
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> stations;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (!std::isfinite(ready[node]) || !std::isfinite(due[node])
            || !std::isfinite(service[node])) {
            throw std::runtime_error(
                "native scheduler exact node metadata is invalid");
        }
        for (std::size_t destination = 0; destination < node_count;
             ++destination) {
            const auto distance = distances[node * node_count + destination];
            if (!std::isfinite(distance) || distance < 0.0) {
                throw std::runtime_error(
                    "native scheduler exact distance is invalid");
            }
        }
        if (kinds[node] == kernels::depot_kind) {
            if (depot >= 0) {
                throw std::runtime_error("native scheduler exact depot is duplicated");
            }
            depot = static_cast<std::int64_t>(node);
        } else if (kinds[node] == kernels::station_kind) {
            stations.push_back(static_cast<std::int64_t>(node));
        } else if (kinds[node] != kernels::customer_kind) {
            throw std::runtime_error("native scheduler exact node kind is invalid");
        }
    }
    if (depot < 0) {
        throw std::runtime_error("native scheduler exact depot is missing");
    }
    if (!std::isfinite(vehicle[0]) || vehicle[0] < 0.0
        || !std::isfinite(vehicle[2]) || vehicle[2] < 0.0
        || !std::isfinite(vehicle[3]) || vehicle[3] < 0.0
        || !std::isfinite(vehicle[4]) || vehicle[4] <= 0.0) {
        throw std::runtime_error(
            "native scheduler exact vehicle parameters are invalid");
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] > offsets[route + 1]) {
            throw std::runtime_error(
                "native scheduler exact offsets are not monotone");
        }
        std::vector<bool> seen(node_count, false);
        for (auto position = offsets[route]; position < offsets[route + 1];
             ++position) {
            const auto node = indices[position];
            if (node < 0 || static_cast<std::size_t>(node) >= node_count
                || kinds[node] != kernels::customer_kind
                || seen[static_cast<std::size_t>(node)]) {
                throw std::runtime_error(
                    "native scheduler exact route indices are invalid");
            }
            seen[static_cast<std::size_t>(node)] = true;
        }
    }
    const auto deadline_absolute =
        input.data<double>(8, protocol::NumericType::float64)[0];
    const auto batch_size =
        input.data<std::int64_t>(9, protocol::NumericType::int64)[0];
    if (std::isnan(deadline_absolute) || deadline_absolute < 0.0
        || batch_size <= 0) {
        throw std::runtime_error(
            "native scheduler exact control values are invalid");
    }
    const auto execution_deadline = std::isfinite(deadline_absolute)
        ? std::max(
              0.0,
              deadline_absolute
                  - std::chrono::duration<double>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count())
        : deadline_absolute;
    kernels::ExactBatchOutput output;
    const auto run_batch = [&](const std::int64_t* batch_offsets,
                               const std::int64_t* batch_indices,
                               const std::size_t batch_route_count,
                               const double batch_deadline) {
        return kernels::run_exact_charging_batch(
            kinds,
            input.data<double>(1, protocol::NumericType::float64),
            input.data<double>(2, protocol::NumericType::float64),
            input.data<double>(3, protocol::NumericType::float64),
            input.data<double>(4, protocol::NumericType::float64),
            input.data<double>(5, protocol::NumericType::float64),
            batch_offsets, batch_indices, node_count, batch_route_count,
            depot, stations, batch_deadline, batch_size);
    };
    output = evrptw::native_parallel::run_exact_charging_parallel(
        pool, offsets, indices, route_count, execution_deadline, run_batch);
    kernels::validate_exact_batch_output(
        output, kinds, offsets, indices, node_count, route_count,
        index_shape.count, depot, batch_size);
    protocol::PayloadBuilder builder(
        protocol::KernelOperation::exact_charging, input.header().request_id);
    builder.add(protocol::NumericType::int64, output.path_offsets.data(),
        output.path_offsets.size(), output.path_offsets.size());
    builder.add(protocol::NumericType::int64, output.path_indices.data(),
        output.path_indices.size(), output.path_indices.size());
    builder.add(protocol::NumericType::int64, output.statuses.data(),
        output.statuses.size(), output.statuses.size());
    builder.add(protocol::NumericType::int64, output.reasons.data(),
        output.reasons.size(), output.reasons.size());
    builder.add(protocol::NumericType::float64, output.metrics.data(),
        output.metrics.size(), route_count, 4);
    builder.add(protocol::NumericType::int64, output.label_counters.data(),
        output.label_counters.size(), route_count, 3);
    builder.add(protocol::NumericType::int64, output.batch_counters.data(),
        output.batch_counters.size(), output.batch_counters.size());
    builder.add(protocol::NumericType::int64, output.completion_order.data(),
        output.completion_order.size(), output.completion_order.size());
    builder.add(protocol::NumericType::int64,
        output.physical_completion_order.data(),
        output.physical_completion_order.size(),
        output.physical_completion_order.size());
    builder.add(protocol::NumericType::int64,
        output.physical_task_receipts.data(),
        output.physical_task_receipts.size(),
        output.physical_task_receipts.size() / 7, 7);
    const std::array<double, 7> telemetry{
        queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
        0.0, 0.0, 0.0};
    builder.add(protocol::NumericType::float64, telemetry.data(),
        telemetry.size(), telemetry.size());
    return builder.finish();
}

std::vector<std::uint8_t> run_screen(
    const protocol::PayloadView& input,
    double queue_wait_seconds,
    std::size_t queue_depth) {
    if (input.header().array_count != 11) {
        throw std::runtime_error("native scheduler screening request shape is invalid");
    }
    const auto node_count = static_cast<std::size_t>(input.descriptor(0).count);
    const auto node_square = protocol::checked_product(
        node_count, node_count,
        "native scheduler screening node count overflows");
    if (node_count == 0 || input.descriptor(1).count != node_count
        || input.descriptor(2).count != node_count
        || input.descriptor(3).count != node_count
        || input.descriptor(4).count != node_count
        || input.descriptor(5).count != node_square
        || input.descriptor(6).count != node_square
        || input.descriptor(5).shape[0] != node_count
        || input.descriptor(5).shape[1] != node_count
        || input.descriptor(6).shape[0] != node_count
        || input.descriptor(6).shape[1] != node_count
        || input.descriptor(7).count != 5 || input.descriptor(9).count != 4
        || input.descriptor(10).count != 6) {
        throw std::runtime_error("native scheduler screening dimensions are invalid");
    }
    const auto* kinds = input.data<std::int64_t>(0, protocol::NumericType::int64);
    std::int64_t depot = -1;
    std::vector<std::int64_t> recharge_nodes;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (kinds[node] == kernels::depot_kind) {
            if (depot >= 0) {
                throw std::runtime_error("native scheduler screening depot is duplicated");
            }
            depot = static_cast<std::int64_t>(node);
            recharge_nodes.push_back(depot);
        } else if (kinds[node] == kernels::station_kind) {
            recharge_nodes.push_back(static_cast<std::int64_t>(node));
        } else if (kinds[node] != kernels::customer_kind) {
            throw std::runtime_error("native scheduler screening node kind is invalid");
        }
    }
    if (depot < 0) {
        throw std::runtime_error("native scheduler screening depot is missing");
    }
    const auto* route = input.data<std::int64_t>(
        8, protocol::NumericType::int64);
    std::vector<bool> seen(node_count, false);
    for (std::size_t index = 0; index < input.descriptor(8).count; ++index) {
        const auto node = route[index];
        if (node < 0 || static_cast<std::size_t>(node) >= node_count
            || kinds[node] != kernels::customer_kind
            || seen[static_cast<std::size_t>(node)]) {
            throw std::runtime_error(
                "native scheduler screening route indices are invalid");
        }
        seen[static_cast<std::size_t>(node)] = true;
    }
    const auto output = kernels::run_screen_route(
        kinds,
        input.data<double>(1, protocol::NumericType::float64),
        input.data<double>(2, protocol::NumericType::float64),
        input.data<double>(3, protocol::NumericType::float64),
        input.data<double>(4, protocol::NumericType::float64),
        input.data<double>(5, protocol::NumericType::float64),
        input.data<std::uint8_t>(6, protocol::NumericType::uint8),
        input.data<double>(7, protocol::NumericType::float64),
        route,
        static_cast<std::size_t>(input.descriptor(8).count), node_count, depot,
        recharge_nodes,
        input.data<double>(9, protocol::NumericType::float64),
        input.data<double>(10, protocol::NumericType::float64));
    protocol::PayloadBuilder builder(
        protocol::KernelOperation::screen_route, input.header().request_id);
    builder.add(protocol::NumericType::int64, output.codes.data(),
        output.codes.size(), output.codes.size());
    builder.add(protocol::NumericType::float64, output.metrics.data(),
        output.metrics.size(), output.metrics.size());
    const std::int64_t queries = output.reachability_queries;
    builder.add(protocol::NumericType::int64, &queries, 1, 1);
    const std::array<double, 7> telemetry{
        queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
        0.0, 0.0, 0.0};
    builder.add(protocol::NumericType::float64, telemetry.data(),
        telemetry.size(), telemetry.size());
    return builder.finish();
}

std::vector<std::uint8_t> run_screen_routes(
    const protocol::PayloadView& input,
    double queue_wait_seconds,
    std::size_t queue_depth,
    NativeWorkPool& pool,
    std::size_t& request_peak_active_tasks) {
    if (input.header().array_count != 12) {
        throw std::runtime_error(
            "native scheduler screening-batch request shape is invalid");
    }
    const auto node_count = static_cast<std::size_t>(input.descriptor(0).count);
    const auto node_square = protocol::checked_product(
        node_count, node_count,
        "native scheduler screening-batch node count overflows");
    const auto& offset_shape = input.descriptor(8);
    const auto& index_shape = input.descriptor(9);
    if (node_count == 0 || input.descriptor(1).count != node_count
        || input.descriptor(2).count != node_count
        || input.descriptor(3).count != node_count
        || input.descriptor(4).count != node_count
        || input.descriptor(5).count != node_square
        || input.descriptor(6).count != node_square
        || input.descriptor(5).shape[0] != node_count
        || input.descriptor(5).shape[1] != node_count
        || input.descriptor(6).shape[0] != node_count
        || input.descriptor(6).shape[1] != node_count
        || input.descriptor(7).count != 5 || offset_shape.count < 2
        || input.descriptor(10).count != 4
        || input.descriptor(11).count != 6) {
        throw std::runtime_error(
            "native scheduler screening-batch dimensions are invalid");
    }
    const auto route_count = static_cast<std::size_t>(offset_shape.count - 1);
    const auto code_count = protocol::checked_product(
        route_count, 16,
        "native scheduler screening-batch code count overflows");
    const auto metric_count = protocol::checked_product(
        route_count, 15,
        "native scheduler screening-batch metric count overflows");
    const auto code_bytes = protocol::checked_product(
        code_count, sizeof(std::int64_t),
        "native scheduler screening-batch code bytes overflow");
    const auto metric_bytes = protocol::checked_product(
        metric_count, sizeof(double),
        "native scheduler screening-batch metric bytes overflow");
    const auto query_bytes = protocol::checked_product(
        route_count, sizeof(std::int64_t),
        "native scheduler screening-batch query bytes overflow");
    if (code_bytes > protocol::maximum_payload_bytes
        || metric_bytes > protocol::maximum_payload_bytes - code_bytes
        || query_bytes
            > protocol::maximum_payload_bytes - code_bytes - metric_bytes) {
        throw std::length_error(
            "native scheduler screening-batch output exceeds its size limit");
    }
    const auto* kinds = input.data<std::int64_t>(
        0, protocol::NumericType::int64);
    const auto* offsets = input.data<std::int64_t>(
        8, protocol::NumericType::int64);
    const auto* indices = input.data<std::int64_t>(
        9, protocol::NumericType::int64);
    if (offsets[0] != 0
        || offsets[route_count] != static_cast<std::int64_t>(index_shape.count)) {
        throw std::runtime_error(
            "native scheduler screening-batch offsets are invalid");
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> recharge_nodes;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (kinds[node] == kernels::depot_kind) {
            if (depot >= 0) {
                throw std::runtime_error(
                    "native scheduler screening-batch depot is duplicated");
            }
            depot = static_cast<std::int64_t>(node);
            recharge_nodes.push_back(depot);
        } else if (kinds[node] == kernels::station_kind) {
            recharge_nodes.push_back(static_cast<std::int64_t>(node));
        } else if (kinds[node] != kernels::customer_kind) {
            throw std::runtime_error(
                "native scheduler screening-batch node kind is invalid");
        }
    }
    if (depot < 0) {
        throw std::runtime_error(
            "native scheduler screening-batch depot is missing");
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] > offsets[route + 1]) {
            throw std::runtime_error(
                "native scheduler screening-batch offsets are not monotone");
        }
        std::vector<bool> seen(node_count, false);
        for (auto position = offsets[route]; position < offsets[route + 1];
             ++position) {
            const auto node = indices[position];
            if (node < 0 || static_cast<std::size_t>(node) >= node_count
                || kinds[node] != kernels::customer_kind
                || seen[static_cast<std::size_t>(node)]) {
                throw std::runtime_error(
                    "native scheduler screening-batch route indices are invalid");
            }
            seen[static_cast<std::size_t>(node)] = true;
        }
    }
    const auto* demands = input.data<double>(
        1, protocol::NumericType::float64);
    const auto* ready = input.data<double>(2, protocol::NumericType::float64);
    const auto* due = input.data<double>(3, protocol::NumericType::float64);
    const auto* service = input.data<double>(4, protocol::NumericType::float64);
    const auto* distances = input.data<double>(
        5, protocol::NumericType::float64);
    const auto* reachable = input.data<std::uint8_t>(
        6, protocol::NumericType::uint8);
    const auto* vehicle = input.data<double>(7, protocol::NumericType::float64);
    const auto* options = input.data<double>(10, protocol::NumericType::float64);
    const auto* incremental = input.data<double>(
        11, protocol::NumericType::float64);
    std::vector<kernels::ScreenOutput> outputs(route_count);
    std::atomic<std::size_t> peak_active{0};
    pool.parallel_for(route_count, [&](std::size_t route) {
        const auto active = pool.active_task_count();
        auto peak = peak_active.load(std::memory_order_relaxed);
        while (active > peak
               && !peak_active.compare_exchange_weak(
                   peak, active, std::memory_order_relaxed)) {
        }
        outputs[route] = kernels::run_screen_route(
            kinds, demands, ready, due, service, distances, reachable, vehicle,
            indices + offsets[route],
            static_cast<std::size_t>(offsets[route + 1] - offsets[route]),
            node_count, depot, recharge_nodes, options, incremental);
    });
    request_peak_active_tasks = peak_active.load(std::memory_order_relaxed);
    std::vector<std::int64_t> codes(code_count);
    std::vector<double> metrics(metric_count);
    std::vector<std::int64_t> queries(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        std::copy(
            outputs[route].codes.begin(), outputs[route].codes.end(),
            codes.begin() + static_cast<std::ptrdiff_t>(route * 16));
        std::copy(
            outputs[route].metrics.begin(), outputs[route].metrics.end(),
            metrics.begin() + static_cast<std::ptrdiff_t>(route * 15));
        queries[route] = outputs[route].reachability_queries;
    }
    protocol::PayloadBuilder builder(
        protocol::KernelOperation::screen_routes, input.header().request_id);
    builder.add(protocol::NumericType::int64, codes.data(), codes.size(),
        route_count, 16);
    builder.add(protocol::NumericType::float64, metrics.data(), metrics.size(),
        route_count, 15);
    builder.add(protocol::NumericType::int64, queries.data(), queries.size(),
        queries.size());
    const std::array<double, 7> telemetry{
        queue_wait_seconds, static_cast<double>(queue_depth),
        static_cast<double>(request_peak_active_tasks),
        static_cast<double>(pool.thread_count()), 0.0, 0.0, 0.0};
    builder.add(protocol::NumericType::float64, telemetry.data(),
        telemetry.size(), telemetry.size());
    return builder.finish();
}

std::vector<std::uint8_t> execute(
    const protocol::PayloadView& input,
    double queue_wait_seconds,
    std::size_t queue_depth,
    const pid_t peer_pid,
    CandidateSessionRegistry& candidate_sessions,
    NativeWorkPool& pool,
    std::optional<std::string>& pending_candidate_transaction,
    std::optional<std::string>& pending_commit_session) {
    switch (input.header().operation) {
    case protocol::KernelOperation::exact_charging:
        return run_exact(input, queue_wait_seconds, queue_depth, pool);
    case protocol::KernelOperation::screen_route:
        return run_screen(input, queue_wait_seconds, queue_depth);
    case protocol::KernelOperation::screen_routes:
        throw std::logic_error(
            "native screening-batch operation requires the shared work pool");
    case protocol::KernelOperation::search_request_receipt: {
        const auto request = evrptw::native_search::request_from_payload(input);
        const auto counts = evrptw::native_search::receipt_counts(request);
        const auto sha256 = request.sha256();
        protocol::PayloadBuilder builder(
            protocol::KernelOperation::search_request_receipt,
            input.header().request_id);
        builder.add(protocol::NumericType::int64, counts.data(), counts.size(),
            counts.size());
        builder.add(protocol::NumericType::uint8, sha256.data(), sha256.size(),
            sha256.size());
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        builder.add(protocol::NumericType::float64, telemetry.data(),
            telemetry.size(), telemetry.size());
        return builder.finish();
    }
    case protocol::KernelOperation::search_initial_state: {
        const auto request = evrptw::native_search::request_from_payload(input);
        const auto state = evrptw::native_search::initialize_state(request);
        const auto sha256 = state.sha256();
        const auto route_count = state.exact.statuses.size();
        protocol::PayloadBuilder builder(
            protocol::KernelOperation::search_initial_state,
            input.header().request_id);
        builder.add(protocol::NumericType::int64,
            state.exact.path_offsets.data(), state.exact.path_offsets.size(),
            state.exact.path_offsets.size());
        builder.add(protocol::NumericType::int64,
            state.exact.path_indices.data(), state.exact.path_indices.size(),
            state.exact.path_indices.size());
        builder.add(protocol::NumericType::int64, state.exact.statuses.data(),
            state.exact.statuses.size(), state.exact.statuses.size());
        builder.add(protocol::NumericType::int64, state.exact.reasons.data(),
            state.exact.reasons.size(), state.exact.reasons.size());
        builder.add(protocol::NumericType::float64, state.exact.metrics.data(),
            state.exact.metrics.size(), route_count, 4);
        builder.add(protocol::NumericType::int64,
            state.exact.label_counters.data(),
            state.exact.label_counters.size(), route_count, 3);
        builder.add(protocol::NumericType::int64,
            state.exact.batch_counters.data(),
            state.exact.batch_counters.size(),
            state.exact.batch_counters.size());
        builder.add(protocol::NumericType::int64,
            state.objective_integer.data(), state.objective_integer.size(),
            state.objective_integer.size());
        builder.add(protocol::NumericType::float64,
            state.objective_float.data(), state.objective_float.size(),
            state.objective_float.size());
        builder.add(protocol::NumericType::int64, state.accounting.data(),
            state.accounting.size(), state.accounting.size());
        builder.add(protocol::NumericType::uint8, sha256.data(), sha256.size(),
            sha256.size());
        builder.add(protocol::NumericType::int64,
            state.exact.completion_order.data(),
            state.exact.completion_order.size(),
            state.exact.completion_order.size());
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        builder.add(protocol::NumericType::float64, telemetry.data(),
            telemetry.size(), telemetry.size());
        return builder.finish();
    }
    case protocol::KernelOperation::candidate_transaction_wire: {
        const auto wire = evrptw::native_search::
            candidate_plan_transaction_wire_from_payload_v2(input);
        const auto decoded = evrptw::native_search::
            decode_candidate_plan_transaction_v2(wire);
        evrptw::native_search::validate_candidate_plan_transaction_v2(
            decoded);
        const auto canonical = evrptw::native_search::
            encode_candidate_plan_transaction_v2(decoded);
        const bool double_values_equal =
            canonical.double_values.size() == wire.double_values.size()
            && (canonical.double_values.empty()
                || std::memcmp(
                       canonical.double_values.data(),
                       wire.double_values.data(),
                       canonical.double_values.size() * sizeof(double)) == 0);
        if (canonical.integer_offsets != wire.integer_offsets
            || canonical.integer_values != wire.integer_values
            || canonical.double_offsets != wire.double_offsets
            || !double_values_equal
            || canonical.byte_offsets != wire.byte_offsets
            || canonical.byte_values != wire.byte_values) {
            throw std::runtime_error(
                "native candidate-plan transaction payload is not canonical");
        }
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        return evrptw::native_search::
            candidate_plan_transaction_wire_payload_v2(
                canonical, input.header().request_id, telemetry);
    }
    case protocol::KernelOperation::candidate_session_open: {
        auto request = evrptw::native_search::request_from_payload(input);
        auto opened = candidate_sessions.open(peer_pid, std::move(request));
        const auto& state = opened.session->initial_state;
        const auto route_count = state.exact.statuses.size();
        const auto sha256 = state.sha256();
        protocol::PayloadBuilder builder(
            protocol::KernelOperation::candidate_session_open,
            input.header().request_id);
        builder.add(protocol::NumericType::int64,
            state.exact.path_offsets.data(), state.exact.path_offsets.size(),
            state.exact.path_offsets.size());
        builder.add(protocol::NumericType::int64,
            state.exact.path_indices.data(), state.exact.path_indices.size(),
            state.exact.path_indices.size());
        builder.add(protocol::NumericType::int64, state.exact.statuses.data(),
            state.exact.statuses.size(), state.exact.statuses.size());
        builder.add(protocol::NumericType::int64, state.exact.reasons.data(),
            state.exact.reasons.size(), state.exact.reasons.size());
        builder.add(protocol::NumericType::float64, state.exact.metrics.data(),
            state.exact.metrics.size(), route_count, 4);
        builder.add(protocol::NumericType::int64,
            state.exact.label_counters.data(),
            state.exact.label_counters.size(), route_count, 3);
        builder.add(protocol::NumericType::int64,
            state.exact.batch_counters.data(),
            state.exact.batch_counters.size(),
            state.exact.batch_counters.size());
        builder.add(protocol::NumericType::int64,
            state.objective_integer.data(), state.objective_integer.size(),
            state.objective_integer.size());
        builder.add(protocol::NumericType::float64,
            state.objective_float.data(), state.objective_float.size(),
            state.objective_float.size());
        builder.add(protocol::NumericType::int64, state.accounting.data(),
            state.accounting.size(), state.accounting.size());
        builder.add(protocol::NumericType::uint8, sha256.data(), sha256.size(),
            sha256.size());
        builder.add(protocol::NumericType::int64,
            state.exact.completion_order.data(),
            state.exact.completion_order.size(),
            state.exact.completion_order.size());
        builder.add(protocol::NumericType::uint8, opened.token.data(),
            opened.token.size(), opened.token.size());
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        builder.add(protocol::NumericType::float64, telemetry.data(),
            telemetry.size(), telemetry.size());
        return builder.finish();
    }
    case protocol::KernelOperation::candidate_session_close: {
        const auto token = evrptw::native_search::
            candidate_session_token_from_payload_v2(input);
        candidate_sessions.close(peer_pid, token);
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        return evrptw::native_search::candidate_session_token_payload_v2(
            protocol::KernelOperation::candidate_session_close, token,
            input.header().request_id, telemetry);
    }
    case protocol::KernelOperation::candidate_transaction_execute: {
        auto transaction = evrptw::native_search::
            candidate_transaction_execute_from_payload_v2(input);
        auto session = candidate_sessions.lookup(
            peer_pid, transaction.token);
        std::unique_lock session_lock(session->mutex, std::try_to_lock);
        if (!session_lock.owns_lock()) {
            throw std::runtime_error(
                "native candidate session already has an active transaction");
        }
        if (!session->transaction_state) {
            throw std::logic_error(
                "native candidate session lost its transaction state");
        }
        if (session->pending_budget_snapshot.has_value()) {
            throw std::runtime_error(
                "native candidate session has an unresolved transaction");
        }
        const auto session_remaining = session->request.config.deadline[1]
            - std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        if (session_remaining <= 0.0) {
            throw evrptw::native_search::CandidateTransactionDeadlineV2(
                evrptw::native_search::
                    CandidateTransactionDeadlinePhaseV2::before_transaction,
                "native candidate session deadline expired before transaction");
        }
        transaction.round.deadline_remaining = std::min(
            transaction.round.deadline_remaining, session_remaining);
        SchedulerCandidateTransactionKernelsV2 kernels(pool);
        const auto budget_snapshot =
            session->transaction_state->budget().snapshot();
        if (transaction.options.defer_commit) {
            pending_candidate_transaction = transaction.token;
        }
        evrptw::native_search::CandidateTransactionExecutionV2 execution;
        if (!transaction.round.ranking_route_offsets.empty()) {
            evrptw::native_search::CandidateTransactionExecutorV2 executor(
                session->request,
                session->transaction_state->exact_cache(),
                session->transaction_state->negative_cache(),
                session->transaction_state->budget(),
                session->transaction_state->attempted_plans(),
                transaction.round.ranking_route_offsets,
                transaction.round.ranking_route_indices,
                kernels);
            execution = executor.execute_with_trace(
                std::move(transaction.round), transaction.options);
        } else {
            evrptw::native_search::CandidateTransactionExecutorV2 executor(
                session->request, *session->transaction_state, kernels);
            execution = executor.execute_with_trace(
                std::move(transaction.round), transaction.options);
        }
        if (transaction.options.defer_commit) {
            session->pending_budget_snapshot = budget_snapshot;
            session->pending_exact_protocol =
                execution.trace.exact_protocol_active;
            session->pending_negative_store =
                execution.trace.negative_store_active;
            session->pending_attempted_mark =
                execution.trace.attempted_mark_active;
            session->transaction_resolution =
                CandidateSessionRegistry::Session::transaction_pending;
        }
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        return evrptw::native_search::
            candidate_transaction_execution_payload_v2(
                execution, input.header().request_id, telemetry);
    }
    case protocol::KernelOperation::candidate_transaction_commit:
    case protocol::KernelOperation::candidate_transaction_rollback: {
        const auto operation = input.header().operation;
        const auto token = evrptw::native_search::
            candidate_session_token_from_payload_v2(input);
        auto session = candidate_sessions.lookup(peer_pid, token);
        std::unique_lock session_lock(session->mutex, std::try_to_lock);
        if (!session_lock.owns_lock()) {
            throw std::runtime_error(
                "native candidate session already has an active transaction");
        }
        if (operation
           == protocol::KernelOperation::candidate_transaction_commit) {
            if (!session->pending_budget_snapshot.has_value()) {
                throw std::runtime_error(
                    "native candidate session has no pending transaction; resolution="
                    + std::to_string(session->transaction_resolution));
            }
            pending_candidate_transaction = token;
            pending_commit_session = token;
        } else {
            session->rollback_pending();
        }
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        return evrptw::native_search::candidate_session_token_payload_v2(
            operation, token, input.header().request_id, telemetry);
    }
    case protocol::KernelOperation::candidate_transaction_status: {
        const auto token = evrptw::native_search::
            candidate_session_token_from_payload_v2(input);
        auto session = candidate_sessions.lookup(peer_pid, token);
        std::unique_lock session_lock(session->mutex, std::try_to_lock);
        if (!session_lock.owns_lock()) {
            throw std::runtime_error(
                "native candidate session already has an active transaction");
        }
        const std::array<double, 7> telemetry{
            queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0,
            0.0, 0.0, 0.0};
        return evrptw::native_search::candidate_session_status_payload_v2(
            token, session->transaction_resolution,
            input.header().request_id, telemetry);
    }
    }
    throw std::runtime_error("native scheduler operation is invalid");
}

void patch_pool_telemetry(
    std::vector<std::uint8_t>& output,
    const NativeWorkPool& pool,
    std::size_t request_peak_active_tasks,
    SchedulerConcurrencySnapshot concurrency,
    pid_t peer_pid) {
    const protocol::PayloadView view(output.data(), output.size());
    if (view.header().array_count == 0) {
        throw std::logic_error("native scheduler output telemetry is missing");
    }
    const auto& descriptor = view.descriptor(view.header().array_count - 1);
    if (descriptor.type != protocol::NumericType::float64
        || descriptor.count != 7) {
        throw std::logic_error("native scheduler output telemetry is invalid");
    }
    auto* values = reinterpret_cast<double*>(
        output.data() + descriptor.offset);
    values[2] = static_cast<double>(request_peak_active_tasks);
    values[3] = static_cast<double>(pool.thread_count());
    values[4] = static_cast<double>(concurrency.peak_active_requests);
    values[5] = static_cast<double>(concurrency.peak_distinct_client_pids);
    values[6] = static_cast<double>(peer_pid);
}

void send_failure(
    int descriptor,
    std::uint64_t request_id,
    std::string_view message,
    protocol::FailureCode code = protocol::FailureCode::unspecified) noexcept {
    try {
        protocol::ControlFrame frame;
        frame.message = protocol::ControlMessage::failure;
        frame.request_id = request_id;
        frame.segment_size = static_cast<std::uint64_t>(code);
        protocol::copy_bounded(message, frame.error.data(), frame.error.size());
        send_exact(descriptor, &frame, sizeof(frame));
    } catch (...) {
    }
}

void handle_connection(
    int descriptor,
    NativeWorkPool& pool,
    std::atomic<bool>& stopping,
    int listening_descriptor,
    double queue_wait_seconds,
    std::size_t queue_depth,
    bool allow_fault_injection,
    SchedulerConcurrencyTelemetry& concurrency_telemetry,
    CandidateSessionRegistry& candidate_sessions) noexcept {
    std::uint64_t request_id = 0;
    pid_t peer_pid = 0;
    std::optional<std::string> pending_open_session;
    std::optional<std::string> pending_candidate_transaction;
    std::optional<std::string> pending_commit_session;
    try {
        ucred peer_credentials{};
        socklen_t peer_credentials_size = sizeof(peer_credentials);
        if (::getsockopt(
                descriptor, SOL_SOCKET, SO_PEERCRED, &peer_credentials,
                &peer_credentials_size) != 0
            || peer_credentials_size != sizeof(peer_credentials)
            || peer_credentials.pid <= 0) {
            throw std::runtime_error(
                "native scheduler peer credentials are invalid");
        }
        peer_pid = peer_credentials.pid;
        protocol::ControlFrame request;
        if (!read_exact(descriptor, &request, sizeof(request))) {
            ::close(descriptor);
            return;
        }
        validate_control(request);
        request_id = request.request_id;
        if (request.message == protocol::ControlMessage::shutdown) {
            if (request.request_id != 0 || request.segment_size != 0
                || !protocol::bounded_string(
                        request.segment_name.data(), request.segment_name.size()).empty()
                || !protocol::bounded_string(
                        request.sha256.data(), request.sha256.size()).empty()
                || !protocol::bounded_string(
                        request.error.data(), request.error.size()).empty()) {
                throw std::runtime_error(
                    "native scheduler shutdown frame is invalid");
            }
            stopping.store(true, std::memory_order_release);
            ::shutdown(listening_descriptor, SHUT_RDWR);
            protocol::ControlFrame released;
            released.message = protocol::ControlMessage::released;
            released.request_id = request_id;
            send_exact(descriptor, &released, sizeof(released));
            ::close(descriptor);
            return;
        }
        const auto test_request =
            request.message == protocol::ControlMessage::test_request;
        if ((request.message != protocol::ControlMessage::request && !test_request)
            || request.request_id == 0
            || request.segment_size < sizeof(protocol::PayloadHeader)
            || request.segment_size > protocol::maximum_payload_bytes) {
            throw std::runtime_error("native scheduler request frame is invalid");
        }
        const auto request_fault = protocol::bounded_string(
            request.error.data(), request.error.size());
        if ((test_request && !allow_fault_injection)
            || (!test_request && !request_fault.empty())) {
            throw std::runtime_error(
                "native scheduler fault injection is not enabled");
        }
        auto injected_fault = request_fault;
        if (!test_request && allow_fault_injection
            && !production_fault.empty()
            && production_fault != "exact_path_offset_oob"
            && production_fault != "deadline_text_screen_failure"
            && production_fault != "candidate_execute_output_failure"
            && production_fault != "candidate_commit_release_loss"
            && production_fault != "candidate_commit_before_apply_crash"
            && production_fault != "screen_response_before_local_apply_crash"
            && !production_fault_consumed.exchange(
                true, std::memory_order_acq_rel)) {
            injected_fault = production_fault;
        }
        if (injected_fault == "pause_before_execute") {
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
        const auto input_name = protocol::bounded_string(
            request.segment_name.data(), request.segment_name.size());
        if (!input_name.starts_with("/evrptw-s52-client-")
            || input_name.find('/', 1) != std::string::npos) {
            throw std::runtime_error(
                "native scheduler input shared-memory identity is invalid");
        }
        auto input = protocol::SharedMapping::open(
            input_name, static_cast<std::size_t>(request.segment_size));
        const std::string_view input_bytes(
            static_cast<const char*>(input.address()), input.size());
        if (protocol::native_sha256_hex(input_bytes)
            != protocol::bounded_string(request.sha256.data(), request.sha256.size())) {
            throw std::runtime_error("native scheduler input hash mismatch");
        }
        const protocol::PayloadView input_view(input.address(), input.size());
        if (input_view.header().request_id != request_id) {
            throw std::runtime_error("native scheduler request identity mismatch");
        }
        if (!test_request && allow_fault_injection
            && production_fault == "exact_path_offset_oob"
            && input_view.header().operation
                == protocol::KernelOperation::exact_charging
            && !production_fault_consumed.exchange(
                true, std::memory_order_acq_rel)) {
            injected_fault = production_fault;
        }
        if (!test_request && allow_fault_injection
            && production_fault == "deadline_text_screen_failure"
            && input_view.header().operation
                == protocol::KernelOperation::screen_routes
            && !production_fault_consumed.exchange(
                true, std::memory_order_acq_rel)) {
            injected_fault = production_fault;
        }
        if (!test_request && allow_fault_injection
            && production_fault == "candidate_execute_output_failure"
            && input_view.header().operation
                == protocol::KernelOperation::candidate_transaction_execute
            && !production_fault_consumed.exchange(
                true, std::memory_order_acq_rel)) {
            injected_fault = production_fault;
        }
        if (!test_request && allow_fault_injection
            && (production_fault == "candidate_commit_release_loss"
                || production_fault == "candidate_commit_before_apply_crash")
            && input_view.header().operation
                == protocol::KernelOperation::candidate_transaction_commit
            && !production_fault_consumed.exchange(
                true, std::memory_order_acq_rel)) {
            injected_fault = production_fault;
        }
        if (!test_request && allow_fault_injection
            && production_fault == "screen_response_before_local_apply_crash"
            && input_view.header().operation
                == protocol::KernelOperation::screen_routes
            && !production_fault_consumed.exchange(
                true, std::memory_order_acq_rel)) {
            injected_fault = production_fault;
        }
        if (injected_fault == "worker_exception") {
            throw std::runtime_error(
                "injected native scheduler worker exception");
        }
        if (injected_fault == "deadline_text_screen_failure") {
            throw std::runtime_error(
                "injected worker failure says deadline expired");
        }
        std::vector<std::uint8_t> output_bytes;
        std::size_t request_peak_active_tasks = 0;
        {
            ActiveRequestGuard active_request(
                concurrency_telemetry, peer_credentials.pid);
            if (input_view.header().operation
                == protocol::KernelOperation::screen_routes) {
                output_bytes = run_screen_routes(
                    input_view, queue_wait_seconds, queue_depth, pool,
                    request_peak_active_tasks);
            } else {
                request_peak_active_tasks = pool.active_task_count();
                output_bytes = execute(
                    input_view, queue_wait_seconds, queue_depth,
                    peer_credentials.pid, candidate_sessions, pool,
                    pending_candidate_transaction, pending_commit_session);
                request_peak_active_tasks = std::max(
                    request_peak_active_tasks, pool.peak_active_task_count());
            }
            // A request thread remains active while executing even when the
            // operation itself has no compute-pool work (for example control
            // transactions).  Keep this lower bound explicit for consumers
            // that use the metric as a liveness/ownership check.
            request_peak_active_tasks = std::max(
                request_peak_active_tasks, std::size_t{1});
        }
        if (injected_fault == "candidate_execute_output_failure") {
            throw std::runtime_error(
                "injected native candidate execute output failure");
        }
        if (injected_fault == "initial_state_path_offset_oob") {
            const protocol::PayloadView output_view(
                output_bytes.data(), output_bytes.size());
            if (output_view.header().operation
                    != protocol::KernelOperation::search_initial_state
                || output_view.descriptor(0).count < 2) {
                throw std::runtime_error(
                    "injected initial-state corruption has the wrong operation");
            }
            const auto& offsets = output_view.descriptor(0);
            auto* offset_values = reinterpret_cast<std::int64_t*>(
                output_bytes.data() + offsets.offset);
            offset_values[offsets.count - 1] = static_cast<std::int64_t>(
                output_view.descriptor(1).count + 1);
        }
        if (injected_fault == "exact_path_offset_oob") {
            const protocol::PayloadView output_view(
                output_bytes.data(), output_bytes.size());
            if (output_view.header().operation
                    != protocol::KernelOperation::exact_charging
                || output_view.descriptor(0).count < 1) {
                throw std::runtime_error(
                    "injected exact corruption has the wrong operation");
            }
            const auto& offsets = output_view.descriptor(0);
            auto* offset_values = reinterpret_cast<std::int64_t*>(
                output_bytes.data() + offsets.offset);
            offset_values[offsets.count - 1] = static_cast<std::int64_t>(
                output_view.descriptor(1).count + 1);
        }
        patch_pool_telemetry(
            output_bytes, pool, request_peak_active_tasks,
            concurrency_telemetry.snapshot(), peer_credentials.pid);
        if (input_view.header().operation
            == protocol::KernelOperation::candidate_session_open) {
            const protocol::PayloadView output_view(
                output_bytes.data(), output_bytes.size());
            const auto* token = output_view.data<std::uint8_t>(
                12, protocol::NumericType::uint8);
            pending_open_session.emplace(
                reinterpret_cast<const char*>(token),
                static_cast<std::size_t>(output_view.descriptor(12).count));
        }
        const auto output_name = "/evrptw-s52-kernel-" + std::to_string(::getpid())
            + "-" + scheduler_run_nonce + "-"
            + std::to_string(segment_counter.fetch_add(1));
        auto output = protocol::SharedMapping::create(output_name, output_bytes.size());
        std::memcpy(output.address(), output_bytes.data(), output_bytes.size());
        const auto output_sha = protocol::native_sha256_hex(std::string_view(
            reinterpret_cast<const char*>(output_bytes.data()), output_bytes.size()));
        protocol::ControlFrame response;
        response.message = protocol::ControlMessage::response;
        response.request_id = request_id;
        response.segment_size = output_bytes.size();
        protocol::copy_bounded(output.name(), response.segment_name.data(),
            response.segment_name.size());
        protocol::copy_bounded(output_sha, response.sha256.data(), response.sha256.size());
        send_exact(descriptor, &response, sizeof(response));
        if (injected_fault == "pause_after_response_before_ack") {
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
        if (injected_fault == "screen_response_before_local_apply_crash") {
            ::_exit(86);
        }
        protocol::ControlFrame acknowledgement;
        if (!read_exact(descriptor, &acknowledgement, sizeof(acknowledgement))) {
            throw std::runtime_error("native scheduler acknowledgement is missing");
        }
        validate_control(acknowledgement);
        if (acknowledgement.message != protocol::ControlMessage::acknowledgement
            || acknowledgement.request_id != request_id
            || protocol::bounded_string(
                   acknowledgement.segment_name.data(),
                   acknowledgement.segment_name.size()) != output.name()
            || protocol::bounded_string(
                   acknowledgement.sha256.data(), acknowledgement.sha256.size())
                != output_sha) {
            throw std::runtime_error("native scheduler acknowledgement mismatch");
        }
        if (injected_fault == "candidate_commit_before_apply_crash") {
            if (!pending_commit_session.has_value()) {
                throw std::logic_error(
                    "candidate commit crash injection lacks a pending commit");
            }
            ::_exit(86);
        }
        if (pending_commit_session.has_value()) {
            candidate_sessions.commit_pending(
                peer_pid, *pending_commit_session);
            pending_candidate_transaction.reset();
            pending_commit_session.reset();
        }
        if (injected_fault == "candidate_commit_release_loss") {
            throw std::runtime_error(
                "injected candidate commit release loss");
        }
        protocol::ControlFrame released;
        released.message = protocol::ControlMessage::released;
        released.request_id = request_id;
        send_exact(descriptor, &released, sizeof(released));
        pending_open_session.reset();
    } catch (const std::exception& error) {
        std::string failure(error.what());
        if (pending_candidate_transaction.has_value()) {
            if (!candidate_sessions.rollback_pending_noexcept(
                    peer_pid, *pending_candidate_transaction)) {
                failure.append("; pending transaction rollback failed");
                std::cerr << failure << '\n';
            }
        }
        if (pending_open_session.has_value()) {
            candidate_sessions.rollback_open_noexcept(
                peer_pid, *pending_open_session);
        }
        const auto code = dynamic_cast<const evrptw::native_search::
            CandidateTransactionDeadlineV2*>(&error) == nullptr
            ? protocol::FailureCode::unspecified
            : protocol::FailureCode::deadline;
        send_failure(descriptor, request_id, failure, code);
    }
    ::close(descriptor);
}

int make_listener(const std::string& socket_path) {
    if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::invalid_argument("native scheduler socket path is too long");
    }
    const auto descriptor = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (descriptor < 0) {
        throw std::runtime_error("native scheduler could not create its socket");
    }
    ::unlink(socket_path.c_str());
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (::bind(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address))
            != 0
        || ::chmod(socket_path.c_str(), 0600) != 0
        || ::listen(descriptor, 64) != 0) {
        ::close(descriptor);
        throw std::runtime_error("native scheduler could not bind/listen");
    }
    return descriptor;
}

template <typename Values>
void write_json_array(const Values& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << values[index];
    }
    std::cout << ']';
}


using TaskReceiptFileDescriptor =
    evrptw::native_telemetry::TaskReceiptFileDescriptor;
using TaskReceiptSpool = evrptw::native_telemetry::TaskReceiptSpool;

void emit_runtime_statistics(
    const NativeWorkPool::Statistics& work,
    const NativeRequestQueue::Statistics& request,
    const SchedulerConcurrencySnapshot concurrency,
    const std::int64_t worker_threads,
    const std::int64_t request_threads,
    const TaskReceiptFileDescriptor& task_receipts,
    const ProcessIoCounters terminal_io) {
    std::cout << std::setprecision(17)
        << "{\"schema_version\":\"stage05.2-native-scheduler-runtime-v4\""
        << ",\"worker_threads\":" << worker_threads
        << ",\"request_threads\":" << request_threads
        << ",\"receipt_writer_threads\":1"
        << ",\"peak_active_requests\":" << concurrency.peak_active_requests
        << ",\"peak_distinct_client_pids\":"
        << concurrency.peak_distinct_client_pids
        << ",\"request_queue\":{\"pending\":" << request.pending_requests
        << ",\"peak_pending\":" << request.peak_pending_requests
        << ",\"queue_full_count\":" << request.queue_full_count
        << ",\"rejected_count\":" << request.rejected_count
        << ",\"completed\":" << request.completed_requests
        << ",\"total_wait_seconds\":" << request.total_wait_seconds
        << ",\"maximum_wait_seconds\":" << request.maximum_wait_seconds
        << ",\"total_service_seconds\":" << request.total_service_seconds
        << ",\"maximum_service_seconds\":" << request.maximum_service_seconds
        << ",\"wait_histogram\":";
    write_json_array(request.wait_histogram);
    std::cout << ",\"service_histogram\":";
    write_json_array(request.service_histogram);
    std::cout << "},\"work_queue\":{\"pending\":" << work.pending_tasks
        << ",\"active\":" << work.active_tasks
        << ",\"peak_pending\":" << work.peak_pending_tasks
        << ",\"peak_active\":" << work.peak_active_tasks
        << ",\"queue_full_count\":" << work.queue_full_count
        << ",\"rejected_count\":" << work.rejected_count
        << ",\"completed\":" << work.completed_tasks
        << ",\"total_wait_seconds\":" << work.total_wait_seconds
        << ",\"maximum_wait_seconds\":" << work.maximum_wait_seconds
        << ",\"total_service_seconds\":" << work.total_service_seconds
        << ",\"maximum_service_seconds\":" << work.maximum_service_seconds
        << ",\"wait_histogram\":";
    write_json_array(work.wait_histogram);
    std::cout << ",\"service_histogram\":";
    write_json_array(work.service_histogram);
    std::cout << "},\"terminal_process_io\":{\"read_bytes\":"
        << terminal_io.read_bytes
        << ",\"write_bytes\":" << terminal_io.write_bytes
        << "},\"task_receipts\":{\"schema_version\":"
        << "\"stage05.2-native-work-task-receipts-v3\""
        << ",\"path\":\"" << task_receipts.path << "\""
        << ",\"sha256\":\"" << task_receipts.sha256 << "\""
        << ",\"bytes\":" << task_receipts.bytes
        << ",\"count\":" << task_receipts.count
        << ",\"storage_model\":\"bounded_async_fifo_stream\""
        << ",\"receipt_batch_capacity\":"
        << task_receipts.receipt_batch_capacity
        << ",\"queue_bound_batches\":"
        << task_receipts.queue_bound_batches
        << ",\"peak_queued_batches\":"
        << task_receipts.peak_queued_batches
        << ",\"submitted_batches\":"
        << task_receipts.submitted_batches
        << ",\"completed_batches\":"
        << task_receipts.completed_batches
        << ",\"dropped_count\":" << task_receipts.dropped_count
        << ",\"producer_wait_seconds\":"
        << task_receipts.producer_wait_seconds
        << ",\"writer_wall_seconds\":"
        << task_receipts.writer_wall_seconds
        << ",\"writer_cpu_seconds\":"
        << task_receipts.writer_cpu_seconds
        << ",\"serialization_seconds\":"
        << task_receipts.serialization_seconds
        << ",\"write_seconds\":" << task_receipts.write_seconds
        << ",\"file_fsync_seconds\":"
        << task_receipts.file_fsync_seconds
        << ",\"atomic_publish_seconds\":"
        << task_receipts.atomic_publish_seconds
        << ",\"parent_fsync_seconds\":"
        << task_receipts.parent_fsync_seconds
        << "}}\n";
    std::cout.flush();
    if (!std::cout) {
        throw std::runtime_error(
            "native scheduler runtime statistics could not be written");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--build-attestation") {
        std::cout
            << "{\"schema_version\":2,\"revision\":\""
            << EVRPTW_BUILD_GIT_REVISION
            << "\",\"git_tree\":\"" << EVRPTW_BUILD_GIT_TREE
            << "\",\"source_manifest_sha256\":\""
            << EVRPTW_BUILD_SOURCE_MANIFEST_SHA256
            << "\",\"tracked_file_count\":"
            << EVRPTW_BUILD_TRACKED_FILE_COUNT
            << ",\"source_dirty\":"
            << (EVRPTW_BUILD_SOURCE_DIRTY ? "true" : "false")
            << ",\"development_override\":"
            << (EVRPTW_BUILD_DEVELOPMENT_OVERRIDE ? "true" : "false")
            << ",\"cpp_source_kind\":\"" << EVRPTW_BUILD_CPP_SOURCE_KIND
            << "\""
            << ",\"performance_profile\":\""
            << EVRPTW_BUILD_PERFORMANCE_PROFILE
            << "\",\"compiler_id\":\"" << EVRPTW_BUILD_COMPILER_ID
            << "\",\"compiler_version\":\""
            << EVRPTW_BUILD_COMPILER_VERSION
            << "\",\"interprocedural_optimization\":"
            << (EVRPTW_BUILD_INTERPROCEDURAL_OPTIMIZATION ? "true" : "false")
            << ",\"host_native\":"
            << (EVRPTW_BUILD_HOST_NATIVE ? "true" : "false")
            << "}\n";
        return 0;
    }
    if (argc < 4 || argc > 12) {
        std::cerr << "usage: evrptw_native_scheduler SOCKET WORKER_THREADS "
                     "[REQUEST_THREADS] RUN_NONCE [--cpu-affinity=CPU[,CPU...]] "
                     "[--task-receipts=PATH] "
                     "[--enable-fault-injection]\n";
        return 2;
    }
    const std::string socket_path(argv[1]);
    int listener = -1;
    try {
        const auto worker_threads = std::stoll(argv[2]);
        std::int64_t request_thread_count = 6;
        std::size_t option_index = 4;
        const auto looks_like_nonce = [](const std::string_view value) {
            return value.size() == 16
                && value.find_first_not_of("0123456789abcdef")
                    == std::string::npos;
        };
        // The explicit form has the nonce in argv[4]; keying off that
        // position keeps legacy invocations unambiguous even when a caller
        // passes a request-thread value that happens to look like a nonce.
        if (argc >= 5 && looks_like_nonce(argv[4])) {
            request_thread_count = std::stoll(argv[3]);
            scheduler_run_nonce = argv[4];
            option_index = 5;
        } else {
            scheduler_run_nonce = argv[3];
        }
        const bool valid_nonce = scheduler_run_nonce.size() == 16
            && scheduler_run_nonce.find_first_not_of("0123456789abcdef")
                == std::string::npos;
        if (!valid_nonce) {
            throw std::invalid_argument("native scheduler run nonce is invalid");
        }
        bool allow_fault_injection = false;
        std::optional<std::string> cpu_affinity_spec;
        std::optional<std::filesystem::path> task_receipt_path;
        for (std::size_t index = option_index;
             index < static_cast<std::size_t>(argc); ++index) {
            const std::string_view option(argv[index]);
            if (option == "--enable-fault-injection") {
                allow_fault_injection = true;
            } else if (option.starts_with("--production-fault=")) {
                production_fault = option.substr(
                    std::string_view("--production-fault=").size());
            } else if (option.starts_with("--cpu-affinity=")) {
                if (cpu_affinity_spec.has_value()) {
                    throw std::invalid_argument(
                        "native scheduler CPU affinity was specified twice");
                }
                cpu_affinity_spec = std::string(option.substr(
                    std::string_view("--cpu-affinity=").size()));
            } else if (option.starts_with("--task-receipts=")) {
                if (task_receipt_path.has_value()) {
                    throw std::invalid_argument(
                        "native scheduler task receipts were specified twice");
                }
                const auto value = option.substr(
                    std::string_view("--task-receipts=").size());
                if (value.empty()) {
                    throw std::invalid_argument(
                        "native scheduler task-receipt path is empty");
                }
                task_receipt_path = std::filesystem::path(value);
            } else {
                throw std::invalid_argument("native scheduler option is invalid");
            }
        }
        if (!production_fault.empty()
            && (!allow_fault_injection
                || (production_fault != "pause_before_execute"
                    && production_fault
                        != "initial_state_path_offset_oob"
                    && production_fault != "exact_path_offset_oob"
                    && production_fault
                        != "deadline_text_screen_failure"
                    && production_fault
                        != "candidate_execute_output_failure"
                    && production_fault
                        != "candidate_commit_release_loss"
                    && production_fault
                        != "candidate_commit_before_apply_crash"
                    && production_fault
                        != "screen_response_before_local_apply_crash"))) {
            throw std::invalid_argument(
                "native scheduler production fault is invalid");
        }
        if (worker_threads <= 0 || worker_threads > 1024) {
            throw std::invalid_argument(
                "native scheduler compute thread count must be in [1, 1024]");
        }
        if (request_thread_count <= 0 || request_thread_count > 64) {
            throw std::invalid_argument(
                "native scheduler request thread count must be in [1, 64]");
        }
        if (cpu_affinity_spec.has_value()) {
            apply_cpu_affinity(*cpu_affinity_spec);
        }
        if (!task_receipt_path.has_value()) {
            task_receipt_path = std::filesystem::path(
                socket_path + ".task-receipts.jsonl");
        }
        TaskReceiptSpool task_receipt_spool(*task_receipt_path);
        NativeWorkPool pool(
            worker_threads,
            static_cast<std::size_t>(worker_threads)
                * static_cast<std::size_t>(request_thread_count),
            [&task_receipt_spool](const NativeWorkPool::TaskReceipt& receipt) {
                task_receipt_spool.record(receipt);
            },
            [&task_receipt_spool]() {
                task_receipt_spool.flush();
            });
        NativeRequestQueue queue;
        SchedulerConcurrencyTelemetry concurrency_telemetry;
        CandidateSessionRegistry candidate_sessions;
        std::atomic<bool> stopping{false};
        listener = make_listener(socket_path);
        std::vector<std::thread> request_threads;
        struct RequestThreadGuard {
            NativeRequestQueue& queue;
            std::vector<std::thread>& threads;

            ~RequestThreadGuard() noexcept {
                queue.stop();
                for (auto& thread : threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
            }
        } request_thread_guard{queue, request_threads};
        request_threads.reserve(static_cast<std::size_t>(request_thread_count));
        for (std::int64_t index = 0;
             index < request_thread_count; ++index) {
            request_threads.emplace_back([&]() {
                set_thread_name("s52-request");
                while (const auto request = queue.take()) {
                    const auto service_started = std::chrono::steady_clock::now();
                    const auto queue_wait_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - request->submitted_at).count();
                    handle_connection(
                        request->descriptor, pool, stopping, listener,
                        queue_wait_seconds, request->queue_depth_on_submit,
                        allow_fault_injection, concurrency_telemetry,
                        candidate_sessions);
                    queue.record_service(
                        std::chrono::steady_clock::now() - service_started);
                }
            });
        }
        while (!stopping.load(std::memory_order_acquire)) {
            const auto connection = ::accept(listener, nullptr, nullptr);
            if (connection < 0) {
                if (stopping.load(std::memory_order_acquire)) {
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("native scheduler accept failed");
            }
            timeval timeout{};
            timeout.tv_sec = 5;
            if (::setsockopt(
                    connection, SOL_SOCKET, SO_RCVTIMEO,
                    &timeout, sizeof(timeout)) != 0
                || ::setsockopt(
                    connection, SOL_SOCKET, SO_SNDTIMEO,
                    &timeout, sizeof(timeout)) != 0) {
                ::close(connection);
                continue;
            }
            if (!queue.submit(connection)) {
                send_failure(connection, 0, "native scheduler request queue is full");
                ::close(connection);
            }
        }
        queue.stop();
        for (auto& request_thread : request_threads) {
            if (request_thread.joinable()) {
                request_thread.join();
            }
        }
        pool.wait_until_idle();
        const auto work_statistics = pool.statistics();
        const auto task_receipts = task_receipt_spool.finalize(
            work_statistics.completed_tasks,
            work_statistics.task_receipt_dropped_count);
        const auto terminal_io = read_terminal_process_io();
        emit_runtime_statistics(
            work_statistics, queue.statistics(),
            concurrency_telemetry.snapshot(), worker_threads,
            request_thread_count, task_receipts, terminal_io);
        ::close(listener);
        listener = -1;
        ::unlink(socket_path.c_str());
        return 0;
    } catch (const std::exception& error) {
        if (listener >= 0) {
            ::close(listener);
        }
        ::unlink(socket_path.c_str());
        std::cerr << "native scheduler failure: " << error.what() << '\n';
        return 1;
    }
}

#else

int main() { return 2; }

#endif
