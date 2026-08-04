#pragma once

#ifdef __linux__

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

#include "native_kernel_protocol.hpp"
#include "native_search_core.hpp"
#include "native_sha256.hpp"
#include "native_solver_kernels.hpp"

namespace evrptw::native_client {

namespace protocol = evrptw::native_protocol;
namespace kernels = evrptw::native_kernels;

inline std::atomic<std::uint64_t> request_counter{1};
inline std::atomic<std::uint64_t> segment_counter{1};
inline thread_local double transaction_deadline_absolute =
    std::numeric_limits<double>::infinity();

struct KernelClientTelemetry final {
    double queue_wait_seconds = 0.0;
    std::size_t peak_queue_depth = 0;
    std::size_t peak_active_tasks = 0;
    std::size_t pool_thread_count = 0;
    std::size_t request_count = 0;
    std::size_t screening_batch_request_count = 0;
    std::size_t initial_state_request_count = 0;
};

inline thread_local KernelClientTelemetry telemetry;
class KernelClientTelemetryCollector final {
public:
    void record(
        const double* values,
        bool screening_batch,
        bool initial_state) {
        std::lock_guard lock(mutex_);
        telemetry_.queue_wait_seconds += values[0];
        telemetry_.peak_queue_depth = std::max(
            telemetry_.peak_queue_depth,
            static_cast<std::size_t>(values[1]));
        telemetry_.peak_active_tasks = std::max(
            telemetry_.peak_active_tasks,
            static_cast<std::size_t>(values[2]));
        telemetry_.pool_thread_count = static_cast<std::size_t>(values[3]);
        ++telemetry_.request_count;
        telemetry_.screening_batch_request_count += screening_batch ? 1U : 0U;
        telemetry_.initial_state_request_count += initial_state ? 1U : 0U;
    }

    KernelClientTelemetry snapshot() const {
        std::lock_guard lock(mutex_);
        return telemetry_;
    }

private:
    mutable std::mutex mutex_;
    KernelClientTelemetry telemetry_;
};

inline thread_local KernelClientTelemetryCollector* telemetry_collector = nullptr;

inline void reset_telemetry() noexcept { telemetry = {}; }

inline KernelClientTelemetry telemetry_snapshot() {
    return telemetry_collector == nullptr
        ? telemetry : telemetry_collector->snapshot();
}

inline void record_telemetry(
    const protocol::PayloadView& output,
    std::size_t index,
    bool screening_batch = false,
    bool initial_state = false) {
    const auto& descriptor = output.descriptor(index);
    if (descriptor.type != protocol::NumericType::float64
        || descriptor.count != 4 || descriptor.dimensions != 1
        || descriptor.shape[0] != 4 || descriptor.shape[1] != 0) {
        throw std::runtime_error("native kernel scheduler telemetry is invalid");
    }
    const auto* values = output.data<double>(
        index, protocol::NumericType::float64);
    if (!std::isfinite(values[0]) || values[0] < 0.0
        || !std::isfinite(values[1]) || values[1] < 1.0
        || values[1] != static_cast<double>(static_cast<std::size_t>(values[1]))
        || !std::isfinite(values[2]) || values[2] < 1.0 || values[2] > 24.0
        || values[2] != static_cast<double>(static_cast<std::size_t>(values[2]))
        || values[3] != 24.0) {
        throw std::runtime_error("native kernel scheduler telemetry values are invalid");
    }
    if (telemetry_collector != nullptr) {
        telemetry_collector->record(values, screening_batch, initial_state);
    } else {
        telemetry.queue_wait_seconds += values[0];
        telemetry.peak_queue_depth = std::max(
            telemetry.peak_queue_depth, static_cast<std::size_t>(values[1]));
        telemetry.peak_active_tasks = std::max(
            telemetry.peak_active_tasks, static_cast<std::size_t>(values[2]));
        telemetry.pool_thread_count = static_cast<std::size_t>(values[3]);
        ++telemetry.request_count;
        telemetry.screening_batch_request_count += screening_batch ? 1U : 0U;
        telemetry.initial_state_request_count += initial_state ? 1U : 0U;
    }
}

inline bool read_exact(int descriptor, void* output, std::size_t size) {
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

inline void send_exact(int descriptor, const void* input, std::size_t size) {
    const auto* cursor = static_cast<const std::uint8_t*>(input);
    while (size > 0) {
        const auto sent = ::send(descriptor, cursor, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            throw std::runtime_error("native kernel client could not send control data");
        }
        cursor += sent;
        size -= static_cast<std::size_t>(sent);
    }
}

class Socket final {
public:
    explicit Socket(
        std::string_view path,
        double deadline_absolute = transaction_deadline_absolute)
        : deadline_absolute_(deadline_absolute) {
        if (path.empty() || path.size() >= sizeof(sockaddr_un::sun_path)) {
            throw std::invalid_argument("native kernel scheduler socket is invalid");
        }
        descriptor_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (descriptor_ < 0) {
            throw std::runtime_error("native kernel client could not create its socket");
        }
        if (!configure_timeout()) {
            ::close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error(
                "native kernel client could not configure socket timeouts");
        }
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.data(), path.size());
        address.sun_path[path.size()] = '\0';
        if (::connect(
                descriptor_, reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) != 0) {
            ::close(descriptor_);
            descriptor_ = -1;
            throw std::runtime_error("native kernel client could not connect");
        }
    }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    ~Socket() noexcept {
        if (descriptor_ >= 0) {
            ::close(descriptor_);
        }
    }
    int get() const noexcept { return descriptor_; }

    [[nodiscard]] bool deadline_expired() const noexcept {
        return std::isfinite(deadline_absolute_)
            && monotonic_seconds() >= deadline_absolute_;
    }

    void refresh_timeout() {
        if (!configure_timeout()) {
            throw std::runtime_error(
                "native kernel client could not refresh socket timeout");
        }
    }

private:
    [[nodiscard]] static double monotonic_seconds() noexcept {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool configure_timeout() noexcept {
        const auto remaining = std::isfinite(deadline_absolute_)
            ? std::max(0.001, deadline_absolute_ - monotonic_seconds())
            : 125.0;
        const auto bounded = std::min(125.0, remaining);
        timeval timeout{};
        timeout.tv_sec = static_cast<decltype(timeout.tv_sec)>(bounded);
        timeout.tv_usec = static_cast<decltype(timeout.tv_usec)>(
            (bounded - static_cast<double>(timeout.tv_sec)) * 1'000'000.0);
        return ::setsockopt(
                   descriptor_, SOL_SOCKET, SO_RCVTIMEO,
                   &timeout, sizeof(timeout)) == 0
            && ::setsockopt(
                   descriptor_, SOL_SOCKET, SO_SNDTIMEO,
                   &timeout, sizeof(timeout)) == 0;
    }

    int descriptor_ = -1;
    double deadline_absolute_ = std::numeric_limits<double>::infinity();
};

inline protocol::PayloadView transact(
    std::string_view socket_path,
    std::vector<std::uint8_t> input_bytes,
    protocol::SharedMapping& output_mapping,
    protocol::ControlFrame& response,
    Socket& socket) {
    const auto request_id = [&]() {
        protocol::PayloadView view(input_bytes.data(), input_bytes.size());
        return view.header().request_id;
    }();
    const auto segment_name = "/evrptw-s52-client-" + std::to_string(::getpid())
        + "-" + std::to_string(segment_counter.fetch_add(1));
    auto input_mapping = protocol::SharedMapping::create(
        segment_name, input_bytes.size());
    std::memcpy(input_mapping.address(), input_bytes.data(), input_bytes.size());
    protocol::ControlFrame request;
    request.message = protocol::ControlMessage::request;
    request.request_id = request_id;
    request.segment_size = input_bytes.size();
    protocol::copy_bounded(input_mapping.name(), request.segment_name.data(),
        request.segment_name.size());
    const auto input_sha = protocol::native_sha256_hex(std::string_view(
        reinterpret_cast<const char*>(input_bytes.data()), input_bytes.size()));
    protocol::copy_bounded(input_sha, request.sha256.data(), request.sha256.size());
    static_cast<void>(socket_path);
    send_exact(socket.get(), &request, sizeof(request));
    if (!read_exact(socket.get(), &response, sizeof(response))) {
        throw std::runtime_error("native kernel scheduler returned a partial response");
    }
    if (socket.deadline_expired()) {
        throw std::runtime_error(
            "native kernel scheduler response crossed its deadline");
    }
    if (response.magic != protocol::kernel_magic
        || response.version != protocol::kernel_protocol_version
        || response.request_id != request_id) {
        throw std::runtime_error("native kernel scheduler response identity mismatch");
    }
    if (response.message == protocol::ControlMessage::failure) {
        throw std::runtime_error(protocol::bounded_string(
            response.error.data(), response.error.size()));
    }
    if (response.message != protocol::ControlMessage::response
        || response.segment_size < sizeof(protocol::PayloadHeader)
        || response.segment_size > protocol::maximum_payload_bytes) {
        throw std::runtime_error("native kernel scheduler response is invalid");
    }
    const auto output_name = protocol::bounded_string(
        response.segment_name.data(), response.segment_name.size());
    if (!output_name.starts_with("/evrptw-s52-kernel-")
        || output_name.find('/', 1) != std::string::npos) {
        throw std::runtime_error(
            "native kernel scheduler output identity is invalid");
    }
    output_mapping = protocol::SharedMapping::open(
        output_name, static_cast<std::size_t>(response.segment_size));
    const std::string_view output_bytes(
        static_cast<const char*>(output_mapping.address()), output_mapping.size());
    const auto output_sha = protocol::bounded_string(
        response.sha256.data(), response.sha256.size());
    if (protocol::native_sha256_hex(output_bytes) != output_sha) {
        throw std::runtime_error("native kernel scheduler output hash mismatch");
    }
    return protocol::PayloadView(output_mapping.address(), output_mapping.size());
}

inline void acknowledge(
    Socket& socket,
    const protocol::ControlFrame& response) {
    if (socket.deadline_expired()) {
        throw std::runtime_error(
            "native kernel scheduler acknowledgement crossed its deadline");
    }
    socket.refresh_timeout();
    protocol::ControlFrame acknowledgement;
    acknowledgement.message = protocol::ControlMessage::acknowledgement;
    acknowledgement.request_id = response.request_id;
    acknowledgement.segment_name = response.segment_name;
    acknowledgement.sha256 = response.sha256;
    send_exact(socket.get(), &acknowledgement, sizeof(acknowledgement));
    protocol::ControlFrame released;
    if (!read_exact(socket.get(), &released, sizeof(released))
        || released.magic != protocol::kernel_magic
        || released.version != protocol::kernel_protocol_version
        || released.message != protocol::ControlMessage::released
        || released.request_id != response.request_id) {
        throw std::runtime_error("native kernel scheduler release receipt is invalid");
    }
    if (socket.deadline_expired()) {
        throw std::runtime_error(
            "native kernel scheduler release crossed its deadline");
    }
}

struct SearchRequestReceipt final {
    std::array<std::int64_t, 8> counts{};
    std::string sha256;
};

inline SearchRequestReceipt search_request_receipt(
    std::string_view socket_path,
    const evrptw::native_search::RequestV2& request) {
    const auto request_id = request_counter.fetch_add(1);
    Socket socket(
        socket_path,
        std::min(
            request.config.deadline[1], transaction_deadline_absolute));
    protocol::SharedMapping output_mapping;
    protocol::ControlFrame response;
    const auto output = transact(
        socket_path,
        evrptw::native_search::request_payload(request, request_id),
        output_mapping,
        response,
        socket);
    if (output.header().operation
            != protocol::KernelOperation::search_request_receipt
        || output.header().request_id != response.request_id
        || output.header().array_count != 3
        || output.descriptor(0).count != 8
        || output.descriptor(1).count != 64) {
        throw std::runtime_error(
            "native search-request receipt schema is invalid");
    }
    SearchRequestReceipt receipt;
    const auto* counts = output.data<std::int64_t>(
        0, protocol::NumericType::int64);
    std::copy(counts, counts + receipt.counts.size(), receipt.counts.begin());
    const auto* sha256 = output.data<std::uint8_t>(
        1, protocol::NumericType::uint8);
    receipt.sha256.assign(
        reinterpret_cast<const char*>(sha256),
        static_cast<std::size_t>(output.descriptor(1).count));
    if (receipt.sha256 != request.sha256()) {
        throw std::runtime_error(
            "native search-request receipt hash mismatch");
    }
    record_telemetry(output, 2);
    acknowledge(socket, response);
    return receipt;
}

inline evrptw::native_search::InitialStateV2 search_initial_state(
    std::string_view socket_path,
    const evrptw::native_search::RequestV2& request) {
    const auto request_id = request_counter.fetch_add(1);
    Socket socket(
        socket_path,
        std::min(request.config.deadline[1], transaction_deadline_absolute));
    if (socket.deadline_expired()) {
        throw std::runtime_error(
            "native initial-search-state deadline expired before IPC");
    }
    protocol::SharedMapping output_mapping;
    protocol::ControlFrame response;
    const auto output = transact(
        socket_path,
        evrptw::native_search::request_payload(
            request, request_id,
            protocol::KernelOperation::search_initial_state),
        output_mapping, response, socket);
    const auto route_count = request.problem.route_count();
    for (const auto index : {
             std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3},
             std::size_t{6}, std::size_t{7}, std::size_t{8}, std::size_t{9},
             std::size_t{10}, std::size_t{11}, std::size_t{12}}) {
        const auto& descriptor = output.descriptor(index);
        if (descriptor.dimensions != 1
            || descriptor.shape[0] != descriptor.count
            || descriptor.shape[1] != 0) {
            throw std::runtime_error(
                "native initial-search-state vector descriptor is invalid");
        }
    }
    if (output.header().operation
            != protocol::KernelOperation::search_initial_state
        || output.header().request_id != response.request_id
        || output.header().array_count != 13
        || output.descriptor(0).count != route_count + 1
        || output.descriptor(2).count != route_count
        || output.descriptor(3).count != route_count
        || output.descriptor(4).count != route_count * 4
        || output.descriptor(4).dimensions != 2
        || output.descriptor(4).shape[0] != route_count
        || output.descriptor(4).shape[1] != 4
        || output.descriptor(5).count != route_count * 3
        || output.descriptor(5).dimensions != 2
        || output.descriptor(5).shape[0] != route_count
        || output.descriptor(5).shape[1] != 3
        || output.descriptor(6).count != 10
        || output.descriptor(7).count != 2
        || output.descriptor(8).count != 2
        || output.descriptor(9).count != 4
        || output.descriptor(10).count != 64
        || output.descriptor(11).count != route_count) {
        throw std::runtime_error(
            "native initial-search-state response schema is invalid");
    }
    evrptw::native_search::InitialStateV2 state;
    state.request_sha256 = request.sha256();
    const auto copy_vector = [&output]<typename T>(
        std::size_t index, protocol::NumericType type) {
        const auto count = static_cast<std::size_t>(
            output.descriptor(index).count);
        const auto* values = output.data<T>(index, type);
        return std::vector<T>(values, values + count);
    };
    state.exact.path_offsets = copy_vector.template operator()<std::int64_t>(
        0, protocol::NumericType::int64);
    state.exact.path_indices = copy_vector.template operator()<std::int64_t>(
        1, protocol::NumericType::int64);
    state.exact.statuses = copy_vector.template operator()<std::int64_t>(
        2, protocol::NumericType::int64);
    state.exact.reasons = copy_vector.template operator()<std::int64_t>(
        3, protocol::NumericType::int64);
    state.exact.metrics = copy_vector.template operator()<double>(
        4, protocol::NumericType::float64);
    state.exact.label_counters =
        copy_vector.template operator()<std::int64_t>(
            5, protocol::NumericType::int64);
    state.exact.batch_counters =
        copy_vector.template operator()<std::int64_t>(
            6, protocol::NumericType::int64);
    state.exact.completion_order =
        copy_vector.template operator()<std::int64_t>(
            11, protocol::NumericType::int64);
    const auto* objective_integer = output.data<std::int64_t>(
        7, protocol::NumericType::int64);
    std::copy(objective_integer, objective_integer + 2,
        state.objective_integer.begin());
    const auto* objective_float = output.data<double>(
        8, protocol::NumericType::float64);
    std::copy(objective_float, objective_float + 2,
        state.objective_float.begin());
    const auto* accounting = output.data<std::int64_t>(
        9, protocol::NumericType::int64);
    std::copy(accounting, accounting + 4, state.accounting.begin());
    const auto* sha256 = output.data<std::uint8_t>(
        10, protocol::NumericType::uint8);
    const std::string remote_sha256(
        reinterpret_cast<const char*>(sha256), 64);
    state.validate(request);
    if (state.sha256() != remote_sha256) {
        throw std::runtime_error(
            "native initial-search-state response hash mismatch");
    }
    if (std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count()
        >= request.config.deadline[1]) {
        throw std::runtime_error(
            "native initial-search-state response arrived at or after deadline");
    }
    record_telemetry(output, 12, false, true);
    acknowledge(socket, response);
    return state;
}

inline kernels::ExactBatchOutput exact_charging(
    std::string_view socket_path,
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
    std::size_t order_count,
    double deadline_remaining,
    std::int64_t batch_size) {
    const auto request_id = request_counter.fetch_add(1);
    protocol::PayloadBuilder builder(
        protocol::KernelOperation::exact_charging, request_id);
    if (std::isnan(deadline_remaining)) {
        throw std::invalid_argument("native exact deadline must not be NaN");
    }
    const auto deadline_absolute =
        std::isinf(deadline_remaining) && deadline_remaining > 0.0
        ? deadline_remaining
        : std::chrono::duration<double>(
              std::chrono::steady_clock::now().time_since_epoch()).count()
            + std::max(0.0, deadline_remaining);
    builder.add(protocol::NumericType::int64, node_kinds, node_count, node_count);
    builder.add(protocol::NumericType::float64, ready, node_count, node_count);
    builder.add(protocol::NumericType::float64, due, node_count, node_count);
    builder.add(protocol::NumericType::float64, service, node_count, node_count);
    builder.add(protocol::NumericType::float64, distances,
        node_count * node_count, node_count, node_count);
    builder.add(protocol::NumericType::float64, vehicle, 5, 5);
    builder.add(protocol::NumericType::int64, order_offsets,
        route_count + 1, route_count + 1);
    builder.add(protocol::NumericType::int64, order_indices,
        order_count, order_count);
    builder.add(protocol::NumericType::float64, &deadline_absolute, 1, 1);
    builder.add(protocol::NumericType::int64, &batch_size, 1, 1);
    Socket socket(
        socket_path,
        std::min(deadline_absolute, transaction_deadline_absolute));
    protocol::SharedMapping output_mapping;
    protocol::ControlFrame response;
    const auto output = transact(
        socket_path, builder.finish(), output_mapping, response, socket);
    if (output.header().operation != protocol::KernelOperation::exact_charging
        || output.header().request_id != response.request_id
        || output.header().array_count != 9) {
        throw std::runtime_error("native exact output schema is invalid");
    }
    const auto one_dimensional = [&](const std::size_t index,
                                     const protocol::NumericType type,
                                     const std::uint64_t count) {
        const auto& item = output.descriptor(index);
        return item.type == type && item.dimensions == 1
            && item.count == count && item.shape[0] == count
            && item.shape[1] == 0;
    };
    const auto two_dimensional = [&](const std::size_t index,
                                     const protocol::NumericType type,
                                     const std::uint64_t first,
                                     const std::uint64_t second) {
        const auto& item = output.descriptor(index);
        return item.type == type && item.dimensions == 2
            && item.count == first * second && item.shape[0] == first
            && item.shape[1] == second;
    };
    const auto& path_descriptor = output.descriptor(1);
    if (!one_dimensional(
            0, protocol::NumericType::int64,
            static_cast<std::uint64_t>(route_count + 1))
        || path_descriptor.type != protocol::NumericType::int64
        || path_descriptor.dimensions != 1
        || path_descriptor.shape[0] != path_descriptor.count
        || path_descriptor.shape[1] != 0
        || !one_dimensional(
            2, protocol::NumericType::int64,
            static_cast<std::uint64_t>(route_count))
        || !one_dimensional(
            3, protocol::NumericType::int64,
            static_cast<std::uint64_t>(route_count))
        || !two_dimensional(
            4, protocol::NumericType::float64,
            static_cast<std::uint64_t>(route_count), 4)
        || !two_dimensional(
            5, protocol::NumericType::int64,
            static_cast<std::uint64_t>(route_count), 3)
        || !one_dimensional(6, protocol::NumericType::int64, 10)
        || output.descriptor(7).type != protocol::NumericType::int64
        || output.descriptor(7).dimensions != 1
        || output.descriptor(7).shape[0] != output.descriptor(7).count
        || output.descriptor(7).shape[1] != 0
        || output.descriptor(7).count > route_count
        || !one_dimensional(8, protocol::NumericType::float64, 4)) {
        throw std::runtime_error("native exact output schema is invalid");
    }
    kernels::ExactBatchOutput result;
    const auto copy_int64 = [&](std::size_t index) {
        const auto count = static_cast<std::size_t>(output.descriptor(index).count);
        const auto* data = output.data<std::int64_t>(
            index, protocol::NumericType::int64);
        return std::vector<std::int64_t>(data, data + count);
    };
    const auto copy_float64 = [&](std::size_t index) {
        const auto count = static_cast<std::size_t>(output.descriptor(index).count);
        const auto* data = output.data<double>(index, protocol::NumericType::float64);
        return std::vector<double>(data, data + count);
    };
    result.path_offsets = copy_int64(0);
    result.path_indices = copy_int64(1);
    result.statuses = copy_int64(2);
    result.reasons = copy_int64(3);
    result.metrics = copy_float64(4);
    result.label_counters = copy_int64(5);
    result.batch_counters = copy_int64(6);
    result.completion_order = copy_int64(7);
    std::int64_t depot = -1;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (node_kinds[node] == kernels::depot_kind) {
            if (depot >= 0) {
                throw std::runtime_error(
                    "native exact output schema is invalid: request has multiple depots");
            }
            depot = static_cast<std::int64_t>(node);
        }
    }
    kernels::validate_exact_batch_output(
        result, node_kinds, order_offsets, order_indices, node_count,
        route_count, order_count, depot, batch_size);
    record_telemetry(output, 8);
    acknowledge(socket, response);
    return result;
}

inline kernels::ScreenOutput screen_route(
    std::string_view socket_path,
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
    const double* options,
    const double* incremental) {
    const auto request_id = request_counter.fetch_add(1);
    protocol::PayloadBuilder builder(
        protocol::KernelOperation::screen_route, request_id);
    builder.add(protocol::NumericType::int64, kinds, node_count, node_count);
    builder.add(protocol::NumericType::float64, demands, node_count, node_count);
    builder.add(protocol::NumericType::float64, ready, node_count, node_count);
    builder.add(protocol::NumericType::float64, due, node_count, node_count);
    builder.add(protocol::NumericType::float64, service, node_count, node_count);
    builder.add(protocol::NumericType::float64, distances,
        node_count * node_count, node_count, node_count);
    builder.add(protocol::NumericType::uint8, reachable,
        node_count * node_count, node_count, node_count);
    builder.add(protocol::NumericType::float64, vehicle, 5, 5);
    builder.add(protocol::NumericType::int64, route, route_size, route_size);
    builder.add(protocol::NumericType::float64, options, 4, 4);
    builder.add(protocol::NumericType::float64, incremental, 6, 6);
    Socket socket(socket_path);
    protocol::SharedMapping output_mapping;
    protocol::ControlFrame response;
    const auto output = transact(
        socket_path, builder.finish(), output_mapping, response, socket);
    if (output.header().operation != protocol::KernelOperation::screen_route
        || output.header().request_id != response.request_id
        || output.header().array_count != 4
        || output.descriptor(0).count != 16
        || output.descriptor(1).count != 15
        || output.descriptor(2).count != 1) {
        throw std::runtime_error("native screening output schema is invalid");
    }
    kernels::ScreenOutput result;
    const auto* codes = output.data<std::int64_t>(0, protocol::NumericType::int64);
    const auto* metrics = output.data<double>(1, protocol::NumericType::float64);
    result.codes.assign(codes, codes + 16);
    result.metrics.assign(metrics, metrics + 15);
    result.reachability_queries = output.data<std::int64_t>(
        2, protocol::NumericType::int64)[0];
    record_telemetry(output, 3);
    acknowledge(socket, response);
    return result;
}

inline std::vector<kernels::ScreenOutput> screen_routes(
    std::string_view socket_path,
    const std::int64_t* kinds,
    const double* demands,
    const double* ready,
    const double* due,
    const double* service,
    const double* distances,
    const std::uint8_t* reachable,
    const double* vehicle,
    const std::int64_t* route_offsets,
    const std::int64_t* route_indices,
    std::size_t route_count,
    std::size_t route_index_count,
    std::size_t node_count,
    const double* options,
    const double* incremental) {
    const auto request_id = request_counter.fetch_add(1);
    protocol::PayloadBuilder builder(
        protocol::KernelOperation::screen_routes, request_id);
    builder.add(protocol::NumericType::int64, kinds, node_count, node_count);
    builder.add(protocol::NumericType::float64, demands, node_count, node_count);
    builder.add(protocol::NumericType::float64, ready, node_count, node_count);
    builder.add(protocol::NumericType::float64, due, node_count, node_count);
    builder.add(protocol::NumericType::float64, service, node_count, node_count);
    builder.add(protocol::NumericType::float64, distances,
        node_count * node_count, node_count, node_count);
    builder.add(protocol::NumericType::uint8, reachable,
        node_count * node_count, node_count, node_count);
    builder.add(protocol::NumericType::float64, vehicle, 5, 5);
    builder.add(protocol::NumericType::int64, route_offsets,
        route_count + 1, route_count + 1);
    builder.add(protocol::NumericType::int64, route_indices,
        route_index_count, route_index_count);
    builder.add(protocol::NumericType::float64, options, 4, 4);
    builder.add(protocol::NumericType::float64, incremental, 6, 6);
    Socket socket(socket_path);
    protocol::SharedMapping output_mapping;
    protocol::ControlFrame response;
    const auto output = transact(
        socket_path, builder.finish(), output_mapping, response, socket);
    const auto expected_code_count = protocol::checked_product(
        route_count, 16, "native screening-batch code count overflows");
    const auto expected_metric_count = protocol::checked_product(
        route_count, 15, "native screening-batch metric count overflows");
    if (output.header().operation != protocol::KernelOperation::screen_routes
        || output.header().request_id != response.request_id
        || output.header().array_count != 4
        || output.descriptor(0).count != expected_code_count
        || output.descriptor(1).count != expected_metric_count
        || output.descriptor(2).count != route_count) {
        throw std::runtime_error("native screening-batch output schema is invalid");
    }
    const auto* codes = output.data<std::int64_t>(
        0, protocol::NumericType::int64);
    const auto* metrics = output.data<double>(
        1, protocol::NumericType::float64);
    const auto* queries = output.data<std::int64_t>(
        2, protocol::NumericType::int64);
    std::vector<kernels::ScreenOutput> results(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        results[route].codes.assign(codes + route * 16, codes + (route + 1) * 16);
        results[route].metrics.assign(
            metrics + route * 15, metrics + (route + 1) * 15);
        results[route].reachability_queries = queries[route];
    }
    record_telemetry(output, 3, true);
    acknowledge(socket, response);
    return results;
}

inline void test_fault(
    std::string_view socket_path,
    std::string_view fault) {
    if (fault != "request_hash_mismatch"
        && fault != "client_disconnect_after_request"
        && fault != "ack_loss"
        && fault != "invalid_ack"
        && fault != "worker_exception"
        && fault != "pause_before_execute"
        && fault != "pause_after_response_before_ack"
        && fault != "descriptor_count_overflow"
        && fault != "route_index_oob"
        && fault != "wrong_rank"
        && fault != "trailing_payload"
        && fault != "oversized_control"
        && fault != "shared_memory_identity") {
        throw std::invalid_argument("unknown native scheduler fault injection");
    }
    const auto request_id = request_counter.fetch_add(1);
    const std::array<std::int64_t, 2> kinds{
        kernels::depot_kind, kernels::customer_kind};
    const std::array<double, 2> ready{0.0, 0.0};
    const std::array<double, 2> due{1.0e9, 1.0e9};
    const std::array<double, 2> service{0.0, 0.0};
    const std::array<double, 4> distances{0.0, 1.0, 1.0, 0.0};
    const std::array<double, 5> vehicle{1.0, 1.0, 1.0, 1.0, 1.0};
    const bool invalid_route = fault == "route_index_oob";
    const std::array<std::int64_t, 2> offsets{0, 1};
    const std::array<std::int64_t, 1> route_indices{
        invalid_route ? 99 : 1};
    const double deadline_absolute = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count() + 10.0;
    const std::int64_t batch_size = 1;
    protocol::PayloadBuilder builder(
        protocol::KernelOperation::exact_charging, request_id);
    builder.add(protocol::NumericType::int64, kinds.data(), 2, 2);
    builder.add(protocol::NumericType::float64, ready.data(), 2, 2);
    builder.add(protocol::NumericType::float64, due.data(), 2, 2);
    builder.add(protocol::NumericType::float64, service.data(), 2, 2);
    builder.add(protocol::NumericType::float64, distances.data(), 4, 2, 2);
    builder.add(protocol::NumericType::float64, vehicle.data(), 5, 5);
    builder.add(protocol::NumericType::int64, offsets.data(), 2, 2);
    builder.add(
        protocol::NumericType::int64, route_indices.data(), 1, 1);
    builder.add(
        protocol::NumericType::float64, &deadline_absolute, 1, 1);
    builder.add(protocol::NumericType::int64, &batch_size, 1, 1);
    auto input_bytes = builder.finish();
    if (fault == "descriptor_count_overflow") {
        auto* header = reinterpret_cast<protocol::PayloadHeader*>(
            input_bytes.data());
        header->arrays[0].count = std::numeric_limits<std::uint64_t>::max();
        header->arrays[0].shape[0] = header->arrays[0].count;
    } else if (fault == "wrong_rank") {
        auto* header = reinterpret_cast<protocol::PayloadHeader*>(
            input_bytes.data());
        header->arrays[1].dimensions = 2;
        header->arrays[1].shape[0] = 1;
        header->arrays[1].shape[1] = 2;
    } else if (fault == "trailing_payload") {
        input_bytes.push_back(0U);
    }
    const auto segment_name = "/evrptw-s52-client-" + std::to_string(::getpid())
        + "-" + std::to_string(segment_counter.fetch_add(1));
    auto input_mapping = protocol::SharedMapping::create(
        segment_name, input_bytes.size());
    std::memcpy(input_mapping.address(), input_bytes.data(), input_bytes.size());
    protocol::ControlFrame request;
    request.message = protocol::ControlMessage::test_request;
    request.request_id = request_id;
    request.segment_size = fault == "oversized_control"
        ? protocol::maximum_payload_bytes + 1ULL
        : input_bytes.size();
    protocol::copy_bounded(input_mapping.name(), request.segment_name.data(),
        request.segment_name.size());
    if (fault == "shared_memory_identity") {
        protocol::copy_bounded(
            "/not-evrptw-owned", request.segment_name.data(),
            request.segment_name.size());
    }
    auto input_sha = protocol::native_sha256_hex(std::string_view(
        reinterpret_cast<const char*>(input_bytes.data()), input_bytes.size()));
    if (fault == "request_hash_mismatch") {
        input_sha.front() = input_sha.front() == '0' ? '1' : '0';
    }
    protocol::copy_bounded(input_sha, request.sha256.data(), request.sha256.size());
    protocol::copy_bounded(fault, request.error.data(), request.error.size());
    Socket socket(socket_path);
    send_exact(socket.get(), &request, sizeof(request));
    if (fault == "client_disconnect_after_request") {
        throw std::runtime_error("native scheduler client disconnect without fallback");
    }
    protocol::ControlFrame response;
    if (!read_exact(socket.get(), &response, sizeof(response))) {
        throw std::runtime_error("native scheduler returned a partial IPC frame");
    }
    if (response.magic != protocol::kernel_magic
        || response.version != protocol::kernel_protocol_version
        || response.request_id != request_id) {
        throw std::runtime_error("native scheduler fault response identity mismatch");
    }
    if (response.message == protocol::ControlMessage::failure) {
        throw std::runtime_error(protocol::bounded_string(
            response.error.data(), response.error.size()));
    }
    if (response.message != protocol::ControlMessage::response) {
        throw std::runtime_error("native scheduler fault response is invalid");
    }
    const auto output_name = protocol::bounded_string(
        response.segment_name.data(), response.segment_name.size());
    auto output_mapping = protocol::SharedMapping::open(
        output_name, static_cast<std::size_t>(response.segment_size));
    const std::string_view output_bytes(
        static_cast<const char*>(output_mapping.address()), output_mapping.size());
    if (protocol::native_sha256_hex(output_bytes)
        != protocol::bounded_string(response.sha256.data(), response.sha256.size())) {
        throw std::runtime_error("native scheduler fault output hash mismatch");
    }
    if (fault == "ack_loss") {
        throw std::runtime_error(
            "native scheduler acknowledgement loss without fallback");
    }
    protocol::ControlFrame acknowledgement;
    acknowledgement.message = protocol::ControlMessage::acknowledgement;
    acknowledgement.request_id = response.request_id;
    acknowledgement.segment_name = response.segment_name;
    acknowledgement.sha256 = response.sha256;
    acknowledgement.sha256[0] = acknowledgement.sha256[0] == '0' ? '1' : '0';
    send_exact(socket.get(), &acknowledgement, sizeof(acknowledgement));
    protocol::ControlFrame released;
    if (!read_exact(socket.get(), &released, sizeof(released))) {
        throw std::runtime_error(
            "native scheduler did not confirm output release");
    }
    throw std::runtime_error("native scheduler did not confirm output release");
}

}  // namespace evrptw::native_client

#endif
