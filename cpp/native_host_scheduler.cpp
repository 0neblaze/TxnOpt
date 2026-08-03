#ifdef __linux__

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <vector>

#include "native_concurrency.hpp"
#include "native_kernel_protocol.hpp"
#include "native_sha256.hpp"
#include "native_solver_kernels.hpp"

namespace protocol = evrptw::native_protocol;
namespace kernels = evrptw::native_kernels;

namespace {

std::atomic<std::uint64_t> segment_counter{0};
std::string scheduler_run_nonce;
std::string production_fault;
std::atomic<bool> production_fault_consumed{false};

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
    std::size_t queue_depth) {
    if (input.header().array_count != 10) {
        throw std::runtime_error("native scheduler exact request shape is invalid");
    }
    const auto& kind_shape = input.descriptor(0);
    const auto& distance_shape = input.descriptor(4);
    const auto& vehicle_shape = input.descriptor(5);
    const auto& offset_shape = input.descriptor(6);
    const auto& index_shape = input.descriptor(7);
    const auto node_count = static_cast<std::size_t>(kind_shape.count);
    if (kind_shape.dimensions != 1 || distance_shape.dimensions != 2
        || node_count == 0
        || input.descriptor(1).count != kind_shape.count
        || input.descriptor(2).count != kind_shape.count
        || input.descriptor(3).count != kind_shape.count
        || distance_shape.shape[0] != kind_shape.count
        || distance_shape.shape[1] != kind_shape.count
        || vehicle_shape.count != 5 || offset_shape.count < 1
        || input.descriptor(8).count != 1 || input.descriptor(9).count != 1) {
        throw std::runtime_error("native scheduler exact dimensions are invalid");
    }
    const auto route_count = static_cast<std::size_t>(offset_shape.count - 1);
    const auto* kinds = input.data<std::int64_t>(0, protocol::NumericType::int64);
    const auto* offsets = input.data<std::int64_t>(6, protocol::NumericType::int64);
    const auto* indices = input.data<std::int64_t>(7, protocol::NumericType::int64);
    if (offsets[0] != 0
        || offsets[route_count] != static_cast<std::int64_t>(index_shape.count)) {
        throw std::runtime_error("native scheduler exact offsets are invalid");
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> stations;
    for (std::size_t node = 0; node < node_count; ++node) {
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
    const auto deadline =
        input.data<double>(8, protocol::NumericType::float64)[0];
    const auto batch_size =
        input.data<std::int64_t>(9, protocol::NumericType::int64)[0];
    if (!std::isfinite(deadline) || deadline <= 0.0 || batch_size <= 0) {
        throw std::runtime_error(
            "native scheduler exact control values are invalid");
    }
    const auto output = kernels::run_exact_charging_batch(
        kinds,
        input.data<double>(1, protocol::NumericType::float64),
        input.data<double>(2, protocol::NumericType::float64),
        input.data<double>(3, protocol::NumericType::float64),
        input.data<double>(4, protocol::NumericType::float64),
        input.data<double>(5, protocol::NumericType::float64),
        offsets, indices, node_count, route_count, depot, stations,
        deadline, batch_size);
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
    const std::array<double, 4> telemetry{
        queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0};
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
    const std::array<double, 4> telemetry{
        queue_wait_seconds, static_cast<double>(queue_depth), 0.0, 24.0};
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
    const std::array<double, 4> telemetry{
        queue_wait_seconds, static_cast<double>(queue_depth),
        static_cast<double>(request_peak_active_tasks),
        static_cast<double>(pool.thread_count())};
    builder.add(protocol::NumericType::float64, telemetry.data(),
        telemetry.size(), telemetry.size());
    return builder.finish();
}

std::vector<std::uint8_t> execute(
    const protocol::PayloadView& input,
    double queue_wait_seconds,
    std::size_t queue_depth) {
    switch (input.header().operation) {
    case protocol::KernelOperation::exact_charging:
        return run_exact(input, queue_wait_seconds, queue_depth);
    case protocol::KernelOperation::screen_route:
        return run_screen(input, queue_wait_seconds, queue_depth);
    case protocol::KernelOperation::screen_routes:
        throw std::logic_error(
            "native screening-batch operation requires the shared work pool");
    }
    throw std::runtime_error("native scheduler operation is invalid");
}

void patch_pool_telemetry(
    std::vector<std::uint8_t>& output,
    const NativeWorkPool& pool,
    std::size_t request_peak_active_tasks) {
    const protocol::PayloadView view(output.data(), output.size());
    if (view.header().array_count == 0) {
        throw std::logic_error("native scheduler output telemetry is missing");
    }
    const auto& descriptor = view.descriptor(view.header().array_count - 1);
    if (descriptor.type != protocol::NumericType::float64
        || descriptor.count != 4) {
        throw std::logic_error("native scheduler output telemetry is invalid");
    }
    auto* values = reinterpret_cast<double*>(
        output.data() + descriptor.offset);
    values[2] = static_cast<double>(request_peak_active_tasks);
    values[3] = static_cast<double>(pool.thread_count());
}

void send_failure(
    int descriptor, std::uint64_t request_id, std::string_view message) noexcept {
    try {
        protocol::ControlFrame frame;
        frame.message = protocol::ControlMessage::failure;
        frame.request_id = request_id;
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
    bool allow_fault_injection) noexcept {
    std::uint64_t request_id = 0;
    try {
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
        if (injected_fault == "worker_exception") {
            throw std::runtime_error(
                "injected native scheduler worker exception");
        }
        std::vector<std::uint8_t> output_bytes;
        std::size_t request_peak_active_tasks = 0;
        if (input_view.header().operation
            == protocol::KernelOperation::screen_routes) {
            output_bytes = run_screen_routes(
                input_view, queue_wait_seconds, queue_depth, pool,
                request_peak_active_tasks);
        } else {
            pool.parallel_for(1, [&](std::size_t) {
                request_peak_active_tasks = pool.active_task_count();
                output_bytes = execute(input_view, queue_wait_seconds, queue_depth);
                request_peak_active_tasks = std::max(
                    request_peak_active_tasks, pool.active_task_count());
            });
        }
        patch_pool_telemetry(
            output_bytes, pool, request_peak_active_tasks);
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
        protocol::ControlFrame released;
        released.message = protocol::ControlMessage::released;
        released.request_id = request_id;
        send_exact(descriptor, &released, sizeof(released));
    } catch (const std::exception& error) {
        send_failure(descriptor, request_id, error.what());
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: evrptw_native_scheduler SOCKET WORKER_THREADS "
                     "RUN_NONCE [--enable-fault-injection]\n";
        return 2;
    }
    const std::string socket_path(argv[1]);
    int listener = -1;
    try {
        const auto worker_threads = std::stoll(argv[2]);
        scheduler_run_nonce = argv[3];
        const bool valid_nonce = scheduler_run_nonce.size() == 16
            && scheduler_run_nonce.find_first_not_of("0123456789abcdef")
                == std::string::npos;
        if (!valid_nonce) {
            throw std::invalid_argument("native scheduler run nonce is invalid");
        }
        bool allow_fault_injection = false;
        for (int index = 4; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--enable-fault-injection") {
                allow_fault_injection = true;
            } else if (option.starts_with("--production-fault=")) {
                production_fault = option.substr(
                    std::string_view("--production-fault=").size());
            } else {
                throw std::invalid_argument("native scheduler option is invalid");
            }
        }
        if (!production_fault.empty()
            && (!allow_fault_injection
                || production_fault != "pause_before_execute")) {
            throw std::invalid_argument(
                "native scheduler production fault is invalid");
        }
        if (worker_threads != 24) {
            throw std::invalid_argument(
                "native scheduler requires exactly 24 compute threads");
        }
        NativeWorkPool pool(worker_threads);
        NativeRequestQueue queue;
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
        request_threads.reserve(6);
        for (std::size_t index = 0; index < 6; ++index) {
            request_threads.emplace_back([&]() {
                while (const auto request = queue.take()) {
                    const auto queue_wait_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - request->submitted_at).count();
                    handle_connection(
                        request->descriptor, pool, stopping, listener,
                        queue_wait_seconds, request->queue_depth_on_submit,
                        allow_fault_injection);
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
