#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "native_sha256.hpp"
#include "native_kernel_protocol.hpp"
#include "native_solver_kernels.hpp"

namespace evrptw::native_search {

struct ProblemV2 final {
    std::vector<std::int64_t> node_kind;
    std::vector<double> demand;
    std::vector<double> ready_time;
    std::vector<double> due_date;
    std::vector<double> service_time;
    std::vector<double> distance;
    std::vector<std::uint8_t> reachable;
    std::array<double, 5> vehicle{};
    std::vector<std::int64_t> lexical_rank;
    std::vector<std::int64_t> node_name_offsets;
    std::vector<std::uint8_t> node_name_bytes;
    std::vector<std::int64_t> initial_route_offsets;
    std::vector<std::int64_t> initial_route_indices;

    [[nodiscard]] std::size_t node_count() const noexcept {
        return node_kind.size();
    }

    [[nodiscard]] std::size_t route_count() const noexcept {
        return initial_route_offsets.empty()
            ? 0U : initial_route_offsets.size() - 1U;
    }

    void validate() const {
        const auto nodes = node_count();
        if (nodes == 0 || demand.size() != nodes || ready_time.size() != nodes
            || due_date.size() != nodes || service_time.size() != nodes
            || lexical_rank.size() != nodes
            || distance.size() != checked_square(nodes)
            || reachable.size() != checked_square(nodes)) {
            throw std::invalid_argument(
                "native search problem arrays do not share one node shape");
        }
        if (node_name_offsets.size() != nodes + 1
            || node_name_offsets.front() != 0
            || node_name_offsets.back()
                != static_cast<std::int64_t>(node_name_bytes.size())) {
            throw std::invalid_argument(
                "native search node-name SoA does not span its byte buffer");
        }
        validate_offsets(
            node_name_offsets, node_name_bytes.size(), "node_name_offsets",
            true);
        for (std::size_t node = 0; node < nodes; ++node) {
            if (node_name_offsets[node] == node_name_offsets[node + 1]) {
                throw std::invalid_argument(
                    "native search node names must be non-empty");
            }
        }
        if (initial_route_offsets.size() < 2
            || initial_route_indices.empty()) {
            throw std::invalid_argument(
                "native search requires a non-empty warm start");
        }
        validate_offsets(
            initial_route_offsets, initial_route_indices.size(),
            "initial_route_offsets", false);
        std::unordered_set<std::int64_t> ranks;
        std::unordered_set<std::int64_t> expected_customers;
        std::unordered_set<std::int64_t> warm_customers;
        ranks.reserve(nodes);
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto kind = node_kind[node];
            if (kind == native_kernels::customer_kind) {
                expected_customers.insert(static_cast<std::int64_t>(node));
            } else if (kind != native_kernels::depot_kind
                && kind != native_kernels::station_kind) {
                throw std::invalid_argument(
                    "native search node kind is invalid");
            }
            const auto rank = lexical_rank[node];
            if (rank < 0 || rank >= static_cast<std::int64_t>(nodes)
                || !ranks.insert(rank).second) {
                throw std::invalid_argument(
                    "native search lexical rank must be a permutation");
            }
            if (!std::isfinite(demand[node]) || demand[node] < 0.0
                || !std::isfinite(ready_time[node])
                || !std::isfinite(due_date[node])
                || !std::isfinite(service_time[node])
                || service_time[node] < 0.0 || ready_time[node] > due_date[node]) {
                throw std::invalid_argument(
                    "native search node metrics are invalid");
            }
        }
        for (const auto value : vehicle) {
            if (!std::isfinite(value) || value <= 0.0) {
                throw std::invalid_argument(
                    "native search vehicle metrics must be finite and positive");
            }
        }
        for (const auto value : distance) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "native search distance matrix is invalid");
            }
        }
        if (std::any_of(
                reachable.begin(), reachable.end(),
                [](std::uint8_t value) { return value > 1U; })) {
            throw std::invalid_argument(
                "native search reachability matrix must be binary");
        }
        for (const auto customer : initial_route_indices) {
            if (customer < 0 || customer >= static_cast<std::int64_t>(nodes)
                || node_kind[static_cast<std::size_t>(customer)]
                    != native_kernels::customer_kind
                || !warm_customers.insert(customer).second) {
                throw std::invalid_argument(
                    "native search warm start contains an invalid customer");
            }
        }
        if (warm_customers != expected_customers) {
            throw std::invalid_argument(
                "native search warm start must cover every customer exactly once");
        }
    }

private:
    [[nodiscard]] static std::size_t checked_square(std::size_t value) {
        if (value > 0
            && value > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("native search node square overflows");
        }
        return value * value;
    }

    static void validate_offsets(
        std::span<const std::int64_t> offsets,
        std::size_t terminal,
        std::string_view name,
        bool allow_empty_rows) {
        if (offsets.empty() || offsets.front() != 0
            || offsets.back() != static_cast<std::int64_t>(terminal)) {
            throw std::invalid_argument(std::string(name) + " boundary is invalid");
        }
        for (std::size_t index = 0; index + 1 < offsets.size(); ++index) {
            if (offsets[index] < 0
                || (allow_empty_rows
                    ? offsets[index] > offsets[index + 1]
                    : offsets[index] >= offsets[index + 1])) {
                throw std::invalid_argument(
                    std::string(name) + " is not monotone");
            }
        }
    }
};

struct ConfigV2 final {
    std::array<std::int64_t, 5> search_control{};
    double deadline_remaining = 0.0;
    std::array<std::int64_t, 13> protocol_control{};
    std::array<double, 2> protocol_options{};
    std::array<std::int64_t, 15> stage04_integer{};
    std::array<double, 15> stage04_float{};
    std::array<std::int64_t, 24> operator_integer{};
    std::array<double, 7> operator_float{};

    void validate() const {
        if (search_control[0] < 0 || search_control[1] <= 0
            || search_control[2] <= 0 || search_control[3] <= 0
            || search_control[4] < -1 || !std::isfinite(deadline_remaining)
            || deadline_remaining <= 0.0) {
            throw std::invalid_argument(
                "native search base control is invalid");
        }
        if ((protocol_control[0] != 0 && protocol_control[0] != 1)
            || protocol_control[1] <= 0 || protocol_control[2] <= 0
            || (protocol_control[3] != 1 && protocol_control[3] != 4)
            || protocol_control[4] < 10 || protocol_control[5] < 0) {
            throw std::invalid_argument(
                "native search Candidate Control configuration is invalid");
        }
        if (!(protocol_options[0] > 0.0 && protocol_options[0] <= 1.0)
            || !std::isfinite(protocol_options[1])
            || protocol_options[1] <= 0.0) {
            throw std::invalid_argument(
                "native search protocol options are invalid");
        }
        if (std::any_of(
                stage04_float.begin(), stage04_float.end(),
                [](double value) { return !std::isfinite(value); })
            || std::any_of(
                operator_float.begin(), operator_float.end(),
                [](double value) { return !std::isfinite(value); })) {
            throw std::invalid_argument(
                "native search floating configuration is non-finite");
        }
    }
};

struct RequestV2 final {
    ProblemV2 problem;
    ConfigV2 config;

    void validate() const {
        problem.validate();
        config.validate();
    }

    [[nodiscard]] std::string sha256() const {
        validate();
        std::string evidence("stage05.2-native-search-request-v2");
        append(evidence, problem.node_kind);
        append(evidence, problem.demand);
        append(evidence, problem.ready_time);
        append(evidence, problem.due_date);
        append(evidence, problem.service_time);
        append(evidence, problem.distance);
        append(evidence, problem.reachable);
        append(evidence, problem.vehicle);
        append(evidence, problem.lexical_rank);
        append(evidence, problem.node_name_offsets);
        append(evidence, problem.node_name_bytes);
        append(evidence, problem.initial_route_offsets);
        append(evidence, problem.initial_route_indices);
        append(evidence, config.search_control);
        append_scalar(evidence, config.deadline_remaining);
        append(evidence, config.protocol_control);
        append(evidence, config.protocol_options);
        append(evidence, config.stage04_integer);
        append(evidence, config.stage04_float);
        append(evidence, config.operator_integer);
        append(evidence, config.operator_float);
        return native_protocol::native_sha256_hex(evidence);
    }

private:
    template <typename Container>
    static void append(std::string& output, const Container& values) {
        const auto count = static_cast<std::uint64_t>(values.size());
        append_scalar(output, count);
        if (!values.empty()) {
            const auto bytes = values.size() * sizeof(typename Container::value_type);
            output.append(
                reinterpret_cast<const char*>(values.data()), bytes);
        }
    }

    template <typename T>
    static void append_scalar(std::string& output, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        output.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
};

inline std::array<std::int64_t, 8> receipt_counts(
    const RequestV2& request) {
    request.validate();
    return {
        static_cast<std::int64_t>(request.problem.node_count()),
        static_cast<std::int64_t>(request.problem.route_count()),
        static_cast<std::int64_t>(
            request.problem.initial_route_indices.size()),
        static_cast<std::int64_t>(request.problem.node_name_bytes.size()),
        static_cast<std::int64_t>(request.problem.distance.size()),
        request.config.search_control[1],
        request.config.protocol_control[2],
        request.config.search_control[3],
    };
}

inline std::vector<std::uint8_t> request_payload(
    const RequestV2& request,
    std::uint64_t request_id) {
    request.validate();
    native_protocol::PayloadBuilder builder(
        native_protocol::KernelOperation::search_request_receipt,
        request_id);
    const auto& problem = request.problem;
    const auto nodes = problem.node_count();
    builder.add(native_protocol::NumericType::int64, problem.node_kind.data(),
        problem.node_kind.size(), problem.node_kind.size());
    builder.add(native_protocol::NumericType::float64, problem.demand.data(),
        problem.demand.size(), problem.demand.size());
    builder.add(native_protocol::NumericType::float64, problem.ready_time.data(),
        problem.ready_time.size(), problem.ready_time.size());
    builder.add(native_protocol::NumericType::float64, problem.due_date.data(),
        problem.due_date.size(), problem.due_date.size());
    builder.add(native_protocol::NumericType::float64, problem.service_time.data(),
        problem.service_time.size(), problem.service_time.size());
    builder.add(native_protocol::NumericType::float64, problem.distance.data(),
        problem.distance.size(), nodes, nodes);
    builder.add(native_protocol::NumericType::uint8, problem.reachable.data(),
        problem.reachable.size(), nodes, nodes);
    builder.add(native_protocol::NumericType::float64, problem.vehicle.data(),
        problem.vehicle.size(), problem.vehicle.size());
    builder.add(native_protocol::NumericType::int64, problem.lexical_rank.data(),
        problem.lexical_rank.size(), problem.lexical_rank.size());
    builder.add(native_protocol::NumericType::int64,
        problem.node_name_offsets.data(), problem.node_name_offsets.size(),
        problem.node_name_offsets.size());
    builder.add(native_protocol::NumericType::uint8,
        problem.node_name_bytes.data(), problem.node_name_bytes.size(),
        problem.node_name_bytes.size());
    builder.add(native_protocol::NumericType::int64,
        problem.initial_route_offsets.data(),
        problem.initial_route_offsets.size(),
        problem.initial_route_offsets.size());
    builder.add(native_protocol::NumericType::int64,
        problem.initial_route_indices.data(),
        problem.initial_route_indices.size(),
        problem.initial_route_indices.size());
    const auto& config = request.config;
    builder.add(native_protocol::NumericType::int64,
        config.search_control.data(), config.search_control.size(),
        config.search_control.size());
    builder.add(native_protocol::NumericType::float64,
        &config.deadline_remaining, 1, 1);
    builder.add(native_protocol::NumericType::int64,
        config.protocol_control.data(), config.protocol_control.size(),
        config.protocol_control.size());
    builder.add(native_protocol::NumericType::float64,
        config.protocol_options.data(), config.protocol_options.size(),
        config.protocol_options.size());
    builder.add(native_protocol::NumericType::int64,
        config.stage04_integer.data(), config.stage04_integer.size(),
        config.stage04_integer.size());
    builder.add(native_protocol::NumericType::float64,
        config.stage04_float.data(), config.stage04_float.size(),
        config.stage04_float.size());
    builder.add(native_protocol::NumericType::int64,
        config.operator_integer.data(), config.operator_integer.size(),
        config.operator_integer.size());
    builder.add(native_protocol::NumericType::float64,
        config.operator_float.data(), config.operator_float.size(),
        config.operator_float.size());
    return builder.finish();
}

inline RequestV2 request_from_payload(
    const native_protocol::PayloadView& payload) {
    if (payload.header().operation
            != native_protocol::KernelOperation::search_request_receipt
        || payload.header().array_count != 21) {
        throw std::runtime_error(
            "native search request payload envelope is invalid");
    }
    const auto require_vector = [&payload](std::size_t index) {
        const auto& descriptor = payload.descriptor(index);
        if (descriptor.dimensions != 1
            || descriptor.shape[0] != descriptor.count
            || descriptor.shape[1] != 0) {
            throw std::runtime_error(
                "native search request vector descriptor is invalid");
        }
    };
    for (const auto index : {
             std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3},
             std::size_t{4}, std::size_t{7}, std::size_t{8}, std::size_t{9},
             std::size_t{10}, std::size_t{11}, std::size_t{12},
             std::size_t{13}, std::size_t{14}, std::size_t{15},
             std::size_t{16}, std::size_t{17}, std::size_t{18},
             std::size_t{19}, std::size_t{20}}) {
        require_vector(index);
    }
    const auto copy_vector = [&payload]<typename T>(
        std::size_t index,
        native_protocol::NumericType type) {
        const auto count = static_cast<std::size_t>(
            payload.descriptor(index).count);
        const auto* values = payload.data<T>(index, type);
        return std::vector<T>(values, values + count);
    };
    const auto copy_array = [&payload]<typename T, std::size_t Size>(
        std::size_t index,
        native_protocol::NumericType type) {
        if (payload.descriptor(index).count != Size
            || payload.descriptor(index).dimensions != 1) {
            throw std::runtime_error(
                "native search request fixed array is invalid");
        }
        std::array<T, Size> output{};
        const auto* values = payload.data<T>(index, type);
        std::copy(values, values + Size, output.begin());
        return output;
    };
    RequestV2 request;
    auto& problem = request.problem;
    problem.node_kind = copy_vector.template operator()<std::int64_t>(
        0, native_protocol::NumericType::int64);
    problem.demand = copy_vector.template operator()<double>(
        1, native_protocol::NumericType::float64);
    problem.ready_time = copy_vector.template operator()<double>(
        2, native_protocol::NumericType::float64);
    problem.due_date = copy_vector.template operator()<double>(
        3, native_protocol::NumericType::float64);
    problem.service_time = copy_vector.template operator()<double>(
        4, native_protocol::NumericType::float64);
    problem.distance = copy_vector.template operator()<double>(
        5, native_protocol::NumericType::float64);
    problem.reachable = copy_vector.template operator()<std::uint8_t>(
        6, native_protocol::NumericType::uint8);
    problem.vehicle = copy_array.template operator()<double, 5>(
        7, native_protocol::NumericType::float64);
    problem.lexical_rank = copy_vector.template operator()<std::int64_t>(
        8, native_protocol::NumericType::int64);
    problem.node_name_offsets = copy_vector.template operator()<std::int64_t>(
        9, native_protocol::NumericType::int64);
    problem.node_name_bytes = copy_vector.template operator()<std::uint8_t>(
        10, native_protocol::NumericType::uint8);
    problem.initial_route_offsets =
        copy_vector.template operator()<std::int64_t>(
            11, native_protocol::NumericType::int64);
    problem.initial_route_indices =
        copy_vector.template operator()<std::int64_t>(
            12, native_protocol::NumericType::int64);
    auto& config = request.config;
    config.search_control = copy_array.template operator()<std::int64_t, 5>(
        13, native_protocol::NumericType::int64);
    config.deadline_remaining = payload.data<double>(
        14, native_protocol::NumericType::float64)[0];
    config.protocol_control = copy_array.template operator()<std::int64_t, 13>(
        15, native_protocol::NumericType::int64);
    config.protocol_options = copy_array.template operator()<double, 2>(
        16, native_protocol::NumericType::float64);
    config.stage04_integer = copy_array.template operator()<std::int64_t, 15>(
        17, native_protocol::NumericType::int64);
    config.stage04_float = copy_array.template operator()<double, 15>(
        18, native_protocol::NumericType::float64);
    config.operator_integer = copy_array.template operator()<std::int64_t, 24>(
        19, native_protocol::NumericType::int64);
    config.operator_float = copy_array.template operator()<double, 7>(
        20, native_protocol::NumericType::float64);
    const auto nodes = problem.node_count();
    if (payload.descriptor(5).dimensions != 2
        || payload.descriptor(5).shape[0] != nodes
        || payload.descriptor(5).shape[1] != nodes
        || payload.descriptor(6).dimensions != 2
        || payload.descriptor(6).shape[0] != nodes
        || payload.descriptor(6).shape[1] != nodes
        || payload.descriptor(14).count != 1
        || payload.descriptor(14).dimensions != 1) {
        throw std::runtime_error(
            "native search request matrix/control shape is invalid");
    }
    request.validate();
    return request;
}

}  // namespace evrptw::native_search
