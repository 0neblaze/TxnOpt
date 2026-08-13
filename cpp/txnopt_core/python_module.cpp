#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "concurrency.hpp"
#include "round_transaction.hpp"
#include "../txnopt_cases/evrptw/exact_kernels.hpp"
#include "../txnopt_cases/evrptw/parallel_exact.hpp"

namespace py = pybind11;

namespace txnopt::native {

namespace evrptw_native = txnopt::cases::evrptw::native;

constexpr auto protocol_version = "txnopt-native-round-v1";

std::uint64_t route_key_checksum(
    const std::int64_t* const indices,
    const std::int64_t first,
    const std::int64_t last) noexcept {
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    auto checksum = offset;
    for (auto index = first; index < last; ++index) {
        const auto value = static_cast<std::uint64_t>(indices[index]);
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            checksum ^= (value >> (byte * 8U)) & 0xffU;
            checksum *= prime;
        }
    }
    checksum ^= static_cast<std::uint64_t>(last - first);
    checksum *= prime;
    return checksum;
}

template <typename Value>
py::array_t<Value> checked_array(
    const py::handle value,
    const char* const name,
    const py::ssize_t dimensions) {
    if (!py::isinstance<py::array>(value)) {
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    const auto generic = py::reinterpret_borrow<py::array>(value);
    if (!generic.dtype().is(py::dtype::of<Value>())
        || generic.ndim() != dimensions
        || (generic.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument(
            std::string(name) + " has an invalid dtype, rank, or layout");
    }
    return py::reinterpret_borrow<py::array_t<Value>>(value);
}

template <typename Value>
py::array_t<Value> vector_1d(const std::vector<Value>& values) {
    py::array_t<Value> output(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(
            output.mutable_data(), values.data(), values.size() * sizeof(Value));
    }
    return output;
}

template <typename Value>
py::array_t<Value> vector_2d(
    const std::vector<Value>& values,
    const py::ssize_t rows,
    const py::ssize_t columns) {
    if (rows < 0 || columns <= 0
        || static_cast<std::size_t>(rows * columns) != values.size()) {
        throw std::runtime_error("native result matrix extent is invalid");
    }
    py::array_t<Value> output(std::vector<py::ssize_t>{rows, columns});
    if (!values.empty()) {
        std::memcpy(
            output.mutable_data(), values.data(), values.size() * sizeof(Value));
    }
    return output;
}

class EVRPTWContext final {
public:
    EVRPTWContext(
        const py::handle node_kind,
        const py::handle demand,
        const py::handle ready_time,
        const py::handle due_date,
        const py::handle service_time,
        const py::handle distance,
        const py::handle reachable,
        const py::handle vehicle,
        const std::int64_t worker_count)
        : worker_count_(worker_count) {
        if (worker_count <= 0 || worker_count > 1024) {
            throw std::invalid_argument("worker_count must be in [1, 1024]");
        }
        const auto kinds = checked_array<std::int64_t>(node_kind, "node_kind", 1);
        const auto demands = checked_array<double>(demand, "demand", 1);
        const auto ready = checked_array<double>(ready_time, "ready_time", 1);
        const auto due = checked_array<double>(due_date, "due_date", 1);
        const auto service = checked_array<double>(service_time, "service_time", 1);
        const auto distances = checked_array<double>(distance, "distance", 2);
        const auto reachability = checked_array<std::uint8_t>(
            reachable, "reachable", 2);
        const auto vehicle_values = checked_array<double>(vehicle, "vehicle", 1);
        node_count_ = static_cast<std::size_t>(kinds.shape(0));
        if (node_count_ == 0
            || ready.shape(0) != kinds.shape(0)
            || demands.shape(0) != kinds.shape(0)
            || due.shape(0) != kinds.shape(0)
            || service.shape(0) != kinds.shape(0)
            || distances.shape(0) != kinds.shape(0)
            || distances.shape(1) != kinds.shape(0)
            || reachability.shape(0) != kinds.shape(0)
            || reachability.shape(1) != kinds.shape(0)
            || vehicle_values.shape(0) != 5) {
            throw std::invalid_argument("EVRPTW context array extents do not align");
        }
        kinds_.assign(kinds.data(), kinds.data() + node_count_);
        demands_.assign(demands.data(), demands.data() + node_count_);
        ready_.assign(ready.data(), ready.data() + node_count_);
        due_.assign(due.data(), due.data() + node_count_);
        service_.assign(service.data(), service.data() + node_count_);
        distances_.assign(
            distances.data(), distances.data() + node_count_ * node_count_);
        reachable_.assign(
            reachability.data(),
            reachability.data() + node_count_ * node_count_);
        vehicle_.assign(vehicle_values.data(), vehicle_values.data() + 5);
        for (std::size_t index = 0; index < node_count_; ++index) {
            if (kinds_[index] == evrptw_native::depot_kind) {
                if (depot_ >= 0) {
                    throw std::invalid_argument("EVRPTW context has multiple depots");
                }
                depot_ = static_cast<std::int64_t>(index);
                recharge_nodes_.push_back(static_cast<std::int64_t>(index));
            } else if (kinds_[index] == evrptw_native::station_kind) {
                stations_.push_back(static_cast<std::int64_t>(index));
                recharge_nodes_.push_back(static_cast<std::int64_t>(index));
            } else if (kinds_[index] != evrptw_native::customer_kind) {
                throw std::invalid_argument("EVRPTW context contains an unknown node kind");
            }
            if (!std::isfinite(ready_[index]) || !std::isfinite(due_[index])
                || !std::isfinite(service_[index])) {
                throw std::invalid_argument("EVRPTW context time data must be finite");
            }
        }
        if (depot_ < 0) {
            throw std::invalid_argument("EVRPTW context requires exactly one depot");
        }
        if (std::any_of(
                vehicle_.begin(), vehicle_.end(),
                [](const double value) { return !std::isfinite(value) || value < 0.0; })
            || vehicle_[4] <= 0.0) {
            throw std::invalid_argument("EVRPTW vehicle values are invalid");
        }
        if (worker_count_ > 1) {
            pool_ = std::make_unique<NativeWorkPool>(worker_count_);
        }
    }

    py::tuple exact_round_v1(
        const py::handle route_offsets,
        const py::handle route_indices,
        const double deadline_seconds,
        const std::int64_t batch_size,
        const std::int64_t work_budget) {
        const auto offsets = checked_array<std::int64_t>(
            route_offsets, "route_offsets", 1);
        const auto indices = checked_array<std::int64_t>(
            route_indices, "route_indices", 1);
        if (offsets.shape(0) < 1 || offsets.data()[0] != 0
            || offsets.data()[offsets.shape(0) - 1] != indices.shape(0)
            || batch_size <= 0
            || work_budget < -1
            || (!std::isfinite(deadline_seconds)
                && !(std::isinf(deadline_seconds) && deadline_seconds > 0.0))
            || deadline_seconds <= 0.0) {
            throw std::invalid_argument(
                "native round control, budget, or offsets are invalid");
        }
        const auto route_count = static_cast<std::size_t>(offsets.shape(0) - 1);
        if (route_count
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("native round has too many routes");
        }
        const auto requested_work = static_cast<std::int64_t>(route_count);
        const auto effective_work_budget = work_budget < 0
            ? requested_work
            : work_budget;
        PreparedRoundTransaction transaction(
            requested_work, effective_work_budget);
        transaction.reserve();
        for (std::size_t route = 0; route < route_count; ++route) {
            const auto first = offsets.data()[route];
            const auto last = offsets.data()[route + 1];
            if (first < 0 || last < first || last > indices.shape(0)) {
                throw std::invalid_argument("route offsets must be monotonic");
            }
        }
        for (py::ssize_t index = 0; index < indices.shape(0); ++index) {
            const auto node = indices.data()[index];
            if (node < 0 || static_cast<std::size_t>(node) >= node_count_
                || kinds_[static_cast<std::size_t>(node)]
                    != evrptw_native::customer_kind) {
                throw std::invalid_argument("route indices must name customer nodes");
            }
        }

        evrptw_native::ExactBatchOutput output;
        std::int64_t screened_routes = 0;
        const auto parallel_route_threshold = static_cast<std::size_t>(
            std::max<std::int64_t>(1, worker_count_ * 2));
        const bool use_parallel = worker_count_ > 1
            && route_count >= parallel_route_threshold;
        transaction.begin_evaluation();
        try {
            py::gil_scoped_release release;
            const double screen_options[4]{1.0, 1e-9, 0.0, 0.0};
            const double incremental[6]{0.0, 0.0, 0.0, 0.0, 1.0, 1.0};
            for (std::size_t route = 0; route < route_count; ++route) {
                const auto first = offsets.data()[route];
                const auto last = offsets.data()[route + 1];
                const auto screened = evrptw_native::run_screen_route(
                    kinds_.data(), demands_.data(), ready_.data(), due_.data(),
                    service_.data(), distances_.data(), reachable_.data(),
                    vehicle_.data(), indices.data() + first,
                    static_cast<std::size_t>(last - first), node_count_, depot_,
                    recharge_nodes_, screen_options, incremental);
                if (screened.codes[0] != 1) {
                    throw std::runtime_error(
                        "Python-admitted EVRPTW route failed native safe screening: reason="
                        + std::to_string(screened.codes[1])
                        + ", check=" + std::to_string(screened.codes[2]));
                }
                ++screened_routes;
            }
            if (!use_parallel) {
                output = run_exact(
                    offsets.data(), indices.data(), route_count,
                    deadline_seconds, batch_size);
            } else {
                output = evrptw_native::run_exact_charging_parallel(
                    *pool_, offsets.data(), indices.data(), route_count,
                    deadline_seconds,
                    [&](const std::int64_t* chunk_offsets,
                        const std::int64_t* chunk_indices,
                        const std::size_t chunk_routes,
                        const double remaining) {
                        return run_exact(
                            chunk_offsets, chunk_indices, chunk_routes,
                            remaining, batch_size);
                    });
            }
            evrptw_native::validate_exact_batch_output(
                output, kinds_.data(), offsets.data(), indices.data(),
                node_count_, route_count,
                static_cast<std::size_t>(indices.shape(0)), depot_, batch_size);
        } catch (...) {
            transaction.abort();
            throw;
        }
        std::vector<std::uint64_t> prepared_cache_keys;
        if (output.batch_counters[3] == 0) {
            prepared_cache_keys.reserve(route_count);
            for (std::size_t route = 0; route < route_count; ++route) {
                prepared_cache_keys.push_back(route_key_checksum(
                    indices.data(), offsets.data()[route],
                    offsets.data()[route + 1]));
            }
        }
        transaction.resolve(
            output.batch_counters[1], output.batch_counters[2],
            output.batch_counters[3], prepared_cache_keys);
        const auto transaction_receipt = transaction.receipt();
        ++round_calls_;
        py::dict receipt;
        receipt["protocol"] = protocol_version;
        receipt["phase"] = std::string(phase_name(transaction_receipt.phase));
        receipt["worker_count"] = worker_count_;
        receipt["scheduled_worker_count"] = use_parallel
            ? static_cast<std::int64_t>(std::min<std::size_t>(
                route_count, static_cast<std::size_t>(worker_count_)))
            : 1;
        receipt["execution_policy"] = worker_count_ == 1
            ? "serial_configured"
            : (use_parallel ? "parallel" : "serial_small_batch");
        receipt["parallel_route_threshold"] = static_cast<std::int64_t>(
            parallel_route_threshold);
        receipt["context_pack_count"] = 1;
        receipt["round_call_count"] = round_calls_;
        receipt["started_work"] = output.batch_counters[1];
        receipt["screened_work"] = screened_routes;
        receipt["completed_work"] = output.batch_counters[2];
        receipt["interrupted_work"] = output.batch_counters[3];
        receipt["budget_limit"] = transaction_receipt.budget_limit;
        receipt["budget_reserved_work"] = transaction_receipt.reserved_work;
        receipt["budget_remaining_work"] = transaction_receipt.remaining_work;
        receipt["prepared_cache_write_count"] =
            transaction_receipt.prepared_cache_write_count;
        receipt["prepared_cache_key_checksum"] =
            transaction_receipt.prepared_cache_key_checksum;
        py::tuple phase_trace(transaction_receipt.phase_trace.size());
        for (std::size_t index = 0;
             index < transaction_receipt.phase_trace.size(); ++index) {
            phase_trace[index] = std::string(
                phase_name(transaction_receipt.phase_trace[index]));
        }
        receipt["phase_trace"] = std::move(phase_trace);
        receipt["semantic_event_count"] = static_cast<std::int64_t>(
            transaction_receipt.phase_trace.size());
        receipt["fallback_count"] = 0;
        receipt["source_revision"] = TXNOPT_BUILD_GIT_REVISION;
        receipt["source_tree"] = TXNOPT_BUILD_GIT_TREE;

        const std::vector<std::int64_t> semantic_counters{
            output.batch_counters[0],
            output.batch_counters[1],
            output.batch_counters[2],
            output.batch_counters[3],
            output.batch_counters[9],
        };
        std::vector<std::int64_t> resolution_order(route_count);
        for (std::size_t route = 0; route < route_count; ++route) {
            resolution_order[route] = static_cast<std::int64_t>(route);
        }

        py::tuple projected(10);
        projected[0] = vector_1d(output.path_offsets);
        projected[1] = vector_1d(output.path_indices);
        projected[2] = vector_1d(output.statuses);
        projected[3] = vector_1d(output.reasons);
        projected[4] = vector_2d(
            output.metrics, static_cast<py::ssize_t>(route_count), 4);
        projected[5] = vector_2d(
            output.label_counters, static_cast<py::ssize_t>(route_count), 3);
        projected[6] = vector_1d(semantic_counters);
        projected[7] = vector_1d(resolution_order);
        projected[8] = vector_2d(
            output.physical_task_receipts,
            static_cast<py::ssize_t>(output.physical_task_receipts.size() / 7),
            7);
        projected[9] = std::move(receipt);
        return projected;
    }

    [[nodiscard]] std::int64_t worker_count() const noexcept {
        return worker_count_;
    }

private:
    evrptw_native::ExactBatchOutput run_exact(
        const std::int64_t* const offsets,
        const std::int64_t* const indices,
        const std::size_t route_count,
        const double deadline_seconds,
        const std::int64_t batch_size) const {
        return evrptw_native::run_exact_charging_batch(
            kinds_.data(), ready_.data(), due_.data(), service_.data(),
            distances_.data(), vehicle_.data(), offsets, indices, node_count_,
            route_count, depot_, stations_, deadline_seconds, batch_size);
    }

    std::size_t node_count_ = 0;
    std::int64_t depot_ = -1;
    std::int64_t worker_count_ = 0;
    std::int64_t round_calls_ = 0;
    std::vector<std::int64_t> kinds_;
    std::vector<double> demands_;
    std::vector<double> ready_;
    std::vector<double> due_;
    std::vector<double> service_;
    std::vector<double> distances_;
    std::vector<std::uint8_t> reachable_;
    std::vector<double> vehicle_;
    std::vector<std::int64_t> stations_;
    std::vector<std::int64_t> recharge_nodes_;
    std::unique_ptr<NativeWorkPool> pool_;
};

}  // namespace txnopt::native

PYBIND11_MODULE(_native, module) {
    module.doc() = "TxnOpt native round protocol";
    module.attr("PROTOCOL_VERSION") = txnopt::native::protocol_version;
    py::dict build_attestation;
    build_attestation["schema_version"] = "txnopt-native-build-attestation-v1";
    build_attestation["source_revision"] = TXNOPT_BUILD_GIT_REVISION;
    build_attestation["source_tree"] = TXNOPT_BUILD_GIT_TREE;
    build_attestation["source_manifest_sha256"] =
        TXNOPT_BUILD_SOURCE_MANIFEST_SHA256;
    build_attestation["tracked_file_count"] = TXNOPT_BUILD_TRACKED_FILE_COUNT;
    build_attestation["source_dirty"] =
        static_cast<bool>(TXNOPT_BUILD_SOURCE_DIRTY);
    build_attestation["development_override"] =
        static_cast<bool>(TXNOPT_BUILD_DEVELOPMENT_OVERRIDE);
    build_attestation["cpp_source_kind"] = TXNOPT_BUILD_CPP_SOURCE_KIND;
    build_attestation["performance_profile"] = TXNOPT_BUILD_PERFORMANCE_PROFILE;
    build_attestation["compiler_id"] = TXNOPT_BUILD_COMPILER_ID;
    build_attestation["compiler_version"] = TXNOPT_BUILD_COMPILER_VERSION;
    module.attr("BUILD_ATTESTATION") = std::move(build_attestation);
    py::class_<txnopt::native::EVRPTWContext>(module, "EVRPTWContext")
        .def(
            py::init<
                py::handle, py::handle, py::handle, py::handle, py::handle,
                py::handle, py::handle, py::handle, std::int64_t>(),
            py::arg("node_kind"), py::arg("demand"), py::arg("ready_time"),
            py::arg("due_date"), py::arg("service_time"), py::arg("distance"),
            py::arg("reachable"), py::arg("vehicle"),
            py::arg("worker_count"))
        .def(
            "exact_round_v1",
            &txnopt::native::EVRPTWContext::exact_round_v1,
            py::arg("route_offsets"), py::arg("route_indices"),
            py::arg("deadline_seconds"), py::arg("batch_size") = 64,
            py::arg("work_budget") = -1)
        .def_property_readonly(
            "worker_count", &txnopt::native::EVRPTWContext::worker_count);
}
