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

#include "native_exact_parallel.hpp"
#include "native_solver_kernels.hpp"

namespace py = pybind11;

namespace txnopt::native {

constexpr auto protocol_version = "txnopt-native-round-v1";

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
            if (kinds_[index] == evrptw::native_kernels::depot_kind) {
                if (depot_ >= 0) {
                    throw std::invalid_argument("EVRPTW context has multiple depots");
                }
                depot_ = static_cast<std::int64_t>(index);
                recharge_nodes_.push_back(static_cast<std::int64_t>(index));
            } else if (kinds_[index] == evrptw::native_kernels::station_kind) {
                stations_.push_back(static_cast<std::int64_t>(index));
                recharge_nodes_.push_back(static_cast<std::int64_t>(index));
            } else if (kinds_[index] != evrptw::native_kernels::customer_kind) {
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
        const std::int64_t batch_size) {
        const auto offsets = checked_array<std::int64_t>(
            route_offsets, "route_offsets", 1);
        const auto indices = checked_array<std::int64_t>(
            route_indices, "route_indices", 1);
        if (offsets.shape(0) < 1 || offsets.data()[0] != 0
            || offsets.data()[offsets.shape(0) - 1] != indices.shape(0)
            || batch_size <= 0
            || (!std::isfinite(deadline_seconds)
                && !(std::isinf(deadline_seconds) && deadline_seconds > 0.0))
            || deadline_seconds <= 0.0) {
            throw std::invalid_argument("native round control or offsets are invalid");
        }
        const auto route_count = static_cast<std::size_t>(offsets.shape(0) - 1);
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
                    != evrptw::native_kernels::customer_kind) {
                throw std::invalid_argument("route indices must name customer nodes");
            }
        }

        evrptw::native_kernels::ExactBatchOutput output;
        std::int64_t screened_routes = 0;
        {
            py::gil_scoped_release release;
            const double screen_options[4]{1.0, 1e-9, 0.0, 0.0};
            const double incremental[6]{0.0, 0.0, 0.0, 0.0, 1.0, 1.0};
            for (std::size_t route = 0; route < route_count; ++route) {
                const auto first = offsets.data()[route];
                const auto last = offsets.data()[route + 1];
                const auto screened = evrptw::native_kernels::run_screen_route(
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
            if (worker_count_ == 1) {
                output = run_exact(
                    offsets.data(), indices.data(), route_count,
                    deadline_seconds, batch_size);
            } else {
                output = evrptw::native_parallel::run_exact_charging_parallel(
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
            evrptw::native_kernels::validate_exact_batch_output(
                output, kinds_.data(), offsets.data(), indices.data(),
                node_count_, route_count,
                static_cast<std::size_t>(indices.shape(0)), depot_, batch_size);
        }
        ++round_calls_;
        py::dict receipt;
        receipt["protocol"] = protocol_version;
        receipt["phase"] = output.batch_counters[3] == 0
            ? "VALIDATED"
            : "INTERRUPTED";
        receipt["worker_count"] = worker_count_;
        receipt["context_pack_count"] = 1;
        receipt["round_call_count"] = round_calls_;
        receipt["started_work"] = output.batch_counters[1];
        receipt["screened_work"] = screened_routes;
        receipt["completed_work"] = output.batch_counters[2];
        receipt["interrupted_work"] = output.batch_counters[3];
        receipt["fallback_count"] = 0;
        receipt["source_revision"] = EVRPTW_BUILD_GIT_REVISION;
        receipt["source_tree"] = EVRPTW_BUILD_GIT_TREE;

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
    evrptw::native_kernels::ExactBatchOutput run_exact(
        const std::int64_t* const offsets,
        const std::int64_t* const indices,
        const std::size_t route_count,
        const double deadline_seconds,
        const std::int64_t batch_size) const {
        return evrptw::native_kernels::run_exact_charging_batch(
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
            py::arg("deadline_seconds"), py::arg("batch_size") = 64)
        .def_property_readonly(
            "worker_count", &txnopt::native::EVRPTWContext::worker_count);
}
