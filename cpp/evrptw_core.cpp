#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Point = std::pair<double, double>;

py::array_t<double> distance_matrix(
    py::array_t<double, py::array::c_style | py::array::forcecast> points) {
    const auto input = points.request();
    if (input.ndim != 2 || input.shape[1] != 2) {
        throw std::invalid_argument("points must be a contiguous n-by-2 float array");
    }
    const auto count = static_cast<std::size_t>(input.shape[0]);
    py::array_t<double> output({input.shape[0], input.shape[0]});
    const auto* coordinates = static_cast<const double*>(input.ptr);
    auto* distances = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        for (std::size_t row = 0; row < count; ++row) {
            for (std::size_t column = 0; column < count; ++column) {
                const auto dx = coordinates[2 * row] - coordinates[2 * column];
                const auto dy = coordinates[2 * row + 1] - coordinates[2 * column + 1];
                distances[row * count + column] = std::hypot(dx, dy);
            }
        }
    }
    return output;
}

double distance(const Point& first, const Point& second) {
    return std::hypot(first.first - second.first, first.second - second.second);
}

double route_distance(const std::vector<Point>& points, const std::vector<std::size_t>& route) {
    if (route.size() < 2) {
        return 0.0;
    }

    double total = 0.0;
    for (std::size_t index = 1; index < route.size(); ++index) {
        if (route[index - 1] >= points.size() || route[index] >= points.size()) {
            throw std::out_of_range("route index is outside the point array");
        }
        total += distance(points[route[index - 1]], points[route[index]]);
    }
    return total;
}

double two_opt_delta(
    const std::vector<Point>& points,
    const std::vector<std::size_t>& route,
    std::size_t first,
    std::size_t second) {
    if (first == 0 || second <= first || second >= route.size() - 1) {
        throw std::invalid_argument("two-opt indices must preserve the route endpoints");
    }

    const auto before = distance(points[route[first - 1]], points[route[first]])
        + distance(points[route[second]], points[route[second + 1]]);
    const auto after = distance(points[route[first - 1]], points[route[second]])
        + distance(points[route[first]], points[route[second + 1]]);
    return after - before;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native kernels for EVRP-TW route evaluation";
    module.def("route_distance", &route_distance, py::arg("points"), py::arg("route"));
    module.def("distance_matrix", &distance_matrix, py::arg("points"));
    module.def(
        "two_opt_delta", &two_opt_delta, py::arg("points"), py::arg("route"),
        py::arg("first"), py::arg("second"));
}
