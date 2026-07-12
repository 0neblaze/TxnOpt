#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using Point = std::pair<double, double>;

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
    module.def(
        "two_opt_delta", &two_opt_delta, py::arg("points"), py::arg("route"),
        py::arg("first"), py::arg("second"));
}
