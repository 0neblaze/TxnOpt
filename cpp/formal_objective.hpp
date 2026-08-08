#pragma once

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <system_error>
#include <tuple>

namespace evrptw::formal_objective {

using Key = std::tuple<std::int64_t, double, double, std::int64_t>;

[[nodiscard]] inline double canonical_component(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            "objective component must be finite and non-negative");
    }
    // Python round(value, 9) performs correctly-rounded decimal conversion.
    // Scaling by 1e9 first is not equivalent near binary half-way values.
    std::array<char, 384> buffer{};
    const auto formatted = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::fixed, 9);
    if (formatted.ec != std::errc{}) {
        throw std::runtime_error(
            "objective component decimal canonicalization failed");
    }
    double canonical = 0.0;
    const auto parsed = std::from_chars(
        buffer.data(), formatted.ptr, canonical, std::chars_format::fixed);
    if (parsed.ec != std::errc{} || parsed.ptr != formatted.ptr) {
        throw std::runtime_error(
            "canonical objective component could not be decoded");
    }
    return canonical;
}

[[nodiscard]] inline Key key(
    std::int64_t vehicle_count,
    double total_distance,
    double total_charging_time,
    std::int64_t charging_count) {
    if (vehicle_count < 0 || charging_count < 0) {
        throw std::invalid_argument(
            "objective integer components must be non-negative");
    }
    return {
        vehicle_count,
        canonical_component(total_distance),
        canonical_component(total_charging_time),
        charging_count,
    };
}

}  // namespace evrptw::formal_objective
