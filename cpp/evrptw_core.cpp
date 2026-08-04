#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <charconv>
#include <cctype>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <descrobject.h>

#include "native_concurrency.hpp"
#include "native_candidate_plan_runtime.hpp"
#include "native_kernel_client.hpp"
#include "native_search_core.hpp"
#include "native_solver_kernels.hpp"

namespace py = pybind11;

using Point = std::pair<double, double>;

template <typename T>
const T* checked_data(const py::array& array);

namespace evrptw::formal_objective {

using Key = std::tuple<std::int64_t, double, double, std::int64_t>;

[[nodiscard]] double canonical_component(double value) {
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

[[nodiscard]] Key key(
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

template <typename IntegerArray, typename FloatArray>
[[nodiscard]] Key key_from_arrays(
    const IntegerArray& integers,
    const FloatArray& floats) {
    return key(
        checked_data<std::int64_t>(integers)[0],
        checked_data<double>(floats)[0],
        checked_data<double>(floats)[1],
        checked_data<std::int64_t>(integers)[1]);
}

}  // namespace evrptw::formal_objective

#ifdef __linux__
thread_local std::string native_kernel_scheduler_endpoint;
thread_local bool native_kernel_scheduler_required = false;
#endif

template <typename Callback>
class ScopeRollback final {
public:
    explicit ScopeRollback(Callback callback)
        : callback_(std::move(callback)) {}

    ScopeRollback(const ScopeRollback&) = delete;
    ScopeRollback& operator=(const ScopeRollback&) = delete;

    ~ScopeRollback() noexcept {
        if (active_) {
            callback_();
        }
    }

    void release() noexcept {
        active_ = false;
    }

    void rollback_now() noexcept {
        if (active_) {
            callback_();
            active_ = false;
        }
    }

private:
    Callback callback_;
    bool active_ = true;
};

#ifdef __linux__
class NativeSchedulerThreadContext final {
public:
    NativeSchedulerThreadContext(
        std::string endpoint,
        bool required,
        evrptw::native_client::KernelClientTelemetryCollector* collector)
        : previous_endpoint_(std::exchange(
              native_kernel_scheduler_endpoint, std::move(endpoint))),
          previous_required_(std::exchange(
              native_kernel_scheduler_required, required)),
          previous_collector_(std::exchange(
              evrptw::native_client::telemetry_collector, collector)) {}

    NativeSchedulerThreadContext(const NativeSchedulerThreadContext&) = delete;
    NativeSchedulerThreadContext& operator=(
        const NativeSchedulerThreadContext&) = delete;

    ~NativeSchedulerThreadContext() noexcept {
        native_kernel_scheduler_endpoint = std::move(previous_endpoint_);
        native_kernel_scheduler_required = previous_required_;
        evrptw::native_client::telemetry_collector = previous_collector_;
    }

private:
    std::string previous_endpoint_;
    bool previous_required_;
    evrptw::native_client::KernelClientTelemetryCollector* previous_collector_;
};
#endif

class PythonRandom {
public:
    explicit PythonRandom(std::uint64_t seed) {
        std::vector<std::uint32_t> key;
        do {
            key.push_back(static_cast<std::uint32_t>(seed & 0xffffffffULL));
            seed >>= 32U;
        } while (seed != 0U);
        init_by_array(key);
    }

    double random() {
        const auto upper = next_u32() >> 5U;
        const auto lower = next_u32() >> 6U;
        return (static_cast<double>(upper) * 67108864.0
                + static_cast<double>(lower))
            * (1.0 / 9007199254740992.0);
    }

    std::uint64_t getrandbits(std::uint32_t bits) {
        if (bits == 0U) {
            return 0U;
        }
        if (bits <= 32U) {
            return static_cast<std::uint64_t>(next_u32() >> (32U - bits));
        }
        if (bits > 64U) {
            throw std::invalid_argument("native PythonRandom supports at most 64 bits");
        }
        const auto low = static_cast<std::uint64_t>(next_u32());
        const auto remaining = bits - 32U;
        const auto high = static_cast<std::uint64_t>(
            next_u32() >> (32U - remaining));
        return low | (high << 32U);
    }

    std::uint64_t randbelow(std::uint64_t upper_bound) {
        if (upper_bound == 0U) {
            throw std::invalid_argument("PythonRandom randbelow bound must be positive");
        }
        std::uint32_t bits = 0U;
        for (auto value = upper_bound; value != 0U; value >>= 1U) {
            ++bits;
        }
        while (true) {
            const auto value = getrandbits(bits);
            if (value < upper_bound) {
                return value;
            }
        }
    }

    std::vector<std::int64_t> sample_indices(
        std::int64_t population_size,
        std::int64_t sample_size) {
        if (population_size < 0 || sample_size < 0 || sample_size > population_size) {
            throw std::invalid_argument("PythonRandom sample size is invalid");
        }
        std::int64_t set_size = 21;
        if (sample_size > 5) {
            auto power = std::int64_t{4};
            const auto target = sample_size * 3;
            while (power < target) {
                power *= 4;
            }
            set_size += power;
        }
        std::vector<std::int64_t> result;
        result.reserve(static_cast<std::size_t>(sample_size));
        if (population_size <= set_size) {
            std::vector<std::int64_t> pool(static_cast<std::size_t>(population_size));
            for (std::int64_t index = 0; index < population_size; ++index) {
                pool[static_cast<std::size_t>(index)] = index;
            }
            for (std::int64_t index = 0; index < sample_size; ++index) {
                const auto selected = static_cast<std::int64_t>(
                    randbelow(static_cast<std::uint64_t>(population_size - index)));
                result.push_back(pool[static_cast<std::size_t>(selected)]);
                pool[static_cast<std::size_t>(selected)] =
                    pool[static_cast<std::size_t>(population_size - index - 1)];
            }
            return result;
        }
        std::unordered_set<std::int64_t> selected_indices;
        for (std::int64_t index = 0; index < sample_size; ++index) {
            auto selected = static_cast<std::int64_t>(
                randbelow(static_cast<std::uint64_t>(population_size)));
            while (selected_indices.contains(selected)) {
                selected = static_cast<std::int64_t>(
                    randbelow(static_cast<std::uint64_t>(population_size)));
            }
            selected_indices.insert(selected);
            result.push_back(selected);
        }
        return result;
    }

    std::size_t weighted_index(const std::vector<double>& weights) {
        if (weights.empty()) {
            throw std::invalid_argument("PythonRandom weighted choice requires weights");
        }
        std::vector<double> cumulative;
        cumulative.reserve(weights.size());
        double total = 0.0;
        for (const auto weight : weights) {
            total += weight;
            cumulative.push_back(total);
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::invalid_argument("PythonRandom total weight must be finite and positive");
        }
        const auto target = random() * total;
        return static_cast<std::size_t>(
            std::upper_bound(cumulative.begin(), cumulative.end() - 1, target)
            - cumulative.begin());
    }

    void shuffle(std::vector<std::int64_t>& values) {
        for (std::size_t index = values.size(); index > 1; --index) {
            const auto selected = static_cast<std::size_t>(randbelow(index));
            std::swap(values[index - 1], values[selected]);
        }
    }

private:
    static constexpr std::size_t state_size = 624;
    static constexpr std::size_t period = 397;
    std::array<std::uint32_t, state_size> state_{};
    std::size_t cursor_ = state_size;

    void init_genrand(std::uint32_t seed) {
        state_[0] = seed;
        for (std::size_t index = 1; index < state_size; ++index) {
            state_[index] = 1812433253U
                    * (state_[index - 1] ^ (state_[index - 1] >> 30U))
                + static_cast<std::uint32_t>(index);
        }
        cursor_ = state_size;
    }

    void init_by_array(const std::vector<std::uint32_t>& key) {
        init_genrand(19650218U);
        auto state_index = std::size_t{1};
        auto key_index = std::size_t{0};
        auto rounds = std::max(state_size, key.size());
        for (; rounds != 0; --rounds) {
            state_[state_index] =
                (state_[state_index]
                 ^ ((state_[state_index - 1] ^ (state_[state_index - 1] >> 30U))
                    * 1664525U))
                + key[key_index] + static_cast<std::uint32_t>(key_index);
            ++state_index;
            ++key_index;
            if (state_index >= state_size) {
                state_[0] = state_[state_size - 1];
                state_index = 1;
            }
            if (key_index >= key.size()) {
                key_index = 0;
            }
        }
        for (rounds = state_size - 1; rounds != 0; --rounds) {
            state_[state_index] =
                (state_[state_index]
                 ^ ((state_[state_index - 1] ^ (state_[state_index - 1] >> 30U))
                    * 1566083941U))
                - static_cast<std::uint32_t>(state_index);
            ++state_index;
            if (state_index >= state_size) {
                state_[0] = state_[state_size - 1];
                state_index = 1;
            }
        }
        state_[0] = 0x80000000U;
    }

    std::uint32_t next_u32() {
        if (cursor_ >= state_size) {
            twist();
        }
        auto value = state_[cursor_++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c5680U;
        value ^= (value << 15U) & 0xefc60000U;
        value ^= value >> 18U;
        return value;
    }

    void twist() {
        constexpr auto upper_mask = std::uint32_t{0x80000000U};
        constexpr auto lower_mask = std::uint32_t{0x7fffffffU};
        constexpr auto matrix = std::uint32_t{0x9908b0dfU};
        for (std::size_t index = 0; index < state_size; ++index) {
            const auto value = (state_[index] & upper_mask)
                | (state_[(index + 1) % state_size] & lower_mask);
            state_[index] = state_[(index + period) % state_size]
                ^ (value >> 1U)
                ^ ((value & 1U) != 0U ? matrix : 0U);
        }
        cursor_ = 0;
    }
};

template <typename T>
py::array checked_array(py::handle array, const char* name, int expected_ndim);

template <typename T>
const T* checked_data(const py::array& array);

template <typename T>
T* checked_data(py::array_t<T>& array);

class NativeSha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        if (finalized_) {
            throw std::logic_error("SHA-256 update after finalize");
        }
        total_bytes_ += size;
        while (size > 0) {
            const auto copied = std::min(size, block_.size() - block_size_);
            std::copy(data, data + copied, block_.begin() + block_size_);
            block_size_ += copied;
            data += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finalize() {
        if (finalized_) {
            return digest_;
        }
        const auto bit_length = static_cast<std::uint64_t>(total_bytes_) * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            std::fill(block_.begin() + block_size_, block_.end(), 0U);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + block_size_, block_.begin() + 56, 0U);
        for (std::size_t byte = 0; byte < 8; ++byte) {
            block_[63 - byte] = static_cast<std::uint8_t>(
                (bit_length >> (byte * 8U)) & 0xffU);
        }
        transform(block_.data());
        for (std::size_t word = 0; word < state_.size(); ++word) {
            for (std::size_t byte = 0; byte < 4; ++byte) {
                digest_[word * 4 + byte] = static_cast<std::uint8_t>(
                    (state_[word] >> ((3 - byte) * 8U)) & 0xffU);
            }
        }
        finalized_ = true;
        return digest_;
    }

private:
    static constexpr std::array<std::uint32_t, 64> constants_ = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<std::uint8_t, 64> block_{};
    std::array<std::uint8_t, 32> digest_{};
    std::size_t block_size_ = 0;
    std::size_t total_bytes_ = 0;
    bool finalized_ = false;

    static std::uint32_t rotate_right(std::uint32_t value, std::uint32_t shift) {
        return (value >> shift) | (value << (32U - shift));
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            words[index] =
                (static_cast<std::uint32_t>(block[index * 4]) << 24U)
                | (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U)
                | (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U)
                | static_cast<std::uint32_t>(block[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto small0 = rotate_right(words[index - 15], 7U)
                ^ rotate_right(words[index - 15], 18U)
                ^ (words[index - 15] >> 3U);
            const auto small1 = rotate_right(words[index - 2], 17U)
                ^ rotate_right(words[index - 2], 19U)
                ^ (words[index - 2] >> 10U);
            words[index] = words[index - 16] + small0 + words[index - 7] + small1;
        }
        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto big1 = rotate_right(e, 6U) ^ rotate_right(e, 11U)
                ^ rotate_right(e, 25U);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto first = h + big1 + choose + constants_[index] + words[index];
            const auto big0 = rotate_right(a, 2U) ^ rotate_right(a, 13U)
                ^ rotate_right(a, 22U);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto second = big0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
};

std::array<std::uint8_t, 32> native_sha256_digest(std::string_view payload) {
    NativeSha256 hasher;
    hasher.update(
        reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
    return hasher.finalize();
}

std::string native_sha256_digest_hex(
    const std::array<std::uint8_t, 32>& digest) {
    constexpr std::string_view hexadecimal = "0123456789abcdef";
    std::string hex;
    hex.reserve(64);
    for (const auto byte : digest) {
        hex.push_back(hexadecimal[byte >> 4U]);
        hex.push_back(hexadecimal[byte & 0x0fU]);
    }
    return hex;
}

std::string native_sha256_hex(std::string_view payload) {
    return native_sha256_digest_hex(native_sha256_digest(payload));
}

std::int64_t stable_int63(std::string_view value) {
    const auto digest = native_sha256_digest(value);
    std::uint64_t result = 0;
    for (std::size_t byte = 0; byte < sizeof(result); ++byte) {
        result |= static_cast<std::uint64_t>(digest[byte]) << (byte * 8U);
    }
    return static_cast<std::int64_t>(result & ((1ULL << 63U) - 1ULL));
}

void append_evidence_u64(std::string& evidence, std::uint64_t value) {
    for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
        evidence.push_back(
            static_cast<char>((value >> (byte * 8U)) & 0xffU));
    }
}

void append_evidence_i64(std::string& evidence, std::int64_t value) {
    append_evidence_u64(evidence, static_cast<std::uint64_t>(value));
}

void append_evidence_f64(std::string& evidence, double value) {
    std::uint64_t bits = 0;
    if (std::isnan(value)) {
        bits = 0x7ff8000000000000ULL;
    } else {
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
    }
    append_evidence_u64(evidence, bits);
}

template <typename T>
void append_evidence_values(
    std::string& evidence,
    const T* values,
    std::size_t count) {
    append_evidence_u64(evidence, static_cast<std::uint64_t>(count));
    for (std::size_t index = 0; index < count; ++index) {
        if constexpr (std::is_same_v<T, double>) {
            append_evidence_f64(evidence, values[index]);
        } else if constexpr (std::is_same_v<T, std::uint8_t>) {
            evidence.push_back(static_cast<char>(values[index]));
        } else {
            append_evidence_i64(
                evidence, static_cast<std::int64_t>(values[index]));
        }
    }
}

template <typename T>
void append_evidence_array(
    std::string& evidence,
    const py::array_t<T>& array) {
    append_evidence_u64(
        evidence, static_cast<std::uint64_t>(array.ndim()));
    for (py::ssize_t dimension = 0; dimension < array.ndim(); ++dimension) {
        append_evidence_u64(
            evidence, static_cast<std::uint64_t>(array.shape(dimension)));
    }
    append_evidence_values(
        evidence, checked_data<T>(array), static_cast<std::size_t>(array.size()));
}

void append_nested_evidence(std::string& evidence, py::handle value) {
    if (value.is_none()) {
        evidence.push_back('N');
        return;
    }
    if (py::isinstance<py::tuple>(value)) {
        evidence.push_back('T');
        const auto tuple = py::reinterpret_borrow<py::tuple>(value);
        append_evidence_u64(evidence, static_cast<std::uint64_t>(tuple.size()));
        for (const auto& item : tuple) {
            append_nested_evidence(evidence, item);
        }
        return;
    }
    if (py::isinstance<py::array>(value)) {
        evidence.push_back('A');
        const auto array = py::reinterpret_borrow<py::array>(value);
        const auto dtype = py::cast<std::string>(array.dtype().attr("str"));
        evidence.append(dtype);
        evidence.push_back('\0');
        if (dtype == "<i8" || dtype == "=i8") {
            append_evidence_array(
                evidence, py::cast<py::array_t<std::int64_t>>(array));
        } else if (dtype == "<f8" || dtype == "=f8") {
            append_evidence_array(
                evidence, py::cast<py::array_t<double>>(array));
        } else if (dtype == "|u1") {
            append_evidence_array(
                evidence, py::cast<py::array_t<std::uint8_t>>(array));
        } else {
            throw std::logic_error(
                "native nested evidence contains an unsupported array dtype");
        }
        return;
    }
    if (py::isinstance<py::str>(value)) {
        evidence.push_back('S');
        const auto encoded = py::cast<std::string>(value);
        append_evidence_u64(evidence, static_cast<std::uint64_t>(encoded.size()));
        evidence.append(encoded);
        return;
    }
    if (py::isinstance<py::bool_>(value)) {
        evidence.push_back('B');
        evidence.push_back(py::cast<bool>(value) ? '\1' : '\0');
        return;
    }
    if (py::isinstance<py::int_>(value)) {
        evidence.push_back('I');
        append_evidence_i64(evidence, py::cast<std::int64_t>(value));
        return;
    }
    if (py::isinstance<py::float_>(value)) {
        evidence.push_back('F');
        append_evidence_f64(evidence, py::cast<double>(value));
        return;
    }
    throw std::logic_error(
        "native nested evidence contains an unsupported value");
}

py::tuple native_sha256_v1(py::handle payload) {
    auto payload_array = checked_array<std::uint8_t>(payload, "payload", 1);
    const auto digest = native_sha256_digest(std::string_view(
        reinterpret_cast<const char*>(checked_data<std::uint8_t>(payload_array)),
        static_cast<std::size_t>(payload_array.size())));
    py::array_t<std::uint8_t> digest_array(digest.size());
    std::copy(digest.begin(), digest.end(), checked_data(digest_array));
    return py::make_tuple(
        std::move(digest_array), native_sha256_digest_hex(digest));
}

py::tuple python_random_golden_v1(
    std::uint64_t seed,
    std::int64_t random_count,
    py::handle randbelow_bounds,
    std::int64_t sample_population,
    std::int64_t sample_size,
    py::handle weights,
    std::int64_t shuffle_size) {
    if (random_count < 0 || shuffle_size < 0) {
        throw std::invalid_argument("PythonRandom vector sizes must be non-negative");
    }
    auto bound_array = checked_array<std::int64_t>(
        randbelow_bounds, "randbelow_bounds", 1);
    auto weight_array = checked_array<double>(weights, "weights", 1);
    PythonRandom random(seed);
    py::array_t<double> random_values(random_count);
    for (std::int64_t index = 0; index < random_count; ++index) {
        checked_data(random_values)[index] = random.random();
    }
    py::array_t<std::int64_t> bounded_values(bound_array.size());
    for (py::ssize_t index = 0; index < bound_array.size(); ++index) {
        const auto bound = checked_data<std::int64_t>(bound_array)[index];
        if (bound <= 0) {
            throw std::invalid_argument("PythonRandom randbelow bounds must be positive");
        }
        checked_data(bounded_values)[index] = static_cast<std::int64_t>(
            random.randbelow(static_cast<std::uint64_t>(bound)));
    }
    const auto sampled = random.sample_indices(sample_population, sample_size);
    py::array_t<std::int64_t> sampled_values(sampled.size());
    std::copy(sampled.begin(), sampled.end(), checked_data(sampled_values));
    std::vector<double> weight_values(
        checked_data<double>(weight_array),
        checked_data<double>(weight_array) + weight_array.size());
    const auto weighted = static_cast<std::int64_t>(random.weighted_index(weight_values));
    std::vector<std::int64_t> shuffled(static_cast<std::size_t>(shuffle_size));
    for (std::int64_t index = 0; index < shuffle_size; ++index) {
        shuffled[static_cast<std::size_t>(index)] = index;
    }
    random.shuffle(shuffled);
    py::array_t<std::int64_t> shuffled_values(shuffled.size());
    std::copy(shuffled.begin(), shuffled.end(), checked_data(shuffled_values));
    return py::make_tuple(
        std::move(random_values),
        std::move(bounded_values),
        std::move(sampled_values),
        weighted,
        std::move(shuffled_values));
}

py::tuple legacy_destroy_v2(
    std::uint64_t seed,
    std::int64_t operation,
    std::int64_t remove_count,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle distance,
    py::handle lexical_rank,
    std::int64_t depot) {
    auto offsets_array = checked_array<std::int64_t>(
        route_offsets, "route_offsets", 1);
    auto indices_array = checked_array<std::int64_t>(
        route_indices, "route_indices", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto lexical_array = checked_array<std::int64_t>(
        lexical_rank, "lexical_rank", 1);
    if (operation < 0 || operation > 2 || remove_count <= 0
        || offsets_array.size() < 2 || distance_array.shape(0) == 0
        || distance_array.shape(0) != distance_array.shape(1)
        || lexical_array.size() != distance_array.shape(0)
        || depot < 0 || depot >= distance_array.shape(0)) {
        throw std::invalid_argument("legacy destroy input/config is invalid");
    }
    const auto route_count = static_cast<std::size_t>(offsets_array.size() - 1);
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* indices = checked_data<std::int64_t>(indices_array);
    const auto* distances = checked_data<double>(distance_array);
    const auto* lexical = checked_data<std::int64_t>(lexical_array);
    const auto node_count = static_cast<std::size_t>(distance_array.shape(0));
    if (offsets[0] != 0 || offsets[route_count] != indices_array.size()) {
        throw std::invalid_argument("legacy destroy route offsets are invalid");
    }
    std::vector<std::vector<std::int64_t>> routes;
    std::vector<std::int64_t> customers;
    std::vector<bool> seen(node_count, false);
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] >= offsets[route + 1]) {
            throw std::invalid_argument("legacy destroy routes must be non-empty");
        }
        routes.emplace_back(
            indices + offsets[route], indices + offsets[route + 1]);
        for (const auto customer : routes.back()) {
            if (customer < 0 || static_cast<std::size_t>(customer) >= node_count
                || customer == depot || seen[static_cast<std::size_t>(customer)]) {
                throw std::invalid_argument(
                    "legacy destroy routes must contain unique non-depot nodes");
            }
            seen[static_cast<std::size_t>(customer)] = true;
            customers.push_back(customer);
        }
    }
    const auto count = std::min<std::size_t>(
        static_cast<std::size_t>(remove_count), customers.size());
    PythonRandom random(seed);
    std::vector<std::int64_t> removed;
    removed.reserve(count);
    if (operation == 0) {
        const auto sampled = random.sample_indices(
            static_cast<std::int64_t>(customers.size()),
            static_cast<std::int64_t>(count));
        for (const auto index : sampled) {
            removed.push_back(customers[static_cast<std::size_t>(index)]);
        }
    } else if (operation == 1) {
        std::vector<std::pair<double, std::int64_t>> contributions;
        contributions.reserve(customers.size());
        for (const auto& route : routes) {
            for (std::size_t position = 0; position < route.size(); ++position) {
                const auto before = position == 0 ? depot : route[position - 1];
                const auto customer = route[position];
                const auto after = position + 1 == route.size()
                    ? depot
                    : route[position + 1];
                const auto saving = distances[
                    static_cast<std::size_t>(before) * node_count
                    + static_cast<std::size_t>(customer)]
                    + distances[static_cast<std::size_t>(customer) * node_count
                                + static_cast<std::size_t>(after)]
                    - distances[static_cast<std::size_t>(before) * node_count
                                + static_cast<std::size_t>(after)];
                contributions.emplace_back(saving, customer);
            }
        }
        std::stable_sort(
            contributions.begin(), contributions.end(),
            [&](const auto& left, const auto& right) {
                if (left.first != right.first) {
                    return left.first > right.first;
                }
                return lexical[left.second] > lexical[right.second];
            });
        for (std::size_t index = 0; index < count; ++index) {
            removed.push_back(contributions[index].second);
        }
    } else {
        const auto anchor = customers[static_cast<std::size_t>(
            random.randbelow(customers.size()))];
        std::vector<std::pair<double, std::int64_t>> related;
        related.reserve(customers.size());
        for (const auto customer : customers) {
            related.emplace_back(
                distances[static_cast<std::size_t>(anchor) * node_count
                          + static_cast<std::size_t>(customer)],
                customer);
        }
        std::stable_sort(
            related.begin(), related.end(),
            [&](const auto& left, const auto& right) {
                if (left.first != right.first) {
                    return left.first < right.first;
                }
                return lexical[left.second] < lexical[right.second];
            });
        for (std::size_t index = 0; index < count; ++index) {
            removed.push_back(related[index].second);
        }
    }
    const std::unordered_set<std::int64_t> removed_set(
        removed.begin(), removed.end());
    std::vector<std::int64_t> partial_offsets{0};
    std::vector<std::int64_t> partial_indices;
    for (const auto& route : routes) {
        const auto before = partial_indices.size();
        for (const auto customer : route) {
            if (!removed_set.contains(customer)) {
                partial_indices.push_back(customer);
            }
        }
        if (partial_indices.size() != before) {
            partial_offsets.push_back(
                static_cast<std::int64_t>(partial_indices.size()));
        }
    }
    py::array_t<std::int64_t> partial_offsets_array(partial_offsets.size());
    py::array_t<std::int64_t> partial_indices_array(partial_indices.size());
    py::array_t<std::int64_t> removed_array(removed.size());
    std::copy(
        partial_offsets.begin(), partial_offsets.end(),
        checked_data(partial_offsets_array));
    std::copy(
        partial_indices.begin(), partial_indices.end(),
        checked_data(partial_indices_array));
    std::copy(removed.begin(), removed.end(), checked_data(removed_array));
    return py::make_tuple(
        std::move(partial_offsets_array),
        std::move(partial_indices_array),
        std::move(removed_array));
}

py::array_t<std::int64_t> native_objective_acceptance_v1(
    py::handle current_integer,
    py::handle current_float,
    py::handle candidate_integer,
    py::handle candidate_float,
    py::handle temperatures,
    py::handle random_draws) {
    auto current_i = checked_array<std::int64_t>(
        current_integer, "current_integer", 2);
    auto current_f = checked_array<double>(current_float, "current_float", 2);
    auto candidate_i = checked_array<std::int64_t>(
        candidate_integer, "candidate_integer", 2);
    auto candidate_f = checked_array<double>(candidate_float, "candidate_float", 2);
    auto temperature_array = checked_array<double>(temperatures, "temperatures", 1);
    auto random_array = checked_array<double>(random_draws, "random_draws", 1);
    const auto rows = current_i.shape(0);
    if (current_i.shape(1) != 2 || candidate_i.shape(0) != rows
        || candidate_i.shape(1) != 2 || current_f.shape(0) != rows
        || current_f.shape(1) != 2 || candidate_f.shape(0) != rows
        || candidate_f.shape(1) != 2 || temperature_array.size() != rows
        || random_array.size() != rows) {
        throw std::invalid_argument("native objective acceptance arrays do not align");
    }
    const auto* current_integer_values = checked_data<std::int64_t>(current_i);
    const auto* current_float_values = checked_data<double>(current_f);
    const auto* candidate_integer_values = checked_data<std::int64_t>(candidate_i);
    const auto* candidate_float_values = checked_data<double>(candidate_f);
    const auto* temperature_values = checked_data<double>(temperature_array);
    const auto* random_values = checked_data<double>(random_array);
    py::array_t<std::int64_t> output(rows);
    auto* accepted = checked_data(output);
    for (py::ssize_t row = 0; row < rows; ++row) {
        const auto current_vehicles = current_integer_values[row * 2];
        const auto candidate_vehicles = candidate_integer_values[row * 2];
        const auto current_charging_count = current_integer_values[row * 2 + 1];
        const auto candidate_charging_count = candidate_integer_values[row * 2 + 1];
        const auto current_distance = current_float_values[row * 2];
        const auto current_charging_time = current_float_values[row * 2 + 1];
        const auto candidate_distance = candidate_float_values[row * 2];
        const auto candidate_charging_time = candidate_float_values[row * 2 + 1];
        if (current_vehicles < 0 || candidate_vehicles < 0
            || current_charging_count < 0 || candidate_charging_count < 0
            || !std::isfinite(current_distance) || current_distance < 0.0
            || !std::isfinite(current_charging_time) || current_charging_time < 0.0
            || !std::isfinite(candidate_distance) || candidate_distance < 0.0
            || !std::isfinite(candidate_charging_time) || candidate_charging_time < 0.0) {
            throw std::invalid_argument("native objective fields are invalid");
        }
        const auto current_key = evrptw::formal_objective::key(
            current_vehicles,
            current_distance,
            current_charging_time,
            current_charging_count);
        const auto candidate_key = evrptw::formal_objective::key(
            candidate_vehicles,
            candidate_distance,
            candidate_charging_time,
            candidate_charging_count);
        if (!std::isfinite(temperature_values[row]) || temperature_values[row] <= 0.0
            || !std::isfinite(random_values[row]) || random_values[row] < 0.0
            || random_values[row] > 1.0) {
            throw std::invalid_argument("native acceptance temperature/draw is invalid");
        }
        if (candidate_vehicles < current_vehicles) {
            accepted[row] = 1;
        } else if (candidate_vehicles > current_vehicles) {
            accepted[row] = 0;
        } else if (candidate_key <= current_key) {
            accepted[row] = 1;
        } else if (std::get<1>(candidate_key) == std::get<1>(current_key)) {
            accepted[row] = 0;
        } else {
            const auto distance_delta = candidate_float_values[row * 2]
                - current_float_values[row * 2];
            accepted[row] = random_values[row]
                    < std::exp(-distance_delta / temperature_values[row])
                ? 1
                : 0;
        }
    }
    return output;
}

py::tuple stage04_segment_update_v1(
    py::handle weights,
    py::handle reward_sums,
    py::handle calls,
    py::handle options) {
    auto weight_array = checked_array<double>(weights, "weights", 1);
    auto reward_array = checked_array<double>(reward_sums, "reward_sums", 1);
    auto call_array = checked_array<std::int64_t>(calls, "calls", 1);
    auto option_array = checked_array<double>(options, "options", 1);
    if (reward_array.size() != weight_array.size()
        || call_array.size() != weight_array.size() || option_array.size() != 4) {
        throw std::invalid_argument("Stage 4 segment arrays do not align");
    }
    const auto* old_weights = checked_data<double>(weight_array);
    const auto* rewards = checked_data<double>(reward_array);
    const auto* call_values = checked_data<std::int64_t>(call_array);
    const auto* config = checked_data<double>(option_array);
    const auto reaction = config[0];
    const auto floor = config[1];
    const auto smoothing = config[2];
    const auto minimum_calls = static_cast<std::int64_t>(config[3]);
    if (!(reaction > 0.0 && reaction <= 1.0) || !(floor > 0.0 && floor < 1.0)
        || !(smoothing >= 0.0 && smoothing <= 1.0) || minimum_calls <= 0) {
        throw std::invalid_argument("Stage 4 segment options are invalid");
    }
    py::array_t<double> updated(weight_array.size());
    py::array_t<std::int64_t> statuses(weight_array.size());
    auto* updated_values = checked_data(updated);
    auto* status_values = checked_data(statuses);
    for (py::ssize_t index = 0; index < weight_array.size(); ++index) {
        if (call_values[index] < 0 || !std::isfinite(old_weights[index])
            || !std::isfinite(rewards[index])) {
            throw std::invalid_argument("Stage 4 segment values are invalid");
        }
        if (call_values[index] < minimum_calls) {
            updated_values[index] = old_weights[index];
            status_values[index] = 0;
            continue;
        }
        const auto average = rewards[index] / static_cast<double>(call_values[index]);
        const auto reacted = std::max(
            floor,
            (1.0 - reaction) * old_weights[index] + reaction * average);
        updated_values[index] = smoothing * old_weights[index]
            + (1.0 - smoothing) * reacted;
        status_values[index] = 1;
    }
    return py::make_tuple(std::move(updated), std::move(statuses));
}

py::array_t<std::int64_t> dynamic_removal_selection_v2(
    std::int64_t customer_count,
    std::int64_t stagnation_iterations,
    std::int64_t iteration,
    py::handle thresholds,
    py::handle fractions,
    bool global_best_reset) {
    auto threshold_array = checked_array<std::int64_t>(
        thresholds, "thresholds", 1);
    auto fraction_array = checked_array<double>(fractions, "fractions", 1);
    if (customer_count < 0 || stagnation_iterations < 0 || iteration < 0
        || threshold_array.size() != 3 || fraction_array.size() != 6) {
        throw std::invalid_argument(
            "dynamic removal selection v2 input/config shape is invalid");
    }
    const auto* threshold_values = checked_data<std::int64_t>(threshold_array);
    const auto* fraction_values = checked_data<double>(fraction_array);
    const auto medium_threshold = threshold_values[0];
    const auto large_threshold = threshold_values[1];
    const auto exploration_period = threshold_values[2];
    if (medium_threshold < 0 || large_threshold <= medium_threshold
        || exploration_period <= 0) {
        throw std::invalid_argument(
            "dynamic removal thresholds are invalid");
    }
    for (std::size_t index = 0; index < 3; ++index) {
        const auto minimum = fraction_values[index * 2];
        const auto maximum = fraction_values[index * 2 + 1];
        if (!std::isfinite(minimum) || !std::isfinite(maximum)
            || minimum <= 0.0 || maximum < minimum || maximum > 1.0) {
            throw std::invalid_argument(
                "dynamic removal fraction bounds are invalid");
        }
    }
    std::int64_t tier = 0;
    std::int64_t trigger = 0;
    if (stagnation_iterations >= large_threshold) {
        tier = 2;
        trigger = 2;
    } else if (stagnation_iterations >= medium_threshold) {
        tier = 1;
        trigger = 1;
    }
    if (iteration > 0 && iteration % exploration_period == 0
        && stagnation_iterations > medium_threshold && tier < 2) {
        ++tier;
        trigger += 3;
    }
    std::int64_t requested = 0;
    std::int64_t lower_bound = 0;
    std::int64_t upper_bound = 0;
    if (customer_count > 1) {
        const auto upper_customer_bound = customer_count - 1;
        lower_bound = std::max<std::int64_t>(
            1,
            std::min<std::int64_t>(
                upper_customer_bound,
                static_cast<std::int64_t>(std::ceil(
                    static_cast<double>(customer_count)
                    * fraction_values[static_cast<std::size_t>(tier) * 2]))));
        upper_bound = std::max<std::int64_t>(
            lower_bound,
            std::min<std::int64_t>(
                upper_customer_bound,
                static_cast<std::int64_t>(std::floor(
                    static_cast<double>(customer_count)
                    * fraction_values[static_cast<std::size_t>(tier) * 2 + 1]))));
        requested = lower_bound;
    } else {
        trigger = 6;
    }
    py::array_t<std::int64_t> output(7);
    auto* values = checked_data(output);
    values[0] = tier;
    values[1] = requested;
    values[2] = lower_bound;
    values[3] = upper_bound;
    values[4] = stagnation_iterations;
    values[5] = trigger;
    values[6] = global_best_reset ? 1 : 0;
    return output;
}

py::tuple changed_candidate_pool_v1(
    std::int64_t operation,
    py::handle route_offsets,
    py::handle route_indices) {
    auto offsets_array = checked_array<std::int64_t>(
        route_offsets, "route_offsets", 1);
    auto indices_array = checked_array<std::int64_t>(
        route_indices, "route_indices", 1);
    if (operation < 0 || operation > 2 || offsets_array.size() < 2) {
        throw std::invalid_argument("changed-candidate operation/routes are invalid");
    }
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* indices = checked_data<std::int64_t>(indices_array);
    const auto route_count = static_cast<std::size_t>(offsets_array.size() - 1);
    if (offsets[0] != 0 || offsets[route_count] != indices_array.size()) {
        throw std::invalid_argument("changed-candidate route offsets do not span indices");
    }
    std::vector<std::vector<std::int64_t>> routes;
    routes.reserve(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] >= offsets[route + 1]) {
            throw std::invalid_argument(
                "changed-candidate routes must be monotone and non-empty");
        }
        routes.emplace_back(
            indices + offsets[route],
            indices + offsets[route + 1]);
    }
    std::vector<std::int64_t> changed_route_indices;
    std::vector<std::int64_t> change_offsets{0};
    std::vector<std::int64_t> change_indices;
    std::vector<std::int64_t> removed_offsets{0};
    std::vector<std::int64_t> removed_indices;
    const auto append = [&](
                            std::size_t left_index,
                            const std::vector<std::int64_t>& left,
                            std::size_t right_index,
                            const std::vector<std::int64_t>& right,
                            const std::vector<std::int64_t>& removed) {
        changed_route_indices.push_back(static_cast<std::int64_t>(left_index));
        changed_route_indices.push_back(static_cast<std::int64_t>(right_index));
        change_indices.insert(change_indices.end(), left.begin(), left.end());
        change_offsets.push_back(static_cast<std::int64_t>(change_indices.size()));
        change_indices.insert(change_indices.end(), right.begin(), right.end());
        change_offsets.push_back(static_cast<std::int64_t>(change_indices.size()));
        removed_indices.insert(removed_indices.end(), removed.begin(), removed.end());
        removed_offsets.push_back(static_cast<std::int64_t>(removed_indices.size()));
    };
    if (operation == 0) {
        for (std::size_t source_index = 0; source_index < route_count; ++source_index) {
            const auto& source = routes[source_index];
            if (source.size() <= 1) {
                continue;
            }
            for (std::size_t source_position = 0;
                 source_position < source.size(); ++source_position) {
                const auto customer = source[source_position];
                auto source_without = source;
                source_without.erase(
                    source_without.begin()
                    + static_cast<std::ptrdiff_t>(source_position));
                for (std::size_t target_index = 0; target_index < route_count;
                     ++target_index) {
                    if (target_index == source_index) {
                        continue;
                    }
                    const auto& target = routes[target_index];
                    for (std::size_t target_position = 0;
                         target_position <= target.size(); ++target_position) {
                        auto target_with = target;
                        target_with.insert(
                            target_with.begin()
                                + static_cast<std::ptrdiff_t>(target_position),
                            customer);
                        if (source_index < target_index) {
                            append(
                                source_index, source_without,
                                target_index, target_with, {customer});
                        } else {
                            append(
                                target_index, target_with,
                                source_index, source_without, {customer});
                        }
                    }
                }
            }
        }
    } else if (operation == 1) {
        for (std::size_t left_index = 0; left_index < route_count; ++left_index) {
            const auto& left = routes[left_index];
            for (std::size_t right_index = left_index + 1;
                 right_index < route_count; ++right_index) {
                const auto& right = routes[right_index];
                for (std::size_t left_position = 0;
                     left_position < left.size(); ++left_position) {
                    for (std::size_t right_position = 0;
                         right_position < right.size(); ++right_position) {
                        auto new_left = left;
                        auto new_right = right;
                        new_left[left_position] = right[right_position];
                        new_right[right_position] = left[left_position];
                        append(
                            left_index, new_left, right_index, new_right,
                            {left[left_position], right[right_position]});
                    }
                }
            }
        }
    } else {
        for (std::size_t left_index = 0; left_index < route_count; ++left_index) {
            const auto& left = routes[left_index];
            for (std::size_t right_index = left_index + 1;
                 right_index < route_count; ++right_index) {
                const auto& right = routes[right_index];
                for (std::size_t left_cut = 1; left_cut < left.size(); ++left_cut) {
                    for (std::size_t right_cut = 1; right_cut < right.size(); ++right_cut) {
                        std::vector<std::int64_t> new_left(
                            left.begin(),
                            left.begin() + static_cast<std::ptrdiff_t>(left_cut));
                        new_left.insert(
                            new_left.end(),
                            right.begin() + static_cast<std::ptrdiff_t>(right_cut),
                            right.end());
                        std::vector<std::int64_t> new_right(
                            right.begin(),
                            right.begin() + static_cast<std::ptrdiff_t>(right_cut));
                        new_right.insert(
                            new_right.end(),
                            left.begin() + static_cast<std::ptrdiff_t>(left_cut),
                            left.end());
                        append(left_index, new_left, right_index, new_right, {});
                    }
                }
            }
        }
    }
    py::array_t<std::int64_t> changed_array(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(changed_route_indices.size() / 2), 2});
    py::array_t<std::int64_t> change_offsets_array(change_offsets.size());
    py::array_t<std::int64_t> change_indices_array(change_indices.size());
    py::array_t<std::int64_t> removed_offsets_array(removed_offsets.size());
    py::array_t<std::int64_t> removed_indices_array(removed_indices.size());
    std::copy(
        changed_route_indices.begin(), changed_route_indices.end(),
        checked_data(changed_array));
    std::copy(
        change_offsets.begin(), change_offsets.end(),
        checked_data(change_offsets_array));
    std::copy(
        change_indices.begin(), change_indices.end(),
        checked_data(change_indices_array));
    std::copy(
        removed_offsets.begin(), removed_offsets.end(),
        checked_data(removed_offsets_array));
    std::copy(
        removed_indices.begin(), removed_indices.end(),
        checked_data(removed_indices_array));
    return py::make_tuple(
        std::move(changed_array),
        std::move(change_offsets_array),
        std::move(change_indices_array),
        std::move(removed_offsets_array),
        std::move(removed_indices_array));
}

py::tuple insertion_candidate_plans_v2_impl(
    py::handle current_route_offsets,
    py::handle current_route_indices,
    std::int64_t customer,
    py::handle demand,
    double load_capacity,
    double epsilon,
    bool allow_new_route) {
    auto offsets_array = checked_array<std::int64_t>(
        current_route_offsets, "current_route_offsets", 1);
    auto indices_array = checked_array<std::int64_t>(
        current_route_indices, "current_route_indices", 1);
    auto demand_array = checked_array<double>(demand, "demand", 1);
    if (offsets_array.size() < 1 || customer < 0
        || customer >= demand_array.size() || !std::isfinite(load_capacity)
        || load_capacity < 0.0 || !std::isfinite(epsilon) || epsilon < 0.0) {
        throw std::invalid_argument("insertion candidate-plan input/config is invalid");
    }
    const auto route_count = static_cast<std::size_t>(offsets_array.size() - 1);
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* indices = checked_data<std::int64_t>(indices_array);
    const auto* demands = checked_data<double>(demand_array);
    if (offsets[0] != 0 || offsets[route_count] != indices_array.size()) {
        throw std::invalid_argument("insertion current routes do not span indices");
    }
    std::vector<std::vector<std::int64_t>> routes;
    routes.reserve(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] >= offsets[route + 1]) {
            throw std::invalid_argument("insertion current routes must be non-empty");
        }
        routes.emplace_back(
            indices + offsets[route], indices + offsets[route + 1]);
        for (const auto node : routes.back()) {
            if (node < 0 || node >= demand_array.size() || node == customer) {
                throw std::invalid_argument(
                    "insertion routes contain an invalid or already-present customer");
            }
        }
    }
    std::vector<std::int64_t> plan_offsets{0};
    std::vector<std::int64_t> route_offsets{0};
    std::vector<std::int64_t> route_indices;
    std::vector<std::int64_t> metadata;
    const auto target_count = route_count + (allow_new_route ? 1U : 0U);
    for (std::size_t target = 0; target < target_count; ++target) {
        const std::vector<std::int64_t> empty;
        const auto& base = target < route_count ? routes[target] : empty;
        double demand_total = 0.0;
        double demand_compensation = 0.0;
        const auto add_demand = [&](double value) {
            const auto next = demand_total + value;
            demand_compensation += std::fabs(demand_total) >= std::fabs(value)
                ? (demand_total - next) + value
                : (value - next) + demand_total;
            demand_total = next;
        };
        add_demand(demands[customer]);
        for (const auto node : base) {
            add_demand(demands[node]);
        }
        const auto total_demand = demand_compensation != 0.0
                && std::isfinite(demand_compensation)
            ? demand_total + demand_compensation
            : demand_total;
        if (total_demand > load_capacity + epsilon) {
            continue;
        }
        for (std::size_t position = 0; position <= base.size(); ++position) {
            auto candidate = base;
            candidate.insert(
                candidate.begin() + static_cast<std::ptrdiff_t>(position), customer);
            for (std::size_t route = 0; route < route_count; ++route) {
                const auto& selected = route == target ? candidate : routes[route];
                route_indices.insert(
                    route_indices.end(), selected.begin(), selected.end());
                route_offsets.push_back(
                    static_cast<std::int64_t>(route_indices.size()));
            }
            if (target == route_count) {
                route_indices.insert(
                    route_indices.end(), candidate.begin(), candidate.end());
                route_offsets.push_back(
                    static_cast<std::int64_t>(route_indices.size()));
            }
            plan_offsets.push_back(
                static_cast<std::int64_t>(route_offsets.size() - 1));
            metadata.push_back(static_cast<std::int64_t>(target));
            metadata.push_back(static_cast<std::int64_t>(position));
        }
    }
    py::array_t<std::int64_t> plan_offsets_array(plan_offsets.size());
    py::array_t<std::int64_t> route_offsets_array(route_offsets.size());
    py::array_t<std::int64_t> route_indices_array(route_indices.size());
    py::array_t<std::int64_t> metadata_array(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(metadata.size() / 2), 2});
    std::copy(
        plan_offsets.begin(), plan_offsets.end(), checked_data(plan_offsets_array));
    std::copy(
        route_offsets.begin(), route_offsets.end(), checked_data(route_offsets_array));
    std::copy(
        route_indices.begin(), route_indices.end(), checked_data(route_indices_array));
    std::copy(metadata.begin(), metadata.end(), checked_data(metadata_array));
    return py::make_tuple(
        std::move(plan_offsets_array),
        std::move(route_offsets_array),
        std::move(route_indices_array),
        std::move(metadata_array));
}

py::tuple insertion_candidate_plans_v2(
    py::handle current_route_offsets,
    py::handle current_route_indices,
    std::int64_t customer,
    py::handle demand,
    double load_capacity,
    double epsilon) {
    return insertion_candidate_plans_v2_impl(
        current_route_offsets, current_route_indices, customer, demand,
        load_capacity, epsilon, true);
}

py::tuple route_merge_candidate_pool_v2(
    py::handle route_offsets,
    py::handle route_indices,
    py::handle route_objective_metrics,
    py::handle demand,
    double load_capacity,
    double epsilon,
    bool pair_pruning,
    bool preserve_duplicates) {
    auto offsets_array = checked_array<std::int64_t>(
        route_offsets, "route_offsets", 1);
    auto indices_array = checked_array<std::int64_t>(
        route_indices, "route_indices", 1);
    auto metrics_array = checked_array<double>(
        route_objective_metrics, "route_objective_metrics", 2);
    auto demand_array = checked_array<double>(demand, "demand", 1);
    if (offsets_array.size() < 3 || metrics_array.shape(1) != 2
        || metrics_array.shape(0) != offsets_array.size() - 1
        || !std::isfinite(load_capacity) || load_capacity < 0.0
        || !std::isfinite(epsilon) || epsilon < 0.0) {
        throw std::invalid_argument("route-merge candidate-pool input/config is invalid");
    }
    const auto route_count = static_cast<std::size_t>(offsets_array.size() - 1);
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* indices = checked_data<std::int64_t>(indices_array);
    const auto* metrics = checked_data<double>(metrics_array);
    const auto* demands = checked_data<double>(demand_array);
    if (offsets[0] != 0 || offsets[route_count] != indices_array.size()) {
        throw std::invalid_argument("route-merge route offsets are invalid");
    }
    std::vector<std::vector<std::int64_t>> routes;
    std::vector<double> route_demands(route_count, 0.0);
    routes.reserve(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] >= offsets[route + 1]) {
            throw std::invalid_argument("route-merge routes must be non-empty");
        }
        routes.emplace_back(
            indices + offsets[route], indices + offsets[route + 1]);
        double total = 0.0;
        double compensation = 0.0;
        for (const auto node : routes.back()) {
            if (node < 0 || node >= demand_array.size()) {
                throw std::invalid_argument("route-merge route contains an unknown node");
            }
            const auto value = demands[node];
            const auto next = total + value;
            compensation += std::fabs(total) >= std::fabs(value)
                ? (total - next) + value
                : (value - next) + total;
            total = next;
        }
        route_demands[route] = compensation != 0.0 && std::isfinite(compensation)
            ? total + compensation
            : total;
        if (!std::isfinite(metrics[route * 2]) || metrics[route * 2] < 0.0
            || !std::isfinite(metrics[route * 2 + 1]) || metrics[route * 2 + 1] < 0.0) {
            throw std::invalid_argument("route-merge objective metrics are invalid");
        }
    }
    struct Pair {
        std::size_t left;
        std::size_t right;
    };
    std::vector<Pair> pairs;
    for (std::size_t left = 0; left < route_count; ++left) {
        for (std::size_t right = left + 1; right < route_count; ++right) {
            pairs.push_back(Pair{left, right});
        }
    }
    std::stable_sort(pairs.begin(), pairs.end(), [&](const Pair& left, const Pair& right) {
        const auto left_size = routes[left.left].size() + routes[left.right].size();
        const auto right_size = routes[right.left].size() + routes[right.right].size();
        if (left_size != right_size) {
            return left_size < right_size;
        }
        const auto left_demand = route_demands[left.left] + route_demands[left.right];
        const auto right_demand = route_demands[right.left] + route_demands[right.right];
        if (left_demand != right_demand) {
            return left_demand < right_demand;
        }
        const auto left_distance = metrics[left.left * 2] + metrics[left.right * 2];
        const auto right_distance = metrics[right.left * 2] + metrics[right.right * 2];
        if (left_distance != right_distance) {
            return left_distance > right_distance;
        }
        const auto left_charging = metrics[left.left * 2 + 1]
            + metrics[left.right * 2 + 1];
        const auto right_charging = metrics[right.left * 2 + 1]
            + metrics[right.right * 2 + 1];
        if (left_charging != right_charging) {
            return left_charging > right_charging;
        }
        return std::tie(left.left, left.right) < std::tie(right.left, right.right);
    });
    std::vector<std::int64_t> candidate_offsets{0};
    std::vector<std::int64_t> candidate_indices;
    std::vector<std::int64_t> metadata;
    std::unordered_set<std::string> seen;
    std::int64_t pruned_pairs = 0;
    std::int64_t pruned_candidates = 0;
    const auto key_for = [](const std::vector<std::int64_t>& route) {
        std::string key;
        key.resize(route.size() * sizeof(std::int64_t));
        if (!route.empty()) {
            std::memcpy(key.data(), route.data(), key.size());
        }
        return key;
    };
    for (const auto& pair : pairs) {
        if (pair_pruning
            && route_demands[pair.left] + route_demands[pair.right]
                > load_capacity + epsilon) {
            ++pruned_pairs;
            pruned_candidates += static_cast<std::int64_t>(
                routes[pair.left].size() + routes[pair.right].size() + 2);
            continue;
        }
        for (const auto& [source_index, target_index] : std::array{
                 std::pair{pair.left, pair.right},
                 std::pair{pair.right, pair.left},
             }) {
            const auto& source = routes[source_index];
            const auto& target = routes[target_index];
            for (std::size_t position = 0; position <= target.size(); ++position) {
                std::vector<std::int64_t> merged(
                    target.begin(), target.begin() + static_cast<std::ptrdiff_t>(position));
                merged.insert(merged.end(), source.begin(), source.end());
                merged.insert(
                    merged.end(),
                    target.begin() + static_cast<std::ptrdiff_t>(position),
                    target.end());
                const auto key = key_for(merged);
                if (!preserve_duplicates && !seen.insert(key).second) {
                    continue;
                }
                if (preserve_duplicates) {
                    seen.insert(key);
                }
                candidate_indices.insert(
                    candidate_indices.end(), merged.begin(), merged.end());
                candidate_offsets.push_back(
                    static_cast<std::int64_t>(candidate_indices.size()));
                metadata.insert(
                    metadata.end(),
                    {static_cast<std::int64_t>(pair.left),
                     static_cast<std::int64_t>(pair.right),
                     static_cast<std::int64_t>(source_index),
                     static_cast<std::int64_t>(target_index),
                     static_cast<std::int64_t>(position)});
            }
        }
    }
    py::array_t<std::int64_t> offsets_output(candidate_offsets.size());
    py::array_t<std::int64_t> indices_output(candidate_indices.size());
    py::array_t<std::int64_t> metadata_output(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(metadata.size() / 5), 5});
    py::array_t<std::int64_t> pruning_output(2);
    std::copy(
        candidate_offsets.begin(), candidate_offsets.end(),
        checked_data(offsets_output));
    std::copy(
        candidate_indices.begin(), candidate_indices.end(),
        checked_data(indices_output));
    std::copy(metadata.begin(), metadata.end(), checked_data(metadata_output));
    checked_data(pruning_output)[0] = pruned_pairs;
    checked_data(pruning_output)[1] = pruned_candidates;
    return py::make_tuple(
        std::move(offsets_output),
        std::move(indices_output),
        std::move(metadata_output),
        std::move(pruning_output));
}

py::tuple assemble_changed_candidate_plans_v1(
    py::handle current_route_offsets,
    py::handle current_route_indices,
    py::handle changed_route_indices,
    py::handle change_offsets,
    py::handle change_indices) {
    auto current_offsets_array = checked_array<std::int64_t>(
        current_route_offsets, "current_route_offsets", 1);
    auto current_indices_array = checked_array<std::int64_t>(
        current_route_indices, "current_route_indices", 1);
    auto changed_routes_array = checked_array<std::int64_t>(
        changed_route_indices, "changed_route_indices", 2);
    auto change_offsets_array = checked_array<std::int64_t>(
        change_offsets, "change_offsets", 1);
    auto change_indices_array = checked_array<std::int64_t>(
        change_indices, "change_indices", 1);
    if (current_offsets_array.size() < 2 || changed_routes_array.shape(1) != 2) {
        throw std::invalid_argument("changed-candidate plan shape is invalid");
    }
    const auto route_count = static_cast<std::size_t>(
        current_offsets_array.size() - 1);
    const auto candidate_count = static_cast<std::size_t>(
        changed_routes_array.shape(0));
    if (change_offsets_array.size()
        != static_cast<py::ssize_t>(candidate_count * 2 + 1)) {
        throw std::invalid_argument("changed-candidate change offsets do not align");
    }
    const auto* current_offsets = checked_data<std::int64_t>(current_offsets_array);
    const auto* current_indices = checked_data<std::int64_t>(current_indices_array);
    const auto* changed_routes = checked_data<std::int64_t>(changed_routes_array);
    const auto* changes = checked_data<std::int64_t>(change_offsets_array);
    const auto* changed_indices = checked_data<std::int64_t>(change_indices_array);
    if (current_offsets[0] != 0
        || current_offsets[route_count] != current_indices_array.size()
        || changes[0] != 0
        || changes[candidate_count * 2] != change_indices_array.size()) {
        throw std::invalid_argument("changed-candidate plan boundary is invalid");
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (current_offsets[route] < 0
            || current_offsets[route] > current_offsets[route + 1]) {
            throw std::invalid_argument("current route offsets must be monotonic");
        }
    }
    for (std::size_t change = 0; change < candidate_count * 2; ++change) {
        if (changes[change] < 0 || changes[change] > changes[change + 1]) {
            throw std::invalid_argument("change offsets must be monotonic");
        }
    }
    std::vector<std::int64_t> plan_offsets{0};
    std::vector<std::int64_t> plan_route_offsets{0};
    std::vector<std::int64_t> plan_route_indices;
    plan_offsets.reserve(candidate_count + 1);
    plan_route_offsets.reserve(candidate_count * route_count + 1);
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        const auto left_route = changed_routes[candidate * 2];
        const auto right_route = changed_routes[candidate * 2 + 1];
        if (left_route < 0 || right_route <= left_route
            || right_route >= static_cast<std::int64_t>(route_count)) {
            throw std::invalid_argument(
                "changed routes must be two ordered current-route indices");
        }
        for (std::size_t route = 0; route < route_count; ++route) {
            const std::int64_t* source = current_indices;
            auto begin = current_offsets[route];
            auto end = current_offsets[route + 1];
            if (static_cast<std::int64_t>(route) == left_route) {
                source = changed_indices;
                begin = changes[candidate * 2];
                end = changes[candidate * 2 + 1];
            } else if (static_cast<std::int64_t>(route) == right_route) {
                source = changed_indices;
                begin = changes[candidate * 2 + 1];
                end = changes[candidate * 2 + 2];
            }
            plan_route_indices.insert(
                plan_route_indices.end(), source + begin, source + end);
            plan_route_offsets.push_back(
                static_cast<std::int64_t>(plan_route_indices.size()));
        }
        plan_offsets.push_back(
            static_cast<std::int64_t>(plan_route_offsets.size() - 1));
    }
    py::array_t<std::int64_t> plan_offsets_output(plan_offsets.size());
    py::array_t<std::int64_t> route_offsets_output(plan_route_offsets.size());
    py::array_t<std::int64_t> route_indices_output(plan_route_indices.size());
    std::copy(
        plan_offsets.begin(), plan_offsets.end(),
        checked_data(plan_offsets_output));
    std::copy(
        plan_route_offsets.begin(), plan_route_offsets.end(),
        checked_data(route_offsets_output));
    std::copy(
        plan_route_indices.begin(), plan_route_indices.end(),
        checked_data(route_indices_output));
    return py::make_tuple(
        std::move(plan_offsets_output), std::move(route_offsets_output),
        std::move(route_indices_output));
}

class PythonFloatSum {
public:
    void add(double value) {
        // Match CPython 3.13's float-specialized sum() exactly.  Stage 5.2
        // compares native and Python raw evidence bit-for-bit, so ordinary
        // left-to-right += accumulation is not semantically equivalent.
        const auto next = total_ + value;
        if (std::fabs(total_) >= std::fabs(value)) {
            compensation_ += (total_ - next) + value;
        } else {
            compensation_ += (value - next) + total_;
        }
        total_ = next;
    }

    [[nodiscard]] double value() const {
        auto result = total_;
        if (compensation_ != 0.0 && std::isfinite(compensation_)) {
            result += compensation_;
        }
        return result;
    }

private:
    double total_ = 0.0;
    double compensation_ = 0.0;
};

py::tuple rank_candidate_plans_v1(
    py::handle plan_offsets,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle route_distance_lower_bounds,
    py::handle current_route_offsets,
    py::handle current_route_indices,
    py::handle lexical_rank,
    py::handle attempted_flags,
    std::int64_t top_k) {
    auto plan_offsets_array = checked_array<std::int64_t>(
        plan_offsets, "plan_offsets", 1);
    auto route_offsets_array = checked_array<std::int64_t>(
        route_offsets, "route_offsets", 1);
    auto route_indices_array = checked_array<std::int64_t>(
        route_indices, "route_indices", 1);
    auto lower_bounds_array = checked_array<double>(
        route_distance_lower_bounds, "route_distance_lower_bounds", 1);
    auto current_offsets_array = checked_array<std::int64_t>(
        current_route_offsets, "current_route_offsets", 1);
    auto current_indices_array = checked_array<std::int64_t>(
        current_route_indices, "current_route_indices", 1);
    auto lexical_array = checked_array<std::int64_t>(
        lexical_rank, "lexical_rank", 1);
    auto attempted_array = checked_array<std::int64_t>(
        attempted_flags, "attempted_flags", 1);
    const auto result = evrptw::native_candidate_plan::rank({
        std::span<const std::int64_t>(
            checked_data<std::int64_t>(plan_offsets_array),
            static_cast<std::size_t>(plan_offsets_array.size())),
        std::span<const std::int64_t>(
            checked_data<std::int64_t>(route_offsets_array),
            static_cast<std::size_t>(route_offsets_array.size())),
        std::span<const std::int64_t>(
            checked_data<std::int64_t>(route_indices_array),
            static_cast<std::size_t>(route_indices_array.size())),
        std::span<const double>(
            checked_data<double>(lower_bounds_array),
            static_cast<std::size_t>(lower_bounds_array.size())),
        std::span<const std::int64_t>(
            checked_data<std::int64_t>(current_offsets_array),
            static_cast<std::size_t>(current_offsets_array.size())),
        std::span<const std::int64_t>(
            checked_data<std::int64_t>(current_indices_array),
            static_cast<std::size_t>(current_indices_array.size())),
        std::span<const std::int64_t>(
            checked_data<std::int64_t>(lexical_array),
            static_cast<std::size_t>(lexical_array.size())),
        std::span<const std::int64_t>(
            checked_data<std::int64_t>(attempted_array),
            static_cast<std::size_t>(attempted_array.size())),
        top_k,
    });
    const auto plan_count = result.ranked.size();
    py::array_t<std::int64_t> ranked_array(result.ranked.size());
    py::array_t<std::int64_t> selected_array(result.selected.size());
    py::array_t<std::int64_t> integer_metrics(
        {static_cast<py::ssize_t>(plan_count), py::ssize_t(2)});
    py::array_t<double> float_metrics(plan_count);
    std::copy(
        result.ranked.begin(), result.ranked.end(), checked_data(ranked_array));
    std::copy(
        result.selected.begin(), result.selected.end(), checked_data(selected_array));
    std::copy(
        result.integer_metrics.begin(), result.integer_metrics.end(),
        checked_data(integer_metrics));
    std::copy(
        result.optimistic_distances.begin(), result.optimistic_distances.end(),
        checked_data(float_metrics));
    return py::make_tuple(
        std::move(ranked_array), std::move(selected_array),
        std::move(integer_metrics), std::move(float_metrics));
}

py::tuple prepare_candidate_plans_v2(
    py::handle plan_offsets,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle expected_customer_indices,
    py::handle node_kind,
    py::handle lexical_rank,
    py::handle complete_customer_indices,
    std::int64_t customer_kind,
    bool allow_partial_customer_coverage) {
    auto plans = checked_array<std::int64_t>(plan_offsets, "plan_offsets", 1);
    auto routes = checked_array<std::int64_t>(route_offsets, "route_offsets", 1);
    auto indices = checked_array<std::int64_t>(route_indices, "route_indices", 1);
    auto expected = checked_array<std::int64_t>(
        expected_customer_indices, "expected_customer_indices", 1);
    auto kinds = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto lexical = checked_array<std::int64_t>(lexical_rank, "lexical_rank", 1);
    auto complete = checked_array<std::int64_t>(
        complete_customer_indices, "complete_customer_indices", 1);
    const auto result = evrptw::native_candidate_plan::prepare({
        {checked_data<std::int64_t>(plans), static_cast<std::size_t>(plans.size())},
        {checked_data<std::int64_t>(routes), static_cast<std::size_t>(routes.size())},
        {checked_data<std::int64_t>(indices), static_cast<std::size_t>(indices.size())},
        {checked_data<std::int64_t>(expected), static_cast<std::size_t>(expected.size())},
        {checked_data<std::int64_t>(kinds), static_cast<std::size_t>(kinds.size())},
        {checked_data<std::int64_t>(lexical), static_cast<std::size_t>(lexical.size())},
        {checked_data<std::int64_t>(complete), static_cast<std::size_t>(complete.size())},
        customer_kind,
        allow_partial_customer_coverage,
    });
    const auto make_array = [](const std::vector<std::int64_t>& values) {
        py::array_t<std::int64_t> array(values.size());
        std::copy(values.begin(), values.end(), checked_data(array));
        return array;
    };
    return py::make_tuple(
        make_array(result.canonical_expected),
        make_array(result.coverage_eligible),
        make_array(result.unique_route_offsets),
        make_array(result.unique_route_indices),
        make_array(result.unique_row_by_route));
}

py::tuple decide_candidate_plans_v2(
    py::handle plan_offsets,
    py::handle coverage_eligible,
    py::handle screening_passed,
    py::handle attempted_flags,
    std::int64_t current_route_count) {
    auto plans = checked_array<std::int64_t>(plan_offsets, "plan_offsets", 1);
    auto coverage = checked_array<std::int64_t>(
        coverage_eligible, "coverage_eligible", 1);
    auto screening = checked_array<std::int64_t>(
        screening_passed, "screening_passed", 1);
    auto attempted = checked_array<std::int64_t>(
        attempted_flags, "attempted_flags", 1);
    const auto result = evrptw::native_candidate_plan::decide({
        {checked_data<std::int64_t>(plans), static_cast<std::size_t>(plans.size())},
        {checked_data<std::int64_t>(coverage), static_cast<std::size_t>(coverage.size())},
        {checked_data<std::int64_t>(screening), static_cast<std::size_t>(screening.size())},
        {checked_data<std::int64_t>(attempted), static_cast<std::size_t>(attempted.size())},
        current_route_count,
    });
    py::array_t<std::int64_t> eligible(result.eligible.size());
    py::array_t<std::int64_t> combined(result.combined_attempted.size());
    std::copy(result.eligible.begin(), result.eligible.end(), checked_data(eligible));
    std::copy(
        result.combined_attempted.begin(), result.combined_attempted.end(),
        checked_data(combined));
    return py::make_tuple(std::move(eligible), std::move(combined));
}

py::array_t<std::int64_t> order_feasible_candidate_plans_v2(
    py::handle plan_offsets,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle objective_integer,
    py::handle objective_float,
    py::handle lexical_rank,
    py::handle feasible_plan_ids) {
    auto plans = checked_array<std::int64_t>(plan_offsets, "plan_offsets", 1);
    auto routes = checked_array<std::int64_t>(route_offsets, "route_offsets", 1);
    auto indices = checked_array<std::int64_t>(route_indices, "route_indices", 1);
    auto integer = checked_array<std::int64_t>(
        objective_integer, "objective_integer", 2);
    auto floating = checked_array<double>(objective_float, "objective_float", 2);
    auto lexical = checked_array<std::int64_t>(lexical_rank, "lexical_rank", 1);
    auto feasible = checked_array<std::int64_t>(
        feasible_plan_ids, "feasible_plan_ids", 1);
    if (integer.shape(1) != 2 || floating.shape(1) != 2) {
        throw std::invalid_argument(
            "feasible-plan objective arrays must have exactly two columns");
    }
    std::vector<double> canonical_floating(
        checked_data<double>(floating),
        checked_data<double>(floating) + floating.size());
    if (plans.size() < 1) {
        throw std::invalid_argument("plan_offsets cannot be empty");
    }
    const auto plan_count = plans.size() - 1;
    const auto* feasible_values = checked_data<std::int64_t>(feasible);
    for (py::ssize_t index = 0; index < feasible.size(); ++index) {
        const auto plan = feasible_values[index];
        if (plan < 0 || plan >= plan_count) {
            throw std::invalid_argument(
                "feasible_plan_ids must identify candidate plans");
        }
        const auto offset = static_cast<std::size_t>(plan) * 2;
        canonical_floating[offset] =
            evrptw::formal_objective::canonical_component(
                canonical_floating[offset]);
        canonical_floating[offset + 1] =
            evrptw::formal_objective::canonical_component(
                canonical_floating[offset + 1]);
    }
    const auto result = evrptw::native_candidate_plan::order_feasible({
        {checked_data<std::int64_t>(plans), static_cast<std::size_t>(plans.size())},
        {checked_data<std::int64_t>(routes), static_cast<std::size_t>(routes.size())},
        {checked_data<std::int64_t>(indices), static_cast<std::size_t>(indices.size())},
        {checked_data<std::int64_t>(integer), static_cast<std::size_t>(integer.size())},
        {canonical_floating.data(), canonical_floating.size()},
        {checked_data<std::int64_t>(lexical), static_cast<std::size_t>(lexical.size())},
        {checked_data<std::int64_t>(feasible), static_cast<std::size_t>(feasible.size())},
    });
    py::array_t<std::int64_t> output(result.size());
    std::copy(result.begin(), result.end(), checked_data(output));
    return output;
}

py::tuple screen_route_batch_transaction_impl(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle candidate_ids,
    py::handle options,
    py::handle incremental,
    py::handle negative_offsets,
    py::handle negative_indices,
    py::handle negative_reason_codes,
    std::int64_t worker_count);

py::tuple exact_charging_batch_numeric(
    py::handle node_kind,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle vehicle,
    py::handle order_offsets,
    py::handle order_indices,
    py::handle deadline_remaining,
    py::handle batch_size);

py::tuple changed_candidate_plan_selection_v1(
    std::int64_t operation,
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle current_route_offsets,
    py::handle current_route_indices,
    py::handle screening_options,
    py::handle negative_offsets,
    py::handle negative_indices,
    py::handle negative_reason_codes,
    py::handle attempted_flags,
    std::int64_t top_k,
    std::int64_t worker_count) {
    if (top_k <= 0 || worker_count <= 0) {
        throw std::invalid_argument(
            "changed candidate-plan selection controls must be positive");
    }
    auto pool = changed_candidate_pool_v1(
        operation, current_route_offsets, current_route_indices);
    auto plans = assemble_changed_candidate_plans_v1(
        current_route_offsets, current_route_indices,
        pool[0], pool[1], pool[2]);
    auto plan_offsets_array = py::cast<py::array_t<std::int64_t>>(plans[0]);
    auto route_offsets_array = py::cast<py::array_t<std::int64_t>>(plans[1]);
    auto route_indices_array = py::cast<py::array_t<std::int64_t>>(plans[2]);
    const auto plan_count = static_cast<std::size_t>(plan_offsets_array.size() - 1);
    const auto route_count = static_cast<std::size_t>(route_offsets_array.size() - 1);
    auto attempted_array = checked_array<std::int64_t>(
        attempted_flags, "attempted_flags", 1);
    if (attempted_array.size() != static_cast<py::ssize_t>(plan_count)) {
        throw std::invalid_argument(
            "changed candidate-plan attempted flags do not align");
    }
    py::array_t<std::int64_t> candidate_ids(route_count);
    py::array_t<double> incremental(
        {static_cast<py::ssize_t>(route_count), py::ssize_t(6)});
    std::fill(checked_data(incremental), checked_data(incremental) + route_count * 6, 0.0);
    for (std::size_t route = 0; route < route_count; ++route) {
        checked_data(candidate_ids)[route] = static_cast<std::int64_t>(route);
    }
    auto screening = screen_route_batch_transaction_impl(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, route_offsets_array, route_indices_array,
        candidate_ids, screening_options, incremental, negative_offsets,
        negative_indices, negative_reason_codes, worker_count);
    auto codes_array = py::cast<py::array_t<std::int64_t>>(screening[3]);
    auto metrics_array = py::cast<py::array_t<double>>(screening[4]);
    const auto* plan_offsets_values = checked_data<std::int64_t>(plan_offsets_array);
    const auto* codes = checked_data<std::int64_t>(codes_array);
    const auto* metrics = checked_data<double>(metrics_array);
    const auto* attempted = checked_data<std::int64_t>(attempted_array);
    py::array_t<std::int64_t> eligible(plan_count);
    py::array_t<std::int64_t> combined_attempted(plan_count);
    py::array_t<double> lower_bounds(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        checked_data(lower_bounds)[route] = metrics[route * 15 + 3];
    }
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        auto valid = true;
        for (auto route = plan_offsets_values[plan];
             route < plan_offsets_values[plan + 1]; ++route) {
            valid = valid && codes[route * 16] == 1;
        }
        checked_data(eligible)[plan] = valid ? 1 : 0;
        checked_data(combined_attempted)[plan] =
            !valid || attempted[plan] != 0 ? 1 : 0;
    }
    auto ranking = rank_candidate_plans_v1(
        plan_offsets_array, route_offsets_array, route_indices_array,
        lower_bounds, current_route_offsets, current_route_indices,
        lexical_rank, combined_attempted, top_k);
    auto ranked_all = py::cast<py::array_t<std::int64_t>>(ranking[0]);
    const auto* ranked_values = checked_data<std::int64_t>(ranked_all);
    std::vector<std::int64_t> rankable;
    rankable.reserve(plan_count);
    for (std::size_t rank = 0; rank < plan_count; ++rank) {
        const auto plan = ranked_values[rank];
        if (checked_data(eligible)[plan] == 1) {
            rankable.push_back(plan);
        }
    }
    py::array_t<std::int64_t> rankable_array(rankable.size());
    std::copy(rankable.begin(), rankable.end(), checked_data(rankable_array));
    std::string evidence = "stage05.2-changed-candidate-plan-selection-v1";
    const auto append_array = [&evidence](const py::array& array) {
        const auto info = array.request();
        evidence.append(
            static_cast<const char*>(info.ptr),
            static_cast<std::size_t>(info.size * info.itemsize));
    };
    append_array(plan_offsets_array);
    append_array(route_offsets_array);
    append_array(route_indices_array);
    append_array(codes_array);
    append_array(metrics_array);
    append_array(eligible);
    append_array(rankable_array);
    append_array(py::cast<py::array>(ranking[1]));
    const auto digest = native_sha256_hex(evidence);
    return py::make_tuple(
        std::move(pool), std::move(plans), std::move(screening),
        std::move(eligible), std::move(rankable_array), ranking[1],
        ranking[2], ranking[3], digest);
}

class NativeSearchEngineV2;

class NativeRouteCacheV2 {
public:
    NativeRouteCacheV2(std::int64_t max_entries, std::int64_t max_memory_bytes)
        : max_entries_(max_entries), max_memory_bytes_(max_memory_bytes) {
        if (max_entries_ <= 0 || max_memory_bytes_ <= 0) {
            throw std::invalid_argument("native route-cache limits must be positive");
        }
    }

    py::tuple lookup_many(py::handle route_offsets, py::handle route_indices) {
        require_no_active_batch("lookup");
        const auto routes = decode_routes(route_offsets, route_indices);
        py::array_t<std::int64_t> hit_flags(routes.size());
        py::array_t<std::uint8_t> hashes(
            {static_cast<py::ssize_t>(routes.size()), py::ssize_t(32)});
        std::fill(
            checked_data(hashes), checked_data(hashes) + routes.size() * 32,
            std::uint8_t{0});
        for (std::size_t index = 0; index < routes.size(); ++index) {
            ++statistics_[0];
            const auto key = route_key(routes[index]);
            const auto newly_seen = !seen_keys_.contains(key);
            if (newly_seen) {
                record_protocol_seen(key);
                const auto [_, inserted] = seen_keys_.insert(key);
                if (!inserted) {
                    throw std::logic_error(
                        "native route-cache seen identity changed during lookup");
                }
            }
            statistics_[10] = static_cast<std::int64_t>(seen_keys_.size());
            const auto found = find_entry(key);
            if (found == entries_.end()) {
                ++statistics_[2];
                checked_data(hit_flags)[index] = 0;
                continue;
            }
            ++statistics_[1];
            checked_data(hit_flags)[index] = 1;
            std::copy(
                found->semantic_hash.begin(), found->semantic_hash.end(),
                checked_data(hashes) + index * 32);
            record_protocol_move(found);
            entries_.splice(entries_.end(), entries_, found);
        }
        return py::make_tuple(
            std::move(hit_flags), std::move(hashes), statistics_array());
    }

    py::tuple begin_store_many_atomic(
        py::handle route_offsets,
        py::handle route_indices,
        py::handle semantic_hashes,
        py::handle entry_bytes) {
        require_no_active_batch("begin_store_many_atomic");
        const auto routes = decode_routes(route_offsets, route_indices);
        auto hashes_array = checked_array<std::uint8_t>(
            semantic_hashes, "semantic_hashes", 2);
        auto bytes_array = checked_array<std::int64_t>(
            entry_bytes, "entry_bytes", 1);
        if (hashes_array.shape(0) != static_cast<py::ssize_t>(routes.size())
            || hashes_array.shape(1) != 32
            || bytes_array.size() != static_cast<py::ssize_t>(routes.size())) {
            throw std::invalid_argument("native route-cache store arrays do not align");
        }
        BatchJournal journal;
        journal.statistics_before = statistics_;
        journal.protocol_operation_count_before = protocol_operation_count();
        journal.active = true;
        journal.inserted_keys.reserve(routes.size());
        journal.evicted_index_nodes.reserve(entries_.size());
        index_.reserve(
            static_cast<std::size_t>(max_entries_) + routes.size());
        std::vector<std::int64_t> statuses(routes.size(), 0);
        std::vector<std::int64_t> eviction_counts(routes.size(), 0);
        const auto* hashes = checked_data<std::uint8_t>(hashes_array);
        const auto* bytes = checked_data<std::int64_t>(bytes_array);
        try {
            for (std::size_t index = 0; index < routes.size(); ++index) {
                if (bytes[index] <= 0) {
                    throw std::invalid_argument(
                        "native route-cache entry bytes must be positive");
                }
                const auto key = route_key(routes[index]);
                const auto existing = find_entry(key);
                if (existing != entries_.end()) {
                    if (!std::equal(
                            existing->semantic_hash.begin(),
                            existing->semantic_hash.end(), hashes + index * 32)) {
                        throw std::runtime_error(
                            "atomic native route-cache semantic conflict");
                    }
                    statuses[index] = 1;
                    continue;
                }
                if (bytes[index] > max_memory_bytes_) {
                    ++statistics_[5];
                    statuses[index] = 2;
                    continue;
                }
                journal.inserted_keys.push_back(key);
                const std::unordered_set<std::string> inserted(
                    journal.inserted_keys.begin(), journal.inserted_keys.end());
                while (!entries_.empty()
                       && (statistics_[6] >= max_entries_
                           || statistics_[8] + bytes[index] > max_memory_bytes_)) {
                    auto evicted = entries_.begin();
                    const auto evicted_key = evicted->key;
                    const auto evicted_bytes = evicted->entry_bytes;
                    const auto retain_eviction = !inserted.contains(evicted_key);
                    if (retain_eviction) {
                        record_protocol_eviction(evicted_key, next_key(evicted));
                    } else {
                        cancel_protocol_insertion(evicted_key);
                    }
                    auto index_node = index_.extract(evicted_key);
                    if (index_node.empty()) {
                        throw std::logic_error(
                            "native route-cache eviction lost its index node");
                    }
                    --statistics_[6];
                    statistics_[8] -= evicted_bytes;
                    ++statistics_[4];
                    ++eviction_counts[index];
                    if (retain_eviction) {
                        journal.evicted_index_nodes.push_back(
                            std::move(index_node));
                        journal.evicted_entries.splice(
                            journal.evicted_entries.end(), entries_, evicted);
                    } else {
                        entries_.erase(evicted);
                    }
                }
                Entry stored;
                stored.key = key;
                stored.route = routes[index];
                std::copy(
                    hashes + index * 32, hashes + (index + 1) * 32,
                    stored.semantic_hash.begin());
                stored.entry_bytes = bytes[index];
                entries_.push_back(std::move(stored));
                auto stored_entry = std::prev(entries_.end());
                try {
                    const auto [_, inserted_entry] = index_.emplace(
                        stored_entry->key, stored_entry);
                    if (!inserted_entry) {
                        throw std::logic_error(
                            "native route-cache insertion duplicated an index key");
                    }
                } catch (...) {
                    entries_.erase(stored_entry);
                    throw;
                }
                record_protocol_insertion(stored_entry->key);
                ++statistics_[3];
                ++statistics_[6];
                statistics_[8] += bytes[index];
                statistics_[7] = std::max(statistics_[7], statistics_[6]);
                statistics_[9] = std::max(statistics_[9], statistics_[8]);
            }
        py::array_t<std::int64_t> status_array(statuses.size());
        py::array_t<std::int64_t> eviction_array(eviction_counts.size());
        std::copy(statuses.begin(), statuses.end(), checked_data(status_array));
        std::copy(
            eviction_counts.begin(), eviction_counts.end(),
            checked_data(eviction_array));
        auto result = py::make_tuple(
            std::move(status_array), std::move(eviction_array), statistics_array());
        active_batch_ = std::move(journal);
        return result;
        } catch (...) {
            rollback_journal(journal);
            throw;
        }
    }

    py::tuple begin_store_exact_many_atomic(
        py::handle route_offsets,
        py::handle route_indices,
        py::handle path_offsets,
        py::handle path_indices,
        py::handle result_statuses,
        py::handle reason_codes,
        py::handle result_metrics,
        py::handle label_counters,
        py::handle semantic_hashes,
        py::handle entry_bytes) {
        const auto routes = decode_routes(route_offsets, route_indices);
        auto path_offsets_array = checked_array<std::int64_t>(
            path_offsets, "path_offsets", 1);
        auto path_indices_array = checked_array<std::int64_t>(
            path_indices, "path_indices", 1);
        auto statuses_array = checked_array<std::int64_t>(
            result_statuses, "result_statuses", 1);
        auto reasons_array = checked_array<std::int64_t>(
            reason_codes, "reason_codes", 1);
        auto metrics_array = checked_array<double>(
            result_metrics, "result_metrics", 2);
        auto labels_array = checked_array<std::int64_t>(
            label_counters, "label_counters", 2);
        const auto count = static_cast<py::ssize_t>(routes.size());
        if (path_offsets_array.size() != count + 1
            || statuses_array.size() != count || reasons_array.size() != count
            || metrics_array.shape(0) != count || metrics_array.shape(1) != 4
            || labels_array.shape(0) != count || labels_array.shape(1) != 3) {
            throw std::invalid_argument(
                "native exact route-cache payload arrays do not align");
        }
        const auto* path_boundaries = checked_data<std::int64_t>(path_offsets_array);
        if (path_boundaries[0] != 0
            || path_boundaries[count] != path_indices_array.size()) {
            throw std::invalid_argument(
                "native exact route-cache path boundary is invalid");
        }
        for (py::ssize_t index = 0; index < count; ++index) {
            if (path_boundaries[index] < 0
                || path_boundaries[index] > path_boundaries[index + 1]) {
                throw std::invalid_argument(
                    "native exact route-cache path offsets must be monotonic");
            }
        }
        auto summary = begin_store_many_atomic(
            route_offsets, route_indices, semantic_hashes, entry_bytes);
        const auto* paths = checked_data<std::int64_t>(path_indices_array);
        const auto* statuses = checked_data<std::int64_t>(statuses_array);
        const auto* reasons = checked_data<std::int64_t>(reasons_array);
        const auto* metrics = checked_data<double>(metrics_array);
        const auto* labels = checked_data<std::int64_t>(labels_array);
        const std::unordered_set<std::string> inserted(
            active_batch_->inserted_keys.begin(), active_batch_->inserted_keys.end());
        try {
            for (std::size_t index = 0; index < routes.size(); ++index) {
                if (statuses[index] < 0 || statuses[index] > 2 || reasons[index] < 0) {
                    throw std::invalid_argument(
                        "native exact route-cache status/reason is invalid");
                }
                const auto key = route_key(routes[index]);
                const auto entry = find_entry(key);
                if (entry == entries_.end()) {
                    // Oversize entries are deliberately not cached.
                    continue;
                }
                ExactPayload payload;
                payload.path.assign(
                    paths + path_boundaries[index],
                    paths + path_boundaries[index + 1]);
                payload.status = statuses[index];
                payload.reason = reasons[index];
                std::copy(
                    metrics + index * 4, metrics + (index + 1) * 4,
                    payload.metrics.begin());
                std::copy(
                    labels + index * 3, labels + (index + 1) * 3,
                    payload.label_counters.begin());
                if (entry->exact_payload.has_value()) {
                    if (entry->exact_payload != payload) {
                        throw std::runtime_error(
                            "atomic native route-cache exact payload conflict");
                    }
                } else if (inserted.contains(key)) {
                    entry->exact_payload = std::move(payload);
                } else {
                    throw std::runtime_error(
                        "existing native route-cache entry lacks exact payload");
                }
            }
        } catch (...) {
            rollback_store_batch();
            throw;
        }
        return summary;
    }

    py::tuple lookup_exact_many(
        py::handle route_offsets,
        py::handle route_indices) {
        require_no_active_batch("lookup_exact_many");
        const auto routes = decode_routes(route_offsets, route_indices);
        py::array_t<std::int64_t> hit_flags(routes.size());
        py::array_t<std::int64_t> statuses(routes.size());
        py::array_t<std::int64_t> reasons(routes.size());
        py::array_t<double> metrics(
            {static_cast<py::ssize_t>(routes.size()), py::ssize_t(4)});
        py::array_t<std::int64_t> labels(
            {static_cast<py::ssize_t>(routes.size()), py::ssize_t(3)});
        py::array_t<std::uint8_t> hashes(
            {static_cast<py::ssize_t>(routes.size()), py::ssize_t(32)});
        std::fill(checked_data(metrics), checked_data(metrics) + routes.size() * 4, 0.0);
        std::fill(checked_data(labels), checked_data(labels) + routes.size() * 3, 0);
        std::fill(checked_data(hashes), checked_data(hashes) + routes.size() * 32, 0);
        std::vector<std::int64_t> path_offsets{0};
        std::vector<std::int64_t> path_indices;
        for (std::size_t index = 0; index < routes.size(); ++index) {
            ++statistics_[0];
            const auto key = route_key(routes[index]);
            const auto newly_seen = !seen_keys_.contains(key);
            if (newly_seen) {
                record_protocol_seen(key);
                const auto [_, inserted] = seen_keys_.insert(key);
                if (!inserted) {
                    throw std::logic_error(
                        "native route-cache seen identity changed during exact lookup");
                }
            }
            statistics_[10] = static_cast<std::int64_t>(seen_keys_.size());
            const auto found = find_entry(key);
            if (found == entries_.end()) {
                ++statistics_[2];
                checked_data(hit_flags)[index] = 0;
                checked_data(statuses)[index] = -1;
                checked_data(reasons)[index] = -1;
                path_offsets.push_back(static_cast<std::int64_t>(path_indices.size()));
                continue;
            }
            if (!found->exact_payload.has_value()) {
                throw std::runtime_error(
                    "native route-cache hit lacks typed exact payload");
            }
            ++statistics_[1];
            checked_data(hit_flags)[index] = 1;
            const auto& payload = *found->exact_payload;
            checked_data(statuses)[index] = payload.status;
            checked_data(reasons)[index] = payload.reason;
            std::copy(
                payload.metrics.begin(), payload.metrics.end(),
                checked_data(metrics) + index * 4);
            std::copy(
                payload.label_counters.begin(), payload.label_counters.end(),
                checked_data(labels) + index * 3);
            std::copy(
                found->semantic_hash.begin(), found->semantic_hash.end(),
                checked_data(hashes) + index * 32);
            path_indices.insert(
                path_indices.end(), payload.path.begin(), payload.path.end());
            path_offsets.push_back(static_cast<std::int64_t>(path_indices.size()));
            record_protocol_move(found);
            entries_.splice(entries_.end(), entries_, found);
        }
        py::array_t<std::int64_t> path_offsets_array(path_offsets.size());
        py::array_t<std::int64_t> path_indices_array(path_indices.size());
        std::copy(
            path_offsets.begin(), path_offsets.end(), checked_data(path_offsets_array));
        std::copy(
            path_indices.begin(), path_indices.end(), checked_data(path_indices_array));
        return py::make_tuple(
            std::move(hit_flags), std::move(path_offsets_array),
            std::move(path_indices_array), std::move(statuses), std::move(reasons),
            std::move(metrics), std::move(labels), std::move(hashes),
            statistics_array());
    }

    void begin_protocol_transaction() {
        require_no_active_batch("begin_protocol_transaction");
        if (protocol_snapshot_.has_value()) {
            throw std::runtime_error(
                "native route-cache protocol transaction is already active");
        }
        ProtocolSnapshot snapshot;
        snapshot.statistics = statistics_;
        protocol_snapshot_ = std::move(snapshot);
    }

    py::array_t<std::int64_t> commit_protocol_transaction() {
        require_no_active_batch("commit_protocol_transaction");
        require_protocol_snapshot("commit_protocol_transaction");
        protocol_snapshot_.reset();
        return statistics_array();
    }

    py::array_t<std::int64_t> rollback_protocol_transaction() {
        require_no_active_batch("rollback_protocol_transaction");
        require_protocol_snapshot("rollback_protocol_transaction");
        prepare_protocol_rollback();
        rollback_protocol_transaction_noexcept();
        return statistics_array();
    }

    void inject_protocol_journal_failure_once() {
        require_protocol_snapshot("inject_protocol_journal_failure_once");
        protocol_journal_failure_injection_ = true;
    }

    py::array_t<std::int64_t> commit_store_batch() {
        require_active_batch("commit_store_batch");
        prepare_store_commit();
        commit_store_batch_noexcept();
        return statistics_array();
    }

    py::array_t<std::int64_t> rollback_store_batch() {
        require_active_batch("rollback_store_batch");
        auto journal = std::move(*active_batch_);
        active_batch_.reset();
        rollback_journal(journal);
        return statistics_array();
    }

    py::tuple snapshot() const {
        std::vector<std::int64_t> offsets{0};
        std::vector<std::int64_t> indices;
        py::array_t<std::uint8_t> hashes(
            {static_cast<py::ssize_t>(entries_.size()), py::ssize_t(32)});
        py::array_t<std::int64_t> entry_bytes(entries_.size());
        std::size_t ordinal = 0;
        for (const auto& entry : entries_) {
            indices.insert(indices.end(), entry.route.begin(), entry.route.end());
            offsets.push_back(static_cast<std::int64_t>(indices.size()));
            std::copy(
                entry.semantic_hash.begin(), entry.semantic_hash.end(),
                checked_data(hashes) + ordinal * 32);
            checked_data(entry_bytes)[ordinal] = entry.entry_bytes;
            ++ordinal;
        }
        py::array_t<std::int64_t> offsets_array(offsets.size());
        py::array_t<std::int64_t> indices_array(indices.size());
        std::copy(offsets.begin(), offsets.end(), checked_data(offsets_array));
        std::copy(indices.begin(), indices.end(), checked_data(indices_array));
        return py::make_tuple(
            std::move(offsets_array), std::move(indices_array), std::move(hashes),
            std::move(entry_bytes), statistics_array());
    }

private:
    friend class NativeSearchEngineV2;
    struct ExactPayload {
        std::vector<std::int64_t> path;
        std::int64_t status = -1;
        std::int64_t reason = -1;
        std::array<double, 4> metrics{};
        std::array<std::int64_t, 3> label_counters{};

        bool operator==(const ExactPayload&) const = default;
    };
    struct Entry {
        std::string key;
        std::vector<std::int64_t> route;
        std::array<std::uint8_t, 32> semantic_hash{};
        std::int64_t entry_bytes = 0;
        std::optional<ExactPayload> exact_payload;
    };
    struct BatchJournal {
        std::vector<std::string> inserted_keys;
        std::list<Entry> evicted_entries;
        std::vector<
            std::unordered_map<
                std::string,
                std::list<Entry>::iterator>::node_type> evicted_index_nodes;
        std::array<std::int64_t, 11> statistics_before{};
        std::size_t protocol_operation_count_before = 0;
        bool active = false;
    };
    enum class ProtocolOperationKind {
        seen,
        move,
        insertion,
        eviction,
    };
    struct ProtocolOperation {
        ProtocolOperationKind kind;
        std::string key;
        std::optional<std::string> next_key;
    };
    struct ProtocolSnapshot {
        std::array<std::int64_t, 11> statistics{};
        std::vector<ProtocolOperation> operations;
        std::list<Entry> retained_entries;
        std::vector<
            std::unordered_map<
                std::string,
                std::list<Entry>::iterator>::node_type> retained_index_nodes;
    };

    std::int64_t max_entries_;
    std::int64_t max_memory_bytes_;
    std::list<Entry> entries_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    std::unordered_set<std::string> seen_keys_;
    std::array<std::int64_t, 11> statistics_{};
    std::optional<BatchJournal> active_batch_;
    std::optional<ProtocolSnapshot> protocol_snapshot_;
    bool protocol_journal_failure_injection_ = false;

    static std::string route_key(const std::vector<std::int64_t>& route) {
        std::string key;
        key.resize((route.size() + 1) * sizeof(std::int64_t));
        const auto length = static_cast<std::int64_t>(route.size());
        std::memcpy(key.data(), &length, sizeof(length));
        if (!route.empty()) {
            std::memcpy(
                key.data() + sizeof(length), route.data(),
                route.size() * sizeof(std::int64_t));
        }
        return key;
    }

    static std::vector<std::vector<std::int64_t>> decode_routes(
        py::handle route_offsets,
        py::handle route_indices) {
        auto offsets_array = checked_array<std::int64_t>(
            route_offsets, "route_offsets", 1);
        auto indices_array = checked_array<std::int64_t>(
            route_indices, "route_indices", 1);
        if (offsets_array.size() < 1) {
            throw std::invalid_argument("native route-cache offsets cannot be empty");
        }
        const auto count = static_cast<std::size_t>(offsets_array.size() - 1);
        const auto* offsets = checked_data<std::int64_t>(offsets_array);
        const auto* indices = checked_data<std::int64_t>(indices_array);
        if (offsets[0] != 0 || offsets[count] != indices_array.size()) {
            throw std::invalid_argument("native route-cache offsets boundary is invalid");
        }
        std::vector<std::vector<std::int64_t>> routes;
        routes.reserve(count);
        for (std::size_t route = 0; route < count; ++route) {
            if (offsets[route] < 0 || offsets[route] > offsets[route + 1]) {
                throw std::invalid_argument(
                    "native route-cache offsets must be monotonic");
            }
            routes.emplace_back(
                indices + offsets[route], indices + offsets[route + 1]);
        }
        return routes;
    }

    std::list<Entry>::iterator find_entry(const std::string& key) {
        const auto found = index_.find(key);
        return found == index_.end() ? entries_.end() : found->second;
    }

    [[nodiscard]] std::optional<std::string> next_key(
        std::list<Entry>::iterator current) const {
        const auto following = std::next(current);
        return following == entries_.end()
            ? std::nullopt
            : std::optional<std::string>(following->key);
    }

    [[nodiscard]] std::size_t protocol_operation_count() const {
        return protocol_snapshot_.has_value()
            ? protocol_snapshot_->operations.size()
            : 0;
    }

    void record_protocol_seen(const std::string& key) {
        if (protocol_snapshot_.has_value()) {
            if (protocol_journal_failure_injection_) {
                protocol_journal_failure_injection_ = false;
                throw std::runtime_error(
                    "injected native route-cache protocol journal failure");
            }
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::seen, key, std::nullopt});
        }
    }

    void record_protocol_move(std::list<Entry>::iterator entry) {
        if (protocol_snapshot_.has_value() && std::next(entry) != entries_.end()) {
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::move,
                entry->key,
                next_key(entry),
            });
        }
    }

    void record_protocol_insertion(const std::string& key) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::insertion, key, std::nullopt});
        }
    }

    void record_protocol_eviction(
        const std::string& key,
        std::optional<std::string> following_key) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::eviction,
                key,
                std::move(following_key),
            });
        }
    }

    void cancel_protocol_insertion(const std::string& key) {
        if (!protocol_snapshot_.has_value()) {
            return;
        }
        auto& operations = protocol_snapshot_->operations;
        const auto found = std::find_if(
            operations.rbegin(), operations.rend(),
            [&](const ProtocolOperation& operation) {
                return operation.kind == ProtocolOperationKind::insertion
                    && operation.key == key;
            });
        if (found == operations.rend()) {
            throw std::logic_error(
                "native route-cache transient eviction lost its insertion journal");
        }
        operations.erase(std::next(found).base());
    }

    void prepare_protocol_rollback() const {
        require_no_active_batch("prepare_protocol_rollback");
        require_protocol_snapshot("prepare_protocol_rollback");
        for (const auto& operation : protocol_snapshot_->operations) {
            if (operation.kind != ProtocolOperationKind::eviction) {
                continue;
            }
            const auto retained_entry = std::find_if(
                protocol_snapshot_->retained_entries.begin(),
                protocol_snapshot_->retained_entries.end(),
                [&](const Entry& entry) { return entry.key == operation.key; });
            const auto retained_node = std::find_if(
                protocol_snapshot_->retained_index_nodes.begin(),
                protocol_snapshot_->retained_index_nodes.end(),
                [&](const auto& node) {
                    return !node.empty() && node.key() == operation.key;
                });
            if (retained_entry == protocol_snapshot_->retained_entries.end()
                || retained_node == protocol_snapshot_->retained_index_nodes.end()) {
                throw std::logic_error(
                    "native route-cache protocol rollback journal is incomplete");
            }
        }
    }

    void rollback_protocol_operations_noexcept(
        ProtocolSnapshot& snapshot) noexcept {
        for (auto operation = snapshot.operations.rbegin();
             operation != snapshot.operations.rend(); ++operation) {
            if (operation->kind == ProtocolOperationKind::seen) {
                seen_keys_.erase(operation->key);
                continue;
            }
            if (operation->kind == ProtocolOperationKind::insertion) {
                const auto found = find_entry(operation->key);
                if (found == entries_.end()) {
                    std::terminate();
                }
                index_.erase(operation->key);
                entries_.erase(found);
                continue;
            }
            if (operation->kind == ProtocolOperationKind::eviction) {
                if (index_.contains(operation->key)) {
                    std::terminate();
                }
                auto position = entries_.end();
                if (operation->next_key.has_value()) {
                    const auto following = index_.find(*operation->next_key);
                    if (following == index_.end()) {
                        std::terminate();
                    }
                    position = following->second;
                }
                const auto retained_entry = std::find_if(
                    snapshot.retained_entries.begin(),
                    snapshot.retained_entries.end(),
                    [&](const Entry& entry) { return entry.key == operation->key; });
                const auto retained_node = std::find_if(
                    snapshot.retained_index_nodes.begin(),
                    snapshot.retained_index_nodes.end(),
                    [&](const auto& node) {
                        return !node.empty() && node.key() == operation->key;
                    });
                if (retained_entry == snapshot.retained_entries.end()
                    || retained_node == snapshot.retained_index_nodes.end()) {
                    std::terminate();
                }
                entries_.splice(position, snapshot.retained_entries, retained_entry);
                const auto restored = index_.insert(std::move(*retained_node));
                if (!restored.inserted) {
                    std::terminate();
                }
                continue;
            }
            const auto found = find_entry(operation->key);
            if (found == entries_.end()) {
                std::terminate();
            }
            auto position = entries_.end();
            if (operation->next_key.has_value()) {
                const auto following = index_.find(*operation->next_key);
                if (following == index_.end()) {
                    std::terminate();
                }
                position = following->second;
            }
            entries_.splice(position, entries_, found);
        }
    }

    void require_no_active_batch(const char* operation) const {
        if (active_batch_.has_value()) {
            throw std::runtime_error(
                std::string("native route-cache ") + operation
                + " is forbidden while a write batch is active");
        }
    }

    void require_active_batch(const char* operation) const {
        if (!active_batch_.has_value() || !active_batch_->active) {
            throw std::runtime_error(
                std::string("native route-cache ") + operation
                + " requires an active write batch");
        }
    }

    void require_protocol_snapshot(const char* operation) const {
        if (!protocol_snapshot_.has_value()) {
            throw std::runtime_error(
                std::string("native route-cache ") + operation
                + " requires an active protocol transaction");
        }
    }

    void rollback_journal(BatchJournal& journal) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.resize(
                journal.protocol_operation_count_before);
        }
        for (const auto& key : journal.inserted_keys) {
            const auto found = index_.find(key);
            if (found == index_.end()) {
                continue;
            }
            entries_.erase(found->second);
            index_.erase(found);
        }
        entries_.splice(entries_.begin(), journal.evicted_entries);
        for (auto& node : journal.evicted_index_nodes) {
            const auto restored = index_.insert(std::move(node));
            if (!restored.inserted) {
                throw std::logic_error(
                    "native route-cache rollback could not restore an index node");
            }
        }
        statistics_ = journal.statistics_before;
    }

    py::array_t<std::int64_t> statistics_array() const {
        py::array_t<std::int64_t> output(statistics_.size());
        std::copy(statistics_.begin(), statistics_.end(), checked_data(output));
        return output;
    }

    void commit_protocol_transaction_noexcept() noexcept {
        protocol_snapshot_.reset();
    }

    void rollback_protocol_transaction_noexcept() noexcept {
        rollback_protocol_operations_noexcept(*protocol_snapshot_);
        statistics_ = protocol_snapshot_->statistics;
        protocol_snapshot_.reset();
    }

    void prepare_store_commit() {
        require_active_batch("prepare_store_commit");
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->retained_index_nodes.reserve(
                protocol_snapshot_->retained_index_nodes.size()
                + active_batch_->evicted_index_nodes.size());
        }
    }

    void commit_store_batch_noexcept() noexcept {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->retained_entries.splice(
                protocol_snapshot_->retained_entries.end(),
                active_batch_->evicted_entries);
            for (auto& node : active_batch_->evicted_index_nodes) {
                protocol_snapshot_->retained_index_nodes.push_back(std::move(node));
            }
        }
        active_batch_.reset();
    }

    void prepare_protocol_commit() const {
        require_no_active_batch("prepare_protocol_commit");
        require_protocol_snapshot("prepare_protocol_commit");
    }

};

class NativeNegativeRouteCacheV2 {
public:
    explicit NativeNegativeRouteCacheV2(std::int64_t capacity)
        : capacity_(capacity) {
        if (capacity_ <= 0) {
            throw std::invalid_argument(
                "native negative route-cache capacity must be positive");
        }
    }

    py::tuple lookup_many(py::handle route_offsets, py::handle route_indices) const {
        const auto routes = decode_routes(route_offsets, route_indices);
        py::array_t<std::int64_t> hit_flags(routes.size());
        py::array_t<std::int64_t> reasons(routes.size());
        for (std::size_t index = 0; index < routes.size(); ++index) {
            const auto found = find_entry(route_key(routes[index]));
            checked_data(hit_flags)[index] = found == entries_.end() ? 0 : 1;
            checked_data(reasons)[index] = found == entries_.end() ? 0 : found->reason;
        }
        return py::make_tuple(
            std::move(hit_flags), std::move(reasons), statistics_array());
    }

    py::array_t<std::int64_t> begin_store_many_atomic(
        py::handle route_offsets,
        py::handle route_indices,
        py::handle reason_codes) {
        if (active_batch_.has_value()) {
            throw std::runtime_error(
                "native negative route-cache already has an active batch");
        }
        const auto routes = decode_routes(route_offsets, route_indices);
        auto reasons_array = checked_array<std::int64_t>(
            reason_codes, "reason_codes", 1);
        if (reasons_array.size() != static_cast<py::ssize_t>(routes.size())) {
            throw std::invalid_argument(
                "native negative route-cache reasons do not align");
        }
        const auto* reasons = checked_data<std::int64_t>(reasons_array);
        std::unordered_set<std::string> input_keys;
        std::vector<Entry> input;
        std::vector<Entry> additions;
        input.reserve(routes.size());
        additions.reserve(routes.size());
        for (std::size_t index = 0; index < routes.size(); ++index) {
            if (reasons[index] <= 0) {
                throw std::invalid_argument(
                    "native negative route-cache reason must be positive");
            }
            Entry entry{
                route_key(routes[index]), routes[index], reasons[index],
            };
            if (!input_keys.insert(entry.key).second) {
                throw std::invalid_argument(
                    "native negative route-cache input routes must be unique");
            }
            const auto existing = find_entry(entry.key);
            if (existing != entries_.end() && existing->reason != entry.reason) {
                throw std::runtime_error(
                    "candidate negative cache reason changed during commit");
            }
            input.push_back(entry);
            if (existing == entries_.end()) {
                additions.push_back(std::move(entry));
            }
        }
        BatchJournal journal;
        journal.added_keys.reserve(additions.size());
        for (const auto& entry : additions) {
            journal.added_keys.push_back(entry.key);
        }
        journal.rollover = entries_.size() + additions.size()
            > static_cast<std::size_t>(capacity_);
        if (journal.rollover) {
            if (input.size() > static_cast<std::size_t>(capacity_)) {
                throw std::runtime_error(
                    "one candidate transaction exceeds the negative route-cache capacity");
            }
            journal.previous_entries = entries_;
            const std::unordered_set<std::string> replacement_keys(
                input_keys.begin(), input_keys.end());
            journal.evicted_count = static_cast<std::int64_t>(std::count_if(
                entries_.begin(), entries_.end(),
                [&](const Entry& entry) {
                    return !replacement_keys.contains(entry.key);
                }));
            entries_ = std::move(input);
        } else {
            entries_.insert(entries_.end(), additions.begin(), additions.end());
        }
        active_batch_ = std::move(journal);
        return batch_summary();
    }

    py::array_t<std::int64_t> commit_store_batch() {
        require_active_batch("commit_store_batch");
        statistics_[0] += static_cast<std::int64_t>(
            active_batch_->added_keys.size());
        statistics_[1] += active_batch_->evicted_count;
        statistics_[2] += active_batch_->rollover ? 1 : 0;
        statistics_[4] = std::max(
            statistics_[4], static_cast<std::int64_t>(entries_.size()));
        statistics_[3] = static_cast<std::int64_t>(entries_.size());
        active_batch_.reset();
        return statistics_array();
    }

    py::array_t<std::int64_t> rollback_store_batch() {
        require_active_batch("rollback_store_batch");
        if (active_batch_->rollover) {
            entries_ = std::move(active_batch_->previous_entries);
        } else {
            const std::unordered_set<std::string> added(
                active_batch_->added_keys.begin(), active_batch_->added_keys.end());
            std::erase_if(
                entries_, [&](const Entry& entry) { return added.contains(entry.key); });
        }
        statistics_[3] = static_cast<std::int64_t>(entries_.size());
        active_batch_.reset();
        return statistics_array();
    }

    py::tuple snapshot() const {
        std::vector<std::int64_t> offsets{0};
        std::vector<std::int64_t> indices;
        py::array_t<std::int64_t> reasons(entries_.size());
        for (std::size_t index = 0; index < entries_.size(); ++index) {
            indices.insert(
                indices.end(), entries_[index].route.begin(), entries_[index].route.end());
            offsets.push_back(static_cast<std::int64_t>(indices.size()));
            checked_data(reasons)[index] = entries_[index].reason;
        }
        py::array_t<std::int64_t> offsets_array(offsets.size());
        py::array_t<std::int64_t> indices_array(indices.size());
        std::copy(offsets.begin(), offsets.end(), checked_data(offsets_array));
        std::copy(indices.begin(), indices.end(), checked_data(indices_array));
        return py::make_tuple(
            std::move(offsets_array), std::move(indices_array), std::move(reasons),
            statistics_array());
    }

private:
    friend class NativeSearchEngineV2;
    struct Entry {
        std::string key;
        std::vector<std::int64_t> route;
        std::int64_t reason;
    };
    struct BatchJournal {
        std::vector<std::string> added_keys;
        std::vector<Entry> previous_entries;
        std::int64_t evicted_count = 0;
        bool rollover = false;
    };

    std::int64_t capacity_;
    std::vector<Entry> entries_;
    // stores, evictions, rollovers, current entries, peak entries
    std::array<std::int64_t, 5> statistics_{};
    std::optional<BatchJournal> active_batch_;

    static std::string route_key(const std::vector<std::int64_t>& route) {
        std::string key;
        key.resize((route.size() + 1) * sizeof(std::int64_t));
        const auto length = static_cast<std::int64_t>(route.size());
        std::memcpy(key.data(), &length, sizeof(length));
        if (!route.empty()) {
            std::memcpy(
                key.data() + sizeof(length), route.data(),
                route.size() * sizeof(std::int64_t));
        }
        return key;
    }

    static std::vector<std::vector<std::int64_t>> decode_routes(
        py::handle route_offsets,
        py::handle route_indices) {
        auto offsets_array = checked_array<std::int64_t>(
            route_offsets, "route_offsets", 1);
        auto indices_array = checked_array<std::int64_t>(
            route_indices, "route_indices", 1);
        if (offsets_array.size() < 1) {
            throw std::invalid_argument(
                "native negative route-cache offsets cannot be empty");
        }
        const auto count = static_cast<std::size_t>(offsets_array.size() - 1);
        const auto* offsets = checked_data<std::int64_t>(offsets_array);
        const auto* indices = checked_data<std::int64_t>(indices_array);
        if (offsets[0] != 0 || offsets[count] != indices_array.size()) {
            throw std::invalid_argument(
                "native negative route-cache offsets boundary is invalid");
        }
        std::vector<std::vector<std::int64_t>> routes;
        routes.reserve(count);
        for (std::size_t route = 0; route < count; ++route) {
            if (offsets[route] < 0 || offsets[route] > offsets[route + 1]) {
                throw std::invalid_argument(
                    "native negative route-cache offsets must be monotonic");
            }
            routes.emplace_back(
                indices + offsets[route], indices + offsets[route + 1]);
        }
        return routes;
    }

    std::vector<Entry>::const_iterator find_entry(const std::string& key) const {
        return std::find_if(
            entries_.begin(), entries_.end(),
            [&](const Entry& entry) { return entry.key == key; });
    }

    void require_active_batch(const char* operation) const {
        if (!active_batch_.has_value()) {
            throw std::runtime_error(
                std::string("native negative route-cache ") + operation
                + " requires an active batch");
        }
    }

    py::array_t<std::int64_t> batch_summary() const {
        py::array_t<std::int64_t> output(3);
        checked_data(output)[0] = static_cast<std::int64_t>(
            active_batch_->added_keys.size());
        checked_data(output)[1] = active_batch_->evicted_count;
        checked_data(output)[2] = active_batch_->rollover ? 1 : 0;
        return output;
    }

    py::array_t<std::int64_t> statistics_array() const {
        auto output_values = statistics_;
        output_values[3] = static_cast<std::int64_t>(entries_.size());
        py::array_t<std::int64_t> output(output_values.size());
        std::copy(output_values.begin(), output_values.end(), checked_data(output));
        return output;
    }

    py::array_t<std::int64_t> projected_statistics_array() const {
        auto projected = statistics_;
        if (active_batch_.has_value()) {
            projected[0] += static_cast<std::int64_t>(
                active_batch_->added_keys.size());
            projected[1] += active_batch_->evicted_count;
            projected[2] += active_batch_->rollover ? 1 : 0;
            projected[3] = static_cast<std::int64_t>(entries_.size());
            projected[4] = std::max(
                projected[4], static_cast<std::int64_t>(entries_.size()));
        }
        py::array_t<std::int64_t> output(projected.size());
        std::copy(projected.begin(), projected.end(), checked_data(output));
        return output;
    }

    void commit_store_batch_noexcept() noexcept {
        statistics_[0] += static_cast<std::int64_t>(
            active_batch_->added_keys.size());
        statistics_[1] += active_batch_->evicted_count;
        statistics_[2] += active_batch_->rollover ? 1 : 0;
        statistics_[3] = static_cast<std::int64_t>(entries_.size());
        statistics_[4] = std::max(
            statistics_[4], static_cast<std::int64_t>(entries_.size()));
        active_batch_.reset();
    }

    void rollback_store_batch_noexcept() noexcept {
        if (active_batch_->rollover) {
            entries_ = std::move(active_batch_->previous_entries);
        } else {
            const auto& added = active_batch_->added_keys;
            std::erase_if(entries_, [&](const Entry& entry) {
                return std::find(added.begin(), added.end(), entry.key)
                    != added.end();
            });
        }
        statistics_[3] = static_cast<std::int64_t>(entries_.size());
        active_batch_.reset();
    }

    void prepare_store_commit() const {
        require_active_batch("prepare_store_commit");
    }
};

class NativeBudgetStateV2 {
public:
    NativeBudgetStateV2(std::int64_t exact_budget, std::int64_t round_budget)
        : exact_budget_(exact_budget), round_budget_(round_budget) {
        if (exact_budget_ == 0 || exact_budget_ < -1 || round_budget_ <= 0) {
            throw std::invalid_argument("native budget limits are invalid");
        }
    }

    py::array_t<std::int64_t> begin_round(
        std::int64_t lane_id, std::int64_t iteration) {
        if (lane_id < 0 || iteration < 0) {
            throw std::invalid_argument("native round identity must be non-negative");
        }
        if (round_active_ && lane_id_ == lane_id && iteration_ == iteration) {
            return state();
        }
        round_active_ = true;
        lane_id_ = lane_id;
        iteration_ = iteration;
        round_used_ = 0;
        return state();
    }

    py::array_t<std::int64_t> begin_shared_iteration_round(
        std::int64_t semantic_lane_id, std::int64_t iteration) {
        if (semantic_lane_id < 0 || iteration < 0) {
            throw std::invalid_argument("native round identity must be non-negative");
        }
        if (round_active_ && iteration_ == iteration) {
            // Candidate Control shares one per-iteration budget across all
            // semantic ALNS lanes.  Preserve the lane that owns the current
            // transaction for replay without resetting the shared counter.
            lane_id_ = semantic_lane_id;
            return state();
        }
        return begin_round(semantic_lane_id, iteration);
    }

    py::array_t<std::int64_t> finish_round() {
        round_active_ = false;
        lane_id_ = -1;
        iteration_ = -1;
        round_used_ = 0;
        return state();
    }

    py::array_t<std::int64_t> reserve_round(
        std::int64_t requested, bool atomic) {
        if (requested <= 0) {
            py::array_t<std::int64_t> output(3);
            checked_data(output)[0] = requested;
            checked_data(output)[1] = 0;
            checked_data(output)[2] = round_remaining();
            return output;
        }
        std::int64_t granted = requested;
        if (round_active_) {
            granted = atomic
                ? (requested <= round_remaining() ? requested : 0)
                : std::min(requested, round_remaining());
        }
        round_used_ += granted;
        py::array_t<std::int64_t> output(3);
        checked_data(output)[0] = requested;
        checked_data(output)[1] = granted;
        checked_data(output)[2] = round_remaining();
        return output;
    }

    py::array_t<std::int64_t> reserve_exact(std::int64_t requested) {
        if (requested <= 0) {
            throw std::invalid_argument("exact-call reservation must be positive");
        }
        const auto granted = exact_budget_ < 0
            ? requested
            : std::min(requested, std::max<std::int64_t>(0, exact_budget_ - started_));
        started_ += granted;
        py::array_t<std::int64_t> output(2);
        checked_data(output)[0] = requested;
        checked_data(output)[1] = granted;
        return output;
    }

    [[nodiscard]] std::int64_t exact_remaining() const {
        return exact_budget_ < 0
            ? -1
            : std::max<std::int64_t>(0, exact_budget_ - started_);
    }

    [[nodiscard]] std::int64_t candidate_round_remaining() const {
        return round_remaining();
    }

    py::array_t<std::int64_t> complete_exact(std::int64_t count) {
        if (count < 0 || completed_ + interrupted_ + count > started_) {
            throw std::runtime_error("invalid completed exact-call count");
        }
        completed_ += count;
        return state();
    }

    py::array_t<std::int64_t> interrupt_exact(std::int64_t count) {
        if (count < 0 || completed_ + interrupted_ + count > started_) {
            throw std::runtime_error("invalid interrupted exact-call count");
        }
        interrupted_ += count;
        return state();
    }

    py::array_t<std::int64_t> snapshot() const {
        return state();
    }

    py::array_t<std::int64_t> restore(py::handle snapshot) {
        auto snapshot_array = checked_array<std::int64_t>(
            snapshot, "snapshot", 1);
        if (snapshot_array.size() != 9) {
            throw std::invalid_argument("native budget snapshot shape is invalid");
        }
        const auto* values = checked_data<std::int64_t>(snapshot_array);
        if ((values[0] != 0 && values[0] != 1) || values[3] < 0
            || values[5] < 0 || values[6] < 0 || values[7] < 0
            || values[6] + values[7] > values[5]
            || (values[0] == 1 && (values[1] < 0 || values[2] < 0))) {
            throw std::invalid_argument("native budget snapshot values are invalid");
        }
        round_active_ = values[0] == 1;
        lane_id_ = values[1];
        iteration_ = values[2];
        round_used_ = values[3];
        started_ = values[5];
        completed_ = values[6];
        interrupted_ = values[7];
        if (!round_active_) {
            lane_id_ = -1;
            iteration_ = -1;
            round_used_ = 0;
        }
        if (round_used_ > round_budget_
            || (exact_budget_ >= 0 && started_ > exact_budget_)) {
            throw std::invalid_argument("native budget snapshot exceeds configured limits");
        }
        return state();
    }

    py::array_t<std::int64_t> state() const {
        py::array_t<std::int64_t> output(9);
        auto* values = checked_data(output);
        values[0] = round_active_ ? 1 : 0;
        values[1] = lane_id_;
        values[2] = iteration_;
        values[3] = round_used_;
        values[4] = round_remaining();
        values[5] = started_;
        values[6] = completed_;
        values[7] = interrupted_;
        values[8] = exact_budget_ >= 0 && started_ >= exact_budget_ ? 1 : 0;
        return output;
    }

    [[nodiscard]] bool budget_reached() const noexcept {
        return exact_budget_ >= 0 && started_ >= exact_budget_;
    }

private:
    friend class NativeSearchEngineV2;
    struct NativeSnapshot {
        bool round_active = false;
        std::int64_t lane_id = -1;
        std::int64_t iteration = -1;
        std::int64_t round_used = 0;
        std::int64_t started = 0;
        std::int64_t completed = 0;
        std::int64_t interrupted = 0;
    };
    std::int64_t exact_budget_;
    std::int64_t round_budget_;
    bool round_active_ = false;
    std::int64_t lane_id_ = -1;
    std::int64_t iteration_ = -1;
    std::int64_t round_used_ = 0;
    std::int64_t started_ = 0;
    std::int64_t completed_ = 0;
    std::int64_t interrupted_ = 0;

    [[nodiscard]] std::int64_t round_remaining() const {
        return round_active_ ? std::max<std::int64_t>(0, round_budget_ - round_used_)
                             : round_budget_;
    }

    [[nodiscard]] NativeSnapshot native_snapshot() const noexcept {
        return NativeSnapshot{
            round_active_, lane_id_, iteration_, round_used_,
            started_, completed_, interrupted_};
    }

    void rollback_preserving_exact_noexcept(
        const NativeSnapshot& snapshot) noexcept {
        rollback_preserving_exact_impl(snapshot, false);
    }

    void rollback_outer_preserving_exact_noexcept(
        const NativeSnapshot& snapshot) noexcept {
        rollback_preserving_exact_impl(snapshot, true);
    }

    void rollback_preserving_exact_impl(
        const NativeSnapshot& snapshot,
        bool restore_round) noexcept {
        const auto started_delta = started_ - snapshot.started;
        const auto completed_delta = completed_ - snapshot.completed;
        const auto interrupted_delta = interrupted_ - snapshot.interrupted;
        if (started_delta < 0 || completed_delta < 0 || interrupted_delta < 0
            || completed_delta + interrupted_delta != started_delta) {
            std::terminate();
        }
        if (restore_round || started_delta == 0) {
            round_active_ = snapshot.round_active;
            lane_id_ = snapshot.lane_id;
            iteration_ = snapshot.iteration;
            round_used_ = snapshot.round_used;
        }
        started_ = snapshot.started + started_delta;
        completed_ = snapshot.completed + completed_delta;
        interrupted_ = snapshot.interrupted + interrupted_delta;
    }
};

class NativeAttemptedPlanSetV2 {
public:
    py::array_t<std::int64_t> lookup(
        py::handle plan_offsets,
        py::handle route_offsets,
        py::handle route_indices) const {
        const auto plans = decode_plans(plan_offsets, route_offsets, route_indices);
        py::array_t<std::int64_t> flags(plans.size());
        for (std::size_t index = 0; index < plans.size(); ++index) {
            checked_data(flags)[index] = attempted_.contains(plan_key(plans[index])) ? 1 : 0;
        }
        return flags;
    }

    py::array_t<std::int64_t> begin_mark_many_atomic(
        py::handle plan_offsets,
        py::handle route_offsets,
        py::handle route_indices,
        py::handle plan_ids) {
        if (active_additions_.has_value()) {
            throw std::runtime_error("native attempted-plan set already has an active batch");
        }
        const auto plans = decode_plans(plan_offsets, route_offsets, route_indices);
        auto ids_array = checked_array<std::int64_t>(plan_ids, "plan_ids", 1);
        const auto* ids = checked_data<std::int64_t>(ids_array);
        std::unordered_set<std::int64_t> unique_ids;
        std::vector<std::int64_t> validated_ids;
        validated_ids.reserve(static_cast<std::size_t>(ids_array.size()));
        for (py::ssize_t ordinal = 0; ordinal < ids_array.size(); ++ordinal) {
            if (ids[ordinal] < 0
                || ids[ordinal] >= static_cast<std::int64_t>(plans.size())
                || !unique_ids.insert(ids[ordinal]).second) {
                throw std::invalid_argument(
                    "native attempted-plan IDs must be unique valid plan rows");
            }
            validated_ids.push_back(ids[ordinal]);
        }
        std::vector<std::string> additions;
        py::array_t<std::int64_t> statuses(ids_array.size());
        for (std::size_t ordinal = 0; ordinal < validated_ids.size(); ++ordinal) {
            const auto key = plan_key(
                plans[static_cast<std::size_t>(validated_ids[ordinal])]);
            if (attempted_.contains(key)) {
                checked_data(statuses)[static_cast<py::ssize_t>(ordinal)] = 0;
            } else {
                attempted_.insert(key);
                additions.push_back(key);
                checked_data(statuses)[static_cast<py::ssize_t>(ordinal)] = 1;
            }
        }
        active_additions_ = std::move(additions);
        return statuses;
    }

    std::int64_t commit_mark_batch() {
        require_active("commit_mark_batch");
        active_additions_.reset();
        return static_cast<std::int64_t>(attempted_.size());
    }

    std::int64_t rollback_mark_batch() {
        require_active("rollback_mark_batch");
        for (const auto& key : *active_additions_) {
            attempted_.erase(key);
        }
        active_additions_.reset();
        return static_cast<std::int64_t>(attempted_.size());
    }

    [[nodiscard]] std::int64_t size() const {
        return static_cast<std::int64_t>(attempted_.size());
    }

private:
    friend class NativeSearchEngineV2;
    using Route = std::vector<std::int64_t>;
    using Plan = std::vector<Route>;
    std::unordered_set<std::string> attempted_;
    std::optional<std::vector<std::string>> active_additions_;

    void commit_mark_batch_noexcept() noexcept {
        active_additions_.reset();
    }

    void rollback_mark_batch_noexcept() noexcept {
        for (const auto& key : *active_additions_) {
            attempted_.erase(key);
        }
        active_additions_.reset();
    }

    void prepare_mark_commit() const {
        require_active("prepare_mark_commit");
    }

    static std::vector<Plan> decode_plans(
        py::handle plan_offsets,
        py::handle route_offsets,
        py::handle route_indices) {
        auto plans_array = checked_array<std::int64_t>(
            plan_offsets, "plan_offsets", 1);
        auto routes_array = checked_array<std::int64_t>(
            route_offsets, "route_offsets", 1);
        auto indices_array = checked_array<std::int64_t>(
            route_indices, "route_indices", 1);
        if (plans_array.size() < 1 || routes_array.size() < 1) {
            throw std::invalid_argument("native attempted-plan offsets cannot be empty");
        }
        const auto plan_count = static_cast<std::size_t>(plans_array.size() - 1);
        const auto route_count = static_cast<std::size_t>(routes_array.size() - 1);
        const auto* plans = checked_data<std::int64_t>(plans_array);
        const auto* routes = checked_data<std::int64_t>(routes_array);
        const auto* indices = checked_data<std::int64_t>(indices_array);
        if (plans[0] != 0 || plans[plan_count] != static_cast<std::int64_t>(route_count)
            || routes[0] != 0 || routes[route_count] != indices_array.size()) {
            throw std::invalid_argument("native attempted-plan boundary is invalid");
        }
        std::vector<Plan> output;
        output.reserve(plan_count);
        for (std::size_t plan = 0; plan < plan_count; ++plan) {
            if (plans[plan] < 0 || plans[plan] > plans[plan + 1]) {
                throw std::invalid_argument("native plan offsets must be monotonic");
            }
            Plan decoded;
            decoded.reserve(static_cast<std::size_t>(plans[plan + 1] - plans[plan]));
            for (auto route = plans[plan]; route < plans[plan + 1]; ++route) {
                if (routes[route] < 0 || routes[route] > routes[route + 1]) {
                    throw std::invalid_argument("native route offsets must be monotonic");
                }
                decoded.emplace_back(
                    indices + routes[route], indices + routes[route + 1]);
            }
            output.push_back(std::move(decoded));
        }
        return output;
    }

    static std::string plan_key(const Plan& plan) {
        std::string key;
        const auto append = [&key](std::int64_t value) {
            const auto position = key.size();
            key.resize(position + sizeof(value));
            std::memcpy(key.data() + position, &value, sizeof(value));
        };
        append(static_cast<std::int64_t>(plan.size()));
        for (const auto& route : plan) {
            append(static_cast<std::int64_t>(route.size()));
            for (const auto node : route) {
                append(node);
            }
        }
        return key;
    }

    void require_active(const char* operation) const {
        if (!active_additions_.has_value()) {
            throw std::runtime_error(
                std::string("native attempted-plan ") + operation
                + " requires an active batch");
        }
    }
};

template <typename T>
struct Stage052ReplayNumericColumn {
    py::array_t<T, py::array::c_style | py::array::forcecast> values;
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> valid;

    explicit Stage052ReplayNumericColumn(const py::handle encoded) {
        const auto tuple = py::cast<py::tuple>(encoded);
        if (tuple.size() != 3) {
            throw std::invalid_argument("native replay column encoding is invalid");
        }
        values = py::cast<decltype(values)>(tuple[0]);
        valid = py::cast<decltype(valid)>(tuple[1]);
        if (values.ndim() != 1 || valid.ndim() != 1 ||
            values.size() != valid.size()) {
            throw std::invalid_argument("native replay column width is invalid");
        }
    }

    [[nodiscard]] bool has(const py::ssize_t index) const {
        return *valid.data(index) != 0;
    }

    [[nodiscard]] T get(const py::ssize_t index) const {
        return *values.data(index);
    }
};

struct Stage052ReplayStringColumn {
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>
        indices;
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> valid;
    std::vector<std::string> dictionary;

    explicit Stage052ReplayStringColumn(const py::handle encoded) {
        const auto tuple = py::cast<py::tuple>(encoded);
        if (tuple.size() != 3) {
            throw std::invalid_argument("native replay string encoding is invalid");
        }
        indices = py::cast<decltype(indices)>(tuple[0]);
        valid = py::cast<decltype(valid)>(tuple[1]);
        dictionary = py::cast<std::vector<std::string>>(tuple[2]);
        if (indices.ndim() != 1 || valid.ndim() != 1 ||
            indices.size() != valid.size()) {
            throw std::invalid_argument("native replay string width is invalid");
        }
    }

    [[nodiscard]] bool has(const py::ssize_t index) const {
        return *valid.data(index) != 0;
    }

    [[nodiscard]] const std::string& get(const py::ssize_t index) const {
        const auto dictionary_index = *indices.data(index);
        if (dictionary_index < 0 ||
            static_cast<std::size_t>(dictionary_index) >= dictionary.size()) {
            throw std::invalid_argument(
                "native replay dictionary index is invalid");
        }
        return dictionary[static_cast<std::size_t>(dictionary_index)];
    }
};

struct Stage052ReplayBatch {
    Stage052ReplayNumericColumn<std::int64_t> event_id;
    Stage052ReplayStringColumn benchmark_axis;
    Stage052ReplayStringColumn lane;
    Stage052ReplayStringColumn event_type;
    Stage052ReplayNumericColumn<double> timestamp_seconds;
    Stage052ReplayNumericColumn<double> started_at;
    Stage052ReplayNumericColumn<double> completed_at;
    Stage052ReplayNumericColumn<std::int64_t> iteration;
    Stage052ReplayNumericColumn<std::int64_t> evaluation_id;
    Stage052ReplayStringColumn cache_key_digest;
    Stage052ReplayStringColumn operation;
    Stage052ReplayStringColumn lookup_result;
    Stage052ReplayStringColumn status;
    Stage052ReplayStringColumn reason;
    Stage052ReplayStringColumn route_key;
    Stage052ReplayNumericColumn<std::int64_t> decision_id;
    Stage052ReplayStringColumn kind;
    Stage052ReplayStringColumn operator_name;
    Stage052ReplayNumericColumn<std::uint8_t> exact_started;
    Stage052ReplayNumericColumn<std::uint8_t> exact_completed;
    Stage052ReplayNumericColumn<std::uint8_t> feasible;
    Stage052ReplayNumericColumn<std::uint8_t> accepted;
    Stage052ReplayNumericColumn<std::uint8_t> global_best;
    Stage052ReplayNumericColumn<std::int64_t> candidate_vehicle_delta;
    Stage052ReplayNumericColumn<std::uint8_t> native_fallback;
    Stage052ReplayStringColumn failure_reason;

    explicit Stage052ReplayBatch(const py::dict& columns)
        : event_id(columns["event_id"]),
          benchmark_axis(columns["benchmark_axis"]),
          lane(columns["lane"]),
          event_type(columns["event_type"]),
          timestamp_seconds(columns["timestamp_seconds"]),
          started_at(columns["started_at"]),
          completed_at(columns["completed_at"]),
          iteration(columns["iteration"]),
          evaluation_id(columns["evaluation_id"]),
          cache_key_digest(columns["cache_key_digest"]),
          operation(columns["operation"]),
          lookup_result(columns["lookup_result"]),
          status(columns["status"]),
          reason(columns["reason"]),
          route_key(columns["route_key"]),
          decision_id(columns["decision_id"]),
          kind(columns["kind"]),
          operator_name(columns["operator"]),
          exact_started(columns["exact_started"]),
          exact_completed(columns["exact_completed"]),
          feasible(columns["feasible"]),
          accepted(columns["accepted"]),
          global_best(columns["global_best"]),
          candidate_vehicle_delta(columns["candidate_vehicle_delta"]),
          native_fallback(columns["native_fallback"]),
          failure_reason(columns["failure_reason"]) {
        const auto row_count = event_id.values.size();
        const std::array<py::ssize_t, 25> widths{{
            benchmark_axis.indices.size(),
            lane.indices.size(),
            event_type.indices.size(),
            timestamp_seconds.values.size(),
            started_at.values.size(),
            completed_at.values.size(),
            iteration.values.size(),
            evaluation_id.values.size(),
            cache_key_digest.indices.size(),
            operation.indices.size(),
            lookup_result.indices.size(),
            status.indices.size(),
            reason.indices.size(),
            route_key.indices.size(),
            decision_id.values.size(),
            kind.indices.size(),
            operator_name.indices.size(),
            exact_started.values.size(),
            exact_completed.values.size(),
            feasible.values.size(),
            accepted.values.size(),
            global_best.values.size(),
            candidate_vehicle_delta.values.size(),
            native_fallback.values.size(),
            failure_reason.indices.size(),
        }};
        if (std::any_of(
                widths.begin(), widths.end(),
                [row_count](const py::ssize_t width) {
                    return width != row_count;
                })) {
            throw std::invalid_argument(
                "native replay columns do not share one row count");
        }
    }
};

void stage052_append_json_string(
    std::string& destination, const std::string_view value) {
    constexpr char hexadecimal[] = "0123456789abcdef";
    destination.push_back('"');
    for (const unsigned char character : value) {
        switch (character) {
            case '"':
                destination += "\\\"";
                break;
            case '\\':
                destination += "\\\\";
                break;
            case '\b':
                destination += "\\b";
                break;
            case '\f':
                destination += "\\f";
                break;
            case '\n':
                destination += "\\n";
                break;
            case '\r':
                destination += "\\r";
                break;
            case '\t':
                destination += "\\t";
                break;
            default:
                if (character < 0x20U) {
                    destination += "\\u00";
                    destination.push_back(hexadecimal[character >> 4U]);
                    destination.push_back(hexadecimal[character & 0x0fU]);
                } else {
                    destination.push_back(static_cast<char>(character));
                }
        }
    }
    destination.push_back('"');
}

void stage052_append_nullable_string(
    std::string& destination,
    const Stage052ReplayStringColumn& column,
    const py::ssize_t index,
    const bool empty_is_null = false) {
    if (!column.has(index) || (empty_is_null && column.get(index).empty())) {
        destination += "null";
        return;
    }
    stage052_append_json_string(destination, column.get(index));
}

template <typename T>
void stage052_append_nullable_integer(
    std::string& destination,
    const Stage052ReplayNumericColumn<T>& column,
    const py::ssize_t index) {
    if (!column.has(index)) {
        destination += "null";
        return;
    }
    destination += std::to_string(column.get(index));
}

void stage052_append_nullable_boolean(
    std::string& destination,
    const Stage052ReplayNumericColumn<std::uint8_t>& column,
    const py::ssize_t index) {
    if (!column.has(index)) {
        destination += "null";
        return;
    }
    destination += column.get(index) != 0 ? "true" : "false";
}

struct Stage052ReplayAxisState {
    std::int64_t budget_seconds = 0;
    std::int64_t last_evaluation_id = 0;
    std::int64_t exact_started = 0;
    std::int64_t exact_completed = 0;
    std::int64_t accepted_candidates = 0;
    std::int64_t global_bests = 0;
    std::unordered_set<std::string> deadline_lanes;
    std::unordered_set<std::string> cache_keys;
    std::unordered_set<std::string> cache_misses;
    std::unordered_set<std::string> completed_exact_keys;
    std::vector<std::pair<std::int64_t, std::string>> ledger_entries;
    std::size_t ledger_index = 0;
    std::int64_t ledger_seen = 0;
    std::string ledger_token_bytes;
    py::object ledger_hasher;
};

class Stage052ReplayState {
   public:
    explicit Stage052ReplayState(
        const py::dict& axis_budgets,
        const py::dict& persistence_ledgers)
        : hasher_(
              py::module_::import("hashlib").attr("sha256")()) {
        if (axis_budgets.empty()) {
            throw std::invalid_argument(
                "axis budgets must not be empty");
        }
        for (const auto& item : axis_budgets) {
            const auto axis = py::cast<std::string>(item.first);
            const auto budget = py::cast<std::int64_t>(item.second);
            if (axis.empty() || budget <= 0) {
                throw std::invalid_argument(
                    "axis budgets must be positive integers");
            }
            Stage052ReplayAxisState state;
            state.budget_seconds = budget;
            states_.emplace(axis, std::move(state));
        }
        if (!persistence_ledgers.empty()) {
            if (persistence_ledgers.size() != states_.size()) {
                throw std::invalid_argument(
                    "persistence ledger axes do not match the shard axes");
            }
            for (const auto& item : persistence_ledgers) {
                const auto axis = py::cast<std::string>(item.first);
                const auto state_iterator = states_.find(axis);
                if (state_iterator == states_.end()) {
                    throw std::invalid_argument(
                        "persistence ledger axes do not match the shard axes");
                }
                const auto entries = py::cast<py::sequence>(item.second);
                if (entries.empty()) {
                    throw std::invalid_argument(
                        "persistence ledger must not be empty");
                }
                auto& state = state_iterator->second;
                state.ledger_entries.reserve(
                    static_cast<std::size_t>(entries.size()));
                for (const auto& raw_entry : entries) {
                    const auto entry = py::cast<py::tuple>(raw_entry);
                    if (entry.size() != 2) {
                        throw std::invalid_argument(
                            "persistence ledger entry is invalid");
                    }
                    const auto row_count =
                        py::cast<std::int64_t>(entry[0]);
                    const auto digest = py::cast<std::string>(entry[1]);
                    if (row_count <= 0 || digest.size() != 64U) {
                        throw std::invalid_argument(
                            "persistence ledger entry is invalid");
                    }
                    state.ledger_entries.emplace_back(row_count, digest);
                }
                state.ledger_hasher =
                    py::module_::import("hashlib").attr("sha256")();
            }
        }
    }

    void consume(const py::dict& encoded_columns) {
        const Stage052ReplayBatch batch(encoded_columns);
        std::string token_bytes;
        token_bytes.reserve(
            static_cast<std::size_t>(batch.event_id.values.size()) * 160U);
        for (py::ssize_t index = 0; index < batch.event_id.values.size();
             ++index) {
            consume_row(batch, index, token_bytes);
        }
        if (!token_bytes.empty()) {
            hasher_.attr("update")(py::bytes(token_bytes));
        }
    }

    [[nodiscard]] py::dict finish() const {
        if (event_count_ <= 0) {
            throw std::invalid_argument(
                "native replay event stream is empty");
        }
        std::vector<std::string> axes;
        axes.reserve(states_.size());
        for (const auto& [axis, unused] : states_) {
            static_cast<void>(unused);
            axes.push_back(axis);
        }
        std::sort(axes.begin(), axes.end());
        py::list axis_counts;
        py::list deadline_axes;
        std::int64_t exact_started = 0;
        std::int64_t exact_completed = 0;
        std::int64_t accepted_candidates = 0;
        std::int64_t global_bests = 0;
        for (const auto& axis : axes) {
            const auto& state = states_.at(axis);
            if (!state.ledger_entries.empty() &&
                (state.ledger_index != state.ledger_entries.size() ||
                 state.ledger_seen != 0)) {
                throw std::invalid_argument(
                    "persistence ledger does not cover the complete event stream: " +
                    axis);
            }
            exact_started += state.exact_started;
            exact_completed += state.exact_completed;
            accepted_candidates += state.accepted_candidates;
            global_bests += state.global_bests;
            if (!state.deadline_lanes.empty()) {
                deadline_axes.append(axis);
            }
            axis_counts.append(py::make_tuple(
                axis, state.exact_started, state.exact_completed));
        }
        py::dict result;
        result["event_count"] = event_count_;
        result["first_event_id"] = first_event_id_;
        result["last_event_id"] = previous_event_id_;
        result["exact_started"] = exact_started;
        result["exact_completed"] = exact_completed;
        result["accepted_candidates"] = accepted_candidates;
        result["global_bests"] = global_bests;
        result["native_fallback_count"] = native_fallback_count_;
        result["deadline_axes"] = deadline_axes;
        result["axis_exact_counts"] = axis_counts;
        result["event_type_counts"] = sorted_counts(event_type_counts_);
        result["failure_reason_counts"] = sorted_counts(failure_reason_counts_);
        result["event_token_sha256"] = hasher_.attr("hexdigest")();
        return result;
    }

   private:
    std::unordered_map<std::string, Stage052ReplayAxisState> states_;
    std::unordered_map<std::string, std::int64_t> event_type_counts_;
    std::unordered_map<std::string, std::int64_t> failure_reason_counts_;
    std::int64_t previous_event_id_ = 0;
    std::int64_t first_event_id_ = 0;
    std::int64_t event_count_ = 0;
    std::int64_t native_fallback_count_ = 0;
    py::object hasher_;

    [[nodiscard]] static py::list sorted_counts(
        const std::unordered_map<std::string, std::int64_t>& counts) {
        std::vector<std::pair<std::string, std::int64_t>> ordered(
            counts.begin(), counts.end());
        std::sort(ordered.begin(), ordered.end());
        py::list result;
        for (const auto& [key, value] : ordered) {
            result.append(py::make_tuple(key, value));
        }
        return result;
    }

    static bool truth(
        const Stage052ReplayNumericColumn<std::uint8_t>& column,
        const py::ssize_t index) {
        return column.has(index) && column.get(index) != 0;
    }

    static std::string value_or_empty(
        const Stage052ReplayStringColumn& column,
        const py::ssize_t index) {
        return column.has(index) ? column.get(index) : std::string{};
    }

    static bool contains_casefold_fallback(const std::string& text) {
        std::string lowered(text);
        std::transform(
            lowered.begin(), lowered.end(), lowered.begin(),
            [](const unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
        return lowered.find("fallback") != std::string::npos;
    }

    static void append_event_token(
        std::string& destination,
        const Stage052ReplayBatch& batch,
        const py::ssize_t index) {
        const auto& event_type = batch.event_type.get(index);
        destination.push_back('[');
        stage052_append_json_string(destination, event_type);
        destination.push_back(',');
        stage052_append_nullable_string(
            destination, batch.benchmark_axis, index);
        destination.push_back(',');
        stage052_append_nullable_string(destination, batch.lane, index);
        destination.push_back(',');
        stage052_append_nullable_integer(destination, batch.iteration, index);
        destination.push_back(',');
        stage052_append_nullable_string(
            destination, batch.operator_name, index);
        destination.push_back(',');
        stage052_append_nullable_string(destination, batch.route_key, index);
        destination.push_back(',');
        stage052_append_nullable_integer(destination, batch.decision_id, index);
        destination.push_back(',');
        stage052_append_nullable_string(
            destination, batch.kind, index, true);
        destination.push_back(',');
        stage052_append_nullable_string(
            destination, batch.operation, index, true);
        destination.push_back(',');
        stage052_append_nullable_string(
            destination, batch.status, index, true);
        destination.push_back(',');
        if (event_type == "screening_decision" &&
            (!batch.reason.has(index) || batch.reason.get(index).empty())) {
            stage052_append_json_string(destination, "");
        } else {
            stage052_append_nullable_string(
                destination, batch.reason, index, true);
        }
        destination.push_back(',');
        stage052_append_nullable_boolean(
            destination, batch.exact_started, index);
        destination.push_back(',');
        stage052_append_nullable_boolean(
            destination, batch.exact_completed, index);
        destination.push_back(',');
        stage052_append_nullable_boolean(destination, batch.feasible, index);
        destination += "]\n";
    }

    static void consume_ledger_token(
        Stage052ReplayAxisState& state,
        const std::string_view token,
        const std::string& axis) {
        if (state.ledger_entries.empty()) {
            return;
        }
        if (state.ledger_index >= state.ledger_entries.size()) {
            throw std::invalid_argument(
                "persistence ledger ended before the event stream: " + axis);
        }
        state.ledger_token_bytes.append(token);
        ++state.ledger_seen;
        const auto& [expected_rows, expected_digest] =
            state.ledger_entries[state.ledger_index];
        if (state.ledger_seen == expected_rows) {
            state.ledger_hasher.attr("update")(
                py::bytes(state.ledger_token_bytes));
            const auto observed =
                py::cast<std::string>(
                    state.ledger_hasher.attr("hexdigest")());
            if (observed != expected_digest) {
                throw std::invalid_argument(
                    "persistence batch digest does not replay: " + axis + "/" +
                    std::to_string(state.ledger_index));
            }
            ++state.ledger_index;
            state.ledger_seen = 0;
            state.ledger_token_bytes.clear();
            state.ledger_hasher =
                py::module_::import("hashlib").attr("sha256")();
        } else if (state.ledger_seen > expected_rows) {
            throw std::invalid_argument(
                "persistence batch row count overflow: " + axis + "/" +
                std::to_string(state.ledger_index));
        }
    }

    void consume_row(
        const Stage052ReplayBatch& batch,
        const py::ssize_t index,
        std::string& token_bytes) {
        if (!batch.event_id.has(index)) {
            throw std::invalid_argument(
                "event IDs are not strictly increasing");
        }
        const auto event_id = batch.event_id.get(index);
        if (event_id <= previous_event_id_) {
            throw std::invalid_argument(
                "event IDs are not strictly increasing");
        }
        if (first_event_id_ == 0) {
            first_event_id_ = event_id;
        }
        previous_event_id_ = event_id;
        ++event_count_;

        auto axis = value_or_empty(batch.benchmark_axis, index);
        const auto lane = value_or_empty(batch.lane, index);
        if (axis.empty()) {
            axis = lane.substr(0, lane.find(':'));
        }
        auto state_iterator = states_.find(axis);
        if (state_iterator == states_.end()) {
            throw std::invalid_argument(
                "event refers to an unknown benchmark axis: " + axis);
        }
        auto& state = state_iterator->second;
        const auto& event_type = batch.event_type.get(index);
        ++event_type_counts_[event_type];
        const auto token_start = token_bytes.size();
        append_event_token(token_bytes, batch, index);
        consume_ledger_token(
            state,
            std::string_view(token_bytes).substr(token_start),
            axis);

        const auto failure_reason =
            value_or_empty(batch.failure_reason, index);
        const bool fallback =
            truth(batch.native_fallback, index) ||
            (event_type == "execution_error" &&
             contains_casefold_fallback(failure_reason));
        if (fallback) {
            ++native_fallback_count_;
            throw std::invalid_argument(
                "native/protocol fallback observed on " + axis);
        }
        if (!failure_reason.empty()) {
            ++failure_reason_counts_[failure_reason];
        }

        if (event_type == "deadline_boundary") {
            if (!batch.timestamp_seconds.has(index) ||
                !std::isfinite(batch.timestamp_seconds.get(index)) ||
                std::abs(
                    batch.timestamp_seconds.get(index) -
                    static_cast<double>(state.budget_seconds)) > 1.0) {
                throw std::invalid_argument(
                    "deadline boundary timestamp mismatch on " + axis);
            }
            state.deadline_lanes.insert(lane);
        }

        if (event_type == "route_evaluation") {
            if (!batch.evaluation_id.has(index) ||
                batch.evaluation_id.get(index) !=
                    state.last_evaluation_id + 1) {
                throw std::invalid_argument(
                    "route evaluation ordering mismatch on " + axis);
            }
            state.last_evaluation_id = batch.evaluation_id.get(index);
            if (truth(batch.exact_started, index)) {
                if (state.deadline_lanes.contains(lane)) {
                    throw std::invalid_argument(
                        "exact work started after deadline on " + axis);
                }
                ++state.exact_started;
                if (!batch.started_at.has(index) ||
                    !std::isfinite(batch.started_at.get(index)) ||
                    batch.started_at.get(index) < 0.0) {
                    throw std::invalid_argument(
                        "invalid exact start time on " + axis);
                }
                if (truth(batch.exact_completed, index)) {
                    ++state.exact_completed;
                    if (!batch.completed_at.has(index) ||
                        !std::isfinite(batch.completed_at.get(index)) ||
                        batch.completed_at.get(index) <
                            batch.started_at.get(index) ||
                        batch.completed_at.get(index) >
                            static_cast<double>(state.budget_seconds)) {
                        throw std::invalid_argument(
                            "exact completion crosses deadline on " + axis);
                    }
                    const auto digest =
                        value_or_empty(batch.cache_key_digest, index);
                    if (digest.empty() ||
                        !state.cache_misses.contains(digest)) {
                        throw std::invalid_argument(
                            "exact completion lacks a preceding cache miss on " +
                            axis);
                    }
                    state.cache_misses.erase(digest);
                    state.completed_exact_keys.insert(digest);
                }
                if (value_or_empty(batch.status, index) ==
                    "interrupted_deadline") {
                    state.deadline_lanes.insert(lane);
                }
            }
        }

        if (event_type == "cache_event") {
            const auto operation = value_or_empty(batch.operation, index);
            const auto digest =
                value_or_empty(batch.cache_key_digest, index);
            if (operation == "store") {
                if (state.deadline_lanes.contains(lane)) {
                    throw std::invalid_argument(
                        "cache store observed after deadline on " + axis);
                }
                if (digest.empty()) {
                    throw std::invalid_argument(
                        "cache store lacks key on " + axis);
                }
                if (!state.completed_exact_keys.contains(digest)) {
                    throw std::invalid_argument(
                        "cache store precedes exact completion on " + axis);
                }
                state.completed_exact_keys.erase(digest);
                state.cache_keys.insert(digest);
            } else if (operation == "evict") {
                if (!state.cache_keys.contains(digest)) {
                    throw std::invalid_argument(
                        "cache eviction refers to an absent key on " + axis);
                }
                state.cache_keys.erase(digest);
            } else if (operation == "lookup_result") {
                const auto result =
                    value_or_empty(batch.lookup_result, index);
                if (result == "hit" &&
                    !state.cache_keys.contains(digest)) {
                    throw std::invalid_argument(
                        "cache hit precedes store on " + axis);
                }
                if (result == "miss") {
                    if (digest.empty()) {
                        throw std::invalid_argument(
                            "cache miss lacks key on " + axis);
                    }
                    state.cache_misses.insert(digest);
                }
            }
        }

        if (event_type == "candidate_state" &&
            truth(batch.accepted, index)) {
            if (state.deadline_lanes.contains(lane)) {
                throw std::invalid_argument(
                    "candidate accepted after deadline on " + axis);
            }
            if (!batch.candidate_vehicle_delta.has(index) ||
                batch.candidate_vehicle_delta.get(index) > 0) {
                throw std::invalid_argument(
                    "accepted candidate violates vehicle-first policy on " +
                    axis);
            }
            ++state.accepted_candidates;
            if (truth(batch.global_best, index)) {
                ++state.global_bests;
            }
        }
    }
};

struct Stage052CanonicalSignature {
    std::array<std::uint64_t, 32> values{};
    py::ssize_t count = 0;
};

struct Stage052ScreeningDefinitionCacheEntry {
    py::tuple key;
    Stage052CanonicalSignature signature;
    py::object definition_id;
};

struct Stage052ScreeningDefinitionCache {
    std::unordered_multimap<std::size_t, Stage052ScreeningDefinitionCacheEntry>
        definitions;
    std::deque<
        std::pair<std::size_t, Stage052ScreeningDefinitionCacheEntry*>>
        insertion_order;
    PyTypeObject* check_type = nullptr;
    py::object check_type_owner;
    std::array<py::object, 4> check_descriptors;
    std::array<Py_ssize_t, 4> check_offsets{{-1, -1, -1, -1}};
    std::size_t capacity = 262144;
};

constexpr const char* STAGE052_SCREENING_CACHE_CAPSULE =
    "evrptw.stage052_screening_definition_cache";

struct Stage052DefinitionIdentityStore {
    std::unordered_map<std::int64_t, std::array<std::uint8_t, 32>> identities;
    std::size_t capacity = 0;
};

constexpr const char* STAGE052_DEFINITION_IDENTITY_STORE_CAPSULE =
    "evrptw.stage052_definition_identity_store";

py::capsule create_stage052_definition_identity_store(
    const py::ssize_t capacity) {
    if (capacity <= 0 || capacity > 4194304) {
        throw std::invalid_argument(
            "Stage 5.2 definition identity capacity must be in 1..4194304");
    }
    auto store = std::make_unique<Stage052DefinitionIdentityStore>();
    store->identities.max_load_factor(0.8F);
    store->identities.reserve(
        std::min<std::size_t>(
            static_cast<std::size_t>(capacity),
            262144));
    store->capacity = static_cast<std::size_t>(capacity);
    py::capsule capsule(
        store.get(),
        STAGE052_DEFINITION_IDENTITY_STORE_CAPSULE,
        [](PyObject* capsule) {
            auto* owned = static_cast<Stage052DefinitionIdentityStore*>(
                PyCapsule_GetPointer(
                    capsule,
                    STAGE052_DEFINITION_IDENTITY_STORE_CAPSULE));
            if (owned == nullptr) {
                PyErr_Clear();
                return;
            }
            delete owned;
        });
    store.release();
    return capsule;
}

std::array<std::uint8_t, 32> stage052_definition_digest(
    const py::handle definition) {
    const auto encoded = py::cast<py::bytes>(definition.attr("digest"));
    const auto value = py::cast<std::string>(encoded);
    if (value.size() != 32) {
        throw std::invalid_argument(
            "Stage 5.2 screening definition digest must contain 32 bytes");
    }
    std::array<std::uint8_t, 32> digest{};
    std::memcpy(digest.data(), value.data(), digest.size());
    return digest;
}

py::tuple register_stage052_definition_identities(
    const py::object& identity_store,
    const py::sequence& definitions) {
    auto* store = static_cast<Stage052DefinitionIdentityStore*>(
        PyCapsule_GetPointer(
            identity_store.ptr(),
            STAGE052_DEFINITION_IDENTITY_STORE_CAPSULE));
    if (store == nullptr) {
        PyErr_Clear();
        throw std::invalid_argument(
            "Stage 5.2 native definition identity store is invalid");
    }

    struct PendingIdentity {
        std::array<std::uint8_t, 32> digest;
        py::object definition;
    };
    std::unordered_map<std::int64_t, PendingIdentity> unique;
    unique.reserve(static_cast<std::size_t>(definitions.size()));
    std::vector<std::int64_t> order;
    order.reserve(static_cast<std::size_t>(definitions.size()));
    for (const py::handle raw_definition : definitions) {
        const auto definition =
            py::reinterpret_borrow<py::object>(raw_definition);
        const auto definition_id =
            py::cast<std::int64_t>(definition.attr("definition_id"));
        const auto digest = stage052_definition_digest(definition);
        const auto existing = unique.find(definition_id);
        if (existing == unique.end()) {
            order.push_back(definition_id);
            unique.emplace(
                definition_id,
                PendingIdentity{digest, definition});
        } else {
            if (existing->second.digest != digest) {
                throw std::invalid_argument(
                    "screening definition ID collision");
            }
            existing->second.definition = definition;
        }
    }

    std::size_t missing = 0;
    for (const auto definition_id : order) {
        const auto& pending = unique.at(definition_id);
        const auto stored = store->identities.find(definition_id);
        if (stored == store->identities.end()) {
            ++missing;
        } else if (stored->second != pending.digest) {
            throw std::invalid_argument(
                "screening definition ID collision");
        }
    }
    if (missing > store->capacity - store->identities.size()) {
        throw std::invalid_argument(
            "screening definition identity bound exceeded");
    }

    py::tuple inserted(static_cast<py::ssize_t>(missing));
    py::ssize_t inserted_index = 0;
    for (const auto definition_id : order) {
        auto& pending = unique.at(definition_id);
        const auto [_, was_inserted] =
            store->identities.emplace(definition_id, pending.digest);
        if (was_inserted) {
            inserted[inserted_index] = std::move(pending.definition);
            ++inserted_index;
        }
    }
    return inserted;
}

py::ssize_t stage052_definition_identity_store_size(
    const py::object& identity_store) {
    auto* store = static_cast<Stage052DefinitionIdentityStore*>(
        PyCapsule_GetPointer(
            identity_store.ptr(),
            STAGE052_DEFINITION_IDENTITY_STORE_CAPSULE));
    if (store == nullptr) {
        PyErr_Clear();
        throw std::invalid_argument(
            "Stage 5.2 native definition identity store is invalid");
    }
    return static_cast<py::ssize_t>(store->identities.size());
}

py::capsule create_stage052_screening_definition_cache(
    const py::ssize_t capacity = 262144) {
    if (capacity <= 0 || capacity > 262144) {
        throw std::invalid_argument(
            "Stage 5.2 screening definition cache capacity must be in 1..262144");
    }
    auto cache = std::make_unique<Stage052ScreeningDefinitionCache>();
    cache->definitions.max_load_factor(0.8F);
    cache->definitions.reserve(static_cast<std::size_t>(capacity));
    cache->capacity = static_cast<std::size_t>(capacity);
    py::capsule capsule(
        cache.get(),
        STAGE052_SCREENING_CACHE_CAPSULE,
        [](PyObject* capsule) {
            auto* owned = static_cast<Stage052ScreeningDefinitionCache*>(
                PyCapsule_GetPointer(capsule, STAGE052_SCREENING_CACHE_CAPSULE));
            if (owned == nullptr) {
                PyErr_Clear();
                return;
            }
            delete owned;
        });
    cache.release();
    return capsule;
}

Stage052ScreeningDefinitionCache* stage052_screening_definition_cache(
    const py::object& definition_cache) {
    auto* cache = static_cast<Stage052ScreeningDefinitionCache*>(
        PyCapsule_GetPointer(
            definition_cache.ptr(), STAGE052_SCREENING_CACHE_CAPSULE));
    if (cache == nullptr) {
        PyErr_Clear();
        throw std::invalid_argument(
            "Stage 5.2 native screening definition cache is invalid");
    }
    return cache;
}

py::ssize_t stage052_screening_definition_cache_size(
    const py::object& definition_cache) {
    return static_cast<py::ssize_t>(
        stage052_screening_definition_cache(definition_cache)
            ->definitions.size());
}

py::ssize_t stage052_screening_definition_cache_capacity(
    const py::object& definition_cache) {
    return static_cast<py::ssize_t>(
        stage052_screening_definition_cache(definition_cache)->capacity);
}

void append_stage052_signature(
    Stage052CanonicalSignature& signature,
    const std::uint64_t value) {
    if (signature.count >= static_cast<py::ssize_t>(signature.values.size())) {
        throw std::invalid_argument(
            "Stage 5.2 screening canonical signature exceeds its fixed domain");
    }
    signature.values[static_cast<std::size_t>(signature.count)] = value;
    ++signature.count;
}

std::uint64_t stage052_double_bits(PyObject* value) {
    const double number = PyFloat_AsDouble(value);
    if (number == -1.0 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(number));
    std::memcpy(&bits, &number, sizeof(bits));
    return bits;
}

void append_stage052_typed_value_signature(
    Stage052CanonicalSignature& signature,
    PyObject* value) {
    if (value == Py_None) {
        append_stage052_signature(signature, 0);
    } else if (PyBool_Check(value)) {
        append_stage052_signature(signature, value == Py_True ? 2 : 1);
    } else if (PyLong_Check(value) || PyFloat_Check(value)) {
        append_stage052_signature(signature, 3);
        append_stage052_signature(signature, stage052_double_bits(value));
    } else if (PyUnicode_Check(value)) {
        const Py_hash_t value_hash = PyObject_Hash(value);
        if (value_hash == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        append_stage052_signature(signature, 4);
        append_stage052_signature(
            signature, static_cast<std::uint64_t>(value_hash));
    } else {
        // Unsupported check values canonicalise to three null value columns.
        append_stage052_signature(signature, 5);
    }
}

std::pair<PyObject*, bool> stage052_screening_check_field(
    PyObject* check,
    const py::ssize_t field_index,
    Stage052ScreeningDefinitionCache* cache) {
    if (PyTuple_Check(check) && PyTuple_GET_SIZE(check) == 4) {
        return {PyTuple_GET_ITEM(check, field_index), false};
    }
    if (cache != nullptr) {
        if (cache->check_type == nullptr) {
            constexpr std::array<const char*, 4> names{
                "check", "status", "value", "reason"};
            std::array<py::object, 4> descriptors;
            std::array<Py_ssize_t, 4> offsets{{-1, -1, -1, -1}};
            bool descriptors_available = true;
            for (py::ssize_t index = 0; index < 4; ++index) {
                PyObject* descriptor = PyObject_GetAttrString(
                    reinterpret_cast<PyObject*>(Py_TYPE(check)),
                    names[static_cast<std::size_t>(index)]);
                if (descriptor == nullptr) {
                    PyErr_Clear();
                    descriptors_available = false;
                    break;
                }
                descriptors[static_cast<std::size_t>(index)] =
                    py::reinterpret_steal<py::object>(descriptor);
                if (Py_IS_TYPE(descriptor, &PyMemberDescr_Type)) {
                    const auto* member_descriptor =
                        reinterpret_cast<PyMemberDescrObject*>(descriptor);
                    if (member_descriptor->d_member != nullptr
                        && member_descriptor->d_member->type == Py_T_OBJECT_EX) {
                        offsets[static_cast<std::size_t>(index)] =
                            member_descriptor->d_member->offset;
                    }
                }
                if (offsets[static_cast<std::size_t>(index)] < 0
                    && Py_TYPE(descriptor)->tp_descr_get == nullptr) {
                    descriptors_available = false;
                    break;
                }
            }
            if (descriptors_available) {
                cache->check_type_owner =
                    py::reinterpret_borrow<py::object>(
                        reinterpret_cast<PyObject*>(Py_TYPE(check)));
                cache->check_type = Py_TYPE(check);
                cache->check_descriptors = std::move(descriptors);
                cache->check_offsets = offsets;
            }
        }
        if (Py_TYPE(check) == cache->check_type) {
            const Py_ssize_t offset =
                cache->check_offsets[static_cast<std::size_t>(field_index)];
            if (offset >= 0) {
                auto** slot = reinterpret_cast<PyObject**>(
                    reinterpret_cast<char*>(check) + offset);
                if (*slot == nullptr) {
                    throw std::invalid_argument(
                        "Stage 5.2 screening check slot is empty");
                }
                return {*slot, false};
            }
            const py::object& descriptor =
                cache->check_descriptors[static_cast<std::size_t>(field_index)];
            descrgetfunc get_value =
                Py_TYPE(descriptor.ptr())->tp_descr_get;
            if (get_value == nullptr) {
                throw std::invalid_argument(
                    "Stage 5.2 screening check descriptor is invalid");
            }
            PyObject* value = get_value(
                descriptor.ptr(),
                check,
                reinterpret_cast<PyObject*>(cache->check_type));
            if (value == nullptr) {
                throw py::error_already_set();
            }
            return {value, true};
        }
    }
    constexpr std::array<const char*, 4> names{
        "check", "status", "value", "reason"};
    PyObject* value = PyObject_GetAttrString(
        check, names[static_cast<std::size_t>(field_index)]);
    if (value == nullptr) {
        throw py::error_already_set();
    }
    return {value, true};
}

Stage052CanonicalSignature stage052_screening_key_signature(
    const std::array<PyObject*, 16>& fields,
    const py::ssize_t field_count,
    Stage052ScreeningDefinitionCache* cache = nullptr) {
    Stage052CanonicalSignature signature;
    if (field_count == 5) {
        append_stage052_typed_value_signature(signature, fields[4]);
        return signature;
    }
    if (field_count != 16) {
        throw std::invalid_argument(
            "Stage 5.2 screening cache key has an invalid field count");
    }
    append_stage052_signature(signature, stage052_double_bits(fields[6]));
    if (fields[7] == Py_None) {
        append_stage052_signature(signature, 0);
    } else {
        append_stage052_signature(signature, 1);
        append_stage052_signature(signature, stage052_double_bits(fields[7]));
    }
    append_stage052_signature(signature, stage052_double_bits(fields[8]));
    append_stage052_typed_value_signature(signature, fields[9]);
    append_stage052_signature(signature, stage052_double_bits(fields[11]));
    append_stage052_typed_value_signature(signature, fields[12]);
    append_stage052_typed_value_signature(signature, fields[13]);
    append_stage052_signature(signature, stage052_double_bits(fields[14]));
    if (!PyTuple_Check(fields[15])) {
        throw std::invalid_argument("Stage 5.2 deferred screening checks are invalid");
    }
    const py::ssize_t check_count = PyTuple_GET_SIZE(fields[15]);
    if (check_count > 8) {
        throw std::invalid_argument(
            "one screening decision exceeds the fixed eight-check domain");
    }
    append_stage052_signature(
        signature, static_cast<std::uint64_t>(check_count));
    for (py::ssize_t index = 0; index < check_count; ++index) {
        PyObject* check = PyTuple_GET_ITEM(fields[15], index);
        const auto [value, owns_value] =
            stage052_screening_check_field(check, 2, cache);
        try {
            append_stage052_typed_value_signature(signature, value);
        } catch (...) {
            if (owns_value) {
                Py_DECREF(value);
            }
            throw;
        }
        if (owns_value) {
            Py_DECREF(value);
        }
    }
    return signature;
}

bool stage052_signature_equal(
    const Stage052CanonicalSignature& left,
    const Stage052CanonicalSignature& right) {
    if (left.count != right.count) {
        return false;
    }
    for (py::ssize_t index = 0; index < left.count; ++index) {
        if (left.values[static_cast<std::size_t>(index)]
            != right.values[static_cast<std::size_t>(index)]) {
            return false;
        }
    }
    return true;
}

std::size_t stage052_screening_key_hash(
    const std::array<PyObject*, 16>& fields,
    const py::ssize_t field_count,
    const Stage052CanonicalSignature& signature,
    Stage052ScreeningDefinitionCache* cache) {
    std::size_t hash =
        static_cast<std::size_t>(field_count) ^ 0x9e3779b97f4a7c15ULL;
    const auto mix_value = [&hash](const std::size_t value) {
        hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    };
    const auto mix_object = [&mix_value](PyObject* value) {
        const Py_hash_t field_hash = PyObject_Hash(value);
        if (field_hash == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        mix_value(static_cast<std::size_t>(field_hash));
    };
    for (py::ssize_t index = 0; index < 4; ++index) {
        mix_object(fields[index]);
    }
    if (field_count == 5) {
        mix_object(fields[4]);
    } else {
        mix_object(fields[4]);
        mix_object(fields[5]);
        mix_object(fields[10]);
        const py::ssize_t check_count = PyTuple_GET_SIZE(fields[15]);
        for (py::ssize_t check_index = 0;
             check_index < check_count;
             ++check_index) {
            PyObject* check = PyTuple_GET_ITEM(fields[15], check_index);
            for (const py::ssize_t check_field : {0, 1, 3}) {
                const auto [value, owns_value] =
                    stage052_screening_check_field(
                        check, check_field, cache);
                try {
                    mix_object(value);
                } catch (...) {
                    if (owns_value) {
                        Py_DECREF(value);
                    }
                    throw;
                }
                if (owns_value) {
                    Py_DECREF(value);
                }
            }
        }
    }
    for (py::ssize_t index = 0; index < signature.count; ++index) {
        mix_value(static_cast<std::size_t>(
            signature.values[static_cast<std::size_t>(index)]));
    }
    return hash;
}

bool stage052_object_equal(PyObject* previous, PyObject* current) {
    if (previous == current) {
        return true;
    }
    const int equal = PyObject_RichCompareBool(previous, current, Py_EQ);
    if (equal < 0) {
        throw py::error_already_set();
    }
    return equal == 1;
}

bool stage052_screening_key_equal(
    const Stage052ScreeningDefinitionCacheEntry& stored,
    const std::array<PyObject*, 16>& fields,
    const py::ssize_t field_count,
    const Stage052CanonicalSignature& signature,
    Stage052ScreeningDefinitionCache* cache) {
    if (!stage052_signature_equal(stored.signature, signature)
        || PyTuple_GET_SIZE(stored.key.ptr()) != field_count) {
        return false;
    }
    std::array<py::ssize_t, 7> compared_fields{{0, 1, 2, 3, 4, 5, 10}};
    const py::ssize_t compared_count = field_count == 5 ? 5 : 7;
    for (py::ssize_t position = 0; position < compared_count; ++position) {
        const py::ssize_t index =
            compared_fields[static_cast<std::size_t>(position)];
        if (!stage052_object_equal(
                PyTuple_GET_ITEM(stored.key.ptr(), index),
                fields[index])) {
            return false;
        }
    }
    if (field_count == 5) {
        return true;
    }
    PyObject* stored_checks = PyTuple_GET_ITEM(stored.key.ptr(), 15);
    const py::ssize_t check_count = PyTuple_GET_SIZE(fields[15]);
    if (!PyTuple_Check(stored_checks)
        || PyTuple_GET_SIZE(stored_checks) != check_count) {
        return false;
    }
    for (py::ssize_t check_index = 0;
         check_index < check_count;
         ++check_index) {
        PyObject* previous_check =
            PyTuple_GET_ITEM(stored_checks, check_index);
        PyObject* current_check =
            PyTuple_GET_ITEM(fields[15], check_index);
        for (const py::ssize_t check_field : {0, 1, 2, 3}) {
            const auto [previous, owns_previous] =
                stage052_screening_check_field(
                    previous_check, check_field, cache);
            PyObject* current = nullptr;
            bool owns_current = false;
            try {
                const auto resolved_current =
                    stage052_screening_check_field(
                        current_check, check_field, cache);
                current = resolved_current.first;
                owns_current = resolved_current.second;
            } catch (...) {
                if (owns_previous) {
                    Py_DECREF(previous);
                }
                throw;
            }
            bool equal = false;
            try {
                equal = stage052_object_equal(previous, current);
            } catch (...) {
                if (owns_previous) {
                    Py_DECREF(previous);
                }
                if (owns_current) {
                    Py_DECREF(current);
                }
                throw;
            }
            if (owns_previous) {
                Py_DECREF(previous);
            }
            if (owns_current) {
                Py_DECREF(current);
            }
            if (!equal) {
                return false;
            }
        }
    }
    return true;
}

py::tuple stage052_screening_key_tuple(
    const std::array<PyObject*, 16>& fields,
    const py::ssize_t field_count) {
    py::tuple key(field_count);
    for (py::ssize_t index = 0; index < field_count; ++index) {
        key[index] = py::reinterpret_borrow<py::object>(fields[index]);
    }
    return key;
}

py::tuple stage052_route_sequence_from_key(const py::object& route_key) {
    if (!PyUnicode_Check(route_key.ptr())) {
        throw std::invalid_argument("Stage 5.2 canonical route key must be text");
    }
    const py::ssize_t key_length = PyUnicode_GetLength(route_key.ptr());
    if (key_length < 0) {
        throw py::error_already_set();
    }
    const py::str prefix("route:");
    const int has_prefix =
        PyUnicode_Tailmatch(route_key.ptr(), prefix.ptr(), 0, key_length, -1);
    if (has_prefix < 0) {
        throw py::error_already_set();
    }
    if (has_prefix == 0) {
        throw std::invalid_argument(
            "cannot reconstruct route dictionary entry from canonical key");
    }
    const py::object encoded = py::reinterpret_steal<py::object>(
        PyUnicode_Substring(route_key.ptr(), 6, key_length));
    if (!encoded) {
        throw py::error_already_set();
    }
    if (PyUnicode_GetLength(encoded.ptr()) == 0) {
        return py::tuple();
    }
    const py::str delimiter("|");
    const py::object raw_tokens = py::reinterpret_steal<py::object>(
        PyUnicode_Split(encoded.ptr(), delimiter.ptr(), -1));
    if (!raw_tokens) {
        throw py::error_already_set();
    }
    const py::list tokens = py::reinterpret_borrow<py::list>(raw_tokens);
    const auto token_count = static_cast<py::ssize_t>(py::len(tokens));
    py::tuple sequence(token_count);
    for (py::ssize_t index = 0; index < token_count; ++index) {
        const py::object token = tokens[index];
        const py::ssize_t token_length = PyUnicode_GetLength(token.ptr());
        const py::ssize_t separator =
            PyUnicode_FindChar(token.ptr(), ':', 0, token_length, 1);
        if (separator <= 0) {
            if (separator < 0 && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            throw std::invalid_argument("invalid canonical route key token");
        }
        std::size_t declared_length = 0;
        for (py::ssize_t digit_index = 0; digit_index < separator; ++digit_index) {
            const auto digit = PyUnicode_ReadChar(token.ptr(), digit_index);
            if (digit < '0' || digit > '9') {
                throw std::invalid_argument("invalid canonical route key length");
            }
            declared_length =
                declared_length * 10 + static_cast<std::size_t>(digit - '0');
        }
        const py::object customer = py::reinterpret_steal<py::object>(
            PyUnicode_Substring(token.ptr(), separator + 1, token_length));
        if (!customer) {
            throw py::error_already_set();
        }
        const py::ssize_t customer_length = PyUnicode_GetLength(customer.ptr());
        if (customer_length < 0) {
            throw py::error_already_set();
        }
        if (declared_length != static_cast<std::size_t>(customer_length)) {
            throw std::invalid_argument("canonical route key length mismatch");
        }
        sequence[index] = customer;
    }
    return sequence;
}

template <typename T>
py::array checked_array(py::handle array, const char* name, int expected_ndim) {
    if (!py::isinstance<py::array>(array)) {
        throw std::invalid_argument(std::string(name) + " must be a NumPy array");
    }
    auto value = py::reinterpret_borrow<py::array>(array);
    if (!py::dtype::of<T>().is(value.dtype())) {
        throw std::invalid_argument(std::string(name) + " must have the requested numeric dtype");
    }
    auto info = value.request();
    if (info.ndim != expected_ndim) {
        throw std::invalid_argument(std::string(name) + " has an unexpected number of dimensions");
    }
    if ((value.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    }
    return value;
}

template <typename T>
const T* checked_data(const py::array& array) {
    return static_cast<const T*>(array.request().ptr);
}

template <typename T>
T* checked_data(py::array_t<T>& array) {
    return static_cast<T*>(array.request().ptr);
}

template <typename T>
py::array_t<T> owned_array_copy(
    py::handle array,
    const char* name,
    int expected_ndim) {
    const auto source = checked_array<T>(array, name, expected_ndim);
    const auto info = source.request();
    py::array_t<T> output(info.shape);
    std::copy(
        checked_data<T>(source),
        checked_data<T>(source) + info.size,
        checked_data(output));
    return output;
}

double distance(const Point& first, const Point& second) {
    return std::hypot(first.first - second.first, first.second - second.second);
}

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

py::tuple pack_stage052_screening_occurrences(
    const py::sequence& events,
    const py::dict& definition_cache,
    py::dict negative_evidence_cache,
    const std::int64_t first_event_id) {
    py::list event_ids;
    py::list definition_ids;
    py::list started_at;
    py::list completed_at;
    py::list iterations;
    py::list decision_ids;
    py::list misses;
    py::list non_screening_indices;
    py::dict pending_indices;

    const py::ssize_t event_count = py::len(events);
    for (py::ssize_t event_index = 0; event_index < event_count; ++event_index) {
        const py::object event = events[event_index];
        if (!PyTuple_Check(event.ptr()) || PyTuple_GET_SIZE(event.ptr()) != 2) {
            non_screening_indices.append(event_index);
            continue;
        }
        const py::object axis_name =
            py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(event.ptr(), 0));
        const py::object raw_values =
            py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(event.ptr(), 1));
        if (!PyUnicode_Check(axis_name.ptr())) {
            non_screening_indices.append(event_index);
            continue;
        }
        if (!PyTuple_Check(raw_values.ptr()) || PyTuple_GET_SIZE(raw_values.ptr()) != 20) {
            throw std::invalid_argument(
                "Stage 5.2 deferred screening values must contain twenty fields");
        }
        const py::tuple values = py::reinterpret_borrow<py::tuple>(raw_values);
        py::tuple key(5);
        key[0] = values[1];
        key[1] = axis_name;
        key[2] = values[2];
        key[3] = values[4];
        if (values[15].ptr() == Py_True && !values[19].is_none()) {
            PyObject* cached_evidence =
                PyDict_GetItemWithError(negative_evidence_cache.ptr(), values[1].ptr());
            if (cached_evidence == nullptr) {
                if (PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                if (py::len(negative_evidence_cache) >= 262144) {
                    throw std::invalid_argument(
                        "Stage 5.2 native negative evidence cache exceeds its hard limit");
                }
                py::tuple evidence(12);
                for (py::ssize_t index = 0; index < 12; ++index) {
                    evidence[index] = values[7 + index];
                }
                negative_evidence_cache[values[1]] = std::move(evidence);
            } else {
                if (!PyTuple_Check(cached_evidence)
                    || PyTuple_GET_SIZE(cached_evidence) != 12) {
                    throw std::invalid_argument(
                        "Stage 5.2 native negative evidence cache is invalid");
                }
                bool all_evidence_fields_identical = true;
                for (py::ssize_t index = 0; index < 12; ++index) {
                    PyObject* previous = PyTuple_GET_ITEM(cached_evidence, index);
                    PyObject* current = values[7 + index].ptr();
                    if (previous == current) {
                        continue;
                    }
                    all_evidence_fields_identical = false;
                    const int equal = PyObject_RichCompareBool(previous, current, Py_EQ);
                    if (equal < 0) {
                        throw py::error_already_set();
                    }
                    if (equal == 0) {
                        throw std::invalid_argument(
                            "negative screening cache returned inconsistent evidence for one route");
                    }
                }
                if (!all_evidence_fields_identical) {
                    std::array<PyObject*, 16> previous_fields{};
                    std::array<PyObject*, 16> current_fields{};
                    for (py::ssize_t index = 0; index < 12; ++index) {
                        previous_fields[4 + index] =
                            PyTuple_GET_ITEM(cached_evidence, index);
                        current_fields[4 + index] = values[7 + index].ptr();
                    }
                    const Stage052CanonicalSignature previous_signature =
                        stage052_screening_key_signature(previous_fields, 16);
                    const Stage052CanonicalSignature current_signature =
                        stage052_screening_key_signature(current_fields, 16);
                    if (!stage052_signature_equal(
                            previous_signature, current_signature)) {
                        throw std::invalid_argument(
                            "negative screening cache returned inconsistent evidence for one route");
                    }
                }
            }
            key[4] = values[19];
        } else {
            py::tuple evidence(12);
            for (py::ssize_t index = 0; index < 12; ++index) {
                evidence[index] = values[7 + index];
            }
            key[4] = std::move(evidence);
        }

        const auto occurrence_index = py::len(event_ids);
        event_ids.append(first_event_id + event_index);
        started_at.append(values[5]);
        completed_at.append(values[6]);
        iterations.append(values[3]);
        decision_ids.append(values[0]);
        PyObject* cached = PyDict_GetItemWithError(definition_cache.ptr(), key.ptr());
        if (cached != nullptr) {
            if (!PyLong_Check(cached) || PyBool_Check(cached)) {
                throw std::invalid_argument(
                    "Stage 5.2 screening definition cache value must be an integer");
            }
            definition_ids.append(py::reinterpret_borrow<py::object>(cached));
            continue;
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        definition_ids.append(py::none());
        PyObject* pending = PyDict_GetItemWithError(pending_indices.ptr(), key.ptr());
        if (pending == nullptr) {
            if (PyErr_Occurred()) {
                throw py::error_already_set();
            }
            py::list indices;
            indices.append(occurrence_index);
            pending_indices[key] = indices;
            misses.append(py::make_tuple(key, event_index, indices));
        } else {
            py::reinterpret_borrow<py::list>(pending).append(occurrence_index);
        }
    }
    return py::make_tuple(
        py::make_tuple(
            std::move(event_ids),
            std::move(definition_ids),
            std::move(started_at),
            std::move(completed_at),
            std::move(iterations),
            std::move(decision_ids)),
        std::move(misses),
        std::move(non_screening_indices));
}

py::tuple pack_stage052_screening_transactions(
    const py::sequence& events,
    const py::object& definition_cache,
    py::dict negative_evidence_cache,
    py::dict lane_ids,
    py::dict operator_ids,
    py::dict route_ids,
    const py::object& resolve_route_id,
    const py::object& stable_dictionary_id,
    const py::object& definition_identity,
    const std::int64_t first_event_id) {
    auto* native_definition_cache =
        static_cast<Stage052ScreeningDefinitionCache*>(
            PyCapsule_GetPointer(
                definition_cache.ptr(), STAGE052_SCREENING_CACHE_CAPSULE));
    if (native_definition_cache == nullptr) {
        PyErr_Clear();
        throw std::invalid_argument(
            "Stage 5.2 native screening definition cache is invalid");
    }
    py::tuple columns(6);
    for (py::ssize_t index = 0; index < 6; ++index) {
        columns[index] = py::list();
    }
    auto event_ids = py::reinterpret_borrow<py::list>(columns[0]);
    auto definition_ids = py::reinterpret_borrow<py::list>(columns[1]);
    auto started_at = py::reinterpret_borrow<py::list>(columns[2]);
    auto completed_at = py::reinterpret_borrow<py::list>(columns[3]);
    auto iterations = py::reinterpret_borrow<py::list>(columns[4]);
    auto decision_ids = py::reinterpret_borrow<py::list>(columns[5]);
    py::list pending_definitions;
    py::list non_screening_indices;
    py::list observed_routes;

    const auto cached_dictionary_id = [&stable_dictionary_id](
                                          py::dict cache,
                                          const py::object& value,
                                          const char* prefix) -> py::object {
        PyObject* cached = PyDict_GetItemWithError(cache.ptr(), value.ptr());
        if (cached != nullptr) {
            return py::reinterpret_borrow<py::object>(cached);
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        const py::object identifier = stable_dictionary_id(py::str(prefix) + py::str(value));
        cache[value] = identifier;
        return identifier;
    };
    const auto checked_float = [](const py::object& value) -> py::object {
        const double converted = PyFloat_AsDouble(value.ptr());
        if (converted == -1.0 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        return py::float_(converted);
    };
    const py::ssize_t event_count = py::len(events);
    for (py::ssize_t event_index = 0; event_index < event_count; ++event_index) {
        const py::object event = events[event_index];
        if (!PyTuple_Check(event.ptr()) || PyTuple_GET_SIZE(event.ptr()) != 2) {
            non_screening_indices.append(event_index);
            continue;
        }
        const py::object axis_name =
            py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(event.ptr(), 0));
        const py::object raw_values =
            py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(event.ptr(), 1));
        if (!PyUnicode_Check(axis_name.ptr())) {
            non_screening_indices.append(event_index);
            continue;
        }
        if (!PyTuple_Check(raw_values.ptr()) || PyTuple_GET_SIZE(raw_values.ptr()) != 20) {
            throw std::invalid_argument(
                "Stage 5.2 deferred screening values must contain twenty fields");
        }
        const py::tuple values = py::reinterpret_borrow<py::tuple>(raw_values);
        std::array<PyObject*, 16> occurrence_key_fields{};
        occurrence_key_fields[0] = values[1].ptr();
        occurrence_key_fields[1] = axis_name.ptr();
        occurrence_key_fields[2] = values[2].ptr();
        occurrence_key_fields[3] = values[4].ptr();
        py::ssize_t occurrence_key_field_count = 16;
        if (values[15].ptr() == Py_True && !values[19].is_none()) {
            bool trusted_producer_token = false;
            if (PyLong_Check(values[19].ptr()) && !PyBool_Check(values[19].ptr())) {
                const unsigned long long token =
                    PyLong_AsUnsignedLongLong(values[19].ptr());
                if (token == static_cast<unsigned long long>(-1)
                    && PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                if (token == 0) {
                    throw std::invalid_argument(
                        "Stage 5.2 negative evidence token must be positive");
                }
                trusted_producer_token = true;
            } else if (
                PyTuple_Check(values[19].ptr())
                && PyTuple_GET_SIZE(values[19].ptr()) == 2) {
                PyObject* token = PyTuple_GET_ITEM(values[19].ptr(), 0);
                PyObject* signature = PyTuple_GET_ITEM(values[19].ptr(), 1);
                if (
                    PyLong_Check(token)
                    && !PyBool_Check(token)
                    && PyBytes_Check(signature)
                    && PyBytes_GET_SIZE(signature) > 0) {
                    const unsigned long long converted =
                        PyLong_AsUnsignedLongLong(token);
                    if (
                        converted == static_cast<unsigned long long>(-1)
                        && PyErr_Occurred()) {
                        throw py::error_already_set();
                    }
                    if (converted == 0) {
                        throw std::invalid_argument(
                            "Stage 5.2 negative evidence token must be positive");
                    }
                    trusted_producer_token = true;
                }
            }
            if (!trusted_producer_token) {
                PyObject* cached_evidence =
                    PyDict_GetItemWithError(negative_evidence_cache.ptr(), values[1].ptr());
                if (cached_evidence == nullptr) {
                    if (PyErr_Occurred()) {
                        throw py::error_already_set();
                    }
                    if (py::len(negative_evidence_cache) >= 262144) {
                        throw std::invalid_argument(
                            "Stage 5.2 native negative evidence cache exceeds its hard limit");
                    }
                    py::tuple evidence(12);
                    for (py::ssize_t index = 0; index < 12; ++index) {
                        evidence[index] = values[7 + index];
                    }
                    negative_evidence_cache[values[1]] = std::move(evidence);
                } else {
                    if (!PyTuple_Check(cached_evidence)
                        || PyTuple_GET_SIZE(cached_evidence) != 12) {
                        throw std::invalid_argument(
                            "Stage 5.2 native negative evidence cache is invalid");
                    }
                    for (py::ssize_t index = 0; index < 12; ++index) {
                        PyObject* previous = PyTuple_GET_ITEM(cached_evidence, index);
                        PyObject* current = values[7 + index].ptr();
                        if (previous == current) {
                            continue;
                        }
                        const int equal =
                            PyObject_RichCompareBool(previous, current, Py_EQ);
                        if (equal < 0) {
                            throw py::error_already_set();
                        }
                        if (equal == 0) {
                            throw std::invalid_argument(
                                "negative screening cache returned inconsistent evidence "
                                "for one route");
                        }
                    }
                    std::array<PyObject*, 16> previous_fields{};
                    std::array<PyObject*, 16> current_fields{};
                    for (py::ssize_t index = 0; index < 12; ++index) {
                        previous_fields[4 + index] =
                            PyTuple_GET_ITEM(cached_evidence, index);
                        current_fields[4 + index] = values[7 + index].ptr();
                    }
                    const Stage052CanonicalSignature previous_signature =
                        stage052_screening_key_signature(previous_fields, 16);
                    const Stage052CanonicalSignature current_signature =
                        stage052_screening_key_signature(current_fields, 16);
                    if (!stage052_signature_equal(
                            previous_signature, current_signature)) {
                        throw std::invalid_argument(
                            "negative screening cache returned inconsistent evidence "
                            "for one route");
                    }
                }
            }
            occurrence_key_fields[4] = values[19].ptr();
            occurrence_key_field_count = 5;
        } else {
            for (py::ssize_t index = 0; index < 12; ++index) {
                occurrence_key_fields[4 + index] = values[7 + index].ptr();
            }
        }
        const Stage052CanonicalSignature occurrence_key_signature =
            stage052_screening_key_signature(
                occurrence_key_fields,
                occurrence_key_field_count,
                native_definition_cache);
        const std::size_t occurrence_key_hash = stage052_screening_key_hash(
            occurrence_key_fields,
            occurrence_key_field_count,
            occurrence_key_signature,
            native_definition_cache);

        event_ids.append(first_event_id + event_index);
        started_at.append(values[5]);
        completed_at.append(values[6]);
        iterations.append(values[3]);
        decision_ids.append(values[0]);
        const auto cached_range =
            native_definition_cache->definitions.equal_range(occurrence_key_hash);
        const Stage052ScreeningDefinitionCacheEntry* cached_definition = nullptr;
        for (auto cached = cached_range.first; cached != cached_range.second; ++cached) {
            if (stage052_screening_key_equal(
                    cached->second,
                    occurrence_key_fields,
                    occurrence_key_field_count,
                    occurrence_key_signature,
                    native_definition_cache)) {
                cached_definition = &cached->second;
                break;
            }
        }
        if (cached_definition != nullptr) {
            if (!PyLong_Check(cached_definition->definition_id.ptr())
                || PyBool_Check(cached_definition->definition_id.ptr())) {
                throw std::invalid_argument(
                    "Stage 5.2 native screening cache value is invalid");
            }
            definition_ids.append(cached_definition->definition_id);
            continue;
        }

        const py::object raw_checks = values[18];
        if (!PyTuple_Check(raw_checks.ptr())) {
            throw std::invalid_argument("Stage 5.2 deferred screening checks are invalid");
        }
        const py::tuple checks = py::reinterpret_borrow<py::tuple>(raw_checks);
        if (py::len(checks) > 8) {
            throw std::invalid_argument(
                "one screening decision exceeds the fixed eight-check domain");
        }
        py::list check_payloads;
        const auto check_count = static_cast<py::ssize_t>(py::len(checks));
        for (py::ssize_t check_index = 0; check_index < check_count; ++check_index) {
            const py::object check = checks[check_index];
            py::object name;
            py::object status;
            py::object value;
            py::object reason;
            if (PyTuple_Check(check.ptr()) && PyTuple_GET_SIZE(check.ptr()) == 4) {
                name = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(check.ptr(), 0));
                status = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(check.ptr(), 1));
                value = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(check.ptr(), 2));
                reason = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(check.ptr(), 3));
            } else {
                name = check.attr("check");
                status = check.attr("status");
                value = check.attr("value");
                reason = check.attr("reason");
            }
            py::object value_bool = py::none();
            py::object value_float = py::none();
            py::object value_text = py::none();
            if (PyBool_Check(value.ptr())) {
                value_bool = value;
            } else if (PyLong_Check(value.ptr()) || PyFloat_Check(value.ptr())) {
                value_float = checked_float(value);
            } else if (PyUnicode_Check(value.ptr())) {
                value_text = value;
            }
            py::dict payload;
            payload["check"] = name;
            payload["status"] = status;
            payload["value_bool"] = value_bool;
            payload["value_float"] = value_float;
            payload["value_text"] = value_text;
            payload["reason"] = reason;
            check_payloads.append(std::move(payload));
        }

        const py::str lane = py::str(axis_name) + py::str(":") + py::str(values[2]);
        const py::object lane_id =
            cached_dictionary_id(lane_ids, lane, "lane:");
        const py::object operator_id =
            cached_dictionary_id(operator_ids, values[4], "operator:");
        PyObject* cached_route = PyDict_GetItemWithError(route_ids.ptr(), values[1].ptr());
        py::object route_id;
        bool route_was_resolved = false;
        if (cached_route != nullptr) {
            route_id = py::reinterpret_borrow<py::object>(cached_route);
        } else {
            if (PyErr_Occurred()) {
                throw py::error_already_set();
            }
            route_id = resolve_route_id(values[1]);
            route_was_resolved = true;
        }
        if (route_was_resolved) {
            observed_routes.append(
                py::make_tuple(
                    values[1],
                    route_id,
                    stage052_route_sequence_from_key(values[1])));
        }

        const py::object demand = checked_float(values[9]);
        const py::object distance_increment_lower_bound =
            values[10].is_none() ? py::none() : checked_float(values[10]);
        const py::object distance_lower_bound = checked_float(values[11]);
        const py::object min_time_window_slack = checked_float(values[14]);
        const py::object structural_energy_lower_bound = checked_float(values[17]);
        py::dict definition;
        definition["lane_id"] = lane_id;
        definition["operator_id"] = operator_id;
        definition["route_id"] = route_id;
        definition["status"] = values[7];
        definition["reason"] = values[8];
        definition["benchmark_axis"] = axis_name;
        definition["demand"] = demand;
        definition["distance_increment_lower_bound"] = distance_increment_lower_bound;
        definition["distance_lower_bound"] = distance_lower_bound;
        definition["exact_call_blocked"] = values[12];
        definition["first_failed_check"] = values[13];
        definition["min_time_window_slack"] = min_time_window_slack;
        definition["negative_cache_hit"] = values[15];
        definition["single_segment_reachable"] = values[16];
        definition["structural_energy_lower_bound"] = structural_energy_lower_bound;
        definition["checks"] = check_payloads;
        const py::tuple identity =
            py::cast<py::tuple>(definition_identity(definition));
        if (py::len(identity) != 3) {
            throw std::invalid_argument(
                "Stage 5.2 screening definition identity is invalid");
        }
        const py::object definition_id = identity[0];
        if (native_definition_cache->definitions.size()
            >= native_definition_cache->capacity) {
            if (native_definition_cache->insertion_order.empty()) {
                throw std::invalid_argument(
                    "Stage 5.2 screening definition cache is unexpectedly empty");
            }
            const auto oldest = native_definition_cache->insertion_order.front();
            native_definition_cache->insertion_order.pop_front();
            const auto oldest_range =
                native_definition_cache->definitions.equal_range(oldest.first);
            bool removed = false;
            for (auto candidate = oldest_range.first;
                 candidate != oldest_range.second;
                 ++candidate) {
                if (&candidate->second == oldest.second) {
                    native_definition_cache->definitions.erase(candidate);
                    removed = true;
                    break;
                }
            }
            if (!removed) {
                throw std::invalid_argument(
                    "Stage 5.2 screening definition cache eviction is invalid");
            }
        }
        const auto inserted = native_definition_cache->definitions.emplace(
            occurrence_key_hash,
            Stage052ScreeningDefinitionCacheEntry{
                stage052_screening_key_tuple(
                    occurrence_key_fields, occurrence_key_field_count),
                occurrence_key_signature,
                definition_id,
            });
        try {
            native_definition_cache->insertion_order.emplace_back(
                occurrence_key_hash, &inserted->second);
        } catch (...) {
            native_definition_cache->definitions.erase(inserted);
            throw;
        }
        definition_ids.append(definition_id);
        py::tuple row(17);
        row[0] = definition_id;
        row[1] = lane_id;
        row[2] = operator_id;
        row[3] = route_id;
        row[4] = values[7];
        row[5] = values[8];
        row[6] = axis_name;
        row[7] = demand;
        row[8] = distance_increment_lower_bound;
        row[9] = distance_lower_bound;
        row[10] = values[12];
        row[11] = values[13];
        row[12] = min_time_window_slack;
        row[13] = values[15];
        row[14] = values[16];
        row[15] = structural_energy_lower_bound;
        row[16] = std::move(check_payloads);
        pending_definitions.append(
            py::make_tuple(
                definition_id,
                identity[1],
                identity[2],
                std::move(definition),
                std::move(row)));
    }
    return py::make_tuple(
        std::move(columns),
        std::move(pending_definitions),
        std::move(non_screening_indices),
        std::move(observed_routes));
}

py::tuple pack_stage052_neighborhood_events(
    const py::sequence& events,
    py::dict lane_ids,
    py::dict operator_ids,
    py::dict extras_cache,
    const py::object& allowed_fields,
    const py::object& missing_extra,
    const py::object& stable_dictionary_id,
    const py::object& json_text,
    const std::int64_t first_event_id) {
    py::tuple columns(36);
    for (py::ssize_t index = 0; index < 36; ++index) {
        columns[index] = py::list();
    }
    py::list non_neighborhood_indices;
    py::list empty_list;
    py::object none = py::none();
    const std::array<const char*, 21> extra_fields = {
        "aggregate_count",
        "benchmark_axis",
        "candidate_objective_key",
        "candidate_pool_hash",
        "chain_depth",
        "constraint_category",
        "distance_improvement",
        "exact_route_evaluations",
        "new_routes_created",
        "prefilter_passed",
        "ranking_score",
        "removal_size_actual",
        "removal_size_requested",
        "removal_tier",
        "removal_trigger",
        "reset_observed",
        "segment_length",
        "selection_rank",
        "stagnation_iterations",
        "track",
        "vehicle_reduction",
    };
    const auto append = [&columns](const py::ssize_t column, const py::object& value) {
        py::reinterpret_borrow<py::list>(columns[column]).append(value);
    };
    const auto get = [](PyObject* mapping, const char* key) -> PyObject* {
        return PyDict_GetItemString(mapping, key);
    };
    const auto string_value = [](PyObject* value) -> py::object {
        if (value == nullptr) {
            return py::str("");
        }
        return py::str(py::reinterpret_borrow<py::object>(value));
    };
    const auto optional_float = [&none](PyObject* value) -> py::object {
        if (value == nullptr || value == Py_None || PyBool_Check(value)) {
            return none;
        }
        if (PyFloat_Check(value) || PyLong_Check(value)) {
            const double converted = PyFloat_AsDouble(value);
            if (converted == -1.0 && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            return py::float_(converted);
        }
        const py::str text(py::reinterpret_borrow<py::object>(value));
        PyObject* converted = PyFloat_FromString(text.ptr());
        if (converted != nullptr) {
            return py::reinterpret_steal<py::object>(converted);
        }
        PyErr_Clear();
        return none;
    };
    const auto optional_int = [&none](PyObject* value) -> py::object {
        if (value == nullptr || value == Py_None || PyBool_Check(value)) {
            return none;
        }
        if (PyLong_Check(value)) {
            return py::reinterpret_borrow<py::object>(value);
        }
        const py::str text(py::reinterpret_borrow<py::object>(value));
        PyObject* converted = PyLong_FromUnicodeObject(text.ptr(), 10);
        if (converted != nullptr) {
            return py::reinterpret_steal<py::object>(converted);
        }
        PyErr_Clear();
        return none;
    };
    const auto optional_bool = [&none](PyObject* value) -> py::object {
        if (value == Py_True || value == Py_False) {
            return py::reinterpret_borrow<py::object>(value);
        }
        return none;
    };
    const auto dictionary_id = [&stable_dictionary_id](
                                   py::dict& dictionary,
                                   const py::object& text,
                                   const char* prefix) -> py::object {
        PyObject* cached = PyDict_GetItemWithError(dictionary.ptr(), text.ptr());
        if (cached != nullptr) {
            return py::reinterpret_borrow<py::object>(cached);
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        const py::object value = stable_dictionary_id(
            py::str(std::string(prefix) + py::cast<std::string>(text)));
        dictionary[text] = value;
        return value;
    };

    const py::ssize_t event_count = py::len(events);
    for (py::ssize_t event_index = 0; event_index < event_count; ++event_index) {
        const py::object event = events[event_index];
        if (!PyDict_Check(event.ptr())) {
            non_neighborhood_indices.append(event_index);
            continue;
        }
        PyObject* event_type = get(event.ptr(), "event_type");
        PyObject* record_type = get(event.ptr(), "record_type");
        const bool is_neighborhood =
            (event_type != nullptr && PyUnicode_Check(event_type)
             && PyUnicode_CompareWithASCIIString(event_type, "neighborhood_event") == 0)
            || (event_type == nullptr && record_type != nullptr
                && PyUnicode_Check(record_type)
                && PyUnicode_CompareWithASCIIString(record_type, "neighborhood_event") == 0);
        if (!is_neighborhood) {
            non_neighborhood_indices.append(event_index);
            continue;
        }
        PyObject* key = nullptr;
        PyObject* value = nullptr;
        Py_ssize_t position = 0;
        bool supported = true;
        while (PyDict_Next(event.ptr(), &position, &key, &value)) {
            const int allowed = PySet_Contains(allowed_fields.ptr(), key);
            if (allowed < 0) {
                throw py::error_already_set();
            }
            if (allowed == 0) {
                non_neighborhood_indices.append(event_index);
                supported = false;
                break;
            }
        }
        if (!supported) {
            continue;
        }

        py::tuple extra_values(extra_fields.size());
        py::dict extras;
        for (std::size_t index = 0; index < extra_fields.size(); ++index) {
            PyObject* raw = get(event.ptr(), extra_fields[index]);
            if (raw == nullptr) {
                extra_values[index] = missing_extra;
            } else {
                const py::object borrowed = py::reinterpret_borrow<py::object>(raw);
                extra_values[index] = borrowed;
                extras[py::str(extra_fields[index])] = borrowed;
            }
        }
        py::object extras_json;
        PyObject* cached_extras =
            PyDict_GetItemWithError(extras_cache.ptr(), extra_values.ptr());
        if (cached_extras != nullptr) {
            extras_json = py::reinterpret_borrow<py::object>(cached_extras);
        } else {
            if (PyErr_Occurred()) {
                PyErr_Clear();
            }
            extras_json = py::len(extras) == 0 ? py::str("") : json_text(extras);
            if (PyObject_Hash(extra_values.ptr()) != -1) {
                extras_cache[extra_values] = extras_json;
                if (py::len(extras_cache) > 65536) {
                    const py::object iterator = py::iter(extras_cache);
                    PyObject* first = PyIter_Next(iterator.ptr());
                    if (first == nullptr) {
                        if (PyErr_Occurred()) {
                            throw py::error_already_set();
                        }
                        throw std::runtime_error(
                            "Stage 5.2 neighborhood extras cache is unexpectedly empty");
                    }
                    const py::object first_key =
                        py::reinterpret_steal<py::object>(first);
                    extras_cache.attr("pop")(first_key);
                }
            } else {
                PyErr_Clear();
            }
        }

        const py::object lane = string_value(get(event.ptr(), "lane"));
        const py::object operator_name = string_value(get(event.ptr(), "operator"));
        PyObject* timestamp = get(event.ptr(), "timestamp_seconds");
        if (timestamp == nullptr || (!PyFloat_Check(timestamp) && !PyLong_Check(timestamp))) {
            timestamp = get(event.ptr(), "started_at");
        }
        if (timestamp == nullptr || (!PyFloat_Check(timestamp) && !PyLong_Check(timestamp))) {
            timestamp = get(event.ptr(), "timestamp");
        }
        py::object timestamp_value = none;
        if (timestamp != nullptr && (PyFloat_Check(timestamp) || PyLong_Check(timestamp))) {
            const double converted = PyFloat_AsDouble(timestamp);
            if (converted == -1.0 && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            timestamp_value = py::float_(converted);
        }
        PyObject* status = get(event.ptr(), "status");
        append(0, py::int_(first_event_id + event_index));
        append(
            1,
            record_type == nullptr
                ? py::str("neighborhood_event")
                : string_value(record_type));
        append(2, py::str("neighborhood_event"));
        append(3, timestamp_value);
        append(4, optional_float(get(event.ptr(), "started_at")));
        append(5, optional_float(get(event.ptr(), "completed_at")));
        append(6, optional_float(get(event.ptr(), "duration_seconds")));
        append(7, dictionary_id(lane_ids, lane, "lane:"));
        append(8, optional_int(get(event.ptr(), "iteration")));
        append(9, dictionary_id(operator_ids, operator_name, "operator:"));
        append(10, none);
        append(11, empty_list);
        append(12, empty_list);
        append(13, empty_list);
        append(14, none);
        append(15, none);
        append(16, string_value(status));
        append(17, string_value(get(event.ptr(), "kind")));
        append(18, string_value(get(event.ptr(), "operation")));
        append(19, string_value(get(event.ptr(), "reason")));
        append(20, string_value(get(event.ptr(), "failure_reason")));
        append(21, optional_bool(get(event.ptr(), "feasible")));
        append(22, optional_bool(get(event.ptr(), "exact_started")));
        append(23, optional_bool(get(event.ptr(), "exact_completed")));
        append(24, optional_bool(get(event.ptr(), "candidate_feasible")));
        append(25, optional_bool(get(event.ptr(), "accepted")));
        append(26, optional_bool(get(event.ptr(), "global_best")));
        append(27, optional_int(get(event.ptr(), "current_vehicle_count")));
        append(28, optional_int(get(event.ptr(), "candidate_vehicle_count")));
        append(29, optional_int(get(event.ptr(), "candidate_vehicle_delta")));
        append(30, string_value(get(event.ptr(), "cache_key_digest")));
        append(31, optional_int(get(event.ptr(), "evaluation_id")));
        append(32, optional_int(get(event.ptr(), "decision_id")));
        append(33, string_value(get(event.ptr(), "route_change_status")));
        append(
            34,
            get(event.ptr(), "propagation_status") == nullptr
                ? string_value(status)
                : string_value(get(event.ptr(), "propagation_status")));
        append(35, extras_json);
    }
    return py::make_tuple(
        std::move(columns),
        std::move(non_neighborhood_indices));
}

py::tuple pack_stage052_deferred_sparse_events(
    const py::sequence& events,
    py::dict route_ids,
    py::dict lane_ids,
    py::dict operator_ids,
    py::dict route_evaluation_extras_cache,
    py::dict cache_event_extras_cache,
    const py::object& resolve_route_id,
    const py::object& stable_dictionary_id,
    const py::object& json_text,
    const std::int64_t first_event_id) {
    py::tuple columns(36);
    for (py::ssize_t index = 0; index < 36; ++index) {
        columns[index] = py::list();
    }
    py::list remaining;
    py::list observed_routes;
    py::dict observed_route_keys;
    py::list empty_list;
    py::object none = py::none();

    const auto append = [&columns](const py::ssize_t column, const py::object& value) {
        py::reinterpret_borrow<py::list>(columns[column]).append(value);
    };
    const auto dictionary_id = [&stable_dictionary_id](
                                   py::dict& dictionary,
                                   const py::object& value,
                                   const char* prefix) -> py::object {
        PyObject* cached = PyDict_GetItemWithError(dictionary.ptr(), value.ptr());
        if (cached != nullptr) {
            return py::reinterpret_borrow<py::object>(cached);
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        const py::object identifier = stable_dictionary_id(
            py::str(std::string(prefix) + py::cast<std::string>(value)));
        dictionary[value] = identifier;
        return identifier;
    };
    const auto route_id = [
                              &route_ids,
                              &resolve_route_id,
                              &observed_routes,
                              &observed_route_keys
                          ](const py::object& key) -> py::object {
        PyObject* cached = PyDict_GetItemWithError(route_ids.ptr(), key.ptr());
        py::object identifier;
        if (cached != nullptr) {
            identifier = py::reinterpret_borrow<py::object>(cached);
        } else {
            if (PyErr_Occurred()) {
                throw py::error_already_set();
            }
            identifier = resolve_route_id(key);
        }
        if (!PyLong_Check(identifier.ptr()) || PyBool_Check(identifier.ptr())) {
            throw std::invalid_argument(
                "Stage 5.2 deferred sparse route ID must be an integer");
        }
        const int already_observed =
            PyDict_Contains(observed_route_keys.ptr(), key.ptr());
        if (already_observed < 0) {
            throw py::error_already_set();
        }
        if (already_observed == 0) {
            observed_route_keys[key] = py::none();
            observed_routes.append(
                py::make_tuple(
                    key,
                    identifier,
                    stage052_route_sequence_from_key(key)));
        }
        return identifier;
    };
    const auto cached_json = [&json_text](
                                 py::dict& cache,
                                 const py::tuple& key,
                                 py::dict extras) -> py::object {
        PyObject* cached = PyDict_GetItemWithError(cache.ptr(), key.ptr());
        if (cached != nullptr) {
            return py::reinterpret_borrow<py::object>(cached);
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        const py::object encoded = json_text(extras);
        cache[key] = encoded;
        if (py::len(cache) > 65536) {
            const py::object iterator = py::iter(cache);
            PyObject* first = PyIter_Next(iterator.ptr());
            if (first == nullptr) {
                if (PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                throw std::runtime_error(
                    "Stage 5.2 deferred sparse extras cache is unexpectedly empty");
            }
            const py::object first_key = py::reinterpret_steal<py::object>(first);
            cache.attr("pop")(first_key);
        }
        return encoded;
    };
    const auto append_common_tail = [
                                        &append,
                                        &empty_list,
                                        &none
                                    ](
                                        const py::object& status,
                                        const py::object& kind,
                                        const py::object& operation,
                                        const py::object& failure_reason,
                                        const py::object& feasible,
                                        const py::object& exact_started,
                                        const py::object& exact_completed,
                                        const py::object& cache_key_digest,
                                        const py::object& evaluation_id,
                                        const py::object& route_change_status,
                                        const py::object& extras_json) {
        append(11, empty_list);
        append(12, empty_list);
        append(13, empty_list);
        append(14, none);
        append(15, none);
        append(16, status);
        append(17, kind);
        append(18, operation);
        append(19, py::str(""));
        append(20, failure_reason);
        append(21, feasible);
        append(22, exact_started);
        append(23, exact_completed);
        append(24, none);
        append(25, none);
        append(26, none);
        append(27, none);
        append(28, none);
        append(29, none);
        append(30, cache_key_digest);
        append(31, evaluation_id);
        append(32, none);
        append(33, route_change_status);
        append(34, status);
        append(35, extras_json);
    };

    const py::ssize_t event_count = py::len(events);
    for (py::ssize_t event_index = 0; event_index < event_count; ++event_index) {
        const py::object event = events[event_index];
        if (!PyTuple_Check(event.ptr())) {
            if (PyDict_Check(event.ptr())) {
                PyObject* event_type = PyDict_GetItemString(event.ptr(), "event_type");
                if (event_type != nullptr && PyUnicode_Check(event_type)
                    && PyUnicode_CompareWithASCIIString(
                           event_type, "screening_decision") == 0) {
                    continue;
                }
            }
            const auto position = py::len(py::reinterpret_borrow<py::list>(columns[0]));
            for (py::ssize_t column = 0; column < 36; ++column) {
                append(column, none);
            }
            remaining.append(py::make_tuple(event_index, position));
            continue;
        }
        const auto tuple_size = PyTuple_GET_SIZE(event.ptr());
        if (tuple_size == 2 || tuple_size == 8) {
            continue;
        }
        if (tuple_size != 3) {
            const auto position = py::len(py::reinterpret_borrow<py::list>(columns[0]));
            for (py::ssize_t column = 0; column < 36; ++column) {
                append(column, none);
            }
            remaining.append(py::make_tuple(event_index, position));
            continue;
        }
        const py::object marker =
            py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(event.ptr(), 0));
        const py::object axis_name =
            py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(event.ptr(), 1));
        const py::object raw_values =
            py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(event.ptr(), 2));
        if (!PyUnicode_Check(marker.ptr()) || !PyUnicode_Check(axis_name.ptr())) {
            throw std::invalid_argument(
                "Stage 5.2 deferred sparse event marker and axis must be text");
        }
        if (!PyTuple_Check(raw_values.ptr())) {
            throw std::invalid_argument(
                "Stage 5.2 deferred sparse event values must be a tuple");
        }
        const py::tuple values = py::reinterpret_borrow<py::tuple>(raw_values);
        const bool is_route_evaluation =
            PyUnicode_CompareWithASCIIString(marker.ptr(), "route_evaluation") == 0;
        const bool is_cache_event =
            PyUnicode_CompareWithASCIIString(marker.ptr(), "cache_event") == 0;
        if (!is_route_evaluation && !is_cache_event) {
            const auto position = py::len(py::reinterpret_borrow<py::list>(columns[0]));
            for (py::ssize_t column = 0; column < 36; ++column) {
                append(column, none);
            }
            remaining.append(py::make_tuple(event_index, position));
            continue;
        }
        const auto value_count = py::len(values);
        if ((is_route_evaluation && value_count != 20)
            || (is_cache_event && value_count != 18 && value_count != 20)) {
            throw std::invalid_argument(
                "Stage 5.2 deferred sparse event has an invalid field count");
        }

        const py::object key = values[is_route_evaluation ? 1 : 0];
        const py::object raw_lane = values[is_route_evaluation ? 2 : 1];
        const py::object operator_name = values[is_route_evaluation ? 4 : 3];
        if (!PyUnicode_Check(key.ptr()) || !PyUnicode_Check(raw_lane.ptr())
            || !PyUnicode_Check(operator_name.ptr())) {
            throw std::invalid_argument(
                "Stage 5.2 deferred sparse route, lane, and operator must be text");
        }
        py::object lane = raw_lane;
        if (is_route_evaluation) {
            lane = py::str(
                py::cast<std::string>(axis_name) + ":" + py::cast<std::string>(raw_lane));
        }
        py::object identifier = none;
        if (is_route_evaluation || py::len(py::reinterpret_borrow<py::str>(key)) > 0) {
            identifier = route_id(key);
        }
        const py::object lane_identifier =
            dictionary_id(lane_ids, lane, "lane:");
        const py::object operator_identifier =
            dictionary_id(operator_ids, operator_name, "operator:");

        append(0, py::int_(first_event_id + event_index));
        append(1, py::str(is_route_evaluation ? "route_evaluation" : "cache_event"));
        append(2, py::str(is_route_evaluation ? "route_evaluation" : "cache_event"));
        if (is_route_evaluation) {
            py::tuple extras_key(5);
            extras_key[0] = axis_name;
            extras_key[1] = values[16];
            extras_key[2] = values[14];
            extras_key[3] = values[13];
            extras_key[4] = values[15];
            py::dict extras;
            extras["benchmark_axis"] = axis_name;
            extras["deadline_boundary"] = values[16];
            extras["labels_expanded"] = values[14];
            extras["labels_generated"] = values[13];
            extras["labels_pruned"] = values[15];
            const py::object extras_json = cached_json(
                route_evaluation_extras_cache, extras_key, std::move(extras));
            append(3, values[6]);
            append(4, values[6]);
            append(5, values[7]);
            append(6, values[8]);
            append(7, lane_identifier);
            append(8, values[3]);
            append(9, operator_identifier);
            append(10, identifier);
            append_common_tail(
                values[19],
                values[5],
                py::str(""),
                values[12],
                values[11],
                values[9],
                values[10],
                values[17],
                values[0],
                values[18],
                extras_json);
            continue;
        }

        const auto extras_presence_index = value_count - 1;
        if (PyBool_Check(values[extras_presence_index].ptr())
            || !PyLong_Check(values[extras_presence_index].ptr())) {
            throw std::invalid_argument(
                "Stage 5.2 deferred cache extras presence must be an integer");
        }
        const auto extras_presence =
            PyLong_AsLongLong(values[extras_presence_index].ptr());
        if (extras_presence == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        const auto extra_count = extras_presence_index - 11;
        py::tuple extras_key(2 + extra_count);
        extras_key[0] = axis_name;
        extras_key[1] = values[extras_presence_index];
        for (std::size_t index = 0; index < extra_count; ++index) {
            extras_key[2 + index] = values[11 + index];
        }
        py::dict extras;
        extras["benchmark_axis"] = axis_name;
        const std::array<const char*, 8> extra_names = {
            "current_bytes",
            "current_entries",
            "entry_bytes",
            "lookup_current_bytes",
            "lookup_current_entries",
            "lookup_result",
            "pending_result_digest",
            "existing_result_digest",
        };
        for (std::size_t index = 0; index < extra_count; ++index) {
            if ((extras_presence & (std::int64_t{1} << index)) != 0) {
                extras[py::str(extra_names[static_cast<std::size_t>(index)])] =
                    values[11 + index];
            }
        }
        const py::object extras_json = cached_json(
            cache_event_extras_cache, extras_key, std::move(extras));
        append(3, values[4]);
        append(4, values[5]);
        append(5, values[6]);
        append(6, values[7]);
        append(7, lane_identifier);
        append(8, values[2]);
        append(9, operator_identifier);
        append(10, identifier);
        append_common_tail(
            values[8],
            py::str(""),
            values[9],
            py::str(""),
            none,
            none,
            none,
            values[10],
            none,
            py::str(""),
            extras_json);
    }
    return py::make_tuple(
        std::move(columns),
        std::move(remaining),
        std::move(observed_routes));
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

using evrptw::native_kernels::customer_kind;
using evrptw::native_kernels::depot_kind;
using evrptw::native_kernels::exact_epsilon;
using evrptw::native_kernels::screen_reason_legacy_energy;
using evrptw::native_kernels::screen_reason_structure;
using evrptw::native_kernels::station_kind;

py::tuple exact_charging_batch_numeric(
    py::handle node_kind,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle vehicle,
    py::handle order_offsets,
    py::handle order_indices,
    py::handle deadline_remaining,
    py::handle batch_size) {
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto ready_array = checked_array<double>(ready_time, "ready_time", 1);
    auto due_array = checked_array<double>(due_date, "due_date", 1);
    auto service_array = checked_array<double>(service_time, "service_time", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto vehicle_array = checked_array<double>(vehicle, "vehicle", 1);
    auto offsets_array = checked_array<std::int64_t>(order_offsets, "order_offsets", 1);
    auto indices_array = checked_array<std::int64_t>(order_indices, "order_indices", 1);
    auto deadline_array = checked_array<double>(deadline_remaining, "deadline_remaining", 1);
    auto batch_size_array = checked_array<std::int64_t>(batch_size, "batch_size", 1);
    const auto kind_info = kind_array.request();
    const auto ready_info = ready_array.request();
    const auto due_info = due_array.request();
    const auto service_info = service_array.request();
    const auto distance_info = distance_array.request();
    const auto vehicle_info = vehicle_array.request();
    const auto offsets_info = offsets_array.request();
    const auto indices_info = indices_array.request();
    const auto deadline_info = deadline_array.request();
    const auto batch_size_info = batch_size_array.request();
    const auto node_count = static_cast<std::size_t>(kind_info.shape[0]);
    if (node_count == 0) {
        throw std::invalid_argument("node arrays must not be empty");
    }
    if (ready_info.shape[0] != kind_info.shape[0] || due_info.shape[0] != kind_info.shape[0]
        || service_info.shape[0] != kind_info.shape[0]) {
        throw std::invalid_argument("node metadata arrays must share one length");
    }
    if (distance_info.shape[0] != kind_info.shape[0]
        || distance_info.shape[1] != kind_info.shape[0]) {
        throw std::invalid_argument("distance must have shape (n, n)");
    }
    if (vehicle_info.shape[0] != 5) {
        throw std::invalid_argument("vehicle must have shape (5,)");
    }
    if (offsets_info.shape[0] == 0) {
        throw std::invalid_argument("order_offsets must contain at least the initial zero");
    }
    if (deadline_info.shape[0] != 1) {
        throw std::invalid_argument("deadline_remaining must have shape (1,)");
    }
    if (batch_size_info.shape[0] != 1) {
        throw std::invalid_argument("batch_size must have shape (1,)");
    }

    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* ready = checked_data<double>(ready_array);
    const auto* due = checked_data<double>(due_array);
    const auto* service = checked_data<double>(service_array);
    const auto* distances = checked_data<double>(distance_array);
    const auto* vehicle_values = checked_data<double>(vehicle_array);
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* indices = checked_data<std::int64_t>(indices_array);
    const auto* deadline = checked_data<double>(deadline_array);
    const auto* batch = checked_data<std::int64_t>(batch_size_array);
    if (batch[0] <= 0) {
        throw std::invalid_argument("batch_size must be positive");
    }
    if (std::isnan(deadline[0])) {
        throw std::invalid_argument("deadline_remaining must not be NaN");
    }
    if (!std::isfinite(vehicle_values[0]) || vehicle_values[0] < 0.0
        || !std::isfinite(vehicle_values[2]) || vehicle_values[2] < 0.0
        || !std::isfinite(vehicle_values[3]) || vehicle_values[3] < 0.0
        || !std::isfinite(vehicle_values[4]) || vehicle_values[4] <= 0.0) {
        throw std::invalid_argument("vehicle contains invalid exact-charging parameters");
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> stations;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (kinds[node] != depot_kind && kinds[node] != customer_kind
            && kinds[node] != station_kind) {
            throw std::invalid_argument("node_kind contains an unknown code");
        }
        if (kinds[node] == depot_kind) {
            if (depot >= 0) {
                throw std::invalid_argument("node_kind must contain exactly one depot");
            }
            depot = static_cast<std::int64_t>(node);
        } else if (kinds[node] == station_kind) {
            stations.push_back(static_cast<std::int64_t>(node));
        }
        if (!std::isfinite(ready[node]) || !std::isfinite(due[node])
            || !std::isfinite(service[node])) {
            throw std::invalid_argument("node metadata must contain finite values");
        }
        for (std::size_t destination = 0; destination < node_count; ++destination) {
            const auto leg = distances[node * node_count + destination];
            if (!std::isfinite(leg) || leg < 0.0) {
                throw std::invalid_argument("distance must contain finite non-negative values");
            }
        }
    }
    if (depot < 0) {
        throw std::invalid_argument("node_kind must contain exactly one depot");
    }
    const auto route_count = static_cast<std::size_t>(offsets_info.shape[0] - 1);
    if (offsets[0] != 0
        || offsets[route_count] != static_cast<std::int64_t>(indices_info.shape[0])) {
        throw std::invalid_argument("order_offsets must span order_indices exactly");
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] > offsets[route + 1]) {
            throw std::invalid_argument("order_offsets must be monotone");
        }
        std::vector<bool> seen(node_count, false);
        for (auto position = offsets[route]; position < offsets[route + 1]; ++position) {
            const auto node = indices[position];
            if (node < 0 || static_cast<std::size_t>(node) >= node_count
                || kinds[node] != customer_kind) {
                throw std::invalid_argument("order_indices must contain customer node indices");
            }
            if (seen[static_cast<std::size_t>(node)]) {
                throw std::invalid_argument("each customer order must be duplicate-free");
            }
            seen[static_cast<std::size_t>(node)] = true;
        }
    }

    evrptw::native_kernels::ExactBatchOutput result;
    {
        py::gil_scoped_release release;
#ifdef __linux__
        if (native_kernel_scheduler_required
            && native_kernel_scheduler_endpoint.empty()) {
            throw std::runtime_error(
                "host scheduler exact kernel lost its required endpoint");
        }
        if (!native_kernel_scheduler_endpoint.empty()) {
            try {
                result = evrptw::native_client::exact_charging(
                    native_kernel_scheduler_endpoint,
                    kinds, ready, due, service, distances, vehicle_values,
                    offsets, indices, node_count, route_count,
                    static_cast<std::size_t>(indices_info.shape[0]),
                    deadline[0], batch[0]);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    std::string("host scheduler IPC failed without fallback: ")
                    + error.what());
            }
        } else {
#endif
        result = evrptw::native_kernels::run_exact_charging_batch(
            kinds,
            ready,
            due,
            service,
            distances,
            vehicle_values,
            offsets,
            indices,
            node_count,
            route_count,
            depot,
            stations,
            deadline[0],
            batch[0]);
#ifdef __linux__
        }
#endif
    }

    py::array_t<std::int64_t> path_offsets_array(result.path_offsets.size());
    py::array_t<std::int64_t> path_indices_array(result.path_indices.size());
    py::array_t<std::int64_t> status_array(result.statuses.size());
    py::array_t<std::int64_t> reason_array(result.reasons.size());
    py::array_t<double> metrics_array(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(route_count), 4});
    py::array_t<std::int64_t> counters_array(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(route_count), 3});
    py::array_t<std::int64_t> batch_counters_array(result.batch_counters.size());
    std::copy(result.path_offsets.begin(), result.path_offsets.end(), checked_data(path_offsets_array));
    std::copy(result.path_indices.begin(), result.path_indices.end(), checked_data(path_indices_array));
    std::copy(result.statuses.begin(), result.statuses.end(), checked_data(status_array));
    std::copy(result.reasons.begin(), result.reasons.end(), checked_data(reason_array));
    std::copy(result.metrics.begin(), result.metrics.end(), checked_data(metrics_array));
    std::copy(result.label_counters.begin(), result.label_counters.end(), checked_data(counters_array));
    std::copy(
        result.batch_counters.begin(),
        result.batch_counters.end(),
        checked_data(batch_counters_array));
    return py::make_tuple(
        std::move(path_offsets_array),
        std::move(path_indices_array),
        std::move(status_array),
        std::move(reason_array),
        std::move(metrics_array),
        std::move(counters_array),
        std::move(batch_counters_array));
}

namespace {

evrptw::native_kernels::ScreenOutput dispatch_screen_route(
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
    std::int64_t depot,
    const std::vector<std::int64_t>& recharge_nodes,
    const double* options,
    const double* incremental) {
#ifdef __linux__
    if (native_kernel_scheduler_required
        && native_kernel_scheduler_endpoint.empty()) {
        throw std::runtime_error(
            "host scheduler screening kernel lost its required endpoint");
    }
    if (!native_kernel_scheduler_endpoint.empty()) {
        try {
            return evrptw::native_client::screen_route(
                native_kernel_scheduler_endpoint,
                kinds, demands, ready, due, service, distances, reachable,
                vehicle, route, route_size, node_count, options, incremental);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("host scheduler IPC failed without fallback: ")
                + error.what());
        }
    }
#endif
    return evrptw::native_kernels::run_screen_route(
        kinds, demands, ready, due, service, distances, reachable, vehicle,
        route, route_size, node_count, depot, recharge_nodes, options,
        incremental);
}

std::vector<evrptw::native_kernels::ScreenOutput> dispatch_screen_routes(
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
    std::int64_t depot,
    const std::vector<std::int64_t>& recharge_nodes,
    const double* options,
    const double* incremental) {
#ifdef __linux__
    if (native_kernel_scheduler_required
        && native_kernel_scheduler_endpoint.empty()) {
        throw std::runtime_error(
            "host scheduler screening-batch kernel lost its required endpoint");
    }
    if (!native_kernel_scheduler_endpoint.empty()) {
        try {
            return evrptw::native_client::screen_routes(
                native_kernel_scheduler_endpoint, kinds, demands, ready, due,
                service, distances, reachable, vehicle, route_offsets,
                route_indices, route_count, route_index_count, node_count,
                options, incremental);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("host scheduler IPC failed without fallback: ")
                + error.what());
        }
    }
#endif
    std::vector<evrptw::native_kernels::ScreenOutput> outputs(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        outputs[route] = evrptw::native_kernels::run_screen_route(
            kinds, demands, ready, due, service, distances, reachable, vehicle,
            route_indices + route_offsets[route],
            static_cast<std::size_t>(
                route_offsets[route + 1] - route_offsets[route]),
            node_count, depot, recharge_nodes, options, incremental);
    }
    return outputs;
}

struct PropagationOutput {
    std::vector<std::int64_t> codes = std::vector<std::int64_t>(10, 0);
    std::vector<double> metrics = std::vector<double>(3, 0.0);
};

PropagationOutput run_incremental_propagation(
    const std::int64_t* kinds,
    const double* ready,
    const double* due,
    const double* service,
    const double* distances,
    const double* vehicle,
    const std::int64_t* base_chain,
    std::size_t base_size,
    const std::int64_t* candidate_chain,
    std::size_t candidate_size,
    const double* base_edges,
    const double* base_earliest,
    const double* base_latest,
    std::size_t node_count,
    double epsilon) {
    PropagationOutput output;
    auto valid_candidate = candidate_size >= 2 && candidate_chain[0] >= 0
        && static_cast<std::size_t>(candidate_chain[0]) < node_count
        && candidate_chain[candidate_size - 1] == candidate_chain[0];
    if (valid_candidate) {
        valid_candidate = kinds[candidate_chain[0]] == depot_kind;
    }
    std::vector<bool> seen(node_count, false);
    if (valid_candidate) {
        for (std::size_t position = 1; position + 1 < candidate_size; ++position) {
            const auto node = candidate_chain[position];
            if (node < 0 || static_cast<std::size_t>(node) >= node_count
                || kinds[node] != customer_kind
                || seen[static_cast<std::size_t>(node)]) {
                valid_candidate = false;
                break;
            }
            seen[static_cast<std::size_t>(node)] = true;
        }
    }
    if (!valid_candidate) {
        output.codes[0] = 1;
        output.codes[1] = 2;
        output.codes[2] = 1;
        return output;
    }

    const bool unchanged = base_size == candidate_size
        && std::equal(base_chain, base_chain + base_size, candidate_chain);
    if (unchanged) {
        output.codes[0] = 0;
        output.codes[1] = 1;
        output.codes[3] = 1;
        output.codes[4] = 1;
        output.codes[5] = static_cast<std::int64_t>(base_size - 1);
        output.codes[9] = 1;
        PythonFloatSum distance_sum;
        for (std::size_t edge = 0; edge + 1 < base_size; ++edge) {
            distance_sum.add(base_edges[edge]);
        }
        output.metrics[0] = distance_sum.value();
        output.metrics[1] = std::numeric_limits<double>::infinity();
        for (std::size_t position = 1; position + 1 < base_size; ++position) {
            const auto node = base_chain[position];
            const auto latest_arrival = std::min(
                due[node], base_latest[position] - service[node]);
            output.metrics[1] = std::min(
                output.metrics[1], latest_arrival - base_earliest[position]);
        }
        if (!std::isfinite(output.metrics[1])) {
            output.metrics[1] = 0.0;
        }
        output.metrics[2] = base_earliest[base_size - 1];
        return output;
    }

    std::size_t prefix_nodes = 0;
    while (prefix_nodes < base_size && prefix_nodes < candidate_size
           && base_chain[prefix_nodes] == candidate_chain[prefix_nodes]) {
        ++prefix_nodes;
    }
    std::size_t suffix_nodes = 0;
    while (suffix_nodes < base_size - prefix_nodes
           && suffix_nodes < candidate_size - prefix_nodes
           && base_chain[base_size - 1 - suffix_nodes]
               == candidate_chain[candidate_size - 1 - suffix_nodes]) {
        ++suffix_nodes;
    }
    const auto candidate_suffix_start = candidate_size - suffix_nodes;
    const auto prefix_edges = prefix_nodes > 0 ? prefix_nodes - 1 : 0;
    const auto suffix_edges = suffix_nodes > 0 ? suffix_nodes - 1 : 0;
    const auto candidate_edges = candidate_size - 1;
    const auto middle_start = prefix_nodes > 0 ? prefix_nodes - 1 : 0;
    const auto middle_end = std::max(middle_start, candidate_suffix_start - 1);
    PythonFloatSum middle_distance_sum;
    for (std::size_t edge = middle_start;
         edge <= middle_end && edge + 1 < candidate_size;
         ++edge) {
        middle_distance_sum.add(distances[
            static_cast<std::size_t>(candidate_chain[edge]) * node_count
            + static_cast<std::size_t>(candidate_chain[edge + 1])]);
    }
    auto total_distance = middle_distance_sum.value();
    if (prefix_edges > 0) {
        PythonFloatSum prefix_distance_sum;
        for (std::size_t edge = 0; edge < prefix_edges; ++edge) {
            prefix_distance_sum.add(base_edges[edge]);
        }
        total_distance += prefix_distance_sum.value();
    }
    if (suffix_edges > 0) {
        PythonFloatSum suffix_distance_sum;
        const auto suffix_start = base_size - 1 - suffix_edges;
        for (std::size_t edge = suffix_start; edge + 1 < base_size; ++edge) {
            suffix_distance_sum.add(base_edges[edge]);
        }
        total_distance += suffix_distance_sum.value();
    }

    std::vector<double> earliest(candidate_size, 0.0);
    earliest[0] = std::max(0.0, ready[candidate_chain[0]]);
    const auto forward_start = prefix_nodes > 0 ? prefix_nodes - 1 : 0;
    if (prefix_nodes > 0 && prefix_nodes <= base_size) {
        std::copy(base_earliest, base_earliest + prefix_nodes, earliest.begin());
    }
    auto current_time = earliest[forward_start];
    if (kinds[candidate_chain[forward_start]] == customer_kind) {
        current_time += service[candidate_chain[forward_start]];
    }
    bool forward_feasible = true;
    for (std::size_t edge = forward_start; edge + 1 < candidate_size; ++edge) {
        const auto destination = candidate_chain[edge + 1];
        current_time += distances[
            static_cast<std::size_t>(candidate_chain[edge]) * node_count
            + static_cast<std::size_t>(destination)] / vehicle[4];
        current_time = std::max(current_time, ready[destination]);
        earliest[edge + 1] = current_time;
        if (current_time > due[destination] + epsilon) {
            forward_feasible = false;
        }
        if (kinds[destination] == customer_kind) {
            current_time += service[destination];
        }
    }

    std::vector<double> latest(candidate_size, 0.0);
    latest[candidate_size - 1] = due[candidate_chain[candidate_size - 1]];
    const auto backward_start = candidate_suffix_start;
    if (suffix_nodes > 0) {
        std::copy(
            base_latest + (base_size - suffix_nodes),
            base_latest + base_size,
            latest.begin() + static_cast<std::ptrdiff_t>(candidate_suffix_start));
    }
    auto latest_departure = latest[backward_start];
    bool backward_feasible = true;
    for (std::size_t reverse = backward_start; reverse-- > 0;) {
        const auto origin = candidate_chain[reverse];
        const auto destination = candidate_chain[reverse + 1];
        const auto latest_arrival = kinds[destination] == customer_kind
            ? std::min(due[destination], latest_departure - service[destination])
            : std::min(due[destination], latest_departure);
        latest_departure = latest_arrival - distances[
            static_cast<std::size_t>(origin) * node_count
            + static_cast<std::size_t>(destination)] / vehicle[4];
        latest[reverse] = latest_departure;
        if (kinds[origin] == customer_kind
            && latest_departure < ready[origin] - epsilon) {
            backward_feasible = false;
        }
    }

    auto min_slack = std::numeric_limits<double>::infinity();
    for (std::size_t position = 1; position + 1 < candidate_size; ++position) {
        const auto node = candidate_chain[position];
        const auto latest_arrival = std::min(
            due[node], latest[position] - service[node]);
        min_slack = std::min(min_slack, latest_arrival - earliest[position]);
    }
    if (!std::isfinite(min_slack)) {
        min_slack = 0.0;
    }
    output.codes[0] = 0;
    output.codes[1] = 0;
    output.codes[3] = forward_feasible ? 1 : 0;
    output.codes[4] = backward_feasible ? 1 : 0;
    output.codes[5] = static_cast<std::int64_t>(prefix_edges);
    output.codes[6] = static_cast<std::int64_t>(suffix_edges);
    output.codes[7] = static_cast<std::int64_t>(candidate_edges - prefix_edges);
    output.codes[8] = static_cast<std::int64_t>(candidate_edges - suffix_edges);
    output.codes[9] = forward_feasible && backward_feasible ? 1 : 0;
    if (!forward_feasible) {
        output.codes[1] = 3;
        output.codes[2] = 2;
    } else if (!backward_feasible) {
        output.codes[1] = 4;
        output.codes[2] = 3;
    } else if (min_slack < -epsilon) {
        output.codes[1] = 5;
        output.codes[2] = 4;
    }
    output.metrics[0] = total_distance;
    output.metrics[1] = min_slack;
    output.metrics[2] = current_time;
    return output;
}

}  // namespace

py::tuple screen_routes_numeric(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle route_indices,
    py::handle options,
    py::handle incremental) {
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto demand_array = checked_array<double>(demand, "demand", 1);
    auto ready_array = checked_array<double>(ready_time, "ready_time", 1);
    auto due_array = checked_array<double>(due_date, "due_date", 1);
    auto service_array = checked_array<double>(service_time, "service_time", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto reachable_array = checked_array<std::uint8_t>(reachable, "reachable", 2);
    auto vehicle_array = checked_array<double>(vehicle, "vehicle", 1);
    auto route_array = checked_array<std::int64_t>(route_indices, "route_indices", 1);
    auto options_array = checked_array<double>(options, "options", 1);
    auto incremental_array = checked_array<double>(incremental, "incremental", 1);
    const auto kind_info = kind_array.request();
    const auto node_count = static_cast<std::size_t>(kind_info.shape[0]);
    if (node_count == 0 || demand_array.request().shape[0] != kind_info.shape[0]
        || ready_array.request().shape[0] != kind_info.shape[0]
        || due_array.request().shape[0] != kind_info.shape[0]
        || service_array.request().shape[0] != kind_info.shape[0]) {
        throw std::invalid_argument("screening node arrays must share one non-zero length");
    }
    if (distance_array.request().shape[0] != kind_info.shape[0]
        || distance_array.request().shape[1] != kind_info.shape[0]
        || reachable_array.request().shape[0] != kind_info.shape[0]
        || reachable_array.request().shape[1] != kind_info.shape[0]) {
        throw std::invalid_argument("distance and reachable must have shape (n, n)");
    }
    if (vehicle_array.request().shape[0] != 5) {
        throw std::invalid_argument("vehicle must have shape (5,)");
    }
    if (options_array.request().shape[0] != 4) {
        throw std::invalid_argument("options must have shape (4,)");
    }
    if (incremental_array.request().shape[0] != 6) {
        throw std::invalid_argument("incremental must have shape (6,)");
    }
    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* demands = checked_data<double>(demand_array);
    const auto* ready = checked_data<double>(ready_array);
    const auto* due = checked_data<double>(due_array);
    const auto* service = checked_data<double>(service_array);
    const auto* distances = checked_data<double>(distance_array);
    const auto* reachable_values = checked_data<std::uint8_t>(reachable_array);
    const auto* vehicle_values = checked_data<double>(vehicle_array);
    const auto* route = checked_data<std::int64_t>(route_array);
    const auto* option_values = checked_data<double>(options_array);
    const auto* incremental_values = checked_data<double>(incremental_array);
    const auto route_size = static_cast<std::size_t>(route_array.request().shape[0]);
    if (option_values[1] <= 0.0 || !std::isfinite(option_values[1])) {
        throw std::invalid_argument("screening epsilon must be finite and positive");
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> recharge_nodes;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (kinds[node] == depot_kind) {
            if (depot >= 0) {
                throw std::invalid_argument("node_kind must contain exactly one depot");
            }
            depot = static_cast<std::int64_t>(node);
            recharge_nodes.push_back(depot);
        } else if (kinds[node] == station_kind) {
            recharge_nodes.push_back(static_cast<std::int64_t>(node));
        } else if (kinds[node] != customer_kind) {
            throw std::invalid_argument("node_kind contains an unknown code");
        }
    }
    if (depot < 0) {
        throw std::invalid_argument("node_kind must contain exactly one depot");
    }
    evrptw::native_kernels::ScreenOutput result;
    {
        py::gil_scoped_release release;
        result = dispatch_screen_route(
            kinds,
            demands,
            ready,
            due,
            service,
            distances,
            reachable_values,
            vehicle_values,
            route,
            route_size,
            node_count,
            depot,
            recharge_nodes,
            option_values,
            incremental_values);
    }
    py::array_t<std::int64_t> codes(result.codes.size());
    py::array_t<double> metrics(result.metrics.size());
    std::copy(result.codes.begin(), result.codes.end(), checked_data(codes));
    std::copy(result.metrics.begin(), result.metrics.end(), checked_data(metrics));
    return py::make_tuple(std::move(codes), std::move(metrics));
}

py::tuple candidate_control_repair_v2(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle partial_route_offsets,
    py::handle partial_route_indices,
    py::handle removed_customer_indices,
    double epsilon,
    std::int64_t route_change_limit,
    bool allow_new_routes) {
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto demand_array = checked_array<double>(demand, "demand", 1);
    auto ready_array = checked_array<double>(ready_time, "ready_time", 1);
    auto due_array = checked_array<double>(due_date, "due_date", 1);
    auto service_array = checked_array<double>(service_time, "service_time", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto reachable_array = checked_array<std::uint8_t>(reachable, "reachable", 2);
    auto vehicle_array = checked_array<double>(vehicle, "vehicle", 1);
    auto lexical_array = checked_array<std::int64_t>(
        lexical_rank, "lexical_rank", 1);
    auto offsets_array = checked_array<std::int64_t>(
        partial_route_offsets, "partial_route_offsets", 1);
    auto indices_array = checked_array<std::int64_t>(
        partial_route_indices, "partial_route_indices", 1);
    auto removed_array = checked_array<std::int64_t>(
        removed_customer_indices, "removed_customer_indices", 1);
    const auto node_count = static_cast<std::size_t>(kind_array.size());
    if (node_count == 0 || demand_array.size() != kind_array.size()
        || ready_array.size() != kind_array.size()
        || due_array.size() != kind_array.size()
        || service_array.size() != kind_array.size()
        || lexical_array.size() != kind_array.size()
        || distance_array.shape(0) != kind_array.size()
        || distance_array.shape(1) != kind_array.size()
        || reachable_array.shape(0) != kind_array.size()
        || reachable_array.shape(1) != kind_array.size()
        || vehicle_array.size() != 5 || offsets_array.size() < 1
        || removed_array.size() < 1 || !std::isfinite(epsilon) || epsilon <= 0.0
        || route_change_limit == 0 || route_change_limit < -1) {
        throw std::invalid_argument(
            "candidate-control repair v2 input/config shape is invalid");
    }
    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* demands = checked_data<double>(demand_array);
    const auto* ready = checked_data<double>(ready_array);
    const auto* due = checked_data<double>(due_array);
    const auto* service = checked_data<double>(service_array);
    const auto* distances = checked_data<double>(distance_array);
    const auto* reachable_values = checked_data<std::uint8_t>(reachable_array);
    const auto* vehicle_values = checked_data<double>(vehicle_array);
    const auto* lexical = checked_data<std::int64_t>(lexical_array);
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* indices = checked_data<std::int64_t>(indices_array);
    const auto* removed = checked_data<std::int64_t>(removed_array);
    if (!std::isfinite(vehicle_values[1]) || vehicle_values[1] < 0.0) {
        throw std::invalid_argument(
            "candidate-control repair v2 load capacity is invalid");
    }
    std::unordered_set<std::int64_t> lexical_values;
    std::int64_t depot = -1;
    std::vector<std::int64_t> recharge_nodes;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (lexical[node] < 0 || lexical[node] >= static_cast<std::int64_t>(node_count)
            || !lexical_values.insert(lexical[node]).second) {
            throw std::invalid_argument(
                "candidate-control repair lexical_rank must be a permutation");
        }
        if (kinds[node] == depot_kind) {
            if (depot >= 0) {
                throw std::invalid_argument(
                    "candidate-control repair requires exactly one depot");
            }
            depot = static_cast<std::int64_t>(node);
            recharge_nodes.push_back(depot);
        } else if (kinds[node] == station_kind) {
            recharge_nodes.push_back(static_cast<std::int64_t>(node));
        } else if (kinds[node] != customer_kind) {
            throw std::invalid_argument(
                "candidate-control repair node_kind contains an unknown code");
        }
    }
    if (depot < 0) {
        throw std::invalid_argument(
            "candidate-control repair requires exactly one depot");
    }
    const auto route_count = static_cast<std::size_t>(offsets_array.size() - 1);
    if (offsets[0] != 0 || offsets[route_count] != indices_array.size()) {
        throw std::invalid_argument(
            "candidate-control repair route offsets do not span indices");
    }
    std::vector<std::vector<std::int64_t>> routes;
    routes.reserve(route_count);
    std::unordered_set<std::int64_t> present;
    for (std::size_t route = 0; route < route_count; ++route) {
        if (offsets[route] < 0 || offsets[route] >= offsets[route + 1]) {
            throw std::invalid_argument(
                "candidate-control repair partial routes must be non-empty");
        }
        routes.emplace_back(
            indices + offsets[route], indices + offsets[route + 1]);
        for (const auto node : routes.back()) {
            if (node < 0 || static_cast<std::size_t>(node) >= node_count
                || kinds[node] != customer_kind || !present.insert(node).second) {
                throw std::invalid_argument(
                    "candidate-control repair partial routes contain invalid customers");
            }
        }
    }
    std::vector<std::int64_t> pending;
    pending.reserve(static_cast<std::size_t>(removed_array.size()));
    for (py::ssize_t index = 0; index < removed_array.size(); ++index) {
        const auto node = removed[index];
        if (node < 0 || static_cast<std::size_t>(node) >= node_count
            || kinds[node] != customer_kind || present.contains(node)
            || std::find(pending.begin(), pending.end(), node) != pending.end()) {
            throw std::invalid_argument(
                "candidate-control repair removed customers are invalid");
        }
        pending.push_back(node);
    }

    struct Option {
        double score;
        std::int64_t customer;
        std::size_t route;
        std::size_t position;
        std::vector<std::int64_t> sequence;
    };
    const auto route_lexical_less = [&](const auto& left, const auto& right) {
        return std::lexicographical_compare(
            left.begin(), left.end(), right.begin(), right.end(),
            [&](std::int64_t lhs, std::int64_t rhs) {
                return lexical[lhs] < lexical[rhs];
            });
    };
    const auto option_less = [&](const Option& left, const Option& right) {
        if (left.score != right.score) {
            return left.score < right.score;
        }
        if (lexical[left.customer] != lexical[right.customer]) {
            return lexical[left.customer] < lexical[right.customer];
        }
        if (left.route != right.route) {
            return left.route < right.route;
        }
        if (left.position != right.position) {
            return left.position < right.position;
        }
        return route_lexical_less(left.sequence, right.sequence);
    };
    std::unordered_set<std::size_t> changed_routes;
    std::int64_t new_routes = 0;
    std::int64_t screening_calls = 0;
    std::int64_t screening_passes = 0;
    std::int64_t screening_rejections = 0;
    std::int64_t failure_code = 0;
    std::array<double, 6> incremental{};
    while (!pending.empty()) {
        std::optional<Option> best;
        for (const auto customer : pending) {
            for (std::size_t route = 0; route < routes.size(); ++route) {
                if (route_change_limit > 0 && !changed_routes.contains(route)
                    && changed_routes.size()
                        >= static_cast<std::size_t>(route_change_limit)) {
                    continue;
                }
                PythonFloatSum demand_sum;
                for (const auto node : routes[route]) {
                    demand_sum.add(demands[node]);
                }
                demand_sum.add(demands[customer]);
                if (demand_sum.value() > vehicle_values[1] + epsilon) {
                    continue;
                }
                PythonFloatSum reference_sum;
                auto origin = depot;
                for (const auto node : routes[route]) {
                    reference_sum.add(distances[
                        static_cast<std::size_t>(origin) * node_count
                        + static_cast<std::size_t>(node)]);
                    origin = node;
                }
                reference_sum.add(distances[
                    static_cast<std::size_t>(origin) * node_count
                    + static_cast<std::size_t>(depot)]);
                for (std::size_t position = 0;
                     position <= routes[route].size(); ++position) {
                    auto candidate = routes[route];
                    candidate.insert(
                        candidate.begin() + static_cast<std::ptrdiff_t>(position),
                        customer);
                    const std::array<double, 4> options{
                        1.0, epsilon, reference_sum.value(), 1.0};
                    const auto screened = dispatch_screen_route(
                        kinds, demands, ready, due, service, distances,
                        reachable_values, vehicle_values, candidate.data(),
                        candidate.size(), node_count, depot, recharge_nodes,
                        options.data(), incremental.data());
                    ++screening_calls;
                    if (screened.codes[0] == 0) {
                        ++screening_rejections;
                        continue;
                    }
                    ++screening_passes;
                    Option option{
                        screened.metrics[4], customer, route, position,
                        std::move(candidate)};
                    if (!best.has_value() || option_less(option, *best)) {
                        best = std::move(option);
                    }
                }
            }
        }
        if (best.has_value()) {
            routes[best->route] = std::move(best->sequence);
            changed_routes.insert(best->route);
            pending.erase(std::find(
                pending.begin(), pending.end(), best->customer));
            continue;
        }
        if (!allow_new_routes) {
            failure_code = 1;
            break;
        }
        const auto customer = *std::min_element(
            pending.begin(), pending.end(), [&](std::int64_t left, std::int64_t right) {
                return lexical[left] < lexical[right];
            });
        const std::vector<std::int64_t> singleton{customer};
        const std::array<double, 4> options{1.0, epsilon, 0.0, 0.0};
        const auto screened = dispatch_screen_route(
            kinds, demands, ready, due, service, distances, reachable_values,
            vehicle_values, singleton.data(), singleton.size(), node_count,
            depot, recharge_nodes, options.data(), incremental.data());
        ++screening_calls;
        if (screened.codes[0] == 0) {
            ++screening_rejections;
            failure_code = 2;
            break;
        }
        ++screening_passes;
        routes.push_back(singleton);
        changed_routes.insert(routes.size() - 1);
        pending.erase(std::find(pending.begin(), pending.end(), customer));
        ++new_routes;
    }

    std::vector<std::int64_t> output_offsets{0};
    std::vector<std::int64_t> output_indices;
    if (failure_code == 0) {
        for (const auto& route : routes) {
            output_indices.insert(
                output_indices.end(), route.begin(), route.end());
            output_offsets.push_back(
                static_cast<std::int64_t>(output_indices.size()));
        }
    }
    py::array_t<std::int64_t> output_offsets_array(output_offsets.size());
    py::array_t<std::int64_t> output_indices_array(output_indices.size());
    py::array_t<std::int64_t> counters(7);
    std::copy(
        output_offsets.begin(), output_offsets.end(),
        checked_data(output_offsets_array));
    std::copy(
        output_indices.begin(), output_indices.end(),
        checked_data(output_indices_array));
    auto* counter_values = checked_data(counters);
    counter_values[0] = failure_code;
    counter_values[1] = new_routes;
    counter_values[2] = static_cast<std::int64_t>(changed_routes.size());
    counter_values[3] = screening_calls;
    counter_values[4] = screening_passes;
    counter_values[5] = screening_rejections;
    counter_values[6] = static_cast<std::int64_t>(pending.size());
    return py::make_tuple(
        std::move(output_offsets_array), std::move(output_indices_array),
        std::move(counters));
}

py::tuple constraint_removal_v2(
    std::int64_t operation,
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle path_offsets,
    py::handle path_indices,
    py::handle result_metrics,
    std::int64_t requested_count,
    std::uint64_t seed) {
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto demand_array = checked_array<double>(demand, "demand", 1);
    auto ready_array = checked_array<double>(ready_time, "ready_time", 1);
    auto due_array = checked_array<double>(due_date, "due_date", 1);
    auto service_array = checked_array<double>(service_time, "service_time", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto reachable_array = checked_array<std::uint8_t>(reachable, "reachable", 2);
    auto vehicle_array = checked_array<double>(vehicle, "vehicle", 1);
    auto lexical_array = checked_array<std::int64_t>(
        lexical_rank, "lexical_rank", 1);
    auto route_offsets_array = checked_array<std::int64_t>(
        route_offsets, "route_offsets", 1);
    auto route_indices_array = checked_array<std::int64_t>(
        route_indices, "route_indices", 1);
    auto path_offsets_array = checked_array<std::int64_t>(
        path_offsets, "path_offsets", 1);
    auto path_indices_array = checked_array<std::int64_t>(
        path_indices, "path_indices", 1);
    auto metrics_array = checked_array<double>(result_metrics, "result_metrics", 2);
    const auto node_count = static_cast<std::size_t>(kind_array.size());
    if (operation < 0 || operation > 3 || requested_count < 0 || node_count == 0
        || demand_array.size() != kind_array.size()
        || ready_array.size() != kind_array.size()
        || due_array.size() != kind_array.size()
        || service_array.size() != kind_array.size()
        || lexical_array.size() != kind_array.size()
        || distance_array.shape(0) != kind_array.size()
        || distance_array.shape(1) != kind_array.size()
        || reachable_array.shape(0) != kind_array.size()
        || reachable_array.shape(1) != kind_array.size()
        || vehicle_array.size() != 5 || route_offsets_array.size() < 2) {
        throw std::invalid_argument(
            "constraint removal v2 input/config shape is invalid");
    }
    const auto route_count = static_cast<std::size_t>(
        route_offsets_array.size() - 1);
    if (path_offsets_array.size() != static_cast<py::ssize_t>(route_count + 1)
        || metrics_array.shape(0) != static_cast<py::ssize_t>(route_count)
        || metrics_array.shape(1) != 4) {
        throw std::invalid_argument(
            "constraint removal v2 exact payload does not align with routes");
    }
    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* demands = checked_data<double>(demand_array);
    const auto* ready = checked_data<double>(ready_array);
    const auto* due = checked_data<double>(due_array);
    const auto* service = checked_data<double>(service_array);
    const auto* distances = checked_data<double>(distance_array);
    const auto* vehicle_values = checked_data<double>(vehicle_array);
    const auto* lexical = checked_data<std::int64_t>(lexical_array);
    const auto* route_boundaries = checked_data<std::int64_t>(route_offsets_array);
    const auto* route_values = checked_data<std::int64_t>(route_indices_array);
    const auto* path_boundaries = checked_data<std::int64_t>(path_offsets_array);
    const auto* path_values = checked_data<std::int64_t>(path_indices_array);
    const auto* metrics = checked_data<double>(metrics_array);
    if (route_boundaries[0] != 0
        || route_boundaries[route_count] != route_indices_array.size()
        || path_boundaries[0] != 0
        || path_boundaries[route_count] != path_indices_array.size()) {
        throw std::invalid_argument(
            "constraint removal v2 route/path boundaries are invalid");
    }
    std::unordered_set<std::int64_t> lexical_values;
    std::vector<std::int64_t> all_instance_customers;
    std::int64_t depot = -1;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (lexical[node] < 0 || lexical[node] >= static_cast<std::int64_t>(node_count)
            || !lexical_values.insert(lexical[node]).second) {
            throw std::invalid_argument(
                "constraint removal lexical_rank must be a permutation");
        }
        if (kinds[node] == customer_kind) {
            all_instance_customers.push_back(static_cast<std::int64_t>(node));
        } else if (kinds[node] == depot_kind) {
            if (depot >= 0) {
                throw std::invalid_argument(
                    "constraint removal requires exactly one depot");
            }
            depot = static_cast<std::int64_t>(node);
        } else if (kinds[node] != depot_kind && kinds[node] != station_kind) {
            throw std::invalid_argument(
                "constraint removal node_kind contains an unknown code");
        }
    }
    if (depot < 0) {
        throw std::invalid_argument(
            "constraint removal requires exactly one depot");
    }
    std::vector<std::vector<std::int64_t>> routes;
    routes.reserve(route_count);
    std::unordered_set<std::int64_t> all_customers;
    for (std::size_t route = 0; route < route_count; ++route) {
        if (route_boundaries[route] < 0
            || route_boundaries[route] >= route_boundaries[route + 1]
            || path_boundaries[route] < 0
            || path_boundaries[route] >= path_boundaries[route + 1]) {
            throw std::invalid_argument(
                "constraint removal routes must be non-empty and monotonic");
        }
        routes.emplace_back(
            route_values + route_boundaries[route],
            route_values + route_boundaries[route + 1]);
        for (const auto customer : routes.back()) {
            if (customer < 0 || static_cast<std::size_t>(customer) >= node_count
                || kinds[customer] != customer_kind
                || !all_customers.insert(customer).second) {
                throw std::invalid_argument(
                    "constraint removal requires unique customer-only routes");
            }
        }
        for (auto position = path_boundaries[route];
             position < path_boundaries[route + 1]; ++position) {
            const auto node = path_values[position];
            if (node < 0 || static_cast<std::size_t>(node) >= node_count) {
                throw std::invalid_argument(
                    "constraint removal exact path contains an invalid node");
            }
        }
        for (std::size_t field = 0; field < 4; ++field) {
            if (!std::isfinite(metrics[route * 4 + field])
                || metrics[route * 4 + field] < 0.0) {
                throw std::invalid_argument(
                    "constraint removal exact metrics are invalid");
            }
        }
    }

    if (all_customers.size() <= 1 || requested_count == 0) {
        py::array_t<std::int64_t> partial_offsets_array(1);
        checked_data(partial_offsets_array)[0] = 0;
        py::array_t<std::int64_t> partial_indices_array(0);
        py::array_t<std::int64_t> removed_output_array(0);
        py::array_t<std::int64_t> score_nodes(0);
        py::array_t<double> score_values(0);
        py::array_t<std::int64_t> score_routes(0);
        py::array_t<std::int64_t> metadata(3);
        checked_data(metadata)[0] = 2;
        checked_data(metadata)[1] = -1;
        checked_data(metadata)[2] = 0;
        return py::make_tuple(
            std::move(partial_offsets_array), std::move(partial_indices_array),
            std::move(removed_output_array), std::move(score_nodes),
            std::move(score_values), std::move(score_routes), std::move(metadata));
    }

    struct Score {
        std::int64_t customer;
        double value;
        std::size_t route;
    };
    std::vector<Score> scores;
    std::int64_t anchor = -1;
    if (operation == 3) {
        auto ordered_customers = std::vector<std::int64_t>(
            all_customers.begin(), all_customers.end());
        std::sort(
            ordered_customers.begin(), ordered_customers.end(),
            [&](std::int64_t left, std::int64_t right) {
                return lexical[left] < lexical[right];
            });
        if (!ordered_customers.empty()) {
            PythonRandom random(seed);
            anchor = ordered_customers[random.randbelow(ordered_customers.size())];
        }
    }
    const auto path_distance = [&](const std::int64_t* values, std::size_t count) {
        PythonFloatSum total;
        for (std::size_t position = 1; position < count; ++position) {
            total.add(distances[
                static_cast<std::size_t>(values[position - 1]) * node_count
                + static_cast<std::size_t>(values[position])]);
        }
        return total.value();
    };
    for (std::size_t route = 0; route < route_count; ++route) {
        const auto* path = path_values + path_boundaries[route];
        const auto path_size = static_cast<std::size_t>(
            path_boundaries[route + 1] - path_boundaries[route]);
        if (operation == 0 || operation == 2) {
            std::vector<std::pair<std::size_t, std::int64_t>> positions;
            for (std::size_t position = 0; position < path_size; ++position) {
                if (kinds[path[position]] == customer_kind) {
                    positions.emplace_back(position, path[position]);
                }
            }
            const auto station_count = static_cast<double>(std::count_if(
                path, path + path_size, [&](std::int64_t node) {
                    return kinds[node] == station_kind;
                }));
            const auto route_pressure = 2.0 * station_count
                + metrics[route * 4 + 2] + 10.0 * metrics[route * 4 + 3];
            for (std::size_t index = 0; index < positions.size(); ++index) {
                const auto left = index == 0 ? std::size_t{0} : positions[index - 1].first;
                const auto right = index + 1 < positions.size()
                    ? positions[index + 1].first
                    : path_size - 1;
                const auto local_distance = path_distance(path + left, right - left + 1);
                const auto direct = distances[
                    static_cast<std::size_t>(path[left]) * node_count
                    + static_cast<std::size_t>(path[right])];
                auto value = local_distance - direct;
                if (operation == 0) {
                    const auto local_stations = static_cast<double>(std::count_if(
                        path + left, path + right + 1, [&](std::int64_t node) {
                            return kinds[node] == station_kind;
                        }));
                    value += 2.0 * local_stations
                        + route_pressure
                            / static_cast<double>(std::max<std::size_t>(1, positions.size()));
                }
                scores.push_back(Score{positions[index].second, value, route});
            }
            continue;
        }
        if (operation == 1) {
            double current_time = std::max(0.0, ready[depot]);
            double battery = vehicle_values[0];
            for (std::size_t position = 1; position < path_size; ++position) {
                const auto origin = path[position - 1];
                const auto destination = path[position];
                const auto leg = distances[
                    static_cast<std::size_t>(origin) * node_count
                    + static_cast<std::size_t>(destination)];
                battery -= leg * vehicle_values[2];
                current_time += leg / vehicle_values[4];
                current_time = std::max(current_time, ready[destination]);
                if (kinds[destination] == station_kind) {
                    const auto charged = vehicle_values[0] - std::max(0.0, battery);
                    current_time += charged * vehicle_values[3];
                    battery = vehicle_values[0];
                } else if (kinds[destination] == customer_kind) {
                    scores.push_back(Score{
                        destination, -(due[destination] - current_time), route});
                    current_time += service[destination];
                }
            }
            continue;
        }
        double maximum_distance = 0.0;
        double maximum_time = 0.0;
        double maximum_demand = 0.0;
        for (const auto customer : all_instance_customers) {
            maximum_distance = std::max(
                maximum_distance,
                distances[static_cast<std::size_t>(anchor) * node_count
                          + static_cast<std::size_t>(customer)]);
            maximum_time = std::max(maximum_time, due[customer]);
            maximum_demand = std::max(maximum_demand, demands[customer]);
        }
        for (const auto customer : routes[route]) {
            const auto normalized_distance = distances[
                static_cast<std::size_t>(anchor) * node_count
                + static_cast<std::size_t>(customer)]
                / std::max(maximum_distance, exact_epsilon);
            const auto time_difference = (
                std::fabs(ready[anchor] - ready[customer])
                + std::fabs(due[anchor] - due[customer]))
                / std::max(maximum_time, exact_epsilon);
            const auto demand_difference = std::fabs(
                demands[anchor] - demands[customer])
                / std::max(maximum_demand, exact_epsilon);
            const auto energy_reachable = [&]() {
                if (anchor == customer) {
                    return true;
                }
                std::vector<std::int64_t> frontier{anchor};
                std::unordered_set<std::int64_t> visited_stations;
                while (!frontier.empty()) {
                    const auto current = frontier.back();
                    frontier.pop_back();
                    if (distances[
                            static_cast<std::size_t>(current) * node_count
                            + static_cast<std::size_t>(customer)]
                            * vehicle_values[2]
                        <= vehicle_values[0] + exact_epsilon) {
                        return true;
                    }
                    if (current != anchor && kinds[current] != depot_kind
                        && kinds[current] != station_kind) {
                        continue;
                    }
                    for (std::size_t station = 0; station < node_count; ++station) {
                        if (kinds[station] != station_kind
                            || static_cast<std::int64_t>(station) == current
                            || visited_stations.contains(
                                static_cast<std::int64_t>(station))) {
                            continue;
                        }
                        if (distances[
                                static_cast<std::size_t>(current) * node_count + station]
                                * vehicle_values[2]
                            <= vehicle_values[0] + exact_epsilon) {
                            visited_stations.insert(static_cast<std::int64_t>(station));
                            frontier.push_back(static_cast<std::int64_t>(station));
                        }
                    }
                }
                return false;
            }();
            const auto energy_penalty = energy_reachable ? 0.0 : 1.0;
            scores.push_back(Score{
                customer,
                normalized_distance + 0.25 * time_difference
                    + 0.25 * demand_difference + energy_penalty,
                route});
        }
    }
    std::stable_sort(scores.begin(), scores.end(), [&](const Score& left, const Score& right) {
        if (left.value != right.value) {
            return operation == 3 ? left.value < right.value : left.value > right.value;
        }
        if (left.route != right.route) {
            return left.route < right.route;
        }
        return lexical[left.customer] < lexical[right.customer];
    });
    const auto actual_count = std::min<std::size_t>(
        {static_cast<std::size_t>(requested_count), scores.size(),
         all_customers.empty() ? std::size_t{0} : all_customers.size() - 1});
    std::unordered_set<std::int64_t> chosen;
    std::vector<std::int64_t> removed_output;
    removed_output.reserve(actual_count);
    for (std::size_t index = 0; index < actual_count; ++index) {
        chosen.insert(scores[index].customer);
        removed_output.push_back(scores[index].customer);
    }
    std::vector<std::int64_t> partial_offsets{0};
    std::vector<std::int64_t> partial_indices;
    for (const auto& route : routes) {
        for (const auto customer : route) {
            if (!chosen.contains(customer)) {
                partial_indices.push_back(customer);
            }
        }
        if (partial_offsets.back() != static_cast<std::int64_t>(partial_indices.size())) {
            partial_offsets.push_back(static_cast<std::int64_t>(partial_indices.size()));
        }
    }
    py::array_t<std::int64_t> partial_offsets_array(partial_offsets.size());
    py::array_t<std::int64_t> partial_indices_array(partial_indices.size());
    py::array_t<std::int64_t> removed_output_array(removed_output.size());
    py::array_t<std::int64_t> score_nodes(scores.size());
    py::array_t<double> score_values(scores.size());
    py::array_t<std::int64_t> score_routes(scores.size());
    py::array_t<std::int64_t> metadata(3);
    std::copy(
        partial_offsets.begin(), partial_offsets.end(),
        checked_data(partial_offsets_array));
    std::copy(
        partial_indices.begin(), partial_indices.end(),
        checked_data(partial_indices_array));
    std::copy(
        removed_output.begin(), removed_output.end(),
        checked_data(removed_output_array));
    for (std::size_t index = 0; index < scores.size(); ++index) {
        checked_data(score_nodes)[index] = scores[index].customer;
        checked_data(score_values)[index] = scores[index].value;
        checked_data(score_routes)[index] = static_cast<std::int64_t>(scores[index].route);
    }
    checked_data(metadata)[0] = scores.empty() ? 1 : 0;
    checked_data(metadata)[1] = anchor;
    checked_data(metadata)[2] = static_cast<std::int64_t>(actual_count);
    return py::make_tuple(
        std::move(partial_offsets_array), std::move(partial_indices_array),
        std::move(removed_output_array), std::move(score_nodes),
        std::move(score_values), std::move(score_routes), std::move(metadata));
}

py::tuple screen_route_batch_transaction_impl(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle candidate_ids,
    py::handle options,
    py::handle incremental,
    py::handle negative_offsets,
    py::handle negative_indices,
    py::handle negative_reason_codes,
    std::int64_t worker_count) {
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto demand_array = checked_array<double>(demand, "demand", 1);
    auto ready_array = checked_array<double>(ready_time, "ready_time", 1);
    auto due_array = checked_array<double>(due_date, "due_date", 1);
    auto service_array = checked_array<double>(service_time, "service_time", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto reachable_array = checked_array<std::uint8_t>(reachable, "reachable", 2);
    auto vehicle_array = checked_array<double>(vehicle, "vehicle", 1);
    auto offsets_array = checked_array<std::int64_t>(route_offsets, "route_offsets", 1);
    auto routes_array = checked_array<std::int64_t>(route_indices, "route_indices", 1);
    auto ids_array = checked_array<std::int64_t>(candidate_ids, "candidate_ids", 1);
    auto options_array = checked_array<double>(options, "options", 1);
    auto incremental_array = checked_array<double>(incremental, "incremental", 2);
    auto negative_offsets_array = checked_array<std::int64_t>(
        negative_offsets, "negative_offsets", 1);
    auto negative_routes_array = checked_array<std::int64_t>(
        negative_indices, "negative_indices", 1);
    auto negative_reasons_array = checked_array<std::int64_t>(
        negative_reason_codes, "negative_reason_codes", 1);

    const auto node_count = static_cast<std::size_t>(kind_array.request().shape[0]);
    const auto candidate_count = static_cast<std::size_t>(ids_array.request().shape[0]);
    const auto negative_count = static_cast<std::size_t>(
        negative_reasons_array.request().shape[0]);
    if (node_count == 0
        || demand_array.request().shape[0] != kind_array.request().shape[0]
        || ready_array.request().shape[0] != kind_array.request().shape[0]
        || due_array.request().shape[0] != kind_array.request().shape[0]
        || service_array.request().shape[0] != kind_array.request().shape[0]) {
        throw std::invalid_argument(
            "batch screening node arrays must share one non-zero length");
    }
    if (distance_array.request().shape[0] != kind_array.request().shape[0]
        || distance_array.request().shape[1] != kind_array.request().shape[0]
        || reachable_array.request().shape[0] != kind_array.request().shape[0]
        || reachable_array.request().shape[1] != kind_array.request().shape[0]) {
        throw std::invalid_argument(
            "batch screening distance and reachable must have shape (n, n)");
    }
    if (vehicle_array.request().shape[0] != 5
        || options_array.request().shape[0] != 4) {
        throw std::invalid_argument(
            "batch screening vehicle/options shape is invalid");
    }
    if (offsets_array.request().shape[0]
            != static_cast<py::ssize_t>(candidate_count + 1)
        || incremental_array.request().shape[0]
            != static_cast<py::ssize_t>(candidate_count)
        || incremental_array.request().shape[1] != 6) {
        throw std::invalid_argument(
            "batch screening candidate arrays do not share one row count");
    }
    if (negative_offsets_array.request().shape[0]
            != static_cast<py::ssize_t>(negative_count + 1)) {
        throw std::invalid_argument(
            "batch screening negative-cache arrays do not share one row count");
    }

    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* demands = checked_data<double>(demand_array);
    const auto* ready = checked_data<double>(ready_array);
    const auto* due = checked_data<double>(due_array);
    const auto* service = checked_data<double>(service_array);
    const auto* distances = checked_data<double>(distance_array);
    const auto* reachable_values = checked_data<std::uint8_t>(reachable_array);
    const auto* vehicle_values = checked_data<double>(vehicle_array);
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* routes = checked_data<std::int64_t>(routes_array);
    const auto* ids = checked_data<std::int64_t>(ids_array);
    const auto* option_values = checked_data<double>(options_array);
    const auto* incremental_values = checked_data<double>(incremental_array);
    const auto* negative_route_offsets = checked_data<std::int64_t>(
        negative_offsets_array);
    const auto* negative_routes = checked_data<std::int64_t>(negative_routes_array);
    const auto* negative_reasons = checked_data<std::int64_t>(
        negative_reasons_array);
    const auto routes_size = static_cast<std::int64_t>(routes_array.request().shape[0]);
    const auto negative_routes_size = static_cast<std::int64_t>(
        negative_routes_array.request().shape[0]);
    if (option_values[1] <= 0.0 || !std::isfinite(option_values[1])) {
        throw std::invalid_argument(
            "batch screening epsilon must be finite and positive");
    }
    if (worker_count <= 0) {
        throw std::invalid_argument("batch screening worker_count must be positive");
    }

    auto validate_offsets = [](
        const std::int64_t* values,
        std::size_t count,
        std::int64_t terminal,
        const char* name) {
        if (values[0] != 0 || values[count] != terminal) {
            throw std::invalid_argument(std::string(name) + " boundary is invalid");
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (values[index] < 0 || values[index] > values[index + 1]) {
                throw std::invalid_argument(
                    std::string(name) + " must be monotonic");
            }
        }
    };
    validate_offsets(offsets, candidate_count, routes_size, "route_offsets");
    validate_offsets(
        negative_route_offsets,
        negative_count,
        negative_routes_size,
        "negative_offsets");

    std::unordered_set<std::int64_t> unique_ids;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        if (!unique_ids.insert(ids[index]).second) {
            throw std::invalid_argument("candidate_ids must be unique");
        }
    }
    std::int64_t depot = -1;
    std::vector<std::int64_t> recharge_nodes;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (kinds[node] == depot_kind) {
            if (depot >= 0) {
                throw std::invalid_argument(
                    "node_kind must contain exactly one depot");
            }
            depot = static_cast<std::int64_t>(node);
            recharge_nodes.push_back(depot);
        } else if (kinds[node] == station_kind) {
            recharge_nodes.push_back(static_cast<std::int64_t>(node));
        } else if (kinds[node] != customer_kind) {
            throw std::invalid_argument("node_kind contains an unknown code");
        }
    }
    if (depot < 0) {
        throw std::invalid_argument("node_kind must contain exactly one depot");
    }

    auto canonical_key = [](
        const std::int64_t* values,
        std::size_t count) {
        std::string key;
        key.reserve((count + 1) * sizeof(std::int64_t));
        const auto append_i64 = [&key](std::int64_t value) {
            const auto bits = static_cast<std::uint64_t>(value);
            for (std::size_t byte = 0; byte < 8; ++byte) {
                key.push_back(static_cast<char>((bits >> (byte * 8)) & 0xffU));
            }
        };
        append_i64(static_cast<std::int64_t>(count));
        for (std::size_t index = 0; index < count; ++index) {
            append_i64(values[index]);
        }
        return key;
    };
    std::unordered_map<std::string, std::int64_t> negative_cache;
    for (std::size_t index = 0; index < negative_count; ++index) {
        if (negative_reasons[index] < screen_reason_structure
            || negative_reasons[index] > screen_reason_legacy_energy) {
            throw std::invalid_argument(
                "negative_reason_codes contains an unknown reason");
        }
        const auto begin = static_cast<std::size_t>(negative_route_offsets[index]);
        const auto end = static_cast<std::size_t>(negative_route_offsets[index + 1]);
        const auto key = canonical_key(negative_routes + begin, end - begin);
        if (!negative_cache.emplace(key, negative_reasons[index]).second) {
            throw std::invalid_argument(
                "negative-cache route identities must be unique");
        }
    }

    std::vector<evrptw::native_kernels::ScreenOutput> outputs(candidate_count);
    std::vector<std::int64_t> statuses(candidate_count, 0);
    std::vector<std::int64_t> duplicate_of(candidate_count, -1);
    std::vector<std::size_t> duplicate_source(
        candidate_count, std::numeric_limits<std::size_t>::max());
    std::unordered_map<std::string, std::size_t> first_by_key;
    std::int64_t duplicate_count = 0;
    std::int64_t negative_hit_count = 0;
    std::int64_t screened_count = 0;
    {
        py::gil_scoped_release release;
        std::vector<std::size_t> screen_indices;
        screen_indices.reserve(candidate_count);
        for (std::size_t index = 0; index < candidate_count; ++index) {
            const auto begin = static_cast<std::size_t>(offsets[index]);
            const auto end = static_cast<std::size_t>(offsets[index + 1]);
            const auto key = canonical_key(routes + begin, end - begin);
            const auto duplicate = first_by_key.find(key);
            if (duplicate != first_by_key.end()) {
                statuses[index] = 1;
                duplicate_of[index] = ids[duplicate->second];
                duplicate_source[index] = duplicate->second;
                ++duplicate_count;
                continue;
            }
            first_by_key.emplace(key, index);
            const auto negative = negative_cache.find(key);
            if (negative != negative_cache.end()) {
                statuses[index] = 2;
                outputs[index].codes[0] = 0;
                outputs[index].codes[1] = negative->second;
                ++negative_hit_count;
                continue;
            }
            screen_indices.push_back(index);
        }
        const auto active_workers = std::min<std::size_t>(
            screen_indices.size(), static_cast<std::size_t>(worker_count));
        std::vector<std::thread> workers;
        workers.reserve(active_workers);
        for (std::size_t worker = 0; worker < active_workers; ++worker) {
            workers.emplace_back([&, worker]() {
                for (std::size_t position = worker;
                     position < screen_indices.size();
                     position += active_workers) {
                    const auto index = screen_indices[position];
                    const auto begin = static_cast<std::size_t>(offsets[index]);
                    const auto end = static_cast<std::size_t>(offsets[index + 1]);
                    outputs[index] = dispatch_screen_route(
                        kinds,
                        demands,
                        ready,
                        due,
                        service,
                        distances,
                        reachable_values,
                        vehicle_values,
                        routes + begin,
                        end - begin,
                        node_count,
                        depot,
                        recharge_nodes,
                        option_values,
                        incremental_values + index * 6);
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        for (std::size_t index = 0; index < candidate_count; ++index) {
            if (statuses[index] == 1) {
                outputs[index] = outputs[duplicate_source[index]];
            }
        }
        screened_count = static_cast<std::int64_t>(screen_indices.size());
    }

    py::array_t<std::int64_t> returned_ids(candidate_count);
    py::array_t<std::int64_t> status_array(candidate_count);
    py::array_t<std::int64_t> duplicate_array(candidate_count);
    py::array_t<std::int64_t> codes_array(
        {static_cast<py::ssize_t>(candidate_count), py::ssize_t(16)});
    py::array_t<double> metrics_array(
        {static_cast<py::ssize_t>(candidate_count), py::ssize_t(15)});
    std::copy(ids, ids + candidate_count, checked_data(returned_ids));
    std::copy(statuses.begin(), statuses.end(), checked_data(status_array));
    std::copy(
        duplicate_of.begin(), duplicate_of.end(), checked_data(duplicate_array));
    auto* code_output = checked_data(codes_array);
    auto* metric_output = checked_data(metrics_array);
    for (std::size_t index = 0; index < candidate_count; ++index) {
        std::copy(
            outputs[index].codes.begin(),
            outputs[index].codes.end(),
            code_output + index * 16);
        std::copy(
            outputs[index].metrics.begin(),
            outputs[index].metrics.end(),
            metric_output + index * 15);
    }
    py::array_t<std::int64_t> counters(5);
    auto* counter_values = checked_data(counters);
    counter_values[0] = static_cast<std::int64_t>(candidate_count);
    counter_values[1] =
        static_cast<std::int64_t>(candidate_count) - duplicate_count;
    counter_values[2] = duplicate_count;
    counter_values[3] = negative_hit_count;
    counter_values[4] = screened_count;

    std::string evidence;
    const auto append_i64 = [&evidence](std::int64_t value) {
        const auto bits = static_cast<std::uint64_t>(value);
        for (std::size_t byte = 0; byte < 8; ++byte) {
            evidence.push_back(
                static_cast<char>((bits >> (byte * 8)) & 0xffU));
        }
    };
    const auto append_f64 = [&append_i64](double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        append_i64(static_cast<std::int64_t>(bits));
    };
    for (std::size_t index = 0; index < candidate_count; ++index) {
        const auto begin = static_cast<std::size_t>(offsets[index]);
        const auto end = static_cast<std::size_t>(offsets[index + 1]);
        append_i64(ids[index]);
        append_i64(statuses[index]);
        append_i64(duplicate_of[index]);
        append_i64(static_cast<std::int64_t>(end - begin));
        for (auto cursor = begin; cursor < end; ++cursor) {
            append_i64(routes[cursor]);
        }
        for (const auto code : outputs[index].codes) {
            append_i64(code);
        }
        for (const auto metric : outputs[index].metrics) {
            append_f64(metric);
        }
    }
    for (std::size_t index = 0; index < 5; ++index) {
        append_i64(counter_values[index]);
    }
    const auto digest = native_sha256_hex(evidence);
    return py::make_tuple(
        std::move(returned_ids),
        std::move(status_array),
        std::move(duplicate_array),
        std::move(codes_array),
        std::move(metrics_array),
        std::move(counters),
        digest);
}

py::tuple screen_route_batch_transaction_v2(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle candidate_ids,
    py::handle options,
    py::handle incremental,
    py::handle negative_offsets,
    py::handle negative_indices,
    py::handle negative_reason_codes) {
    return screen_route_batch_transaction_impl(
        node_kind,
        demand,
        ready_time,
        due_date,
        service_time,
        distance,
        reachable,
        vehicle,
        route_offsets,
        route_indices,
        candidate_ids,
        options,
        incremental,
        negative_offsets,
        negative_indices,
        negative_reason_codes,
        1);
}

py::tuple candidate_round_transaction_impl(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle route_offsets,
    py::handle route_indices,
    py::handle candidate_ids,
    py::handle lexical_rank,
    py::handle options,
    py::handle incremental,
    py::handle negative_offsets,
    py::handle negative_indices,
    py::handle negative_reason_codes,
    py::handle cache_hit_flags,
    py::handle control,
    py::handle deadline_remaining,
    py::handle batch_size,
    py::handle context_ids,
    py::handle resource_receipt,
    std::string_view evidence_domain) {
    const auto started = std::chrono::steady_clock::now();
    auto offsets_array = checked_array<std::int64_t>(
        route_offsets, "route_offsets", 1);
    auto routes_array = checked_array<std::int64_t>(
        route_indices, "route_indices", 1);
    auto ids_array = checked_array<std::int64_t>(
        candidate_ids, "candidate_ids", 1);
    auto lexical_array = checked_array<std::int64_t>(
        lexical_rank, "lexical_rank", 1);
    auto cache_array = checked_array<std::int64_t>(
        cache_hit_flags, "cache_hit_flags", 1);
    auto control_array = checked_array<std::int64_t>(control, "control", 1);
    auto deadline_array = checked_array<double>(
        deadline_remaining, "deadline_remaining", 1);
    auto batch_array = checked_array<std::int64_t>(batch_size, "batch_size", 1);
    auto context_array = checked_array<std::int64_t>(context_ids, "context_ids", 1);
    auto receipt_checked = checked_array<std::int64_t>(
        resource_receipt, "resource_receipt", 1);
    if (!receipt_checked.writeable()) {
        throw std::invalid_argument("resource_receipt must be writable");
    }
    auto receipt_array = py::reinterpret_borrow<py::array_t<std::int64_t>>(
        receipt_checked);
    const auto candidate_count = static_cast<std::size_t>(ids_array.request().shape[0]);
    const auto node_count = static_cast<std::size_t>(
        checked_array<std::int64_t>(node_kind, "node_kind", 1).request().shape[0]);
    if (offsets_array.request().shape[0]
            != static_cast<py::ssize_t>(candidate_count + 1)
        || cache_array.request().shape[0]
            != static_cast<py::ssize_t>(candidate_count)) {
        throw std::invalid_argument(
            "candidate round arrays do not share one candidate row count");
    }
    if (lexical_array.request().shape[0] != static_cast<py::ssize_t>(node_count)) {
        throw std::invalid_argument("lexical_rank must contain one value per node");
    }
    if (control_array.request().shape[0] != 3
        || deadline_array.request().shape[0] != 1
        || batch_array.request().shape[0] != 1
        || context_array.request().shape[0] != 3
        || receipt_array.request().shape[0] != 6) {
        throw std::invalid_argument("candidate round scalar-array shape is invalid");
    }
    const auto* offsets = checked_data<std::int64_t>(offsets_array);
    const auto* routes = checked_data<std::int64_t>(routes_array);
    const auto* ids = checked_data<std::int64_t>(ids_array);
    const auto* lexical = checked_data<std::int64_t>(lexical_array);
    const auto* cache_hits = checked_data<std::int64_t>(cache_array);
    const auto* control_values = checked_data<std::int64_t>(control_array);
    const auto* deadline_values = checked_data<double>(deadline_array);
    const auto* batch_values = checked_data<std::int64_t>(batch_array);
    const auto* context_values = checked_data<std::int64_t>(context_array);
    auto* receipt_values = checked_data<std::int64_t>(receipt_array);
    if (receipt_values[0] != 2
        || receipt_values[1] != 0
        || std::any_of(receipt_values + 2, receipt_values + 6,
                       [](std::int64_t value) { return value != 0; })) {
        throw std::invalid_argument(
            "candidate round resource receipt was not initialized for v2");
    }
    if (control_values[0] <= 0 || control_values[1] < 0
        || control_values[2] <= 0) {
        throw std::invalid_argument(
            "candidate round requires positive top-k/thread count and "
            "non-negative exact budget");
    }
    if (!std::isfinite(deadline_values[0]) || deadline_values[0] <= 0.0) {
        throw std::invalid_argument(
            "candidate round deadline_remaining must be finite and positive");
    }
    if (batch_values[0] <= 0) {
        throw std::invalid_argument("candidate round batch_size must be positive");
    }
    std::unordered_set<std::int64_t> lexical_values;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (lexical[node] < 0
            || lexical[node] >= static_cast<std::int64_t>(node_count)
            || !lexical_values.insert(lexical[node]).second) {
            throw std::invalid_argument("lexical_rank must be a permutation");
        }
    }
    for (std::size_t index = 0; index < candidate_count; ++index) {
        if (cache_hits[index] != 0 && cache_hits[index] != 1) {
            throw std::invalid_argument("cache_hit_flags must contain only zero or one");
        }
    }
    // The receipt is caller-owned, contiguous memory.  It remains readable if
    // a later native or Python validation step throws, so started work cannot
    // be silently refunded with the result payload.
    receipt_values[1] = 1;

    const auto screening_started = std::chrono::steady_clock::now();
    py::tuple screening = screen_route_batch_transaction_impl(
        node_kind,
        demand,
        ready_time,
        due_date,
        service_time,
        distance,
        reachable,
        vehicle,
        route_offsets,
        route_indices,
        candidate_ids,
        options,
        incremental,
        negative_offsets,
        negative_indices,
        negative_reason_codes,
        control_values[2]);
    const auto screening_completed = std::chrono::steady_clock::now();
    auto status_array = py::cast<py::array_t<std::int64_t>>(screening[1]);
    auto duplicate_array = py::cast<py::array_t<std::int64_t>>(screening[2]);
    auto codes_array = py::cast<py::array_t<std::int64_t>>(screening[3]);
    auto metrics_array = py::cast<py::array_t<double>>(screening[4]);
    const auto* statuses = checked_data<std::int64_t>(status_array);
    const auto* duplicates = checked_data<std::int64_t>(duplicate_array);
    const auto* codes = checked_data<std::int64_t>(codes_array);
    const auto* metrics = checked_data<double>(metrics_array);

    struct RankableCandidate {
        std::size_t index;
        double distance_lower_bound;
    };
    std::vector<RankableCandidate> rankable;
    rankable.reserve(candidate_count);
    std::int64_t rejected_count = 0;
    std::int64_t negative_hit_count = 0;
    std::int64_t duplicate_count = 0;
    for (std::size_t index = 0; index < candidate_count; ++index) {
        if (statuses[index] == 1) {
            ++duplicate_count;
            continue;
        }
        if (codes[index * 16] == 0) {
            ++rejected_count;
            if (statuses[index] == 2) {
                ++negative_hit_count;
            }
            continue;
        }
        rankable.push_back({index, metrics[index * 15 + 3]});
    }
    const auto route_less = [&](std::size_t left, std::size_t right) {
        auto left_cursor = offsets[left];
        auto right_cursor = offsets[right];
        const auto left_end = offsets[left + 1];
        const auto right_end = offsets[right + 1];
        while (left_cursor < left_end && right_cursor < right_end) {
            const auto left_rank = lexical[routes[left_cursor]];
            const auto right_rank = lexical[routes[right_cursor]];
            if (left_rank != right_rank) {
                return left_rank < right_rank;
            }
            ++left_cursor;
            ++right_cursor;
        }
        return (left_end - offsets[left]) < (right_end - offsets[right]);
    };
    std::stable_sort(
        rankable.begin(),
        rankable.end(),
        [&](const RankableCandidate& left, const RankableCandidate& right) {
            if (left.distance_lower_bound != right.distance_lower_bound) {
                return left.distance_lower_bound < right.distance_lower_bound;
            }
            if (route_less(left.index, right.index)) {
                return true;
            }
            if (route_less(right.index, left.index)) {
                return false;
            }
            return ids[left.index] < ids[right.index];
        });
    const auto selected_count = std::min<std::size_t>(
        rankable.size(), static_cast<std::size_t>(control_values[0]));

    std::vector<std::int64_t> resolutions(candidate_count, 3);
    std::vector<std::int64_t> sources(candidate_count, -1);
    std::vector<std::int64_t> journal(candidate_count * 3, 0);
    for (std::size_t index = 0; index < candidate_count; ++index) {
        journal[index * 3] = ids[index];
        journal[index * 3 + 2] = -1;
        if (statuses[index] == 1) {
            resolutions[index] = 5;
            sources[index] = duplicates[index];
            journal[index * 3 + 1] = 6;
        } else if (codes[index * 16] == 0) {
            resolutions[index] = 0;
            journal[index * 3 + 1] = 5;
        } else {
            journal[index * 3 + 1] = 4;
        }
    }
    std::vector<std::size_t> exact_indices;
    std::int64_t cache_hit_count = 0;
    for (std::size_t rank = 0; rank < selected_count; ++rank) {
        const auto index = rankable[rank].index;
        if (cache_hits[index] == 1) {
            resolutions[index] = 1;
            journal[index * 3 + 1] = 1;
            ++cache_hit_count;
        } else {
            exact_indices.push_back(index);
        }
    }
    std::int64_t budget_skip_count = 0;
    if (exact_indices.size() > static_cast<std::size_t>(control_values[1])) {
        budget_skip_count = static_cast<std::int64_t>(exact_indices.size());
        for (const auto index : exact_indices) {
            resolutions[index] = 4;
            journal[index * 3 + 1] = 3;
        }
        exact_indices.clear();
    }

    std::vector<std::int64_t> exact_offsets(1, 0);
    std::vector<std::int64_t> exact_routes;
    std::vector<std::int64_t> exact_ids;
    exact_ids.reserve(exact_indices.size());
    for (std::size_t ordinal = 0; ordinal < exact_indices.size(); ++ordinal) {
        const auto index = exact_indices[ordinal];
        resolutions[index] = 2;
        journal[index * 3 + 1] = 2;
        journal[index * 3 + 2] = static_cast<std::int64_t>(ordinal);
        exact_ids.push_back(ids[index]);
        exact_routes.insert(
            exact_routes.end(), routes + offsets[index], routes + offsets[index + 1]);
        exact_offsets.push_back(static_cast<std::int64_t>(exact_routes.size()));
    }
    py::array_t<std::int64_t> exact_offsets_array(exact_offsets.size());
    py::array_t<std::int64_t> exact_routes_array(exact_routes.size());
    std::copy(
        exact_offsets.begin(), exact_offsets.end(), checked_data(exact_offsets_array));
    std::copy(
        exact_routes.begin(), exact_routes.end(), checked_data(exact_routes_array));
    const auto exact_started = std::chrono::steady_clock::now();
    const auto elapsed_before_exact = std::chrono::duration<double>(
        exact_started - started).count();
    py::array_t<double> exact_deadline_array(1);
    checked_data(exact_deadline_array)[0] = std::max(
        0.0, deadline_values[0] - elapsed_before_exact);
    receipt_values[1] = 2;
    receipt_values[2] = static_cast<std::int64_t>(exact_ids.size());
    py::tuple exact_payload;
    try {
        exact_payload = exact_charging_batch_numeric(
            node_kind,
            ready_time,
            due_date,
            service_time,
            distance,
            vehicle,
            exact_offsets_array,
            exact_routes_array,
            exact_deadline_array,
            batch_array);
    } catch (...) {
        receipt_values[4] = receipt_values[2];
        throw;
    }
    auto exact_batch_counters = py::cast<py::array_t<std::int64_t>>(
        exact_payload[6]);
    const auto* exact_counter_values = checked_data<std::int64_t>(
        exact_batch_counters);
    receipt_values[1] = 3;
    receipt_values[2] = exact_counter_values[1];
    receipt_values[3] = exact_counter_values[2];
    receipt_values[4] = exact_counter_values[3];
    const auto exact_completed = std::chrono::steady_clock::now();

    for (std::size_t index = 0; index < candidate_count; ++index) {
        if (statuses[index] != 1) {
            continue;
        }
        const auto source_id = duplicates[index];
        const auto source = std::find(ids, ids + candidate_count, source_id);
        if (source == ids + candidate_count) {
            throw std::runtime_error("native duplicate source identity was lost");
        }
        const auto source_index = static_cast<std::size_t>(source - ids);
        sources[index] = source_id;
        if (resolutions[source_index] == 0) {
            resolutions[index] = 0;
        } else if (resolutions[source_index] == 1
                   || resolutions[source_index] == 2) {
            resolutions[index] = 5;
        } else {
            resolutions[index] = resolutions[source_index];
        }
    }

    py::array_t<std::int64_t> resolution_array(resolutions.size());
    py::array_t<std::int64_t> source_array(sources.size());
    py::array_t<std::int64_t> journal_array(
        {static_cast<py::ssize_t>(candidate_count), py::ssize_t(3)});
    py::array_t<std::int64_t> exact_ids_array(exact_ids.size());
    py::array_t<std::int64_t> completion_array(exact_ids.size());
    std::copy(resolutions.begin(), resolutions.end(), checked_data(resolution_array));
    std::copy(sources.begin(), sources.end(), checked_data(source_array));
    std::copy(journal.begin(), journal.end(), checked_data(journal_array));
    std::copy(exact_ids.begin(), exact_ids.end(), checked_data(exact_ids_array));
    std::copy(exact_ids.begin(), exact_ids.end(), checked_data(completion_array));

    py::array_t<std::int64_t> counters(10);
    auto* counter_values = checked_data(counters);
    counter_values[0] = static_cast<std::int64_t>(candidate_count);
    counter_values[1] = static_cast<std::int64_t>(rankable.size());
    counter_values[2] = static_cast<std::int64_t>(selected_count);
    counter_values[3] = rejected_count;
    counter_values[4] = negative_hit_count;
    counter_values[5] = cache_hit_count;
    counter_values[6] = static_cast<std::int64_t>(exact_ids.size());
    counter_values[7] = budget_skip_count;
    counter_values[8] = duplicate_count;
    counter_values[9] = 0;

    const auto completed = std::chrono::steady_clock::now();
    py::array_t<double> timings(4);
    auto* timing_values = checked_data(timings);
    timing_values[0] = std::chrono::duration<double>(
        screening_completed - screening_started).count();
    timing_values[1] = std::chrono::duration<double>(
        exact_completed - exact_started).count();
    timing_values[2] = std::chrono::duration<double>(completed - started).count();
    timing_values[3] = 0.0;

    std::string evidence(evidence_domain);
    const auto append_i64 = [&evidence](std::int64_t value) {
        const auto bits = static_cast<std::uint64_t>(value);
        for (std::size_t byte = 0; byte < 8; ++byte) {
            evidence.push_back(
                static_cast<char>((bits >> (byte * 8)) & 0xffU));
        }
    };
    for (std::size_t index = 0; index < 3; ++index) {
        append_i64(context_values[index]);
    }
    for (const auto value : resolutions) {
        append_i64(value);
    }
    for (const auto value : sources) {
        append_i64(value);
    }
    for (const auto value : journal) {
        append_i64(value);
    }
    for (const auto value : exact_ids) {
        append_i64(value);
    }
    for (const auto value : exact_ids) {
        append_i64(value);
    }
    for (const auto item : exact_payload) {
        evidence += py::cast<std::string>(
            py::reinterpret_borrow<py::object>(item).attr("tobytes")());
    }
    evidence += py::cast<std::string>(screening[6]);
    const auto digest = native_sha256_hex(evidence);
    timing_values[2] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    receipt_values[1] = 4;
    return py::make_tuple(
        std::move(screening),
        std::move(resolution_array),
        std::move(source_array),
        std::move(journal_array),
        std::move(exact_ids_array),
        std::move(completion_array),
        std::move(exact_payload),
        std::move(counters),
        std::move(timings),
        digest);
}

py::tuple candidate_round_transaction_v1(
    py::handle node_kind, py::handle demand, py::handle ready_time,
    py::handle due_date, py::handle service_time, py::handle distance,
    py::handle reachable, py::handle vehicle, py::handle route_offsets,
    py::handle route_indices, py::handle candidate_ids, py::handle lexical_rank,
    py::handle options, py::handle incremental, py::handle negative_offsets,
    py::handle negative_indices, py::handle negative_reason_codes,
    py::handle cache_hit_flags, py::handle control, py::handle deadline_remaining,
    py::handle batch_size, py::handle context_ids) {
    py::array_t<std::int64_t> resource_receipt(6);
    std::fill(
        checked_data(resource_receipt), checked_data(resource_receipt) + 6, 0);
    checked_data(resource_receipt)[0] = 2;
    return candidate_round_transaction_impl(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, route_offsets, route_indices, candidate_ids,
        lexical_rank, options, incremental, negative_offsets, negative_indices,
        negative_reason_codes, cache_hit_flags, control, deadline_remaining,
        batch_size, context_ids, resource_receipt,
        "stage05.2-candidate-round-transaction-v1");
}

py::tuple candidate_round_transaction_v2(
    py::handle node_kind, py::handle demand, py::handle ready_time,
    py::handle due_date, py::handle service_time, py::handle distance,
    py::handle reachable, py::handle vehicle, py::handle route_offsets,
    py::handle route_indices, py::handle candidate_ids, py::handle lexical_rank,
    py::handle options, py::handle incremental, py::handle negative_offsets,
    py::handle negative_indices, py::handle negative_reason_codes,
    py::handle cache_hit_flags, py::handle control, py::handle deadline_remaining,
    py::handle batch_size, py::handle context_ids,
    py::handle resource_receipt) {
    return candidate_round_transaction_impl(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, route_offsets, route_indices, candidate_ids,
        lexical_rank, options, incremental, negative_offsets, negative_indices,
        negative_reason_codes, cache_hit_flags, control, deadline_remaining,
        batch_size, context_ids, resource_receipt,
        "stage05.2-candidate-round-transaction-v2");
}

py::tuple full_native_alns_v1(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining) {
    const auto started = std::chrono::steady_clock::now();
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto demand_array = checked_array<double>(demand, "demand", 1);
    auto ready_array = checked_array<double>(ready_time, "ready_time", 1);
    auto due_array = checked_array<double>(due_date, "due_date", 1);
    auto service_array = checked_array<double>(service_time, "service_time", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto vehicle_array = checked_array<double>(vehicle, "vehicle", 1);
    auto lexical_array = checked_array<std::int64_t>(
        lexical_rank, "lexical_rank", 1);
    auto initial_offsets_array = checked_array<std::int64_t>(
        initial_route_offsets, "initial_route_offsets", 1);
    auto initial_indices_array = checked_array<std::int64_t>(
        initial_route_indices, "initial_route_indices", 1);
    auto control_array = checked_array<std::int64_t>(control, "control", 1);
    auto deadline_array = checked_array<double>(
        deadline_remaining, "deadline_remaining", 1);
    const auto node_count = static_cast<std::size_t>(kind_array.request().shape[0]);
    if (node_count == 0
        || demand_array.request().shape[0] != kind_array.request().shape[0]
        || ready_array.request().shape[0] != kind_array.request().shape[0]
        || due_array.request().shape[0] != kind_array.request().shape[0]
        || service_array.request().shape[0] != kind_array.request().shape[0]
        || lexical_array.request().shape[0] != kind_array.request().shape[0]) {
        throw std::invalid_argument("full native node arrays must share one length");
    }
    if (distance_array.request().shape[0] != kind_array.request().shape[0]
        || distance_array.request().shape[1] != kind_array.request().shape[0]
        || vehicle_array.request().shape[0] != 5
        || control_array.request().shape[0] != 5
        || deadline_array.request().shape[0] != 1) {
        throw std::invalid_argument("full native fixed-array shape is invalid");
    }
    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* demands = checked_data<double>(demand_array);
    const auto* ready = checked_data<double>(ready_array);
    const auto* due = checked_data<double>(due_array);
    const auto* lexical = checked_data<std::int64_t>(lexical_array);
    const auto* initial_offsets = checked_data<std::int64_t>(initial_offsets_array);
    const auto* initial_indices = checked_data<std::int64_t>(initial_indices_array);
    const auto* vehicle_values = checked_data<double>(vehicle_array);
    const auto* control_values = checked_data<std::int64_t>(control_array);
    const auto* deadline_values = checked_data<double>(deadline_array);
    if (control_values[1] <= 0 || control_values[2] <= 0
        || control_values[3] <= 0 || control_values[4] < -1) {
        throw std::invalid_argument("full native iteration/batch/thread counts must be positive");
    }
    if (!std::isfinite(deadline_values[0]) || deadline_values[0] <= 0.0) {
        throw std::invalid_argument("full native deadline must be finite and positive");
    }
    std::vector<std::int64_t> customers;
    for (std::size_t node = 0; node < node_count; ++node) {
        if (kinds[node] == customer_kind) {
            customers.push_back(static_cast<std::int64_t>(node));
        }
    }
    if (customers.empty()) {
        throw std::invalid_argument("full native solve requires at least one customer");
    }
    std::stable_sort(
        customers.begin(),
        customers.end(),
        [&](std::int64_t left, std::int64_t right) {
            if (due[left] != due[right]) {
                return due[left] < due[right];
            }
            if (ready[left] != ready[right]) {
                return ready[left] < ready[right];
            }
            return lexical[left] < lexical[right];
        });
    std::vector<std::vector<std::int64_t>> routes;
    if (initial_offsets_array.size() > 0) {
        const auto route_count = static_cast<std::size_t>(initial_offsets_array.size() - 1);
        if (initial_offsets[0] != 0
            || initial_offsets[route_count] != initial_indices_array.size()) {
            throw std::invalid_argument("full native initial routes do not span their indices");
        }
        std::vector<bool> seen(node_count, false);
        for (std::size_t route = 0; route < route_count; ++route) {
            if (initial_offsets[route] < 0
                || initial_offsets[route] >= initial_offsets[route + 1]) {
                throw std::invalid_argument("full native initial routes must be non-empty");
            }
            routes.emplace_back(
                initial_indices + initial_offsets[route],
                initial_indices + initial_offsets[route + 1]);
            for (const auto node : routes.back()) {
                if (node < 0 || static_cast<std::size_t>(node) >= node_count
                    || kinds[node] != customer_kind || seen[static_cast<std::size_t>(node)]) {
                    throw std::invalid_argument(
                        "full native initial routes must cover unique customers");
                }
                seen[static_cast<std::size_t>(node)] = true;
            }
        }
        for (const auto customer : customers) {
            if (!seen[static_cast<std::size_t>(customer)]) {
                throw std::invalid_argument(
                    "full native initial routes must cover every customer");
            }
        }
    } else {
        double route_demand = 0.0;
        for (const auto customer : customers) {
            if (routes.empty()
                || route_demand + demands[customer] > vehicle_values[1] + 1e-9) {
                routes.emplace_back();
                route_demand = 0.0;
            }
            routes.back().push_back(customer);
            route_demand += demands[customer];
        }
    }
    const auto pack_routes = [](const std::vector<std::vector<std::int64_t>>& values) {
        std::size_t item_count = 0;
        for (const auto& route : values) {
            item_count += route.size();
        }
        py::array_t<std::int64_t> offsets(values.size() + 1);
        py::array_t<std::int64_t> indices(item_count);
        auto* offset_values = checked_data(offsets);
        auto* index_values = checked_data(indices);
        std::size_t cursor = 0;
        offset_values[0] = 0;
        for (std::size_t route = 0; route < values.size(); ++route) {
            std::copy(values[route].begin(), values[route].end(), index_values + cursor);
            cursor += values[route].size();
            offset_values[route + 1] = static_cast<std::int64_t>(cursor);
        }
        return std::make_pair(std::move(offsets), std::move(indices));
    };
    py::array_t<double> exact_deadline(1);
    py::array_t<std::int64_t> exact_batch_size(1);
    checked_data(exact_batch_size)[0] = control_values[2];
    const auto exact_budget = control_values[4];
    const auto remaining_seconds = [&]() {
        return deadline_values[0] - std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    };
    auto [customer_offsets, customer_indices] = pack_routes(routes);
    if (exact_budget >= 0
        && static_cast<std::int64_t>(routes.size()) > exact_budget) {
        throw std::runtime_error(
            "full native initial route batch does not fit the exact-call budget");
    }
    checked_data(exact_deadline)[0] = remaining_seconds();
    py::tuple exact_payload = exact_charging_batch_numeric(
        node_kind, ready_time, due_date, service_time, distance, vehicle,
        customer_offsets, customer_indices, exact_deadline, exact_batch_size);
    auto exact_status = py::cast<py::array_t<std::int64_t>>(exact_payload[2]);
    auto all_exact_feasible = [](const py::array_t<std::int64_t>& statuses) {
        const auto* values = checked_data<std::int64_t>(statuses);
        for (py::ssize_t index = 0; index < statuses.size(); ++index) {
            if (values[index] != 0) {
                return false;
            }
        }
        return true;
    };
    std::int64_t started_calls = static_cast<std::int64_t>(routes.size());
    std::int64_t completed_calls = 0;
    while (!all_exact_feasible(exact_status)) {
        auto initial_batch_counters = py::cast<py::array_t<std::int64_t>>(
            exact_payload[6]);
        completed_calls += checked_data(initial_batch_counters)[2];
        if (initial_offsets_array.size() > 0) {
            throw std::runtime_error("full native supplied initial routes are not exact-feasible");
        }
        const auto* status_values = checked_data<std::int64_t>(exact_status);
        std::vector<std::vector<std::int64_t>> split_routes;
        for (std::size_t route = 0; route < routes.size(); ++route) {
            if (status_values[route] == 0) {
                split_routes.push_back(routes[route]);
                continue;
            }
            if (routes[route].size() <= 1) {
                throw std::runtime_error(
                    "full native singleton initial route is not exact-feasible");
            }
            const auto middle = routes[route].size() / 2;
            split_routes.emplace_back(routes[route].begin(), routes[route].begin() + middle);
            split_routes.emplace_back(routes[route].begin() + middle, routes[route].end());
        }
        routes = std::move(split_routes);
        if (exact_budget >= 0
            && started_calls + static_cast<std::int64_t>(routes.size()) > exact_budget) {
            throw std::runtime_error(
                "full native initial-route split does not fit the exact-call budget");
        }
        std::tie(customer_offsets, customer_indices) = pack_routes(routes);
        checked_data(exact_deadline)[0] = remaining_seconds();
        exact_payload = exact_charging_batch_numeric(
            node_kind, ready_time, due_date, service_time, distance, vehicle,
            customer_offsets, customer_indices, exact_deadline, exact_batch_size);
        started_calls += static_cast<std::int64_t>(routes.size());
        exact_status = py::cast<py::array_t<std::int64_t>>(exact_payload[2]);
    }
    auto initial_batch_counters = py::cast<py::array_t<std::int64_t>>(exact_payload[6]);
    completed_calls += checked_data(initial_batch_counters)[2];
    std::int64_t accepted_moves = 0;
    std::int64_t improving_moves = 0;
    std::int64_t rejected_moves = 0;
    std::int64_t interrupted_calls = 0;
    std::int64_t completed_iterations = 0;
    std::vector<std::int64_t> trajectory;
    trajectory.reserve(static_cast<std::size_t>(control_values[1]) * 7);
    const auto objective_key = [&](const py::tuple& payload, std::size_t vehicle_count) {
        auto path_array = py::cast<py::array_t<std::int64_t>>(payload[1]);
        auto metrics_array = py::cast<py::array_t<double>>(payload[4]);
        const auto* paths = checked_data<std::int64_t>(path_array);
        const auto* metrics = checked_data<double>(metrics_array);
        double total_distance = 0.0;
        double total_charging_time = 0.0;
        for (py::ssize_t route = 0; route < metrics_array.shape(0); ++route) {
            total_distance += metrics[route * 4];
            total_charging_time += metrics[route * 4 + 3];
        }
        std::int64_t charging_count = 0;
        for (py::ssize_t index = 0; index < path_array.size(); ++index) {
            charging_count += kinds[paths[index]] == station_kind ? 1 : 0;
        }
        return evrptw::formal_objective::key(
            static_cast<std::int64_t>(vehicle_count),
            total_distance,
            total_charging_time,
            charging_count);
    };
    auto current_objective = objective_key(exact_payload, routes.size());
    std::size_t pair_cursor = static_cast<std::size_t>(
        static_cast<std::uint64_t>(control_values[0]) % std::max<std::size_t>(1, routes.size()));
    PythonRandom random(static_cast<std::uint64_t>(control_values[0]));
    for (std::int64_t iteration = 0; iteration < control_values[1]; ++iteration) {
        if (remaining_seconds() <= 0.0) {
            break;
        }
        ++completed_iterations;
        auto candidate_routes = routes;
        const auto operator_id = static_cast<std::int64_t>(random.randbelow(12));
        auto status_code = std::int64_t{0};
        if (routes.size() >= 2) {
            const auto left = pair_cursor % routes.size();
            auto right = (left + 1 + static_cast<std::size_t>(
                random.randbelow(routes.size() - 1)))
                % routes.size();
            if (right == left) {
                right = (right + 1) % routes.size();
            }
            pair_cursor = (pair_cursor + 1) % routes.size();
            if (operator_id == 0 || operator_id == 1 || operator_id == 11) {
                const auto first = std::min(left, right);
                const auto second = std::max(left, right);
                std::vector<std::int64_t> merged = routes[left];
                if (operator_id == 1) {
                    merged.insert(merged.begin(), routes[right].begin(), routes[right].end());
                } else {
                    merged.insert(merged.end(), routes[right].begin(), routes[right].end());
                }
                candidate_routes[first] = std::move(merged);
                candidate_routes.erase(
                    candidate_routes.begin() + static_cast<std::ptrdiff_t>(second));
            } else if (operator_id == 2 && routes[left].size() > 1) {
                const auto node = candidate_routes[left].back();
                candidate_routes[left].pop_back();
                candidate_routes[right].insert(candidate_routes[right].begin(), node);
            } else if (operator_id == 3) {
                std::swap(candidate_routes[left].front(), candidate_routes[right].front());
            } else if (operator_id == 4) {
                const auto left_cut = candidate_routes[left].size() / 2;
                const auto right_cut = candidate_routes[right].size() / 2;
                std::vector<std::int64_t> left_tail(
                    candidate_routes[left].begin() + static_cast<std::ptrdiff_t>(left_cut),
                    candidate_routes[left].end());
                std::vector<std::int64_t> right_tail(
                    candidate_routes[right].begin() + static_cast<std::ptrdiff_t>(right_cut),
                    candidate_routes[right].end());
                candidate_routes[left].erase(
                    candidate_routes[left].begin() + static_cast<std::ptrdiff_t>(left_cut),
                    candidate_routes[left].end());
                candidate_routes[right].erase(
                    candidate_routes[right].begin() + static_cast<std::ptrdiff_t>(right_cut),
                    candidate_routes[right].end());
                candidate_routes[left].insert(
                    candidate_routes[left].end(), right_tail.begin(), right_tail.end());
                candidate_routes[right].insert(
                    candidate_routes[right].end(), left_tail.begin(), left_tail.end());
            } else if (operator_id == 5 && routes[left].size() > 1) {
                const auto segment_size = std::max<std::size_t>(1, routes[left].size() / 3);
                candidate_routes[right].insert(
                    candidate_routes[right].end(),
                    candidate_routes[left].begin(),
                    candidate_routes[left].begin() + static_cast<std::ptrdiff_t>(segment_size));
                candidate_routes[left].erase(
                    candidate_routes[left].begin(),
                    candidate_routes[left].begin() + static_cast<std::ptrdiff_t>(segment_size));
            } else if (operator_id == 6 && routes.size() >= 3) {
                const auto third = (right + 1) % routes.size();
                if (third != left) {
                    const auto first_node = candidate_routes[left].front();
                    candidate_routes[left].front() = candidate_routes[right].front();
                    candidate_routes[right].front() = candidate_routes[third].front();
                    candidate_routes[third].front() = first_node;
                }
            } else {
                auto& route = candidate_routes[left];
                if (route.size() > 1) {
                    if (operator_id == 8) {
                        std::stable_sort(
                            route.begin(), route.end(),
                            [&](std::int64_t a, std::int64_t b) {
                                return std::tie(due[a], ready[a], lexical[a])
                                    < std::tie(due[b], ready[b], lexical[b]);
                            });
                    } else if (operator_id == 9) {
                        std::rotate(route.begin(), route.begin() + 1, route.end());
                    } else {
                        std::reverse(route.begin(), route.end());
                    }
                }
            }
        } else if (!candidate_routes.empty() && candidate_routes[0].size() > 1) {
            auto& route = candidate_routes[0];
            if (operator_id == 8) {
                std::stable_sort(
                    route.begin(), route.end(),
                    [&](std::int64_t a, std::int64_t b) {
                        return std::tie(due[a], ready[a], lexical[a])
                            < std::tie(due[b], ready[b], lexical[b]);
                    });
            } else {
                std::rotate(route.begin(), route.begin() + 1, route.end());
            }
        }
        bool structurally_valid = candidate_routes != routes;
        for (const auto& route : candidate_routes) {
            double route_demand = 0.0;
            for (const auto node : route) {
                route_demand += demands[node];
            }
            structurally_valid = structurally_valid && !route.empty()
                && route_demand <= vehicle_values[1] + 1e-9;
        }
        if (!structurally_valid) {
            ++rejected_moves;
            trajectory.insert(
                trajectory.end(),
                {iteration, operator_id, static_cast<std::int64_t>(routes.size()),
                 started_calls, status_code, 0, 0});
            continue;
        }
        const auto candidate_calls = static_cast<std::int64_t>(candidate_routes.size());
        if (exact_budget >= 0 && started_calls + candidate_calls > exact_budget) {
            ++rejected_moves;
            status_code = 1;
            trajectory.insert(
                trajectory.end(),
                {iteration, operator_id, static_cast<std::int64_t>(routes.size()),
                 started_calls, status_code, 0, 0});
            continue;
        }
        auto [candidate_offsets, candidate_indices] = pack_routes(candidate_routes);
        checked_data(exact_deadline)[0] = remaining_seconds();
        auto candidate_payload = exact_charging_batch_numeric(
            node_kind, ready_time, due_date, service_time, distance, vehicle,
            candidate_offsets, candidate_indices, exact_deadline, exact_batch_size);
        started_calls += candidate_calls;
        auto candidate_status = py::cast<py::array_t<std::int64_t>>(candidate_payload[2]);
        auto candidate_batch_counters = py::cast<py::array_t<std::int64_t>>(
            candidate_payload[6]);
        const auto* batch_counter_values = checked_data(candidate_batch_counters);
        completed_calls += batch_counter_values[2];
        interrupted_calls += batch_counter_values[3];
        if (!all_exact_feasible(candidate_status)) {
            ++rejected_moves;
            status_code = batch_counter_values[3] > 0 ? 5 : 2;
            trajectory.insert(
                trajectory.end(),
                {iteration, operator_id, static_cast<std::int64_t>(routes.size()),
                 started_calls, status_code, 0, 0});
            if (batch_counter_values[3] > 0) {
                break;
            }
            continue;
        }
        const auto candidate_objective = objective_key(
            candidate_payload, candidate_routes.size());
        const bool vehicle_improvement = candidate_routes.size() < routes.size();
        if (!(candidate_objective < current_objective)) {
            ++rejected_moves;
            status_code = 4;
            trajectory.insert(
                trajectory.end(),
                {iteration, operator_id, static_cast<std::int64_t>(routes.size()),
                 started_calls, status_code, 0, 0});
            continue;
        }
        routes = std::move(candidate_routes);
        customer_offsets = std::move(candidate_offsets);
        customer_indices = std::move(candidate_indices);
        exact_payload = std::move(candidate_payload);
        current_objective = candidate_objective;
        ++accepted_moves;
        ++improving_moves;
        status_code = 3;
        trajectory.insert(
            trajectory.end(),
            {iteration, operator_id, static_cast<std::int64_t>(routes.size()),
             started_calls, status_code, 1, vehicle_improvement ? 1 : 0});
    }
    py::array_t<std::int64_t> trajectory_array(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(trajectory.size() / 7), 7});
    std::copy(
        trajectory.begin(), trajectory.end(), checked_data(trajectory_array));
    py::array_t<std::int64_t> counters(8);
    auto* counter_values = checked_data(counters);
    counter_values[0] = completed_iterations;
    counter_values[1] = started_calls;
    counter_values[2] = completed_calls;
    counter_values[3] = accepted_moves;
    counter_values[4] = improving_moves;
    counter_values[5] = rejected_moves;
    counter_values[6] = interrupted_calls;
    counter_values[7] = 0;
    py::array_t<double> timings(4);
    auto* timing_values = checked_data(timings);
    timing_values[0] = 0.0;
    timing_values[1] = 0.0;
    timing_values[2] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    timing_values[3] = 0.0;
    std::string evidence = "stage05.2-full-native-alns-v1";
    evidence += py::cast<std::string>(customer_offsets.attr("tobytes")());
    evidence += py::cast<std::string>(customer_indices.attr("tobytes")());
    for (const auto item : exact_payload) {
        evidence += py::cast<std::string>(
            py::reinterpret_borrow<py::object>(item).attr("tobytes")());
    }
    evidence += py::cast<std::string>(counters.attr("tobytes")());
    evidence += py::cast<std::string>(trajectory_array.attr("tobytes")());
    const auto digest = native_sha256_hex(evidence);
    timing_values[2] = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return py::make_tuple(
        std::move(customer_offsets),
        std::move(customer_indices),
        std::move(exact_payload),
        std::move(counters),
        std::move(timings),
        std::move(trajectory_array),
        digest);
}

py::tuple full_native_initialize_impl_v2(
    py::handle node_kind,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle vehicle,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining,
    bool require_feasible) {
    auto control_array = checked_array<std::int64_t>(control, "control", 1);
    auto offsets_array = checked_array<std::int64_t>(
        initial_route_offsets, "initial_route_offsets", 1);
    if (control_array.size() != 5 || offsets_array.size() < 2) {
        throw std::invalid_argument(
            "full native initialization control/warm-start shape is invalid");
    }
    const auto route_count = static_cast<std::int64_t>(offsets_array.size() - 1);
    const auto* control_values = checked_data<std::int64_t>(control_array);
    if (control_values[2] <= 0 || control_values[4] < -1) {
        throw std::invalid_argument(
            "full native initialization batch/budget values are invalid");
    }
    if (control_values[4] >= 0 && route_count > control_values[4]) {
        throw std::runtime_error(
            "full native warm start does not fit the exact-call budget");
    }
    py::array_t<std::int64_t> batch_size(1);
    checked_data(batch_size)[0] = control_values[2];
    const auto exact_payload = exact_charging_batch_numeric(
        node_kind,
        ready_time,
        due_date,
        service_time,
        distance,
        vehicle,
        initial_route_offsets,
        initial_route_indices,
        deadline_remaining,
        batch_size);
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto status_array = py::cast<py::array_t<std::int64_t>>(exact_payload[2]);
    auto path_array = py::cast<py::array_t<std::int64_t>>(exact_payload[1]);
    auto metrics_array = py::cast<py::array_t<double>>(exact_payload[4]);
    auto batch_counters = py::cast<py::array_t<std::int64_t>>(exact_payload[6]);
    const auto* statuses = checked_data<std::int64_t>(status_array);
    for (py::ssize_t index = 0; index < status_array.size(); ++index) {
        if (require_feasible && statuses[index] != 0) {
            throw std::runtime_error(
                "full native supplied warm start is not exact-feasible");
        }
    }
    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* paths = checked_data<std::int64_t>(path_array);
    const auto* metrics = checked_data<double>(metrics_array);
    double total_distance = 0.0;
    double total_charging_time = 0.0;
    for (py::ssize_t route = 0; route < metrics_array.shape(0); ++route) {
        total_distance += metrics[route * 4];
        total_charging_time += metrics[route * 4 + 3];
    }
    std::int64_t charging_count = 0;
    for (py::ssize_t index = 0; index < path_array.size(); ++index) {
        charging_count += kinds[paths[index]] == station_kind ? 1 : 0;
    }
    py::array_t<std::int64_t> objective_integer(2);
    checked_data(objective_integer)[0] = route_count;
    checked_data(objective_integer)[1] = charging_count;
    py::array_t<double> objective_float(2);
    checked_data(objective_float)[0] = total_distance;
    checked_data(objective_float)[1] = total_charging_time;
    py::array_t<std::int64_t> accounting(4);
    const auto* batch_values = checked_data<std::int64_t>(batch_counters);
    checked_data(accounting)[0] = route_count;
    checked_data(accounting)[1] = batch_values[2];
    checked_data(accounting)[2] = batch_values[3];
    checked_data(accounting)[3] = 0;
    return py::make_tuple(
        exact_payload,
        std::move(objective_integer),
        std::move(objective_float),
        std::move(accounting));
}

py::tuple full_native_initialize_v2(
    py::handle node_kind,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle vehicle,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining) {
    return full_native_initialize_impl_v2(
        node_kind, ready_time, due_date, service_time, distance, vehicle,
        initial_route_offsets, initial_route_indices, control,
        deadline_remaining, true);
}

class NativeExactDeadlineInterruption final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Stage 5.2's full-native causal journal is deliberately a separate ABI
// product from the historical semantic and storage journals.  The stream and
// event codes are integer-only so a consumer never has to reconstruct order
// from phase-specific Python projections.
enum class NativeCausalStreamCode : std::int64_t {
    operator_event = 0,
    candidate_control = 1,
    screening = 2,
    cache = 3,
    exact = 4,
    stage04 = 5,
    deadline = 6,
    termination = 7,
};

enum class NativeCausalEventCode : std::int64_t {
    candidate_plan = 1,
    screening_decision = 2,
    exact_route_result = 3,
    deadline_boundary = 4,
    termination = 5,
    exact_work = 6,
    cache_lookup = 7,
    cache_store = 8,
    operator_outcome = 9,
    stage04_outcome = 10,
    budget_boundary = 11,
};

class NativeCausalJournalV2 {
public:
    static constexpr std::size_t stream_count = 8;

    struct Snapshot {
        std::size_t row_count = 0;
        std::array<std::int64_t, stream_count> stream_counts{};
        bool terminal_recorded = false;
    };

    [[nodiscard]] Snapshot snapshot() const noexcept {
        return Snapshot{event_ids_.size(), stream_counts_, terminal_recorded_};
    }

    void rollback_noexcept(const Snapshot& snapshot) noexcept {
        event_ids_.resize(snapshot.row_count);
        stream_codes_.resize(snapshot.row_count);
        event_codes_.resize(snapshot.row_count);
        lane_ids_.resize(snapshot.row_count);
        operator_ids_.resize(snapshot.row_count);
        iterations_.resize(snapshot.row_count);
        transaction_ids_.resize(snapshot.row_count);
        subject_ids_.resize(snapshot.row_count);
        status_codes_.resize(snapshot.row_count);
        flags_.resize(snapshot.row_count);
        stream_counts_ = snapshot.stream_counts;
        terminal_recorded_ = snapshot.terminal_recorded;
    }

    void append(
        NativeCausalStreamCode stream,
        NativeCausalEventCode event,
        std::int64_t lane_id,
        std::int64_t operator_id,
        std::int64_t iteration,
        std::int64_t transaction_id,
        std::int64_t subject_id,
        std::int64_t status_code,
        std::int64_t flags) {
        const auto stream_value = static_cast<std::int64_t>(stream);
        if (stream_value < 0
            || stream_value >= static_cast<std::int64_t>(stream_count)) {
            throw std::invalid_argument("native causal stream code is invalid");
        }
        const auto event_value = static_cast<std::int64_t>(event);
        if (event_value < static_cast<std::int64_t>(
                NativeCausalEventCode::candidate_plan)
            || event_value > static_cast<std::int64_t>(
                NativeCausalEventCode::budget_boundary)) {
            throw std::invalid_argument("native causal event code is invalid");
        }
        const auto event_matches_stream = [stream, event]() {
            switch (stream) {
            case NativeCausalStreamCode::operator_event:
                return event == NativeCausalEventCode::operator_outcome;
            case NativeCausalStreamCode::candidate_control:
                return event == NativeCausalEventCode::candidate_plan;
            case NativeCausalStreamCode::screening:
                return event == NativeCausalEventCode::screening_decision;
            case NativeCausalStreamCode::cache:
                return event == NativeCausalEventCode::cache_lookup
                    || event == NativeCausalEventCode::cache_store;
            case NativeCausalStreamCode::exact:
                return event == NativeCausalEventCode::exact_route_result
                    || event == NativeCausalEventCode::exact_work;
            case NativeCausalStreamCode::stage04:
                return event == NativeCausalEventCode::stage04_outcome;
            case NativeCausalStreamCode::deadline:
                return event == NativeCausalEventCode::deadline_boundary
                    || event == NativeCausalEventCode::budget_boundary;
            case NativeCausalStreamCode::termination:
                return event == NativeCausalEventCode::termination;
            }
            return false;
        };
        if (!event_matches_stream()) {
            throw std::invalid_argument(
                "native causal event does not belong to its stream");
        }
        if (terminal_recorded_) {
            throw std::logic_error(
                "native causal journal cannot append after termination");
        }
        if (iteration < -1 || status_code < 0 || flags < 0) {
            throw std::invalid_argument("native causal event fields are invalid");
        }
        if (stream == NativeCausalStreamCode::deadline
            || stream == NativeCausalStreamCode::termination) {
            if (lane_id != -1 || operator_id != -1 || iteration < 0
                || transaction_id != -1 || subject_id != -1) {
                throw std::invalid_argument(
                    "native causal boundary fields are invalid");
            }
            if (stream == NativeCausalStreamCode::termination) {
                if (status_code > 3 || flags != 0) {
                    throw std::invalid_argument(
                        "native causal termination fields are invalid");
                }
            } else if (status_code < 1 || status_code > 3 || flags != 1
                || (status_code == 2
                    && event != NativeCausalEventCode::deadline_boundary)
                || (status_code != 2
                    && event != NativeCausalEventCode::budget_boundary)) {
                throw std::invalid_argument(
                    "native causal deadline fields are invalid");
            }
        } else if (lane_id < 0 || operator_id < 0 || transaction_id < 0
            || subject_id < 0) {
            throw std::invalid_argument(
                "native causal transaction fields are invalid");
        }
        if (event_ids_.size()
            > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("native causal journal event ID overflow");
        }
        const auto before = snapshot();
        try {
            const auto event_id = static_cast<std::int64_t>(event_ids_.size());
            event_ids_.push_back(event_id);
            stream_codes_.push_back(stream_value);
            event_codes_.push_back(static_cast<std::int64_t>(event));
            lane_ids_.push_back(lane_id);
            operator_ids_.push_back(operator_id);
            iterations_.push_back(iteration);
            transaction_ids_.push_back(transaction_id);
            subject_ids_.push_back(subject_id);
            status_codes_.push_back(status_code);
            flags_.push_back(flags);
            ++stream_counts_[static_cast<std::size_t>(stream_value)];
            terminal_recorded_ =
                stream == NativeCausalStreamCode::termination;
        } catch (...) {
            rollback_noexcept(before);
            throw;
        }
    }

    [[nodiscard]] py::tuple payload() const {
        const auto make_array = [](const std::vector<std::int64_t>& values) {
            py::array_t<std::int64_t> output(values.size());
            std::copy(values.begin(), values.end(), checked_data(output));
            return output;
        };
        auto event_ids = make_array(event_ids_);
        auto stream_codes = make_array(stream_codes_);
        auto event_codes = make_array(event_codes_);
        auto lane_ids = make_array(lane_ids_);
        auto operator_ids = make_array(operator_ids_);
        auto iterations = make_array(iterations_);
        auto transaction_ids = make_array(transaction_ids_);
        auto subject_ids = make_array(subject_ids_);
        auto status_codes = make_array(status_codes_);
        auto flags = make_array(flags_);
        py::array_t<std::int64_t> stream_counts(stream_count);
        std::copy(
            stream_counts_.begin(), stream_counts_.end(),
            checked_data(stream_counts));
        std::string evidence("stage05.2-native-causal-journal-v2");
        append_evidence_array(evidence, event_ids);
        append_evidence_array(evidence, stream_codes);
        append_evidence_array(evidence, event_codes);
        append_evidence_array(evidence, lane_ids);
        append_evidence_array(evidence, operator_ids);
        append_evidence_array(evidence, iterations);
        append_evidence_array(evidence, transaction_ids);
        append_evidence_array(evidence, subject_ids);
        append_evidence_array(evidence, status_codes);
        append_evidence_array(evidence, flags);
        append_evidence_array(evidence, stream_counts);
        return py::make_tuple(
            std::move(event_ids), std::move(stream_codes),
            std::move(event_codes), std::move(lane_ids),
            std::move(operator_ids), std::move(iterations),
            std::move(transaction_ids), std::move(subject_ids),
            std::move(status_codes), std::move(flags),
            std::move(stream_counts), native_sha256_hex(evidence));
    }

private:
    std::vector<std::int64_t> event_ids_;
    std::vector<std::int64_t> stream_codes_;
    std::vector<std::int64_t> event_codes_;
    std::vector<std::int64_t> lane_ids_;
    std::vector<std::int64_t> operator_ids_;
    std::vector<std::int64_t> iterations_;
    std::vector<std::int64_t> transaction_ids_;
    std::vector<std::int64_t> subject_ids_;
    std::vector<std::int64_t> status_codes_;
    std::vector<std::int64_t> flags_;
    std::array<std::int64_t, stream_count> stream_counts_{};
    bool terminal_recorded_ = false;
};

class NativeSearchEngineV2 {
public:
    NativeSearchEngineV2(
        std::int64_t exact_budget,
        std::int64_t round_budget,
        std::int64_t cache_entries,
        std::int64_t cache_memory_bytes,
        std::int64_t negative_cache_entries,
        std::int64_t proposal_top_k,
        double screening_epsilon,
        std::int64_t worker_count)
        : NativeSearchEngineV2(
              exact_budget, round_budget, cache_entries, cache_memory_bytes,
              negative_cache_entries, proposal_top_k, screening_epsilon,
              worker_count, nullptr) {}

    NativeSearchEngineV2(
        std::int64_t exact_budget,
        std::int64_t round_budget,
        std::int64_t cache_entries,
        std::int64_t cache_memory_bytes,
        std::int64_t negative_cache_entries,
        std::int64_t proposal_top_k,
        double screening_epsilon,
        std::int64_t worker_count,
        std::shared_ptr<NativeWorkPool> shared_work_pool)
        : route_cache_(cache_entries, cache_memory_bytes),
          negative_cache_(negative_cache_entries),
          budget_(exact_budget, round_budget),
          proposal_top_k_(proposal_top_k),
          screening_epsilon_(screening_epsilon),
          worker_count_(worker_count),
          work_pool_(
              shared_work_pool
                  ? std::move(shared_work_pool)
                  : std::make_shared<NativeWorkPool>(worker_count)) {
        if (proposal_top_k_ <= 0) {
            throw std::invalid_argument(
                "full native proposal_top_k must be positive");
        }
        if (!std::isfinite(screening_epsilon_) || screening_epsilon_ <= 0.0) {
            throw std::invalid_argument(
                "full native screening epsilon must be finite and positive");
        }
        const bool remote_scheduler_owns_work =
#ifdef __linux__
            !native_kernel_scheduler_endpoint.empty();
#else
            false;
#endif
        if (worker_count_ < 0
            || (worker_count_ == 0 && !remote_scheduler_owns_work)) {
            throw std::invalid_argument(
                "full native worker_count is invalid for its scheduler mode");
        }
        if (work_pool_->thread_count() < worker_count_) {
            throw std::invalid_argument(
                "native work pool has fewer threads than the solve contract");
        }
    }

    void configure_node_names(
        py::handle name_offsets, py::handle name_bytes) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock() || initialized_) {
            throw std::runtime_error(
                "full native node names must be configured before initialization");
        }
        auto offsets = checked_array<std::int64_t>(
            name_offsets, "node_name_offsets", 1);
        auto bytes = checked_array<std::uint8_t>(
            name_bytes, "node_name_bytes", 1);
        if (offsets.size() < 2 || checked_data<std::int64_t>(offsets)[0] != 0
            || checked_data<std::int64_t>(offsets)[offsets.size() - 1]
                != bytes.size()) {
            throw std::invalid_argument("full native node-name SoA is invalid");
        }
        std::vector<std::string> decoded;
        decoded.reserve(static_cast<std::size_t>(offsets.size() - 1));
        std::unordered_set<std::string> unique;
        for (py::ssize_t index = 0; index + 1 < offsets.size(); ++index) {
            const auto first = checked_data<std::int64_t>(offsets)[index];
            const auto last = checked_data<std::int64_t>(offsets)[index + 1];
            if (first < 0 || first >= last || last > bytes.size()) {
                throw std::invalid_argument(
                    "full native node-name offsets must be non-empty and monotonic");
            }
            std::string name(
                reinterpret_cast<const char*>(checked_data<std::uint8_t>(bytes) + first),
                static_cast<std::size_t>(last - first));
            if (!unique.insert(name).second) {
                throw std::invalid_argument("full native node names must be unique");
            }
            decoded.push_back(std::move(name));
        }
        node_names_ = std::move(decoded);
    }

    void configure_node_names_owned(
        const evrptw::native_search::ProblemV2& problem) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock() || initialized_) {
            throw std::runtime_error(
                "full native node names must be configured before initialization");
        }
        const auto& offsets = problem.node_name_offsets;
        const auto& bytes = problem.node_name_bytes;
        if (offsets.size() < 2 || offsets.front() != 0
            || offsets.back() != static_cast<std::int64_t>(bytes.size())) {
            throw std::invalid_argument("full native node-name SoA is invalid");
        }
        std::vector<std::string> decoded;
        decoded.reserve(offsets.size() - 1);
        std::unordered_set<std::string> unique;
        for (std::size_t index = 0; index + 1 < offsets.size(); ++index) {
            const auto first = offsets[index];
            const auto last = offsets[index + 1];
            if (first < 0 || first >= last
                || last > static_cast<std::int64_t>(bytes.size())) {
                throw std::invalid_argument(
                    "full native node-name offsets must be non-empty and monotonic");
            }
            std::string name(
                reinterpret_cast<const char*>(bytes.data() + first),
                static_cast<std::size_t>(last - first));
            if (!unique.insert(name).second) {
                throw std::invalid_argument("full native node names must be unique");
            }
            decoded.push_back(std::move(name));
        }
        node_names_ = std::move(decoded);
    }

    py::tuple initialize(
        py::handle node_kind,
        py::handle demand,
        py::handle ready_time,
        py::handle due_date,
        py::handle service_time,
        py::handle distance,
        py::handle reachable,
        py::handle vehicle,
        py::handle lexical_rank,
        py::handle initial_route_offsets,
        py::handle initial_route_indices,
        py::handle control,
        py::handle deadline_remaining) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (initialized_) {
            throw std::runtime_error(
                "full native v2 search engine cannot be initialized twice");
        }
        node_kind_ = owned_array_copy<std::int64_t>(node_kind, "node_kind", 1);
        demand_ = owned_array_copy<double>(demand, "demand", 1);
        ready_time_ = owned_array_copy<double>(ready_time, "ready_time", 1);
        due_date_ = owned_array_copy<double>(due_date, "due_date", 1);
        service_time_ = owned_array_copy<double>(service_time, "service_time", 1);
        distance_ = owned_array_copy<double>(distance, "distance", 2);
        reachable_ = owned_array_copy<std::uint8_t>(reachable, "reachable", 2);
        vehicle_ = owned_array_copy<double>(vehicle, "vehicle", 1);
        lexical_rank_ = owned_array_copy<std::int64_t>(
            lexical_rank, "lexical_rank", 1);
        const auto node_count = node_kind_.size();
        if (node_names_.empty()) {
            throw std::invalid_argument(
                "full native node names must be configured before initialization");
        }
        if (node_names_.size() != static_cast<std::size_t>(node_count)) {
            throw std::invalid_argument(
                "full native node names do not align with node arrays");
        }
        if (node_count == 0 || demand_.size() != node_count
            || ready_time_.size() != node_count || due_date_.size() != node_count
            || service_time_.size() != node_count || lexical_rank_.size() != node_count
            || distance_.shape(0) != node_count || distance_.shape(1) != node_count
            || reachable_.shape(0) != node_count || reachable_.shape(1) != node_count
            || vehicle_.size() != 5) {
            throw std::invalid_argument(
                "full native search problem arrays do not share one shape");
        }
        depot_ = -1;
        recharge_nodes_.clear();
        all_customers_.clear();
        std::unordered_set<std::int64_t> lexical_values;
        const auto* kinds = checked_data<std::int64_t>(node_kind_);
        const auto* lexical = checked_data<std::int64_t>(lexical_rank_);
        for (py::ssize_t node = 0; node < node_count; ++node) {
            if (kinds[node] == depot_kind) {
                if (depot_ >= 0) {
                    throw std::invalid_argument(
                        "full native search requires exactly one depot");
                }
                depot_ = node;
                recharge_nodes_.push_back(node);
            } else if (kinds[node] == station_kind) {
                recharge_nodes_.push_back(node);
            } else if (kinds[node] == customer_kind) {
                all_customers_.insert(node);
            } else {
                throw std::invalid_argument(
                    "full native search node_kind contains an unknown code");
            }
            if (lexical[node] < 0 || lexical[node] >= node_count
                || !lexical_values.insert(lexical[node]).second) {
                throw std::invalid_argument(
                    "full native lexical_rank must be a permutation");
            }
        }
        if (depot_ < 0) {
            throw std::invalid_argument(
                "full native search requires exactly one depot");
        }
        if (all_customers_.empty()) {
            throw std::invalid_argument(
                "full native search requires at least one customer");
        }
        auto offsets_array = owned_array_copy<std::int64_t>(
            initial_route_offsets, "initial_route_offsets", 1);
        auto indices_array = owned_array_copy<std::int64_t>(
            initial_route_indices, "initial_route_indices", 1);
        auto control_array = owned_array_copy<std::int64_t>(
            control, "control", 1);
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        if (control_array.size() != 5) {
            throw std::invalid_argument(
                "full native search control must contain five values");
        }
        const auto* search_control = checked_data<std::int64_t>(control_array);
        if (search_control[0] < 0 || search_control[1] <= 0
            || search_control[2] <= 0 || search_control[3] <= 0
            || search_control[4] < -1) {
            throw std::invalid_argument(
                "full native search control values are invalid");
        }
        PythonRandom prepared_rng(
            static_cast<std::uint64_t>(search_control[0]));
        PythonRandom prepared_constraint_rng(
            static_cast<std::uint64_t>(search_control[0]) ^ 0x5EED23ULL);
        if (offsets_array.size() < 2) {
            throw std::invalid_argument(
                "full native warm start requires at least one route");
        }
        const auto route_count = static_cast<std::int64_t>(offsets_array.size() - 1);
        const auto warm_transaction_id = allocate_causal_transaction();
        causal_context_ = {
            stable_int63("initialization"), stable_int63("initial_solution"), -1};
        causal_context_transaction_id_ = warm_transaction_id;
        const auto* warm_offsets = checked_data<std::int64_t>(offsets_array);
        if (warm_offsets[0] != 0
            || warm_offsets[route_count] != indices_array.size()) {
            throw std::invalid_argument(
                "full native warm-start offsets do not span route indices");
        }
        for (std::int64_t route = 0; route < route_count; ++route) {
            if (warm_offsets[route] < 0
                || warm_offsets[route] >= warm_offsets[route + 1]) {
                throw std::invalid_argument(
                    "full native warm-start routes must be non-empty and monotonic");
            }
        }
        std::unordered_set<std::int64_t> warm_customers;
        const auto* warm_indices = checked_data<std::int64_t>(indices_array);
        for (py::ssize_t index = 0; index < indices_array.size(); ++index) {
            const auto customer = warm_indices[index];
            if (!all_customers_.contains(customer)
                || !warm_customers.insert(customer).second) {
                throw std::invalid_argument(
                    "full native warm start must contain every customer exactly once");
            }
        }
        if (warm_customers != all_customers_) {
            throw std::invalid_argument(
                "full native warm start must contain every customer exactly once");
        }
        std::vector<evrptw::native_kernels::ScreenOutput> warm_screening(
            static_cast<std::size_t>(route_count));
        std::vector<std::size_t> warm_rows(
            static_cast<std::size_t>(route_count));
        std::iota(warm_rows.begin(), warm_rows.end(), std::size_t{0});
        std::vector<std::int64_t> warm_negative_hits(
            static_cast<std::size_t>(route_count), 0);
        const std::array<double, 4> warm_screen_options{
            1.0, screening_epsilon_, 0.0, 0.0};
        const std::array<double, 6> warm_no_incremental{};
        const auto* warm_demands = checked_data<double>(demand_);
        const auto* warm_ready = checked_data<double>(ready_time_);
        const auto* warm_due = checked_data<double>(due_date_);
        const auto* warm_service = checked_data<double>(service_time_);
        const auto* warm_distances = checked_data<double>(distance_);
        const auto* warm_reachable = checked_data<std::uint8_t>(reachable_);
        const auto* warm_vehicle = checked_data<double>(vehicle_);
        const auto warm_screening_started = std::chrono::steady_clock::now();
        screening_occupancies_.push_back(
            static_cast<std::int64_t>(route_count));
#ifdef __linux__
        const auto warm_scheduler_endpoint = native_kernel_scheduler_endpoint;
        const auto warm_scheduler_required = native_kernel_scheduler_required;
        auto* warm_telemetry_collector =
            evrptw::native_client::telemetry_collector;
#endif
        {
            py::gil_scoped_release release;
#ifdef __linux__
            if (!warm_scheduler_endpoint.empty()) {
                warm_screening = dispatch_screen_routes(
                    kinds, warm_demands, warm_ready, warm_due, warm_service,
                    warm_distances, warm_reachable, warm_vehicle,
                    warm_offsets, warm_indices,
                    static_cast<std::size_t>(route_count),
                    static_cast<std::size_t>(warm_offsets[route_count]),
                    static_cast<std::size_t>(node_count), depot_, recharge_nodes_,
                    warm_screen_options.data(), warm_no_incremental.data());
            } else
#endif
            {
            work_pool_->parallel_for(
                static_cast<std::size_t>(route_count), [&](std::size_t route) {
#ifdef __linux__
                    NativeSchedulerThreadContext scheduler_context(
                        warm_scheduler_endpoint, warm_scheduler_required,
                        warm_telemetry_collector);
#endif
                    const auto first = warm_offsets[route];
                    const auto last = warm_offsets[route + 1];
                    warm_screening[route] =
                        dispatch_screen_route(
                        kinds,
                        warm_demands,
                        warm_ready,
                        warm_due,
                        warm_service,
                        warm_distances,
                        warm_reachable,
                        warm_vehicle,
                        warm_indices + first,
                        static_cast<std::size_t>(last - first),
                        static_cast<std::size_t>(node_count),
                        depot_, recharge_nodes_, warm_screen_options.data(),
                        warm_no_incremental.data());
                });
            }
        }
        record_screening_outputs(
            warm_screening, warm_rows, warm_rows,
            {
                stable_int63("initialization"),
                stable_int63("initial_solution"),
                -1,
            },
            warm_offsets, warm_indices, warm_negative_hits.data(), 0,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - warm_screening_started).count(),
            warm_transaction_id);
        if (std::any_of(
                warm_screening.begin(), warm_screening.end(),
                [](const evrptw::native_kernels::ScreenOutput& output) {
                    return output.codes[0] != 1;
                })) {
            throw std::runtime_error(
                "full native supplied warm start failed safe screening");
        }
        // Mirror Candidate Control's warm-start cache transaction: the
        // candidate is looked up before exact work and the committed incumbent
        // is looked up again before it is installed into all search lanes.
        auto warm_lookup_before = route_cache_.lookup_exact_many(
            offsets_array, indices_array);
        auto warm_hit_flags_before = py::cast<py::array_t<std::int64_t>>(
            warm_lookup_before[0]);
        append_causal_cache_events(
            {stable_int63("initialization"), stable_int63("initial_solution"), -1},
            warm_transaction_id, warm_hit_flags_before, false);
        if (std::any_of(
                checked_data<std::int64_t>(warm_hit_flags_before),
                checked_data<std::int64_t>(warm_hit_flags_before) + route_count,
                [](std::int64_t value) { return value != 0; })) {
            throw std::logic_error("full native warm-start cache was not empty");
        }
        const auto warm_budget_remaining = budget_.exact_remaining();
        if (warm_budget_remaining >= 0 && route_count > warm_budget_remaining) {
            throw std::runtime_error(
                "full native warm start does not fit the exact-call budget");
        }
        const auto reservation = budget_.reserve_exact(route_count);
        const auto* reserved = checked_data<std::int64_t>(reservation);
        if (reserved[1] != route_count) {
            throw std::runtime_error(
                "full native warm start does not fit the exact-call budget");
        }
        py::tuple initialized;
        const auto warm_exact_started = std::chrono::steady_clock::now();
        append_causal_exact_work(
            {stable_int63("initialization"), stable_int63("initial_solution"), -1},
            warm_transaction_id, route_count);
        try {
            initialized = full_native_initialize_impl_v2(
                node_kind_,
                ready_time_,
                due_date_,
                service_time_,
                distance_,
                vehicle_,
                offsets_array,
                indices_array,
                control_array,
                deadline_array,
                false);
        } catch (...) {
            budget_.interrupt_exact(route_count);
            throw;
        }
        auto exact_payload = py::cast<py::tuple>(initialized[0]);
        auto exact_counters = py::cast<py::array_t<std::int64_t>>(exact_payload[6]);
        const auto* counter_values = checked_data<std::int64_t>(exact_counters);
        if (counter_values[2] != route_count || counter_values[3] != 0) {
            const auto completed = std::clamp<std::int64_t>(
                counter_values[2], 0, route_count);
            budget_.complete_exact(completed);
            budget_.interrupt_exact(route_count - completed);
            throw std::runtime_error(
                "full native warm-start exact transaction did not complete atomically");
        }
        record_exact_backend_metrics(
            exact_payload,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - warm_exact_started).count());
        budget_.complete_exact(route_count);
        record_exact_journal_batch(
            {
                stable_int63("initialization"),
                stable_int63("initial_solution"),
                -1,
            },
            offsets_array,
            indices_array,
            exact_payload,
            warm_transaction_id);

        auto status_array = py::cast<py::array_t<std::int64_t>>(exact_payload[2]);
        auto reason_array = py::cast<py::array_t<std::int64_t>>(exact_payload[3]);
        auto metrics_array = py::cast<py::array_t<double>>(exact_payload[4]);
        auto labels_array = py::cast<py::array_t<std::int64_t>>(exact_payload[5]);
        auto path_offsets_array = py::cast<py::array_t<std::int64_t>>(exact_payload[0]);
        auto path_indices_array = py::cast<py::array_t<std::int64_t>>(exact_payload[1]);
        const auto* route_offsets = checked_data<std::int64_t>(offsets_array);
        const auto* route_indices = checked_data<std::int64_t>(indices_array);
        const auto* path_offsets = checked_data<std::int64_t>(path_offsets_array);
        const auto* path_indices = checked_data<std::int64_t>(path_indices_array);
        const auto* statuses = checked_data<std::int64_t>(status_array);
        const auto* reasons = checked_data<std::int64_t>(reason_array);
        const auto* metrics = checked_data<double>(metrics_array);
        const auto* labels = checked_data<std::int64_t>(labels_array);
        for (std::int64_t route = 0; route < route_count; ++route) {
            if (statuses[route] != 0) {
                throw std::runtime_error(
                    "full native supplied warm start is not exact-feasible");
            }
        }
        py::array_t<std::uint8_t> semantic_hashes(
            {static_cast<py::ssize_t>(route_count), py::ssize_t(32)});
        py::array_t<std::int64_t> entry_bytes(route_count);
        for (std::int64_t route = 0; route < route_count; ++route) {
            std::string evidence("stage05.2-native-route-result-v2");
            append_evidence_values(
                evidence,
                route_indices + route_offsets[route],
                static_cast<std::size_t>(route_offsets[route + 1] - route_offsets[route]));
            append_evidence_values(evidence, statuses + route, 1);
            append_evidence_values(evidence, reasons + route, 1);
            append_evidence_values(evidence, metrics + route * 4, 4);
            append_evidence_values(evidence, labels + route * 3, 3);
            append_evidence_values(
                evidence,
                path_indices + path_offsets[route],
                static_cast<std::size_t>(path_offsets[route + 1] - path_offsets[route]));
            const auto digest = native_sha256_digest(evidence);
            std::copy(
                digest.begin(), digest.end(),
                checked_data(semantic_hashes) + route * 32);
        }
        entry_bytes = exact_entry_bytes(
            path_offsets_array,
            path_indices_array,
            status_array,
            reason_array,
            metrics_array,
            labels_array);
        auto prepared_current_exact = owned_exact_state_copy(exact_payload);
        auto prepared_current_objective_integer = owned_array_copy<std::int64_t>(
            initialized[1], "initial_objective_integer", 1);
        auto prepared_current_objective_float = owned_array_copy<double>(
            initialized[2], "initial_objective_float", 1);
        auto prepared_legacy_offsets = owned_array_copy<std::int64_t>(
            offsets_array, "legacy_initial_route_offsets", 1);
        auto prepared_legacy_indices = owned_array_copy<std::int64_t>(
            indices_array, "legacy_initial_route_indices", 1);
        auto prepared_legacy_exact = owned_exact_state_copy(exact_payload);
        auto prepared_legacy_objective_integer = owned_array_copy<std::int64_t>(
            initialized[1], "legacy_initial_objective_integer", 1);
        auto prepared_legacy_objective_float = owned_array_copy<double>(
            initialized[2], "legacy_initial_objective_float", 1);
        auto prepared_quality_offsets = owned_array_copy<std::int64_t>(
            offsets_array, "quality_initial_route_offsets", 1);
        auto prepared_quality_indices = owned_array_copy<std::int64_t>(
            indices_array, "quality_initial_route_indices", 1);
        auto prepared_quality_exact = owned_exact_state_copy(exact_payload);
        auto prepared_quality_objective_integer = owned_array_copy<std::int64_t>(
            initialized[1], "quality_initial_objective_integer", 1);
        auto prepared_quality_objective_float = owned_array_copy<double>(
            initialized[2], "quality_initial_objective_float", 1);
        auto prepared_best_offsets = offsets_array;
        auto prepared_best_indices = indices_array;
        auto prepared_best_exact = prepared_current_exact;
        auto prepared_best_objective_integer = prepared_current_objective_integer;
        auto prepared_best_objective_float = prepared_current_objective_float;
        route_cache_.begin_store_exact_many_atomic(
            initial_route_offsets,
            initial_route_indices,
            exact_payload[0],
            exact_payload[1],
            exact_payload[2],
            exact_payload[3],
            exact_payload[4],
            exact_payload[5],
            semantic_hashes,
            entry_bytes);
        route_cache_.prepare_store_commit();
        route_cache_.commit_store_batch_noexcept();
        append_causal_cache_events(
            {stable_int63("initialization"), stable_int63("initial_solution"), -1},
            warm_transaction_id, warm_hit_flags_before, true);
        // Candidate Control re-applies safe screening before accepting the
        // newly cached warm incumbent.  Reuse the already proven output as a
        // semantic-only decision; this is not a second physical kernel call.
        record_screening_outputs(
            warm_screening, std::vector<std::size_t>{}, warm_rows,
            {
                stable_int63("initialization"),
                stable_int63("initial_solution"),
                -1,
            },
            warm_offsets, warm_indices, warm_negative_hits.data(), 0, 0.0,
            warm_transaction_id);
        auto warm_lookup_after = route_cache_.lookup_exact_many(
            offsets_array, indices_array);
        auto warm_hit_flags_after = py::cast<py::array_t<std::int64_t>>(
            warm_lookup_after[0]);
        append_causal_cache_events(
            {stable_int63("initialization"), stable_int63("initial_solution"), -1},
            warm_transaction_id, warm_hit_flags_after, false);
        py::array_t<std::int64_t> initial_plan_offsets(2);
        checked_data(initial_plan_offsets)[0] = 0;
        checked_data(initial_plan_offsets)[1] = route_count;
        py::array_t<std::int64_t> initial_plan_ids(1);
        checked_data(initial_plan_ids)[0] = 0;
        attempted_plans_.begin_mark_many_atomic(
            initial_plan_offsets, offsets_array, indices_array, initial_plan_ids);
        attempted_plans_.prepare_mark_commit();
        attempted_plans_.commit_mark_batch_noexcept();
        ControlJournalBatch initial_control_journal;
        initial_control_journal.context = {
            stable_int63("initialization"), stable_int63("initial_solution"), -1};
        initial_control_journal.transaction_id = warm_transaction_id;
        initial_control_journal.plan_offsets = {0, route_count};
        initial_control_journal.route_offsets.assign(
            route_offsets, route_offsets + route_count + 1);
        initial_control_journal.route_indices.assign(
            route_indices, route_indices + indices_array.size());
        initial_control_journal.ranked = {0};
        initial_control_journal.selected = {0};
        initial_control_journal.decision_codes = {2};
        initial_control_journal.statuses = {5};
        initial_control_journal.ranking_integer = {route_count, route_count};
        initial_control_journal.ranking_float = {
            checked_data<double>(prepared_current_objective_float)[0]};
        initial_control_journal.route_resolutions.assign(route_count, 2);
        auto initial_cache_snapshot = route_cache_.snapshot();
        auto initial_cache_statistics = py::cast<py::array_t<std::int64_t>>(
            initial_cache_snapshot[4]);
        initial_control_journal.cache_statistics.assign(
            checked_data<std::int64_t>(initial_cache_statistics),
            checked_data<std::int64_t>(initial_cache_statistics)
                + initial_cache_statistics.size());
        auto initial_budget_state = budget_.state();
        initial_control_journal.budget_state.assign(
            checked_data<std::int64_t>(initial_budget_state),
            checked_data<std::int64_t>(initial_budget_state)
                + initial_budget_state.size());
        initial_control_journal.protocol_flags = {0, 0, 0, 0};
        append_control_journal_causal(
            std::list<ControlJournalBatch>{initial_control_journal});
        control_journal_.push_back(std::move(initial_control_journal));
        current_offsets_ = std::move(offsets_array);
        current_indices_ = std::move(indices_array);
        current_exact_payload_ = std::move(prepared_current_exact);
        current_objective_integer_ =
            std::move(prepared_current_objective_integer);
        current_objective_float_ = std::move(prepared_current_objective_float);
        legacy_offsets_ = std::move(prepared_legacy_offsets);
        legacy_indices_ = std::move(prepared_legacy_indices);
        legacy_exact_payload_ = std::move(prepared_legacy_exact);
        legacy_objective_integer_ =
            std::move(prepared_legacy_objective_integer);
        legacy_objective_float_ = std::move(prepared_legacy_objective_float);
        quality_offsets_ = std::move(prepared_quality_offsets);
        quality_indices_ = std::move(prepared_quality_indices);
        quality_exact_payload_ = std::move(prepared_quality_exact);
        quality_objective_integer_ =
            std::move(prepared_quality_objective_integer);
        quality_objective_float_ = std::move(prepared_quality_objective_float);
        best_offsets_ = std::move(prepared_best_offsets);
        best_indices_ = std::move(prepared_best_indices);
        best_exact_payload_ = std::move(prepared_best_exact);
        best_objective_integer_ = std::move(prepared_best_objective_integer);
        best_objective_float_ = std::move(prepared_best_objective_float);
        batch_size_ = checked_data<std::int64_t>(control_array)[2];
        rng_.emplace(std::move(prepared_rng));
        constraint_rng_.emplace(std::move(prepared_constraint_rng));
        initialized_ = true;
        return initialized;
    }

    py::tuple evaluate_plans(
        py::handle plan_offsets,
        py::handle route_offsets,
        py::handle route_indices,
        py::handle context_ids,
        py::handle deadline_remaining,
        py::handle batch_size,
        py::handle expected_customer_indices) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        const auto transaction_started = std::chrono::steady_clock::now();
        if (!initialized_) {
            throw std::runtime_error(
                "full native search engine must be initialized before plan evaluation");
        }
        if (last_candidate_ready_) {
            throw std::runtime_error(
                "full native search engine has an unapplied candidate");
        }
        auto plans_array = owned_array_copy<std::int64_t>(
            plan_offsets, "plan_offsets", 1);
        auto routes_array = owned_array_copy<std::int64_t>(
            route_offsets, "route_offsets", 1);
        auto indices_array = owned_array_copy<std::int64_t>(
            route_indices, "route_indices", 1);
        auto context_array = owned_array_copy<std::int64_t>(
            context_ids, "context_ids", 1);
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "batch_size", 1);
        auto expected_array = owned_array_copy<std::int64_t>(
            expected_customer_indices, "expected_customer_indices", 1);
        if (context_array.size() != 3 || deadline_array.size() != 1
            || batch_array.size() != 1
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native plan transaction scalar-array shape is invalid");
        }
        const auto* context = checked_data<std::int64_t>(context_array);
        if (context[0] < 0 || context[1] < 0 || context[2] < -1) {
            throw std::invalid_argument(
                "full native plan transaction context IDs are invalid");
        }
        const auto causal_transaction_id = allocate_causal_transaction();
        const auto causal_snapshot = causal_journal_.snapshot();
        causal_context_ = {context[0], context[1], context[2]};
        causal_context_transaction_id_ = causal_transaction_id;
        const auto remaining_seconds = checked_data<double>(deadline_array)[0];
        if (!std::isfinite(remaining_seconds) || remaining_seconds <= 0.0) {
            throw std::invalid_argument(
                "full native plan transaction deadline must be finite and positive");
        }
        const auto plans = NativeAttemptedPlanSetV2::decode_plans(
            plans_array, routes_array, indices_array);
        const auto plan_count = plans.size();
        const auto route_count = static_cast<std::size_t>(routes_array.size() - 1);
        const auto* plan_boundaries = checked_data<std::int64_t>(plans_array);
        const auto* route_boundaries = checked_data<std::int64_t>(routes_array);
        const auto* route_nodes = checked_data<std::int64_t>(indices_array);
        if (plan_count == 0 || route_count == 0) {
            throw std::invalid_argument(
                "full native plan transaction requires a non-empty plan pool");
        }
        std::vector<std::int64_t> complete_customer_indices(
            all_customers_.begin(), all_customers_.end());
        const auto prepared_plans = evrptw::native_candidate_plan::prepare({
            std::span<const std::int64_t>(plan_boundaries, plan_count + 1),
            std::span<const std::int64_t>(route_boundaries, route_count + 1),
            std::span<const std::int64_t>(
                route_nodes, static_cast<std::size_t>(indices_array.size())),
            std::span<const std::int64_t>(
                checked_data<std::int64_t>(expected_array),
                static_cast<std::size_t>(expected_array.size())),
            std::span<const std::int64_t>(
                checked_data<std::int64_t>(node_kind_),
                static_cast<std::size_t>(node_kind_.size())),
            std::span<const std::int64_t>(
                checked_data<std::int64_t>(lexical_rank_),
                static_cast<std::size_t>(lexical_rank_.size())),
            std::span<const std::int64_t>(
                complete_customer_indices.data(), complete_customer_indices.size()),
            customer_kind,
            allow_partial_customer_coverage_,
        });
        const auto& canonical_expected = prepared_plans.canonical_expected;
        py::array_t<std::int64_t> canonical_expected_array(
            canonical_expected.size());
        std::copy(
            canonical_expected.begin(), canonical_expected.end(),
            checked_data(canonical_expected_array));

        auto round_budget_snapshot = budget_.native_snapshot();
        if (!suppress_round_budget_) {
            // Candidate Control owns one exact budget across every ALNS lane
            // in the same iteration while the state retains the real lane for
            // replay and transaction hashing.
            budget_.begin_shared_iteration_round(context[0], context[2]);
        }
        bool round_protocol_active = false;
        bool negative_store_active = false;
        bool attempted_mark_active = false;
        try {
        auto attempted_flags = suppress_attempted_plan_journal_
            ? py::array_t<std::int64_t>(plan_count)
            : attempted_plans_.lookup(plans_array, routes_array, indices_array);
        if (suppress_attempted_plan_journal_) {
            std::fill(
                checked_data(attempted_flags),
                checked_data(attempted_flags) + plan_count,
                std::int64_t{0});
        }
        const auto* attempted = checked_data<std::int64_t>(attempted_flags);
        py::array_t<double> lower_bounds(route_count);
        std::vector<std::int64_t> eligible = prepared_plans.coverage_eligible;
        std::vector<std::vector<std::int64_t>> unique_screen_routes;
        unique_screen_routes.reserve(
            prepared_plans.unique_route_offsets.size() - 1);
        for (std::size_t row = 0;
             row + 1 < prepared_plans.unique_route_offsets.size(); ++row) {
            unique_screen_routes.emplace_back(
                prepared_plans.unique_route_indices.begin()
                    + prepared_plans.unique_route_offsets[row],
                prepared_plans.unique_route_indices.begin()
                    + prepared_plans.unique_route_offsets[row + 1]);
        }
        std::vector<std::size_t> screen_row_by_route(route_count);
        std::transform(
            prepared_plans.unique_row_by_route.begin(),
            prepared_plans.unique_row_by_route.end(),
            screen_row_by_route.begin(),
            [](std::int64_t row) { return static_cast<std::size_t>(row); });
        std::array<double, 4> screen_options{
            1.0, screening_epsilon_, 0.0, 0.0};
        std::array<double, 6> no_incremental{};
        const auto* kinds = checked_data<std::int64_t>(node_kind_);
        const auto* demands = checked_data<double>(demand_);
        const auto* ready = checked_data<double>(ready_time_);
        const auto* due = checked_data<double>(due_date_);
        const auto* service = checked_data<double>(service_time_);
        const auto* distances = checked_data<double>(distance_);
        const auto* reachable = checked_data<std::uint8_t>(reachable_);
        const auto* vehicle = checked_data<double>(vehicle_);
        const auto node_count = static_cast<std::size_t>(node_kind_.size());
        const auto& unique_offsets = prepared_plans.unique_route_offsets;
        const auto& unique_indices = prepared_plans.unique_route_indices;
        py::array_t<std::int64_t> unique_offsets_array(unique_offsets.size());
        py::array_t<std::int64_t> unique_indices_array(unique_indices.size());
        std::copy(
            unique_offsets.begin(), unique_offsets.end(),
            checked_data(unique_offsets_array));
        std::copy(
            unique_indices.begin(), unique_indices.end(),
            checked_data(unique_indices_array));
        py::array_t<std::int64_t> negative_hit_array(unique_screen_routes.size());
        py::array_t<std::int64_t> negative_reason_array(unique_screen_routes.size());
        if (suppress_plan_screening_negative_cache_) {
            std::fill(
                checked_data(negative_hit_array),
                checked_data(negative_hit_array) + negative_hit_array.size(),
                std::int64_t{0});
            std::fill(
                checked_data(negative_reason_array),
                checked_data(negative_reason_array) + negative_reason_array.size(),
                std::int64_t{0});
        } else {
            auto negative_lookup = negative_cache_.lookup_many(
                unique_offsets_array, unique_indices_array);
            negative_hit_array = py::cast<py::array_t<std::int64_t>>(
                negative_lookup[0]);
            negative_reason_array = py::cast<py::array_t<std::int64_t>>(
                negative_lookup[1]);
        }
        const auto* negative_hits = checked_data<std::int64_t>(negative_hit_array);
        const auto* negative_reasons = checked_data<std::int64_t>(
            negative_reason_array);
        std::vector<evrptw::native_kernels::ScreenOutput> screen_outputs(
            unique_screen_routes.size());
        std::vector<std::size_t> screen_rows;
        std::int64_t negative_hit_count = 0;
        for (std::size_t row = 0; row < unique_screen_routes.size(); ++row) {
            if (negative_hits[row] != 0) {
                screen_outputs[row].codes[0] = 0;
                screen_outputs[row].codes[1] = negative_reasons[row];
                ++negative_hit_count;
            } else {
                screen_rows.push_back(row);
            }
        }
        const auto screening_started = std::chrono::steady_clock::now();
        screening_occupancies_.push_back(
            static_cast<std::int64_t>(screen_rows.size()));
#ifdef __linux__
        const auto pool_scheduler_endpoint = native_kernel_scheduler_endpoint;
        const auto pool_scheduler_required = native_kernel_scheduler_required;
        auto* pool_telemetry_collector =
            evrptw::native_client::telemetry_collector;
#endif
        {
            py::gil_scoped_release release;
#ifdef __linux__
            if (!pool_scheduler_endpoint.empty() && !screen_rows.empty()) {
                std::vector<std::int64_t> remote_offsets{0};
                std::vector<std::int64_t> remote_indices;
                for (const auto row : screen_rows) {
                    const auto& sequence = unique_screen_routes[row];
                    remote_indices.insert(
                        remote_indices.end(), sequence.begin(), sequence.end());
                    remote_offsets.push_back(
                        static_cast<std::int64_t>(remote_indices.size()));
                }
                auto remote_outputs = dispatch_screen_routes(
                    kinds, demands, ready, due, service, distances,
                    reachable, vehicle,
                    remote_offsets.data(), remote_indices.data(),
                    screen_rows.size(), remote_indices.size(), node_count,
                    depot_, recharge_nodes_,
                    screen_options.data(), no_incremental.data());
                for (std::size_t position = 0;
                     position < screen_rows.size(); ++position) {
                    screen_outputs[screen_rows[position]] =
                        std::move(remote_outputs[position]);
                }
            } else
#endif
            {
            work_pool_->parallel_for(screen_rows.size(), [&](std::size_t position) {
#ifdef __linux__
                NativeSchedulerThreadContext scheduler_context(
                    pool_scheduler_endpoint, pool_scheduler_required,
                    pool_telemetry_collector);
#endif
                const auto row = screen_rows[position];
                const auto& sequence = unique_screen_routes[row];
                screen_outputs[row] = dispatch_screen_route(
                    kinds, demands, ready, due, service, distances,
                    reachable, vehicle, sequence.data(), sequence.size(),
                    node_count, depot_, recharge_nodes_,
                    screen_options.data(), no_incremental.data());
            });
            }
        }
        record_screening_outputs(
            screen_outputs, screen_rows, screen_row_by_route,
            {context[0], context[1], context[2]},
            route_boundaries, route_nodes, negative_hits, negative_hit_count,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - screening_started).count(),
            causal_transaction_id);
        std::vector<std::int64_t> rejected_offsets{0};
        std::vector<std::int64_t> rejected_indices;
        std::vector<std::int64_t> rejected_reasons;
        for (const auto row : screen_rows) {
            const auto& screen = screen_outputs[row];
            if (screen.codes[0] == 1) {
                continue;
            }
            rejected_indices.insert(
                rejected_indices.end(), unique_screen_routes[row].begin(),
                unique_screen_routes[row].end());
            rejected_offsets.push_back(
                static_cast<std::int64_t>(rejected_indices.size()));
            rejected_reasons.push_back(screen.codes[1]);
        }
        py::array_t<std::int64_t> rejected_offsets_array(rejected_offsets.size());
        py::array_t<std::int64_t> rejected_indices_array(rejected_indices.size());
        py::array_t<std::int64_t> rejected_reasons_array(rejected_reasons.size());
        std::copy(
            rejected_offsets.begin(), rejected_offsets.end(),
            checked_data(rejected_offsets_array));
        std::copy(
            rejected_indices.begin(), rejected_indices.end(),
            checked_data(rejected_indices_array));
        std::copy(
            rejected_reasons.begin(), rejected_reasons.end(),
            checked_data(rejected_reasons_array));
        if (!suppress_plan_screening_negative_cache_
            && !rejected_reasons.empty()) {
            negative_cache_.begin_store_many_atomic(
                rejected_offsets_array, rejected_indices_array,
                rejected_reasons_array);
            negative_store_active = true;
        }
        std::vector<std::int64_t> screening_passed(route_count);
        for (std::size_t route = 0; route < route_count; ++route) {
            const auto& screen = screen_outputs[screen_row_by_route[route]];
            checked_data(lower_bounds)[route] = screen.metrics[3];
            screening_passed[route] = screen.codes[0] == 1 ? 1 : 0;
        }
        const auto decision = evrptw::native_candidate_plan::decide({
            std::span<const std::int64_t>(plan_boundaries, plan_count + 1),
            std::span<const std::int64_t>(eligible.data(), eligible.size()),
            std::span<const std::int64_t>(
                screening_passed.data(), screening_passed.size()),
            std::span<const std::int64_t>(attempted, plan_count),
            static_cast<std::int64_t>(current_offsets_.size() - 1),
        });
        eligible = decision.eligible;
        py::array_t<std::int64_t> combined_attempted(plan_count);
        std::copy(
            decision.combined_attempted.begin(),
            decision.combined_attempted.end(),
            checked_data(combined_attempted));
        auto ranking = rank_candidate_plans_v1(
            plans_array, routes_array, indices_array, lower_bounds,
            current_offsets_, current_indices_, lexical_rank_, combined_attempted,
            proposal_top_k_);
        auto selected_array = py::cast<py::array_t<std::int64_t>>(ranking[1]);
        const auto* selected = checked_data<std::int64_t>(selected_array);
        std::vector<std::int64_t> statuses(plan_count, 2);
        for (std::size_t plan = 0; plan < plan_count; ++plan) {
            statuses[plan] = eligible[plan] == 0 ? 0 : attempted[plan] != 0 ? 1 : 2;
        }
        py::array_t<std::int64_t> objective_integer(
            {static_cast<py::ssize_t>(plan_count), py::ssize_t(2)});
        py::array_t<double> objective_float(
            {static_cast<py::ssize_t>(plan_count), py::ssize_t(2)});
        std::fill(
            checked_data(objective_integer),
            checked_data(objective_integer) + plan_count * 2, -1);
        std::fill(
            checked_data(objective_float),
            checked_data(objective_float) + plan_count * 2,
            std::numeric_limits<double>::quiet_NaN());
        std::vector<std::int64_t> route_resolutions(route_count, 0);
        std::vector<std::optional<NativeRouteCacheV2::ExactPayload>> route_payloads(
            route_count);
        std::vector<std::int64_t> exact_route_rows;
        std::vector<std::int64_t> completion_order;
        std::int64_t feasible_count = 0;
        std::int64_t infeasible_count = 0;
        std::int64_t budget_skip_count = 0;
        std::vector<std::int64_t> feasible_plan_ids;
        std::vector<std::int64_t> completed_plan_ids;

        route_cache_.begin_protocol_transaction();
        round_protocol_active = true;

        for (py::ssize_t selected_ordinal = 0;
             selected_ordinal < selected_array.size(); ++selected_ordinal) {
            const auto plan = static_cast<std::size_t>(selected[selected_ordinal]);
            const auto first_route = static_cast<std::size_t>(plan_boundaries[plan]);
            const auto last_route = static_cast<std::size_t>(plan_boundaries[plan + 1]);
            std::vector<std::int64_t> local_offsets{0};
            std::vector<std::int64_t> local_indices;
            for (auto route = first_route; route < last_route; ++route) {
                local_indices.insert(
                    local_indices.end(),
                    route_nodes + route_boundaries[route],
                    route_nodes + route_boundaries[route + 1]);
                local_offsets.push_back(
                    static_cast<std::int64_t>(local_indices.size()));
            }
            py::array_t<std::int64_t> local_offsets_array(local_offsets.size());
            py::array_t<std::int64_t> local_indices_array(local_indices.size());
            std::copy(
                local_offsets.begin(), local_offsets.end(),
                checked_data(local_offsets_array));
            std::copy(
                local_indices.begin(), local_indices.end(),
                checked_data(local_indices_array));
            bool store_active = false;
            bool exact_reserved = false;
            bool exact_accounted = false;
            bool exact_started = false;
            std::int64_t requested_exact = 0;
            auto budget_snapshot = budget_.snapshot();
            try {
                auto cached = route_cache_.lookup_exact_many(
                    local_offsets_array, local_indices_array);
                auto hit_flags = py::cast<py::array_t<std::int64_t>>(cached[0]);
                auto cached_path_offsets = py::cast<py::array_t<std::int64_t>>(cached[1]);
                auto cached_path_indices = py::cast<py::array_t<std::int64_t>>(cached[2]);
                auto cached_statuses = py::cast<py::array_t<std::int64_t>>(cached[3]);
                auto cached_reasons = py::cast<py::array_t<std::int64_t>>(cached[4]);
                auto cached_metrics = py::cast<py::array_t<double>>(cached[5]);
                auto cached_labels = py::cast<py::array_t<std::int64_t>>(cached[6]);
                const auto* hits = checked_data<std::int64_t>(hit_flags);
                append_causal_cache_events(
                    {context[0], context[1], context[2]},
                    causal_transaction_id, hit_flags, false);
                std::vector<std::int64_t> missing_offsets{0};
                std::vector<std::int64_t> missing_indices;
                std::vector<std::size_t> missing_local_rows;
                std::vector<NativeRouteCacheV2::ExactPayload> local_payloads(
                    last_route - first_route);
                for (std::size_t local = 0; local < last_route - first_route; ++local) {
                    if (hits[local] == 1) {
                        auto& payload = local_payloads[local];
                        const auto* path_offsets = checked_data<std::int64_t>(
                            cached_path_offsets);
                        const auto* paths = checked_data<std::int64_t>(
                            cached_path_indices);
                        payload.path.assign(
                            paths + path_offsets[local],
                            paths + path_offsets[local + 1]);
                        payload.status = checked_data<std::int64_t>(
                            cached_statuses)[local];
                        payload.reason = checked_data<std::int64_t>(
                            cached_reasons)[local];
                        std::copy(
                            checked_data<double>(cached_metrics) + local * 4,
                            checked_data<double>(cached_metrics) + (local + 1) * 4,
                            payload.metrics.begin());
                        std::copy(
                            checked_data<std::int64_t>(cached_labels) + local * 3,
                            checked_data<std::int64_t>(cached_labels) + (local + 1) * 3,
                            payload.label_counters.begin());
                        route_resolutions[first_route + local] = 1;
                        continue;
                    }
                    missing_local_rows.push_back(local);
                    missing_indices.insert(
                        missing_indices.end(),
                        local_indices.begin() + local_offsets[local],
                        local_indices.begin() + local_offsets[local + 1]);
                    missing_offsets.push_back(
                        static_cast<std::int64_t>(missing_indices.size()));
                }
                requested_exact = static_cast<std::int64_t>(missing_local_rows.size());
                const auto exact_remaining = budget_.exact_remaining();
                if (requested_exact > budget_.candidate_round_remaining()
                    || (exact_remaining >= 0 && requested_exact > exact_remaining)) {
                    ++budget_skip_count;
                    statuses[plan] = 3;
                    if (std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - transaction_started).count()
                        >= remaining_seconds) {
                        throw std::runtime_error(
                            "full native plan transaction reached its deadline during budget skip");
                    }
                    continue;
                }
                py::tuple exact_payload;
                if (requested_exact > 0) {
                    const auto round_receipt = budget_.reserve_round(
                        requested_exact, true);
                    const auto exact_receipt = budget_.reserve_exact(requested_exact);
                    if (checked_data<std::int64_t>(round_receipt)[1] != requested_exact
                        || checked_data<std::int64_t>(exact_receipt)[1]
                            != requested_exact) {
                        throw std::logic_error(
                            "full native plan budget changed during atomic reservation");
                    }
                    exact_reserved = true;
                    py::array_t<std::int64_t> missing_offsets_array(
                        missing_offsets.size());
                    py::array_t<std::int64_t> missing_indices_array(
                        missing_indices.size());
                    std::copy(
                        missing_offsets.begin(), missing_offsets.end(),
                        checked_data(missing_offsets_array));
                    std::copy(
                        missing_indices.begin(), missing_indices.end(),
                        checked_data(missing_indices_array));
                    const auto elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - transaction_started).count();
                    const auto exact_remaining = remaining_seconds - elapsed;
                    if (exact_remaining <= 0.0) {
                        throw std::runtime_error(
                            "full native plan transaction reached its deadline before exact work");
                    }
                    py::array_t<double> exact_deadline(1);
                    checked_data(exact_deadline)[0] = exact_remaining;
                    if (exact_kernel_deadline_injection_) {
                        exact_kernel_deadline_injection_ = false;
                        checked_data(exact_deadline)[0] = 0.0;
                    }
                    exact_started = true;
                    append_causal_exact_work(
                        {context[0], context[1], context[2]},
                        causal_transaction_id, requested_exact);
                    const auto candidate_exact_started =
                        std::chrono::steady_clock::now();
                    exact_payload = exact_charging_batch_numeric(
                        node_kind_, ready_time_, due_date_, service_time_, distance_,
                        vehicle_, missing_offsets_array, missing_indices_array,
                        exact_deadline, batch_array);
                    auto batch_counters = py::cast<py::array_t<std::int64_t>>(
                        exact_payload[6]);
                    const auto completed = checked_data<std::int64_t>(
                        batch_counters)[2];
                    const auto interrupted = checked_data<std::int64_t>(
                        batch_counters)[3];
                    record_exact_backend_metrics(
                        exact_payload,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now()
                            - candidate_exact_started).count());
                    budget_.complete_exact(completed);
                    budget_.interrupt_exact(interrupted);
                    exact_accounted = true;
                    if (completed != requested_exact || interrupted != 0) {
                        throw NativeExactDeadlineInterruption(
                            "full native candidate-plan exact batch reached its deadline");
                    }
                    auto exact_path_offsets = py::cast<py::array_t<std::int64_t>>(
                        exact_payload[0]);
                    auto exact_path_indices = py::cast<py::array_t<std::int64_t>>(
                        exact_payload[1]);
                    auto exact_statuses = py::cast<py::array_t<std::int64_t>>(
                        exact_payload[2]);
                    auto exact_reasons = py::cast<py::array_t<std::int64_t>>(
                        exact_payload[3]);
                    auto exact_metrics = py::cast<py::array_t<double>>(
                        exact_payload[4]);
                    auto exact_labels = py::cast<py::array_t<std::int64_t>>(
                        exact_payload[5]);
                    const auto* path_offsets = checked_data<std::int64_t>(
                        exact_path_offsets);
                    const auto* paths = checked_data<std::int64_t>(exact_path_indices);
                    for (std::size_t exact = 0; exact < missing_local_rows.size(); ++exact) {
                        const auto local = missing_local_rows[exact];
                        auto& payload = local_payloads[local];
                        payload.path.assign(
                            paths + path_offsets[exact],
                            paths + path_offsets[exact + 1]);
                        payload.status = checked_data<std::int64_t>(
                            exact_statuses)[exact];
                        payload.reason = checked_data<std::int64_t>(
                            exact_reasons)[exact];
                        std::copy(
                            checked_data<double>(exact_metrics) + exact * 4,
                            checked_data<double>(exact_metrics) + (exact + 1) * 4,
                            payload.metrics.begin());
                        std::copy(
                            checked_data<std::int64_t>(exact_labels) + exact * 3,
                            checked_data<std::int64_t>(exact_labels) + (exact + 1) * 3,
                            payload.label_counters.begin());
                        const auto global_route = first_route + local;
                        route_resolutions[global_route] = 2;
                        exact_route_rows.push_back(
                            static_cast<std::int64_t>(global_route));
                        completion_order.push_back(
                            static_cast<std::int64_t>(global_route));
                    }
                    record_exact_journal_batch(
                        {
                            checked_data<std::int64_t>(context_array)[0],
                            checked_data<std::int64_t>(context_array)[1],
                            checked_data<std::int64_t>(context_array)[2],
                        },
                        missing_offsets_array,
                        missing_indices_array,
                        exact_payload,
                        causal_transaction_id);
                    auto semantic_hashes = exact_semantic_hashes(
                        missing_offsets_array, missing_indices_array, exact_payload);
                    auto entry_bytes = exact_entry_bytes(
                        exact_path_offsets,
                        exact_path_indices,
                        exact_statuses,
                        exact_reasons,
                        exact_metrics,
                        exact_labels);
                    route_cache_.begin_store_exact_many_atomic(
                        missing_offsets_array, missing_indices_array,
                        exact_payload[0], exact_payload[1], exact_payload[2],
                        exact_payload[3], exact_payload[4], exact_payload[5],
                        semantic_hashes, entry_bytes);
                    store_active = true;
                }
                bool feasible = true;
                PythonFloatSum total_distance;
                PythonFloatSum total_charging_time;
                std::int64_t charging_count = 0;
                for (std::size_t local = 0; local < local_payloads.size(); ++local) {
                    const auto& payload = local_payloads[local];
                    feasible = feasible && payload.status == 0;
                    if (payload.status == 0) {
                        total_distance.add(payload.metrics[0]);
                        total_charging_time.add(payload.metrics[3]);
                        for (const auto node : payload.path) {
                            charging_count += kinds[node] == station_kind ? 1 : 0;
                        }
                    }
                    route_payloads[first_route + local] = payload;
                }
                if (feasible) {
                    statuses[plan] = 5;
                    ++feasible_count;
                    feasible_plan_ids.push_back(static_cast<std::int64_t>(plan));
                    checked_data(objective_integer)[plan * 2] =
                        static_cast<std::int64_t>(last_route - first_route);
                    checked_data(objective_integer)[plan * 2 + 1] = charging_count;
                    checked_data(objective_float)[plan * 2] = total_distance.value();
                    checked_data(objective_float)[plan * 2 + 1] =
                        total_charging_time.value();
                } else {
                    statuses[plan] = 4;
                    ++infeasible_count;
                }
                completed_plan_ids.push_back(static_cast<std::int64_t>(plan));
                if (std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - transaction_started).count()
                    >= remaining_seconds) {
                    throw std::runtime_error(
                        "full native plan transaction reached its deadline before commit");
                }
                if (store_active) {
                    route_cache_.commit_store_batch();
                    store_active = false;
                    append_causal_cache_events(
                        {context[0], context[1], context[2]},
                        causal_transaction_id, hit_flags, true);
                }
            } catch (...) {
                if (store_active) {
                    route_cache_.rollback_store_batch();
                }
                if (exact_started && exact_reserved && !exact_accounted) {
                    budget_.interrupt_exact(requested_exact);
                } else if (!exact_started) {
                    budget_.restore(budget_snapshot);
                }
                throw;
            }
        }

        if (std::chrono::duration<double>(
                std::chrono::steady_clock::now() - transaction_started).count()
            >= remaining_seconds) {
            throw std::runtime_error(
                "full native plan transaction reached its deadline before return");
        }
        if (!completed_plan_ids.empty() && !suppress_attempted_plan_journal_) {
            py::array_t<std::int64_t> completed_plan_array(completed_plan_ids.size());
            std::copy(
                completed_plan_ids.begin(), completed_plan_ids.end(),
                checked_data(completed_plan_array));
            attempted_plans_.begin_mark_many_atomic(
                plans_array, routes_array, indices_array, completed_plan_array);
            attempted_mark_active = true;
        }

        py::array_t<std::int64_t> unordered_feasible_plan_ids(
            feasible_plan_ids.size());
        std::copy(
            feasible_plan_ids.begin(), feasible_plan_ids.end(),
            checked_data(unordered_feasible_plan_ids));
        auto canonical_feasible_order = order_feasible_candidate_plans_v2(
            plans_array, routes_array, indices_array, objective_integer,
            objective_float, lexical_rank_, unordered_feasible_plan_ids);
        feasible_plan_ids.assign(
            checked_data<std::int64_t>(canonical_feasible_order),
            checked_data<std::int64_t>(canonical_feasible_order)
                + canonical_feasible_order.size());

        py::array_t<std::int64_t> status_array(statuses.size());
        py::array_t<std::int64_t> resolution_array(route_resolutions.size());
        py::array_t<std::int64_t> exact_rows_array(exact_route_rows.size());
        py::array_t<std::int64_t> completion_array(completion_order.size());
        py::array_t<std::int64_t> feasible_order_array(feasible_plan_ids.size());
        std::copy(statuses.begin(), statuses.end(), checked_data(status_array));
        std::copy(
            route_resolutions.begin(), route_resolutions.end(),
            checked_data(resolution_array));
        std::copy(
            exact_route_rows.begin(), exact_route_rows.end(),
            checked_data(exact_rows_array));
        std::copy(
            completion_order.begin(), completion_order.end(),
            checked_data(completion_array));
        std::copy(
            feasible_plan_ids.begin(), feasible_plan_ids.end(),
            checked_data(feasible_order_array));
        py::array_t<std::int64_t> counters(8);
        checked_data(counters)[0] = static_cast<std::int64_t>(plan_count);
        checked_data(counters)[1] = static_cast<std::int64_t>(selected_array.size());
        checked_data(counters)[2] = feasible_count;
        checked_data(counters)[3] = infeasible_count;
        checked_data(counters)[4] = budget_skip_count;
        checked_data(counters)[5] = static_cast<std::int64_t>(exact_route_rows.size());
        checked_data(counters)[6] = attempted_plans_.size();
        checked_data(counters)[7] = negative_hit_count;
        auto cache_snapshot = route_cache_.snapshot();
        auto cache_statistics = py::cast<py::array_t<std::int64_t>>(
            cache_snapshot[4]);
        auto negative_snapshot = negative_cache_.snapshot();
        auto negative_statistics = negative_cache_.projected_statistics_array();
        auto budget_state = budget_.state();
        ControlJournalBatch control_journal_batch;
        std::copy(
            context, context + 3, control_journal_batch.context.begin());
        control_journal_batch.transaction_id = causal_transaction_id;
        const auto copy_integer_array = [](const py::array_t<std::int64_t>& array) {
            return std::vector<std::int64_t>(
                checked_data<std::int64_t>(array),
                checked_data<std::int64_t>(array) + array.size());
        };
        control_journal_batch.plan_offsets = copy_integer_array(plans_array);
        control_journal_batch.route_offsets = copy_integer_array(routes_array);
        control_journal_batch.route_indices = copy_integer_array(indices_array);
        control_journal_batch.ranked = copy_integer_array(
            py::cast<py::array_t<std::int64_t>>(ranking[0]));
        control_journal_batch.selected = copy_integer_array(selected_array);
        control_journal_batch.statuses = statuses;
        control_journal_batch.ranking_integer = copy_integer_array(
            py::cast<py::array_t<std::int64_t>>(ranking[2]));
        auto ranking_float_array = py::cast<py::array_t<double>>(ranking[3]);
        control_journal_batch.ranking_float.assign(
            checked_data<double>(ranking_float_array),
            checked_data<double>(ranking_float_array) + ranking_float_array.size());
        control_journal_batch.route_resolutions = route_resolutions;
        control_journal_batch.cache_statistics = copy_integer_array(cache_statistics);
        control_journal_batch.budget_state = copy_integer_array(budget_state);
        control_journal_batch.protocol_flags = {
            suppress_attempted_plan_journal_ ? 1 : 0,
            suppress_round_budget_ ? 1 : 0,
            suppress_plan_screening_negative_cache_ ? 1 : 0,
            allow_partial_customer_coverage_ ? 1 : 0,
        };
        std::unordered_set<std::int64_t> selected_plan_ids(
            control_journal_batch.selected.begin(),
            control_journal_batch.selected.end());
        control_journal_batch.decision_codes.reserve(plan_count);
        for (std::size_t plan = 0; plan < plan_count; ++plan) {
            control_journal_batch.decision_codes.push_back(
                eligible[plan] == 0 ? 0
                : attempted[plan] != 0 ? 1
                : selected_plan_ids.contains(static_cast<std::int64_t>(plan)) ? 2
                : 3);
        }
        std::list<ControlJournalBatch> staged_control_journal;
        staged_control_journal.push_back(std::move(control_journal_batch));
        std::string evidence("stage05.2-native-solution-plan-transaction-v2");
        append_evidence_array(evidence, selected_array);
        append_evidence_array(evidence, plans_array);
        append_evidence_array(evidence, routes_array);
        append_evidence_array(evidence, indices_array);
        append_evidence_array(evidence, context_array);
        append_evidence_array(evidence, canonical_expected_array);
        append_evidence_array(evidence, batch_array);
        append_evidence_array(evidence, lower_bounds);
        append_evidence_array(
            evidence, py::cast<py::array_t<std::int64_t>>(ranking[0]));
        append_evidence_array(evidence, status_array);
        append_evidence_array(evidence, objective_integer);
        append_evidence_array(evidence, objective_float);
        append_evidence_array(evidence, resolution_array);
        append_evidence_array(evidence, exact_rows_array);
        append_evidence_array(evidence, completion_array);
        append_evidence_array(evidence, counters);
        append_evidence_array(evidence, cache_statistics);
        append_evidence_array(
            evidence, py::cast<py::array_t<std::uint8_t>>(cache_snapshot[2]));
        append_evidence_array(
            evidence, py::cast<py::array_t<std::int64_t>>(negative_snapshot[0]));
        append_evidence_array(
            evidence, py::cast<py::array_t<std::int64_t>>(negative_snapshot[1]));
        append_evidence_array(
            evidence, py::cast<py::array_t<std::int64_t>>(negative_snapshot[2]));
        append_evidence_array(evidence, negative_statistics);
        append_evidence_array(evidence, budget_state);
        append_evidence_array(evidence, feasible_order_array);
        std::vector<std::int64_t> transaction_path_offsets{0};
        std::vector<std::int64_t> transaction_path_indices;
        py::array_t<std::int64_t> transaction_statuses(route_count);
        py::array_t<std::int64_t> transaction_reasons(route_count);
        py::array_t<double> transaction_metrics(
            {static_cast<py::ssize_t>(route_count), py::ssize_t(4)});
        py::array_t<std::int64_t> transaction_labels(
            {static_cast<py::ssize_t>(route_count), py::ssize_t(3)});
        std::fill(
            checked_data(transaction_statuses),
            checked_data(transaction_statuses) + route_count, -1);
        std::fill(
            checked_data(transaction_reasons),
            checked_data(transaction_reasons) + route_count, -1);
        std::fill(
            checked_data(transaction_metrics),
            checked_data(transaction_metrics) + route_count * 4, 0.0);
        std::fill(
            checked_data(transaction_labels),
            checked_data(transaction_labels) + route_count * 3, 0);
        for (std::size_t route = 0; route < route_payloads.size(); ++route) {
            if (!route_payloads[route].has_value()) {
                transaction_path_offsets.push_back(
                    static_cast<std::int64_t>(transaction_path_indices.size()));
                continue;
            }
            const auto& payload = *route_payloads[route];
            const auto route_id = static_cast<std::int64_t>(route);
            append_evidence_i64(evidence, route_id);
            append_evidence_i64(evidence, payload.status);
            append_evidence_i64(evidence, payload.reason);
            append_evidence_values(
                evidence, payload.metrics.data(), payload.metrics.size());
            append_evidence_values(
                evidence, payload.label_counters.data(),
                payload.label_counters.size());
            append_evidence_values(
                evidence, payload.path.data(), payload.path.size());
            transaction_path_indices.insert(
                transaction_path_indices.end(), payload.path.begin(), payload.path.end());
            transaction_path_offsets.push_back(
                static_cast<std::int64_t>(transaction_path_indices.size()));
            checked_data(transaction_statuses)[route] = payload.status;
            checked_data(transaction_reasons)[route] = payload.reason;
            std::copy(
                payload.metrics.begin(), payload.metrics.end(),
                checked_data(transaction_metrics) + route * 4);
            std::copy(
                payload.label_counters.begin(), payload.label_counters.end(),
                checked_data(transaction_labels) + route * 3);
        }
        py::array_t<std::int64_t> transaction_path_offsets_array(
            transaction_path_offsets.size());
        py::array_t<std::int64_t> transaction_path_indices_array(
            transaction_path_indices.size());
        std::copy(
            transaction_path_offsets.begin(), transaction_path_offsets.end(),
            checked_data(transaction_path_offsets_array));
        std::copy(
            transaction_path_indices.begin(), transaction_path_indices.end(),
            checked_data(transaction_path_indices_array));
        auto transaction_exact_payload = py::make_tuple(
            std::move(transaction_path_offsets_array),
            std::move(transaction_path_indices_array),
            std::move(transaction_statuses), std::move(transaction_reasons),
            std::move(transaction_metrics), std::move(transaction_labels));
        auto result = py::make_tuple(
            std::move(selected_array), std::move(status_array),
            std::move(objective_integer), std::move(objective_float),
            std::move(resolution_array), std::move(exact_rows_array),
            std::move(completion_array), std::move(counters),
            std::move(cache_statistics), std::move(negative_statistics),
            std::move(budget_state),
            std::move(feasible_order_array),
            native_sha256_hex(evidence));
        route_cache_.prepare_protocol_commit();
        route_cache_.prepare_protocol_rollback();
        if (commit_failure_injection_ == 1) {
            commit_failure_injection_ = 0;
            throw std::runtime_error(
                "injected full native route-cache commit preparation failure");
        }
        if (negative_store_active) {
            negative_cache_.prepare_store_commit();
        }
        if (commit_failure_injection_ == 2) {
            commit_failure_injection_ = 0;
            throw std::runtime_error(
                "injected full native negative-cache commit preparation failure");
        }
        if (attempted_mark_active) {
            attempted_plans_.prepare_mark_commit();
        }
        if (commit_failure_injection_ == 3) {
            commit_failure_injection_ = 0;
            throw std::runtime_error(
                "injected full native attempted-plan commit preparation failure");
        }
        if (defer_composite_commit_) {
            if (pending_composite_active_) {
                throw std::logic_error(
                    "full native composite transaction is already pending");
            }
            pending_round_protocol_ = round_protocol_active;
            pending_negative_store_ = negative_store_active;
            pending_attempted_mark_ = attempted_mark_active;
            pending_budget_snapshot_.emplace(round_budget_snapshot);
            pending_candidate_exact_payload_ = transaction_exact_payload;
            pending_candidate_exact_ready_ = !feasible_plan_ids.empty();
            pending_causal_snapshot_ = causal_snapshot;
            append_control_journal_causal(staged_control_journal);
            pending_control_journal_.splice(
                pending_control_journal_.end(), staged_control_journal);
            pending_composite_active_ = true;
            round_protocol_active = false;
            negative_store_active = false;
            attempted_mark_active = false;
            return result;
        }
        append_control_journal_causal(staged_control_journal);
        route_cache_.commit_protocol_transaction_noexcept();
        round_protocol_active = false;
        if (negative_store_active) {
            negative_cache_.commit_store_batch_noexcept();
            negative_store_active = false;
        }
        if (attempted_mark_active) {
            attempted_plans_.commit_mark_batch_noexcept();
            attempted_mark_active = false;
        }
        control_journal_.splice(
            control_journal_.end(), staged_control_journal);
        return result;
        } catch (...) {
            if (attempted_mark_active) {
                attempted_plans_.rollback_mark_batch_noexcept();
            }
            if (negative_store_active) {
                negative_cache_.rollback_store_batch_noexcept();
            }
            if (round_protocol_active) {
                route_cache_.rollback_protocol_transaction_noexcept();
            }
            causal_journal_.rollback_noexcept(causal_snapshot);
            rollback_round_budget_preserving_exact(round_budget_snapshot);
            throw;
        }
    }

    py::tuple legacy_route_elimination_probe(
        std::int64_t iteration,
        std::int64_t max_attempts,
        std::int64_t route_change_limit,
        py::handle deadline_remaining,
        py::handle batch_size,
        bool defer_acceptance) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || iteration < 0 || max_attempts <= 0
            || route_change_limit == 0 || route_change_limit < -1
            || last_candidate_ready_ || legacy_candidate_ready_
            || pending_composite_active_) {
            throw std::invalid_argument(
                "full native legacy route-elimination probe state/config is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "legacy_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "legacy_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native legacy route-elimination deadline/batch is invalid");
        }
        const auto route_count = legacy_offsets_.size() - 1;
        py::array_t<std::int64_t> empty_profile_order(0);
        py::array_t<std::int64_t> empty_attempts(
            py::array::ShapeContainer{0, 6});
        py::array_t<std::int64_t> empty_plan_offsets(1);
        py::array_t<std::int64_t> empty_route_offsets(1);
        py::array_t<std::int64_t> empty_route_indices(0);
        py::array_t<std::int64_t> empty_outcome(5);
        checked_data(empty_plan_offsets)[0] = 0;
        checked_data(empty_route_offsets)[0] = 0;
        std::fill(
            checked_data(empty_outcome), checked_data(empty_outcome) + 5,
            std::int64_t{-1});
        if (route_count <= 1) {
            accumulate_full_stage04_outcome_noexcept(
                2, false, 1, false, false, true);
            return py::make_tuple(
                std::move(empty_profile_order), std::move(empty_attempts),
                std::move(empty_plan_offsets), std::move(empty_route_offsets),
                std::move(empty_route_indices), py::none(),
                std::move(empty_outcome), lane_solution_state(0));
        }

        struct RouteProfile {
            std::int64_t route;
            std::int64_t customer_count;
            double distance;
            double charging_time;
            std::int64_t charging_count;
        };
        auto path_offsets = py::cast<py::array_t<std::int64_t>>(
            legacy_exact_payload_[0]);
        auto path_indices = py::cast<py::array_t<std::int64_t>>(
            legacy_exact_payload_[1]);
        auto statuses = py::cast<py::array_t<std::int64_t>>(
            legacy_exact_payload_[2]);
        auto metrics = py::cast<py::array_t<double>>(
            legacy_exact_payload_[4]);
        const auto* route_boundaries =
            checked_data<std::int64_t>(legacy_offsets_);
        const auto* path_boundaries = checked_data<std::int64_t>(path_offsets);
        const auto* paths = checked_data<std::int64_t>(path_indices);
        const auto* status_values = checked_data<std::int64_t>(statuses);
        const auto* metric_values = checked_data<double>(metrics);
        std::vector<RouteProfile> profiles;
        profiles.reserve(static_cast<std::size_t>(route_count));
        for (std::int64_t route = 0; route < route_count; ++route) {
            if (status_values[route] != 0) {
                throw std::logic_error(
                    "full native legacy lane contains an infeasible incumbent route");
            }
            std::int64_t charging_count = 0;
            for (auto cursor = path_boundaries[route];
                 cursor < path_boundaries[route + 1]; ++cursor) {
                charging_count += checked_data<std::int64_t>(node_kind_)[
                    paths[cursor]] == station_kind ? 1 : 0;
            }
            profiles.push_back(RouteProfile{
                route,
                route_boundaries[route + 1] - route_boundaries[route],
                metric_values[route * 4],
                metric_values[route * 4 + 3],
                charging_count});
        }
        std::stable_sort(
            profiles.begin(), profiles.end(),
            [](const RouteProfile& left, const RouteProfile& right) {
                if (left.customer_count != right.customer_count) {
                    return left.customer_count < right.customer_count;
                }
                if (left.distance != right.distance) {
                    return left.distance > right.distance;
                }
                if (left.charging_time != right.charging_time) {
                    return left.charging_time > right.charging_time;
                }
                if (left.charging_count != right.charging_count) {
                    return left.charging_count > right.charging_count;
                }
                return left.route < right.route;
            });
        const auto attempt_count = std::min<std::int64_t>(
            max_attempts, static_cast<std::int64_t>(profiles.size()));
        py::array_t<std::int64_t> profile_order(attempt_count);
        py::array_t<std::int64_t> attempts(
            {static_cast<py::ssize_t>(attempt_count), py::ssize_t(6)});
        std::fill(
            checked_data(attempts),
            checked_data(attempts) + attempt_count * 6,
            std::int64_t{-1});
        std::vector<std::int64_t> plan_offsets{0};
        std::vector<std::int64_t> packed_route_offsets{0};
        std::vector<std::int64_t> packed_route_indices;
        std::vector<std::int64_t> source_route_by_plan;
        std::unordered_set<std::string> seen_plans;
        const auto* legacy_nodes = checked_data<std::int64_t>(legacy_indices_);
        for (std::int64_t rank = 0; rank < attempt_count; ++rank) {
            const auto source_route = profiles[static_cast<std::size_t>(rank)].route;
            checked_data(profile_order)[rank] = source_route;
            auto* attempt = checked_data(attempts) + rank * 6;
            attempt[0] = source_route;
            attempt[1] = rank + 1;
            attempt[4] = route_boundaries[source_route + 1]
                - route_boundaries[source_route];
            std::vector<std::int64_t> partial_offsets{0};
            std::vector<std::int64_t> partial_indices;
            for (std::int64_t route = 0; route < route_count; ++route) {
                if (route == source_route) {
                    continue;
                }
                partial_indices.insert(
                    partial_indices.end(),
                    legacy_nodes + route_boundaries[route],
                    legacy_nodes + route_boundaries[route + 1]);
                partial_offsets.push_back(
                    static_cast<std::int64_t>(partial_indices.size()));
            }
            py::array_t<std::int64_t> partial_offsets_array(
                partial_offsets.size());
            py::array_t<std::int64_t> partial_indices_array(
                partial_indices.size());
            py::array_t<std::int64_t> removed_indices_array(
                route_boundaries[source_route + 1]
                    - route_boundaries[source_route]);
            std::copy(
                partial_offsets.begin(), partial_offsets.end(),
                checked_data(partial_offsets_array));
            std::copy(
                partial_indices.begin(), partial_indices.end(),
                checked_data(partial_indices_array));
            std::copy(
                legacy_nodes + route_boundaries[source_route],
                legacy_nodes + route_boundaries[source_route + 1],
                checked_data(removed_indices_array));
            auto repair = candidate_control_repair_v2(
                node_kind_, demand_, ready_time_, due_date_, service_time_,
                distance_, reachable_, vehicle_, lexical_rank_,
                partial_offsets_array, partial_indices_array,
                removed_indices_array, screening_epsilon_, route_change_limit,
                false);
            auto repaired_offsets =
                py::cast<py::array_t<std::int64_t>>(repair[0]);
            auto repaired_indices =
                py::cast<py::array_t<std::int64_t>>(repair[1]);
            auto repair_metadata =
                py::cast<py::array_t<std::int64_t>>(repair[2]);
            attempt[2] = checked_data<std::int64_t>(repair_metadata)[0];
            attempt[5] = checked_data<std::int64_t>(repair_metadata)[1];
            if (attempt[2] != 0 || repaired_offsets.size() - 1 != route_count - 1) {
                continue;
            }
            std::string identity("stage05.2-native-legacy-elimination-plan-v2");
            append_evidence_array(identity, repaired_offsets);
            append_evidence_array(identity, repaired_indices);
            if (!seen_plans.insert(identity).second) {
                continue;
            }
            const auto plan_id = static_cast<std::int64_t>(
                source_route_by_plan.size());
            attempt[3] = plan_id;
            source_route_by_plan.push_back(source_route);
            const auto* repaired_boundaries =
                checked_data<std::int64_t>(repaired_offsets);
            const auto* repaired_nodes =
                checked_data<std::int64_t>(repaired_indices);
            for (py::ssize_t route = 0;
                 route + 1 < repaired_offsets.size(); ++route) {
                packed_route_indices.insert(
                    packed_route_indices.end(),
                    repaired_nodes + repaired_boundaries[route],
                    repaired_nodes + repaired_boundaries[route + 1]);
                packed_route_offsets.push_back(
                    static_cast<std::int64_t>(packed_route_indices.size()));
            }
            plan_offsets.push_back(
                static_cast<std::int64_t>(packed_route_offsets.size() - 1));
        }
        py::array_t<std::int64_t> plan_offsets_array(plan_offsets.size());
        py::array_t<std::int64_t> route_offsets_array(
            packed_route_offsets.size());
        py::array_t<std::int64_t> route_indices_array(
            packed_route_indices.size());
        std::copy(
            plan_offsets.begin(), plan_offsets.end(),
            checked_data(plan_offsets_array));
        std::copy(
            packed_route_offsets.begin(), packed_route_offsets.end(),
            checked_data(route_offsets_array));
        std::copy(
            packed_route_indices.begin(), packed_route_indices.end(),
            checked_data(route_indices_array));
        py::array_t<std::int64_t> outcome(5);
        std::fill(
            checked_data(outcome), checked_data(outcome) + 5,
            std::int64_t{-1});
        if (source_route_by_plan.empty()) {
            accumulate_full_stage04_outcome_noexcept(
                2, false, 1, false, false, true);
            return py::make_tuple(
                std::move(profile_order), std::move(attempts),
                std::move(plan_offsets_array), std::move(route_offsets_array),
                std::move(route_indices_array), py::none(),
                std::move(outcome), lane_solution_state(0));
        }
        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        py::array_t<std::int64_t> context(3);
        checked_data(context)[0] = stable_int63("legacy");
        checked_data(context)[1] = stable_int63("route_elimination");
        checked_data(context)[2] = iteration;
        swap_active_with_lane_noexcept(0);
        ScopeRollback restore_lane([this]() noexcept {
            swap_active_with_lane_noexcept(0);
        });
        defer_composite_commit_ = true;
        try {
            auto transaction = evaluate_plans(
                plan_offsets_array, route_offsets_array, route_indices_array,
                context, deadline_array, batch_array, expected_array);
            defer_composite_commit_ = false;
            const auto selected_plan = prepare_first_feasible_candidate(
                plan_offsets_array, route_offsets_array, route_indices_array,
                transaction);
            if (selected_plan.has_value()) {
                commit_pending_composite_noexcept();
                checked_data(outcome)[0] = *selected_plan;
                checked_data(outcome)[4] = source_route_by_plan[
                    static_cast<std::size_t>(*selected_plan)];
                if (defer_acceptance) {
                    legacy_candidate_offsets_ = std::move(last_candidate_offsets_);
                    legacy_candidate_indices_ = std::move(last_candidate_indices_);
                    legacy_candidate_exact_payload_ =
                        std::move(last_candidate_exact_payload_);
                    legacy_candidate_objective_integer_ =
                        std::move(last_candidate_objective_integer_);
                    legacy_candidate_objective_float_ =
                        std::move(last_candidate_objective_float_);
                    last_candidate_ready_ = false;
                    legacy_candidate_ready_ = true;
                    legacy_candidate_operator_ = 2;
                    checked_data(outcome)[1] = -2;
                    checked_data(outcome)[2] = 0;
                    checked_data(outcome)[3] = 0;
                } else {
                    const auto comparison = last_candidate_comparison();
                    auto acceptance = apply_last_candidate(1.0, 1.0);
                    checked_data(outcome)[1] =
                        py::cast<std::int64_t>(acceptance[0]);
                    checked_data(outcome)[2] =
                        py::cast<std::int64_t>(acceptance[1]);
                    checked_data(outcome)[3] =
                        py::cast<std::int64_t>(acceptance[2]);
                    accumulate_full_stage04_outcome_noexcept(
                        2, checked_data(outcome)[1] != 0, comparison,
                        checked_data(outcome)[2] != 0,
                        checked_data(outcome)[3] != 0, true);
                }
            } else {
                commit_pending_composite_noexcept();
                accumulate_full_stage04_outcome_noexcept(
                    2, false, 1, false, false, true);
            }
            restore_lane.rollback_now();
            return py::make_tuple(
                std::move(profile_order), std::move(attempts),
                std::move(plan_offsets_array), std::move(route_offsets_array),
                std::move(route_indices_array), std::move(transaction),
                std::move(outcome), lane_solution_state(0));
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
    }

    py::tuple legacy_route_merge_probe(
        std::int64_t iteration,
        py::handle deadline_remaining,
        py::handle batch_size,
        bool defer_acceptance) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || iteration < 0 || last_candidate_ready_
            || legacy_candidate_ready_ || pending_composite_active_) {
            throw std::invalid_argument(
                "full native route-merge probe state is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "route_merge_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "route_merge_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native route-merge deadline/batch is invalid");
        }
        const auto route_count = legacy_offsets_.size() - 1;
        py::array_t<std::int64_t> metadata(4);
        auto* metadata_values = checked_data(metadata);
        metadata_values[0] = iteration;
        metadata_values[1] = 3;
        metadata_values[2] = route_count;
        metadata_values[3] = 0;
        if (route_count <= 1) {
            accumulate_full_stage04_outcome_noexcept(
                3, false, 1, false, false, true);
            return py::make_tuple(std::move(metadata), lane_solution_state(0));
        }

        auto exact_metrics = py::cast<py::array_t<double>>(
            legacy_exact_payload_[4]);
        py::array_t<double> route_metrics(
            {route_count, py::ssize_t{2}});
        for (py::ssize_t route = 0; route < route_count; ++route) {
            checked_data(route_metrics)[route * 2] =
                checked_data<double>(exact_metrics)[route * 4];
            checked_data(route_metrics)[route * 2 + 1] =
                checked_data<double>(exact_metrics)[route * 4 + 3];
        }
        auto pool = route_merge_candidate_pool_v2(
            legacy_offsets_, legacy_indices_, route_metrics, demand_,
            checked_data<double>(vehicle_)[1], screening_epsilon_, true, false);
        auto candidate_offsets = py::cast<py::array_t<std::int64_t>>(pool[0]);
        auto candidate_indices = py::cast<py::array_t<std::int64_t>>(pool[1]);
        auto candidate_metadata = py::cast<py::array_t<std::int64_t>>(pool[2]);
        const auto candidate_count = candidate_offsets.size() - 1;
        metadata_values[3] = candidate_count;
        py::array_t<std::int64_t> screening_reasons(candidate_count);

        std::vector<std::int64_t> plan_offsets{0};
        std::vector<std::int64_t> route_offsets{0};
        std::vector<std::int64_t> route_indices;
        std::vector<std::int64_t> source_candidates;
        const auto* candidate_boundaries =
            checked_data<std::int64_t>(candidate_offsets);
        const auto* candidate_nodes =
            checked_data<std::int64_t>(candidate_indices);
        const auto* pool_metadata =
            checked_data<std::int64_t>(candidate_metadata);
        const auto* legacy_boundaries =
            checked_data<std::int64_t>(legacy_offsets_);
        const auto* legacy_nodes = checked_data<std::int64_t>(legacy_indices_);
        const auto* kinds = checked_data<std::int64_t>(node_kind_);
        const auto* demands = checked_data<double>(demand_);
        const auto* ready = checked_data<double>(ready_time_);
        const auto* due = checked_data<double>(due_date_);
        const auto* service = checked_data<double>(service_time_);
        const auto* distances = checked_data<double>(distance_);
        const auto* reachable = checked_data<std::uint8_t>(reachable_);
        const auto* vehicle = checked_data<double>(vehicle_);
        const std::array<double, 4> screen_options{
            0.0, screening_epsilon_, 0.0, 0.0};
        const std::array<double, 6> no_incremental{
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const auto merge_screening_started = std::chrono::steady_clock::now();
        std::vector<evrptw::native_kernels::ScreenOutput> merge_screening(
            static_cast<std::size_t>(candidate_count));
        screening_occupancies_.push_back(candidate_count);
#ifdef __linux__
        const auto merge_scheduler_endpoint = native_kernel_scheduler_endpoint;
        const auto merge_scheduler_required = native_kernel_scheduler_required;
        auto* merge_telemetry_collector =
            evrptw::native_client::telemetry_collector;
#endif
        {
            py::gil_scoped_release release;
#ifdef __linux__
            if (!merge_scheduler_endpoint.empty()) {
                merge_screening = dispatch_screen_routes(
                    kinds, demands, ready, due, service, distances,
                    reachable, vehicle,
                    candidate_boundaries, candidate_nodes,
                    static_cast<std::size_t>(candidate_count),
                    static_cast<std::size_t>(
                        candidate_boundaries[candidate_count]),
                    static_cast<std::size_t>(node_kind_.size()),
                    depot_, recharge_nodes_,
                    screen_options.data(), no_incremental.data());
            } else
#endif
            {
            work_pool_->parallel_for(
                static_cast<std::size_t>(candidate_count),
                [&](std::size_t candidate) {
#ifdef __linux__
                    NativeSchedulerThreadContext scheduler_context(
                        merge_scheduler_endpoint, merge_scheduler_required,
                        merge_telemetry_collector);
#endif
                    const auto first = candidate_boundaries[candidate];
                    const auto last = candidate_boundaries[candidate + 1];
                    merge_screening[candidate] =
                        dispatch_screen_route(
                        kinds, demands, ready, due, service, distances,
                        reachable, vehicle, candidate_nodes + first,
                        static_cast<std::size_t>(last - first),
                        static_cast<std::size_t>(node_kind_.size()), depot_,
                        recharge_nodes_, screen_options.data(),
                        no_incremental.data());
                });
            }
        }
        for (py::ssize_t candidate = 0; candidate < candidate_count; ++candidate) {
            const auto first = candidate_boundaries[candidate];
            const auto last = candidate_boundaries[candidate + 1];
            const auto& screen = merge_screening[
                static_cast<std::size_t>(candidate)];
            checked_data(screening_reasons)[candidate] =
                screen.codes[0] == 1 ? 0 : screen.codes[1];
            ++screening_statistics_[0];
            ++screening_statistics_[5];
            if (screen.codes[0] == 1) {
                ++screening_statistics_[1];
                ++screening_statistics_[6];
            } else {
                ++screening_statistics_[2];
                ++screening_statistics_[7];
                ++screening_reason_counts_[screen.codes[1]];
            }
            ScreeningJournalRow screening_journal;
            screening_journal.context = {
                stable_int63("legacy"), stable_int63("route_merge"), iteration};
            screening_journal.route.assign(
                candidate_nodes + first, candidate_nodes + last);
            std::copy(
                screen.codes.begin(), screen.codes.end(),
                screening_journal.codes.begin());
            std::copy(
                screen.metrics.begin(), screen.metrics.end(),
                screening_journal.metrics.begin());
            screening_journal.flags = {1, 0, 1, screen.reachability_queries};
            screening_journal_.push_back(std::move(screening_journal));
            if (screen.codes[0] != 1) {
                continue;
            }
            const auto left = pool_metadata[candidate * 5];
            const auto right = pool_metadata[candidate * 5 + 1];
            for (py::ssize_t route = 0; route < route_count; ++route) {
                if (route == right) {
                    continue;
                }
                if (route == left) {
                    route_indices.insert(
                        route_indices.end(), candidate_nodes + first,
                        candidate_nodes + last);
                } else {
                    route_indices.insert(
                        route_indices.end(), legacy_nodes + legacy_boundaries[route],
                        legacy_nodes + legacy_boundaries[route + 1]);
                }
                route_offsets.push_back(
                    static_cast<std::int64_t>(route_indices.size()));
            }
            plan_offsets.push_back(
                static_cast<std::int64_t>(route_offsets.size() - 1));
            source_candidates.push_back(candidate);
        }
        screening_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - merge_screening_started).count();

        py::array_t<std::int64_t> plan_offsets_array(plan_offsets.size());
        py::array_t<std::int64_t> route_offsets_array(route_offsets.size());
        py::array_t<std::int64_t> route_indices_array(route_indices.size());
        py::array_t<std::int64_t> source_candidates_array(
            source_candidates.size());
        std::copy(
            plan_offsets.begin(), plan_offsets.end(),
            checked_data(plan_offsets_array));
        std::copy(
            route_offsets.begin(), route_offsets.end(),
            checked_data(route_offsets_array));
        std::copy(
            route_indices.begin(), route_indices.end(),
            checked_data(route_indices_array));
        std::copy(
            source_candidates.begin(), source_candidates.end(),
            checked_data(source_candidates_array));
        auto plans = py::make_tuple(
            plan_offsets_array, route_offsets_array, route_indices_array,
            source_candidates_array);
        py::array_t<std::int64_t> outcome(5);
        std::fill(
            checked_data(outcome), checked_data(outcome) + 5,
            std::int64_t{-1});
        if (source_candidates.empty()) {
            accumulate_full_stage04_outcome_noexcept(
                3, false, 1, false, false, true);
            return py::make_tuple(
                std::move(metadata), std::move(pool),
                std::move(screening_reasons), std::move(plans), py::none(),
                std::move(outcome), lane_solution_state(0));
        }

        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        py::array_t<std::int64_t> context(3);
        checked_data(context)[0] = stable_int63("legacy");
        checked_data(context)[1] = stable_int63("route_merge");
        checked_data(context)[2] = iteration;
        swap_active_with_lane_noexcept(0);
        ScopeRollback restore_lane([this]() noexcept {
            swap_active_with_lane_noexcept(0);
        });
        defer_composite_commit_ = true;
        try {
            auto transaction = evaluate_plans(
                plan_offsets_array, route_offsets_array, route_indices_array,
                context, deadline_array, batch_array, expected_array);
            defer_composite_commit_ = false;
            const auto selected = prepare_first_feasible_candidate(
                plan_offsets_array, route_offsets_array, route_indices_array,
                transaction);
            if (selected.has_value()) {
                commit_pending_composite_noexcept();
                checked_data(outcome)[0] = *selected;
                checked_data(outcome)[4] = source_candidates[
                    static_cast<std::size_t>(*selected)];
                if (defer_acceptance) {
                    legacy_candidate_offsets_ =
                        std::move(last_candidate_offsets_);
                    legacy_candidate_indices_ =
                        std::move(last_candidate_indices_);
                    legacy_candidate_exact_payload_ =
                        std::move(last_candidate_exact_payload_);
                    legacy_candidate_objective_integer_ =
                        std::move(last_candidate_objective_integer_);
                    legacy_candidate_objective_float_ =
                        std::move(last_candidate_objective_float_);
                    last_candidate_ready_ = false;
                    legacy_candidate_ready_ = true;
                    legacy_candidate_operator_ = 3;
                    checked_data(outcome)[1] = -2;
                    checked_data(outcome)[2] = 0;
                    checked_data(outcome)[3] = 0;
                } else {
                    const auto comparison = last_candidate_comparison();
                    auto acceptance = apply_last_candidate(1.0, 1.0);
                    checked_data(outcome)[1] =
                        py::cast<std::int64_t>(acceptance[0]);
                    checked_data(outcome)[2] =
                        py::cast<std::int64_t>(acceptance[1]);
                    checked_data(outcome)[3] =
                        py::cast<std::int64_t>(acceptance[2]);
                    accumulate_full_stage04_outcome_noexcept(
                        3, checked_data(outcome)[1] != 0, comparison,
                        checked_data(outcome)[2] != 0,
                        checked_data(outcome)[3] != 0, true);
                }
            } else {
                commit_pending_composite_noexcept();
                accumulate_full_stage04_outcome_noexcept(
                    3, false, 1, false, false, true);
            }
            restore_lane.rollback_now();
            return py::make_tuple(
                std::move(metadata), std::move(pool),
                std::move(screening_reasons), std::move(plans),
                std::move(transaction), std::move(outcome),
                lane_solution_state(0));
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
    }

    py::tuple apply_legacy_candidate(double temperature, double random_draw) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!legacy_candidate_ready_ || last_candidate_ready_
            || pending_composite_active_) {
            throw std::runtime_error(
                "full native legacy lane has no prepared candidate to apply");
        }
        if (!std::isfinite(temperature) || temperature <= 0.0
            || !std::isfinite(random_draw) || random_draw < 0.0
            || random_draw > 1.0) {
            throw std::invalid_argument(
                "full native legacy acceptance inputs are invalid");
        }
        swap_active_with_lane_noexcept(0);
        ScopeRollback restore_lane([this]() noexcept {
            swap_active_with_lane_noexcept(0);
        });
        last_candidate_offsets_ = std::move(legacy_candidate_offsets_);
        last_candidate_indices_ = std::move(legacy_candidate_indices_);
        last_candidate_exact_payload_ =
            std::move(legacy_candidate_exact_payload_);
        last_candidate_objective_integer_ =
            std::move(legacy_candidate_objective_integer_);
        last_candidate_objective_float_ =
            std::move(legacy_candidate_objective_float_);
        last_candidate_ready_ = true;
        legacy_candidate_ready_ = false;
        const auto comparison = last_candidate_comparison();
        const auto operator_index = legacy_candidate_operator_;
        if (operator_index < 0
            || operator_index >= static_cast<std::int64_t>(
                full_operator_totals_.size())) {
            throw std::logic_error(
                "full native legacy candidate lost its operator identity");
        }
        auto outcome = apply_last_candidate(temperature, random_draw);
        accumulate_full_stage04_outcome_noexcept(
            static_cast<std::size_t>(operator_index),
            py::cast<std::int64_t>(outcome[0]) != 0, comparison,
            py::cast<std::int64_t>(outcome[1]) != 0,
            py::cast<std::int64_t>(outcome[2]) != 0, true);
        if (operator_index == 0 || operator_index == 1) {
            if (legacy_candidate_destroy_operator_ < 0
                || legacy_candidate_destroy_operator_ >= 3
                || (operator_index == 0
                    && (legacy_candidate_repair_operator_ < 0
                        || legacy_candidate_repair_operator_ >= 3))) {
                throw std::logic_error(
                    "full native legacy candidate lost its role identities");
            }
            const auto accepted = py::cast<std::int64_t>(outcome[0]) != 0;
            const auto is_global_best = py::cast<std::int64_t>(outcome[1]) != 0;
            const auto vehicle_reduction = py::cast<std::int64_t>(outcome[2]) != 0;
            accumulate_full_stage04_outcome_noexcept(
                static_cast<std::size_t>(14 + legacy_candidate_destroy_operator_),
                accepted, comparison, is_global_best, vehicle_reduction, true);
            if (operator_index == 0) {
                accumulate_full_stage04_outcome_noexcept(
                    static_cast<std::size_t>(
                        17 + legacy_candidate_repair_operator_),
                    accepted, comparison, is_global_best, vehicle_reduction, true);
            }
        }
        legacy_candidate_operator_ = -1;
        legacy_candidate_destroy_operator_ = -1;
        legacy_candidate_repair_operator_ = -1;
        restore_lane.rollback_now();
        return outcome;
    }

    py::tuple legacy_vehicle_count_aware_probe(
        std::int64_t iteration,
        double removal_fraction,
        std::int64_t route_change_limit,
        py::handle deadline_remaining,
        py::handle batch_size,
        bool defer_acceptance,
        bool consume_main_selection = false) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !rng_.has_value() || iteration < 0
            || !std::isfinite(removal_fraction) || removal_fraction <= 0.0
            || removal_fraction > 1.0 || route_change_limit == 0
            || route_change_limit < -1 || last_candidate_ready_
            || legacy_candidate_ready_ || pending_composite_active_) {
            throw std::invalid_argument(
                "full native vehicle-count-aware probe state/config is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "legacy_repair_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "legacy_repair_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native vehicle-count-aware deadline/batch is invalid");
        }
        const auto customer_count = static_cast<std::int64_t>(
            legacy_indices_.size());
        auto remove_count = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(std::ceil(
                static_cast<double>(all_customers_.size()) * removal_fraction)));
        if (all_customers_.size() > 20) {
            remove_count = std::min<std::int64_t>(remove_count, 3);
        }
        remove_count = std::min(remove_count, customer_count);

        auto next_rng = *rng_;
        if (consume_main_selection) {
            const auto selected_main = static_cast<std::int64_t>(
                next_rng.weighted_index({
                    full_operator_weights_[0], full_operator_weights_[1],
                    full_operator_weights_[2], full_operator_weights_[3]}));
            if (selected_main != 1) {
                throw std::logic_error(
                    "full native vehicle-count-aware dispatcher selection changed");
            }
        }
        const auto destroy_operation = static_cast<std::int64_t>(
            next_rng.weighted_index({
                full_operator_weights_[14], full_operator_weights_[15],
                full_operator_weights_[16]}));
        const auto accumulate_vehicle_repair_outcome = [this, destroy_operation](
            bool accepted,
            std::int64_t comparison,
            bool is_global_best,
            bool vehicle_reduction) noexcept {
            accumulate_full_stage04_outcome_noexcept(
                1, accepted, comparison, is_global_best, vehicle_reduction, true);
            accumulate_full_stage04_outcome_noexcept(
                static_cast<std::size_t>(14 + destroy_operation), accepted,
                comparison, is_global_best, vehicle_reduction, true);
        };
        const auto* route_boundaries =
            checked_data<std::int64_t>(legacy_offsets_);
        const auto* route_nodes = checked_data<std::int64_t>(legacy_indices_);
        std::vector<std::int64_t> customers(
            route_nodes, route_nodes + legacy_indices_.size());
        std::vector<std::int64_t> removed;
        removed.reserve(static_cast<std::size_t>(remove_count));
        if (destroy_operation == 0) {
            for (const auto index : next_rng.sample_indices(
                     customer_count, remove_count)) {
                removed.push_back(customers[static_cast<std::size_t>(index)]);
            }
        } else if (destroy_operation == 1) {
            struct Contribution {
                double saving;
                std::int64_t customer;
            };
            std::vector<Contribution> contributions;
            const auto node_count = static_cast<std::size_t>(node_kind_.size());
            const auto* distances = checked_data<double>(distance_);
            for (py::ssize_t route = 0; route + 1 < legacy_offsets_.size(); ++route) {
                auto previous = depot_;
                for (auto cursor = route_boundaries[route];
                     cursor < route_boundaries[route + 1]; ++cursor) {
                    const auto customer = route_nodes[cursor];
                    const auto after = cursor + 1 < route_boundaries[route + 1]
                        ? route_nodes[cursor + 1] : depot_;
                    const auto saving = distances[
                        static_cast<std::size_t>(previous) * node_count
                        + static_cast<std::size_t>(customer)]
                        + distances[
                            static_cast<std::size_t>(customer) * node_count
                            + static_cast<std::size_t>(after)]
                        - distances[
                            static_cast<std::size_t>(previous) * node_count
                            + static_cast<std::size_t>(after)];
                    contributions.push_back({saving, customer});
                    previous = customer;
                }
            }
            std::stable_sort(
                contributions.begin(), contributions.end(),
                [](const Contribution& left, const Contribution& right) {
                    if (left.saving != right.saving) {
                        return left.saving > right.saving;
                    }
                    return left.customer > right.customer;
                });
            for (std::int64_t index = 0; index < remove_count; ++index) {
                removed.push_back(
                    contributions[static_cast<std::size_t>(index)].customer);
            }
        } else {
            const auto anchor = customers[static_cast<std::size_t>(
                next_rng.randbelow(static_cast<std::uint64_t>(customer_count)))];
            const auto node_count = static_cast<std::size_t>(node_kind_.size());
            const auto* distances = checked_data<double>(distance_);
            std::vector<std::pair<double, std::int64_t>> related;
            related.reserve(customers.size());
            for (const auto customer : customers) {
                related.emplace_back(
                    distances[static_cast<std::size_t>(anchor) * node_count
                        + static_cast<std::size_t>(customer)],
                    customer);
            }
            std::sort(related.begin(), related.end());
            for (std::int64_t index = 0; index < remove_count; ++index) {
                removed.push_back(related[static_cast<std::size_t>(index)].second);
            }
        }

        const std::unordered_set<std::int64_t> removed_set(
            removed.begin(), removed.end());
        std::vector<std::int64_t> partial_offsets{0};
        std::vector<std::int64_t> partial_indices;
        for (py::ssize_t route = 0; route + 1 < legacy_offsets_.size(); ++route) {
            const auto before = partial_indices.size();
            for (auto cursor = route_boundaries[route];
                 cursor < route_boundaries[route + 1]; ++cursor) {
                if (!removed_set.contains(route_nodes[cursor])) {
                    partial_indices.push_back(route_nodes[cursor]);
                }
            }
            if (partial_indices.size() != before) {
                partial_offsets.push_back(
                    static_cast<std::int64_t>(partial_indices.size()));
            }
        }
        py::array_t<std::int64_t> removed_array(removed.size());
        py::array_t<std::int64_t> partial_offsets_array(partial_offsets.size());
        py::array_t<std::int64_t> partial_indices_array(partial_indices.size());
        std::copy(removed.begin(), removed.end(), checked_data(removed_array));
        std::copy(
            partial_offsets.begin(), partial_offsets.end(),
            checked_data(partial_offsets_array));
        std::copy(
            partial_indices.begin(), partial_indices.end(),
            checked_data(partial_indices_array));
        auto repair = candidate_control_repair_v2(
            node_kind_, demand_, ready_time_, due_date_, service_time_,
            distance_, reachable_, vehicle_, lexical_rank_,
            partial_offsets_array, partial_indices_array, removed_array,
            screening_epsilon_, route_change_limit, true);
        auto repaired_offsets = py::cast<py::array_t<std::int64_t>>(repair[0]);
        auto repaired_indices = py::cast<py::array_t<std::int64_t>>(repair[1]);
        auto repair_counters = py::cast<py::array_t<std::int64_t>>(repair[2]);
        py::array_t<std::int64_t> metadata(8);
        auto* metadata_values = checked_data(metadata);
        metadata_values[0] = iteration;
        metadata_values[1] = destroy_operation;
        metadata_values[2] = remove_count;
        metadata_values[3] = checked_data<std::int64_t>(repair_counters)[0];
        metadata_values[4] = checked_data<std::int64_t>(repair_counters)[1];
        metadata_values[5] = -1;
        metadata_values[6] = 0;
        metadata_values[7] = -2;
        if (metadata_values[3] != 0) {
            accumulate_vehicle_repair_outcome(false, 1, false, false);
            *rng_ = std::move(next_rng);
            return py::make_tuple(
                std::move(metadata), std::move(removed_array),
                std::move(partial_offsets_array),
                std::move(partial_indices_array), std::move(repair),
                py::none(), lane_solution_state(0));
        }

        py::array_t<std::int64_t> plan_offsets(2);
        checked_data(plan_offsets)[0] = 0;
        checked_data(plan_offsets)[1] = repaired_offsets.size() - 1;
        py::array_t<std::int64_t> context(3);
        checked_data(context)[0] = stable_int63("legacy");
        checked_data(context)[1] = stable_int63("vehicle_count_aware_repair");
        checked_data(context)[2] = iteration;
        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        swap_active_with_lane_noexcept(0);
        ScopeRollback restore_lane([this]() noexcept {
            swap_active_with_lane_noexcept(0);
        });
        suppress_attempted_plan_journal_ = true;
        ScopeRollback restore_attempted_plan_policy([this]() noexcept {
            suppress_attempted_plan_journal_ = false;
        });
        defer_composite_commit_ = true;
        try {
            auto transaction = evaluate_plans(
                plan_offsets, repaired_offsets, repaired_indices, context,
                deadline_array, batch_array, expected_array);
            defer_composite_commit_ = false;
            const auto selected = prepare_first_feasible_candidate(
                plan_offsets, repaired_offsets, repaired_indices, transaction);
            metadata_values[6] = static_cast<std::int64_t>(
                py::cast<py::array_t<std::int64_t>>(transaction[5]).size());
            if (selected.has_value()) {
                commit_pending_composite_noexcept();
                metadata_values[5] = *selected;
                if (defer_acceptance) {
                    legacy_candidate_offsets_ =
                        std::move(last_candidate_offsets_);
                    legacy_candidate_indices_ =
                        std::move(last_candidate_indices_);
                    legacy_candidate_exact_payload_ =
                        std::move(last_candidate_exact_payload_);
                    legacy_candidate_objective_integer_ =
                        std::move(last_candidate_objective_integer_);
                    legacy_candidate_objective_float_ =
                        std::move(last_candidate_objective_float_);
                    last_candidate_ready_ = false;
                    legacy_candidate_ready_ = true;
                    legacy_candidate_operator_ = 1;
                    legacy_candidate_destroy_operator_ = destroy_operation;
                    legacy_candidate_repair_operator_ = -1;
                } else {
                    const auto comparison = last_candidate_comparison();
                    const auto draw = next_rng.random();
                    auto acceptance = apply_last_candidate(
                        stage04_initial_temperature_, draw);
                    metadata_values[7] = py::cast<std::int64_t>(acceptance[0]);
                    accumulate_vehicle_repair_outcome(
                        metadata_values[7] != 0, comparison,
                        py::cast<std::int64_t>(acceptance[1]) != 0,
                        py::cast<std::int64_t>(acceptance[2]) != 0);
                }
            } else {
                commit_pending_composite_noexcept();
                accumulate_vehicle_repair_outcome(false, 1, false, false);
                metadata_values[7] = 0;
            }
            *rng_ = std::move(next_rng);
            suppress_attempted_plan_journal_ = false;
            restore_attempted_plan_policy.release();
            restore_lane.rollback_now();
            return py::make_tuple(
                std::move(metadata), std::move(removed_array),
                std::move(partial_offsets_array),
                std::move(partial_indices_array), std::move(repair),
                std::move(transaction), lane_solution_state(0));
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
    }

    py::tuple legacy_vehicle_reduction_refinement(
        std::int64_t iteration,
        std::int64_t evaluation_budget,
        py::handle deadline_remaining,
        py::handle batch_size) {
        const auto refinement_started = std::chrono::steady_clock::now();
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !legacy_candidate_ready_ || iteration < 0
            || evaluation_budget <= 0 || last_candidate_ready_
            || pending_composite_active_ || suppress_attempted_plan_journal_) {
            throw std::invalid_argument(
                "full native legacy refinement state/config is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "refinement_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "refinement_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native legacy refinement deadline/batch is invalid");
        }
        const auto total_deadline = checked_data<double>(deadline_array)[0];
        const auto next_deadline = [&]() {
            const auto remaining = total_deadline - std::chrono::duration<double>(
                std::chrono::steady_clock::now() - refinement_started).count();
            if (remaining <= 0.0) {
                throw std::runtime_error(
                    "full native legacy refinement reached its deadline");
            }
            py::array_t<double> output(1);
            checked_data(output)[0] = remaining;
            return output;
        };
        const auto entry_budget = budget_.native_snapshot();
        const auto route_count = legacy_candidate_offsets_.size() - 1;
        const auto* candidate_boundaries =
            checked_data<std::int64_t>(legacy_candidate_offsets_);
        const auto* candidate_nodes =
            checked_data<std::int64_t>(legacy_candidate_indices_);
        std::vector<std::vector<std::int64_t>> candidate_routes;
        candidate_routes.reserve(static_cast<std::size_t>(route_count));
        for (std::int64_t route = 0; route < route_count; ++route) {
            candidate_routes.emplace_back(
                candidate_nodes + candidate_boundaries[route],
                candidate_nodes + candidate_boundaries[route + 1]);
        }
        const auto customer_count = static_cast<std::int64_t>(all_customers_.size());
        const auto removal_count = std::max<std::int64_t>(
            1, std::min<std::int64_t>(3, customer_count - 1));
        struct Contribution {
            double saving;
            std::int64_t customer;
        };
        std::vector<Contribution> contributions;
        for (const auto& route : candidate_routes) {
            for (std::size_t position = 0; position < route.size(); ++position) {
                const auto before = position == 0 ? depot_ : route[position - 1];
                const auto customer = route[position];
                const auto after = position + 1 == route.size()
                    ? depot_ : route[position + 1];
                const auto node_count = static_cast<std::size_t>(node_kind_.size());
                const auto* distances = checked_data<double>(distance_);
                const auto saving = distances[before * node_count + customer]
                    + distances[customer * node_count + after]
                    - distances[before * node_count + after];
                contributions.push_back({saving, customer});
            }
        }
        std::stable_sort(
            contributions.begin(), contributions.end(),
            [&](const Contribution& left, const Contribution& right) {
                if (left.saving != right.saving) {
                    return left.saving > right.saving;
                }
                return checked_data<std::int64_t>(lexical_rank_)[left.customer]
                    > checked_data<std::int64_t>(lexical_rank_)[right.customer];
            });
        std::vector<std::int64_t> removed;
        removed.reserve(static_cast<std::size_t>(removal_count));
        for (std::int64_t index = 0; index < removal_count; ++index) {
            removed.push_back(contributions[static_cast<std::size_t>(index)].customer);
        }
        const std::unordered_set<std::int64_t> removed_set(
            removed.begin(), removed.end());
        std::vector<std::vector<std::int64_t>> sequences;
        for (const auto& route : candidate_routes) {
            std::vector<std::int64_t> partial;
            for (const auto customer : route) {
                if (!removed_set.contains(customer)) {
                    partial.push_back(customer);
                }
            }
            if (!partial.empty()) {
                sequences.push_back(std::move(partial));
            }
        }
        const auto pack_plan = [](const auto& routes) {
            py::array_t<std::int64_t> plan_offsets(2);
            py::array_t<std::int64_t> route_offsets(routes.size() + 1);
            std::size_t index_count = 0;
            for (const auto& route : routes) {
                index_count += route.size();
            }
            py::array_t<std::int64_t> route_indices(index_count);
            checked_data(plan_offsets)[0] = 0;
            checked_data(plan_offsets)[1] =
                static_cast<std::int64_t>(routes.size());
            checked_data(route_offsets)[0] = 0;
            std::size_t cursor = 0;
            for (std::size_t route = 0; route < routes.size(); ++route) {
                std::copy(
                    routes[route].begin(), routes[route].end(),
                    checked_data(route_indices) + cursor);
                cursor += routes[route].size();
                checked_data(route_offsets)[route + 1] =
                    static_cast<std::int64_t>(cursor);
            }
            return py::make_tuple(
                std::move(plan_offsets), std::move(route_offsets),
                std::move(route_indices));
        };
        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        py::array_t<std::int64_t> context(3);
        checked_data(context)[0] = stable_int63("legacy");
        checked_data(context)[1] =
            stable_int63("vehicle_reduction_refinement");
        checked_data(context)[2] = iteration;
        suppress_attempted_plan_journal_ = true;
        allow_partial_customer_coverage_ = true;
        ScopeRollback restore_journal([this]() noexcept {
            suppress_attempted_plan_journal_ = false;
            allow_partial_customer_coverage_ = false;
        });
        const auto exact_used = [&]() {
            return budget_.native_snapshot().started - entry_budget.started;
        };
        std::vector<std::int64_t> evaluation_plan_offsets{0};
        std::vector<std::int64_t> evaluation_route_offsets{0};
        std::vector<std::int64_t> evaluation_route_indices;
        std::vector<std::int64_t> evaluation_exact_deltas;
        struct Option {
            double delta;
            std::int64_t route;
            std::int64_t position;
            std::vector<std::vector<std::int64_t>> routes;
        };
        const auto route_lexical_less = [&](const auto& left, const auto& right) {
            return std::lexicographical_compare(
                left.begin(), left.end(), right.begin(), right.end(),
                [&](std::int64_t lhs, std::int64_t rhs) {
                    return checked_data<std::int64_t>(lexical_rank_)[lhs]
                        < checked_data<std::int64_t>(lexical_rank_)[rhs];
                });
        };
        const auto plan_lexical_less = [&](const auto& left, const auto& right) {
            return std::lexicographical_compare(
                left.begin(), left.end(), right.begin(), right.end(),
                route_lexical_less);
        };
        auto evaluate = [&](const auto& routes) -> std::optional<double> {
            if (exact_used() >= evaluation_budget) {
                return std::nullopt;
            }
            auto packed = pack_plan(routes);
            const auto exact_before = budget_.native_snapshot().started;
            std::vector<std::int64_t> partial_expected;
            for (const auto& route : routes) {
                partial_expected.insert(
                    partial_expected.end(), route.begin(), route.end());
            }
            std::stable_sort(
                partial_expected.begin(), partial_expected.end(),
                [&](std::int64_t left, std::int64_t right) {
                    return checked_data<std::int64_t>(lexical_rank_)[left]
                        < checked_data<std::int64_t>(lexical_rank_)[right];
                });
            py::array_t<std::int64_t> partial_expected_array(
                partial_expected.size());
            std::copy(
                partial_expected.begin(), partial_expected.end(),
                checked_data(partial_expected_array));
            auto transaction = evaluate_plans(
                packed[0], packed[1], packed[2], context,
                next_deadline(), batch_array, partial_expected_array);
            for (const auto& route : routes) {
                evaluation_route_indices.insert(
                    evaluation_route_indices.end(), route.begin(), route.end());
                evaluation_route_offsets.push_back(
                    static_cast<std::int64_t>(evaluation_route_indices.size()));
            }
            evaluation_plan_offsets.push_back(
                static_cast<std::int64_t>(evaluation_route_offsets.size() - 1));
            evaluation_exact_deltas.push_back(
                budget_.native_snapshot().started - exact_before);
            auto feasible =
                py::cast<py::array_t<std::int64_t>>(transaction[11]);
            if (feasible.size() == 0) {
                return std::nullopt;
            }
            auto objective = py::cast<py::array_t<double>>(transaction[3]);
            return checked_data<double>(objective)[0];
        };
        std::int64_t failure_code = 0;
        while (!removed.empty()) {
            struct CustomerOptions {
                std::int64_t customer;
                std::vector<Option> options;
            };
            std::vector<CustomerOptions> options_by_customer;
            for (const auto customer : removed) {
                CustomerOptions customer_options{customer, {}};
                const auto old_total = evaluate(sequences);
                if (!old_total.has_value() && exact_used() >= evaluation_budget) {
                    failure_code = 2;
                    break;
                }
                if (!old_total.has_value()) {
                    options_by_customer.push_back(std::move(customer_options));
                    continue;
                }
                for (std::size_t route = 0; route < sequences.size(); ++route) {
                    PythonFloatSum demand_sum;
                    for (const auto node : sequences[route]) {
                        demand_sum.add(checked_data<double>(demand_)[node]);
                    }
                    demand_sum.add(checked_data<double>(demand_)[customer]);
                    if (demand_sum.value()
                        > checked_data<double>(vehicle_)[1] + screening_epsilon_) {
                        continue;
                    }
                    for (std::size_t position = 0;
                         position <= sequences[route].size(); ++position) {
                        if (exact_used() >= evaluation_budget) {
                            failure_code = 2;
                            break;
                        }
                        auto candidate = sequences;
                        candidate[route].insert(
                            candidate[route].begin()
                                + static_cast<std::ptrdiff_t>(position),
                            customer);
                        const auto candidate_total = evaluate(candidate);
                        if (candidate_total.has_value()) {
                            customer_options.options.push_back(Option{
                                *candidate_total - *old_total,
                                static_cast<std::int64_t>(route),
                                static_cast<std::int64_t>(position),
                                std::move(candidate)});
                        }
                    }
                    if (failure_code != 0) {
                        break;
                    }
                }
                std::stable_sort(
                    customer_options.options.begin(),
                    customer_options.options.end(),
                    [&](const Option& left, const Option& right) {
                        if (left.delta != right.delta) {
                            return left.delta < right.delta;
                        }
                        if (left.route != right.route) {
                            return left.route < right.route;
                        }
                        if (left.position != right.position) {
                            return left.position < right.position;
                        }
                        return plan_lexical_less(left.routes, right.routes);
                    });
                options_by_customer.push_back(std::move(customer_options));
                if (failure_code != 0) {
                    break;
                }
            }
            if (failure_code != 0) {
                break;
            }
            std::optional<std::size_t> selected_customer;
            double selected_regret = -std::numeric_limits<double>::infinity();
            for (std::size_t index = 0; index < options_by_customer.size(); ++index) {
                const auto& candidate = options_by_customer[index];
                if (candidate.options.empty()) {
                    continue;
                }
                const auto regret = candidate.options.size() < 2
                    ? std::numeric_limits<double>::infinity()
                    : candidate.options[1].delta - candidate.options[0].delta;
                if (!selected_customer.has_value()
                    || regret > selected_regret
                    || (regret == selected_regret
                        && checked_data<std::int64_t>(lexical_rank_)[candidate.customer]
                            > checked_data<std::int64_t>(lexical_rank_)[
                                options_by_customer[*selected_customer].customer])) {
                    selected_customer = index;
                    selected_regret = regret;
                }
            }
            if (!selected_customer.has_value()) {
                failure_code = 1;
                break;
            }
            const auto customer =
                options_by_customer[*selected_customer].customer;
            sequences = std::move(
                options_by_customer[*selected_customer].options[0].routes);
            removed.erase(
                std::find(removed.begin(), removed.end(), customer));
        }
        auto final_plan = pack_plan(sequences);
        py::array_t<std::int64_t> metadata(4);
        checked_data(metadata)[0] = failure_code;
        checked_data(metadata)[1] = 0;
        checked_data(metadata)[2] = exact_used();
        checked_data(metadata)[3] = removal_count;
        py::array_t<std::int64_t> removed_array(removal_count);
        for (std::int64_t index = 0; index < removal_count; ++index) {
            checked_data(removed_array)[index] =
                contributions[static_cast<std::size_t>(index)].customer;
        }
        if (failure_code == 0) {
            defer_composite_commit_ = true;
            try {
                auto final_transaction = evaluate_plans(
                    final_plan[0], final_plan[1], final_plan[2], context,
                    next_deadline(), batch_array, expected_array);
                defer_composite_commit_ = false;
                const auto selected = prepare_first_feasible_candidate(
                    py::cast<py::array_t<std::int64_t>>(final_plan[0]),
                    py::cast<py::array_t<std::int64_t>>(final_plan[1]),
                    py::cast<py::array_t<std::int64_t>>(final_plan[2]),
                    final_transaction);
                if (!selected.has_value()) {
                    throw std::logic_error(
                        "full native refinement lost its final feasible plan");
                }
                const auto refined_key =
                    evrptw::formal_objective::key_from_arrays(
                    last_candidate_objective_integer_,
                    last_candidate_objective_float_);
                const auto legacy_key =
                    evrptw::formal_objective::key_from_arrays(
                    legacy_candidate_objective_integer_,
                    legacy_candidate_objective_float_);
                commit_pending_composite_noexcept();
                if (refined_key < legacy_key) {
                    legacy_candidate_offsets_ = std::move(last_candidate_offsets_);
                    legacy_candidate_indices_ = std::move(last_candidate_indices_);
                    legacy_candidate_exact_payload_ =
                        std::move(last_candidate_exact_payload_);
                    legacy_candidate_objective_integer_ =
                        std::move(last_candidate_objective_integer_);
                    legacy_candidate_objective_float_ =
                        std::move(last_candidate_objective_float_);
                    checked_data(metadata)[1] = 1;
                }
                last_candidate_ready_ = false;
            } catch (...) {
                defer_composite_commit_ = false;
                if (pending_composite_active_) {
                    rollback_pending_composite();
                }
                last_candidate_ready_ = false;
                throw;
            }
        }
        suppress_attempted_plan_journal_ = false;
        allow_partial_customer_coverage_ = false;
        restore_journal.release();
        py::array_t<std::int64_t> evaluation_plan_offsets_array(
            evaluation_plan_offsets.size());
        py::array_t<std::int64_t> evaluation_route_offsets_array(
            evaluation_route_offsets.size());
        py::array_t<std::int64_t> evaluation_route_indices_array(
            evaluation_route_indices.size());
        py::array_t<std::int64_t> evaluation_exact_deltas_array(
            evaluation_exact_deltas.size());
        std::copy(
            evaluation_plan_offsets.begin(), evaluation_plan_offsets.end(),
            checked_data(evaluation_plan_offsets_array));
        std::copy(
            evaluation_route_offsets.begin(), evaluation_route_offsets.end(),
            checked_data(evaluation_route_offsets_array));
        std::copy(
            evaluation_route_indices.begin(), evaluation_route_indices.end(),
            checked_data(evaluation_route_indices_array));
        std::copy(
            evaluation_exact_deltas.begin(), evaluation_exact_deltas.end(),
            checked_data(evaluation_exact_deltas_array));
        auto evaluation_journal = py::make_tuple(
            std::move(evaluation_plan_offsets_array),
            std::move(evaluation_route_offsets_array),
            std::move(evaluation_route_indices_array),
            std::move(evaluation_exact_deltas_array));
        const auto refinement_selected = checked_data(metadata)[1] != 0;
        accumulate_full_stage04_outcome_noexcept(
            13, refinement_selected, refinement_selected ? -1 : 1,
            refinement_selected, refinement_selected, false);
        return py::make_tuple(
            std::move(metadata), std::move(removed_array),
            std::move(final_plan), std::move(evaluation_journal));
    }

    py::tuple quality_changed_probe(
        std::int64_t operation,
        std::int64_t iteration,
        py::handle deadline_remaining,
        py::handle batch_size) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || operation < 0 || operation > 2 || iteration < 0
            || last_candidate_ready_ || pending_composite_active_) {
            throw std::invalid_argument(
                "full native quality changed-route probe state/config is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "quality_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "quality_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native quality relocate deadline/batch is invalid");
        }
        auto pool = changed_candidate_pool_v1(
            operation, quality_offsets_, quality_indices_);
        auto changed_routes = py::cast<py::array_t<std::int64_t>>(pool[0]);
        auto change_offsets = py::cast<py::array_t<std::int64_t>>(pool[1]);
        auto change_indices = py::cast<py::array_t<std::int64_t>>(pool[2]);
        const auto candidate_count = changed_routes.shape(0);
        const auto route_count = quality_offsets_.size() - 1;
        const auto* base_offsets = checked_data<std::int64_t>(quality_offsets_);
        const auto* base_indices = checked_data<std::int64_t>(quality_indices_);
        const auto* changed = checked_data<std::int64_t>(changed_routes);
        const auto* changes = checked_data<std::int64_t>(change_offsets);
        const auto* changed_indices = checked_data<std::int64_t>(change_indices);
        std::vector<std::int64_t> plan_offsets{0};
        std::vector<std::int64_t> route_offsets{0};
        std::vector<std::int64_t> route_indices;
        std::unordered_set<std::string> seen_plans;
        for (py::ssize_t candidate = 0; candidate < candidate_count; ++candidate) {
            std::string identity("stage05.2-native-quality-plan-v2");
            std::vector<std::vector<std::int64_t>> routes;
            routes.reserve(static_cast<std::size_t>(route_count));
            for (py::ssize_t route = 0; route < route_count; ++route) {
                const auto first_changed = changed[candidate * 2];
                const auto second_changed = changed[candidate * 2 + 1];
                const std::int64_t* begin = nullptr;
                const std::int64_t* end = nullptr;
                if (route == first_changed) {
                    begin = changed_indices + changes[candidate * 2];
                    end = changed_indices + changes[candidate * 2 + 1];
                } else if (route == second_changed) {
                    begin = changed_indices + changes[candidate * 2 + 1];
                    end = changed_indices + changes[candidate * 2 + 2];
                } else {
                    begin = base_indices + base_offsets[route];
                    end = base_indices + base_offsets[route + 1];
                }
                routes.emplace_back(begin, end);
                const auto length = static_cast<std::int64_t>(end - begin);
                append_evidence_values(identity, &length, 1);
                append_evidence_values(
                    identity, begin, static_cast<std::size_t>(length));
            }
            if (!seen_plans.insert(identity).second) {
                continue;
            }
            for (const auto& route : routes) {
                route_indices.insert(
                    route_indices.end(), route.begin(), route.end());
                route_offsets.push_back(
                    static_cast<std::int64_t>(route_indices.size()));
            }
            plan_offsets.push_back(
                static_cast<std::int64_t>(route_offsets.size() - 1));
        }
        py::array_t<std::int64_t> plan_offsets_array(plan_offsets.size());
        py::array_t<std::int64_t> route_offsets_array(route_offsets.size());
        py::array_t<std::int64_t> route_indices_array(route_indices.size());
        std::copy(
            plan_offsets.begin(), plan_offsets.end(),
            checked_data(plan_offsets_array));
        std::copy(
            route_offsets.begin(), route_offsets.end(),
            checked_data(route_offsets_array));
        std::copy(
            route_indices.begin(), route_indices.end(),
            checked_data(route_indices_array));
        if (plan_offsets.size() == 1) {
            py::array_t<std::int64_t> outcome(4);
            std::fill(
                checked_data(outcome), checked_data(outcome) + 4,
                std::int64_t{0});
            checked_data(outcome)[0] = -1;
            accumulate_full_stage04_outcome_noexcept(
                static_cast<std::size_t>(operation + 4), false, 1,
                false, false, true);
            return py::make_tuple(
                std::move(pool), py::none(), std::move(outcome),
                lane_solution_state(1));
        }
        py::array_t<std::int64_t> context(3);
        checked_data(context)[0] = stable_int63("quality_shadow");
        constexpr std::array<std::string_view, 3> operation_names{
            "relocate", "swap", "two_opt_star"};
        checked_data(context)[1] = stable_int63(
            operation_names[static_cast<std::size_t>(operation)]);
        checked_data(context)[2] = iteration;
        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        swap_active_with_lane_noexcept(1);
        ScopeRollback restore_lane([this]() noexcept {
            swap_active_with_lane_noexcept(1);
        });
        defer_composite_commit_ = true;
        try {
            auto transaction = evaluate_plans(
                plan_offsets_array, route_offsets_array, route_indices_array,
                context, deadline_array, batch_array, expected_array);
            defer_composite_commit_ = false;
            py::array_t<std::int64_t> outcome(4);
            std::fill(checked_data(outcome), checked_data(outcome) + 4, 0);
            checked_data(outcome)[0] = -1;
            const auto selected_plan = prepare_first_feasible_candidate(
                plan_offsets_array, route_offsets_array, route_indices_array,
                transaction);
            if (selected_plan.has_value()) {
                commit_pending_composite_noexcept();
                checked_data(outcome)[0] = *selected_plan;
                const auto comparison = last_candidate_comparison();
                if (!budget_.budget_reached()) {
                    auto acceptance = apply_last_candidate(1.0, 1.0);
                    checked_data(outcome)[1] =
                        py::cast<std::int64_t>(acceptance[0]);
                    checked_data(outcome)[2] =
                        py::cast<std::int64_t>(acceptance[1]);
                    checked_data(outcome)[3] =
                        py::cast<std::int64_t>(acceptance[2]);
                } else {
                    last_candidate_ready_ = false;
                }
                accumulate_full_stage04_outcome_noexcept(
                    static_cast<std::size_t>(operation + 4),
                    checked_data(outcome)[1] != 0, comparison,
                    checked_data(outcome)[2] != 0,
                    checked_data(outcome)[3] != 0, true);
            } else {
                commit_pending_composite_noexcept();
                accumulate_full_stage04_outcome_noexcept(
                    static_cast<std::size_t>(operation + 4), false, 1,
                    false, false, true);
            }
            restore_lane.rollback_now();
            return py::make_tuple(
                std::move(pool), std::move(transaction), std::move(outcome),
                lane_solution_state(1));
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
    }

    py::tuple legacy_standard_probe(
        std::int64_t iteration,
        double removal_fraction,
        std::int64_t route_change_limit,
        py::handle deadline_remaining,
        py::handle batch_size) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !rng_.has_value() || iteration < 3
            || !std::isfinite(removal_fraction) || removal_fraction <= 0.0
            || removal_fraction > 1.0 || route_change_limit == 0
            || route_change_limit < -1 || last_candidate_ready_
            || legacy_candidate_ready_ || pending_composite_active_) {
            throw std::invalid_argument(
                "full native standard probe state/config is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "legacy_standard_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "legacy_standard_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native standard deadline/batch is invalid");
        }
        const auto entry_exact_started = budget_.native_snapshot().started;
        auto next_rng = *rng_;
        const std::vector<double> main_weights{
            full_operator_weights_[0], full_operator_weights_[1],
            full_operator_weights_[2], full_operator_weights_[3]};
        const auto selected_main = static_cast<std::int64_t>(
            next_rng.weighted_index(main_weights));
        if (selected_main != 0) {
            throw std::runtime_error(
                "full native weighted legacy follow-up selected an unimplemented operator");
        }
        const auto destroy = static_cast<std::int64_t>(
            next_rng.weighted_index({
                full_operator_weights_[14], full_operator_weights_[15],
                full_operator_weights_[16]}));
        auto remove_count = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(std::ceil(
                static_cast<double>(all_customers_.size()) * removal_fraction)));
        if (all_customers_.size() > 20) {
            remove_count = std::min<std::int64_t>(remove_count, 3);
        }
        remove_count = std::min<std::int64_t>(
            remove_count, legacy_indices_.size());
        std::vector<std::int64_t> customers(
            checked_data<std::int64_t>(legacy_indices_),
            checked_data<std::int64_t>(legacy_indices_) + legacy_indices_.size());
        std::vector<std::int64_t> removed;
        if (destroy == 0) {
            const auto sampled = next_rng.sample_indices(
                legacy_indices_.size(), remove_count);
            for (const auto index : sampled) {
                removed.push_back(customers[static_cast<std::size_t>(index)]);
            }
        } else if (destroy == 1) {
            std::vector<std::pair<double, std::int64_t>> contributions;
            const auto node_count = static_cast<std::size_t>(node_kind_.size());
            const auto* distances = checked_data<double>(distance_);
            const auto* route_offsets = checked_data<std::int64_t>(legacy_offsets_);
            for (py::ssize_t route = 0; route + 1 < legacy_offsets_.size(); ++route) {
                for (auto cursor = route_offsets[route];
                     cursor < route_offsets[route + 1]; ++cursor) {
                    const auto before = cursor == route_offsets[route]
                        ? depot_ : customers[static_cast<std::size_t>(cursor - 1)];
                    const auto customer = customers[static_cast<std::size_t>(cursor)];
                    const auto after = cursor + 1 == route_offsets[route + 1]
                        ? depot_ : customers[static_cast<std::size_t>(cursor + 1)];
                    const auto saving = distances[
                        static_cast<std::size_t>(before) * node_count
                        + static_cast<std::size_t>(customer)]
                        + distances[
                            static_cast<std::size_t>(customer) * node_count
                            + static_cast<std::size_t>(after)]
                        - distances[
                            static_cast<std::size_t>(before) * node_count
                            + static_cast<std::size_t>(after)];
                    contributions.emplace_back(saving, customer);
                }
            }
            std::stable_sort(
                contributions.begin(), contributions.end(),
                [&](const auto& left, const auto& right) {
                    if (left.first != right.first) {
                        return left.first > right.first;
                    }
                    return checked_data<std::int64_t>(lexical_rank_)[left.second]
                        > checked_data<std::int64_t>(lexical_rank_)[right.second];
                });
            for (std::int64_t index = 0; index < remove_count; ++index) {
                removed.push_back(
                    contributions[static_cast<std::size_t>(index)].second);
            }
        } else {
            const auto anchor = customers[static_cast<std::size_t>(
                next_rng.randbelow(customers.size()))];
            const auto node_count = static_cast<std::size_t>(node_kind_.size());
            const auto* distances = checked_data<double>(distance_);
            std::vector<std::pair<double, std::int64_t>> related;
            for (const auto customer : customers) {
                related.emplace_back(
                    distances[static_cast<std::size_t>(anchor) * node_count
                              + static_cast<std::size_t>(customer)],
                    customer);
            }
            std::stable_sort(
                related.begin(), related.end(),
                [&](const auto& left, const auto& right) {
                    if (left.first != right.first) {
                        return left.first < right.first;
                    }
                    return checked_data<std::int64_t>(lexical_rank_)[left.second]
                        < checked_data<std::int64_t>(lexical_rank_)[right.second];
                });
            for (std::int64_t index = 0; index < remove_count; ++index) {
                removed.push_back(related[static_cast<std::size_t>(index)].second);
            }
        }
        const auto repair_operation = static_cast<std::int64_t>(
            next_rng.weighted_index({
                full_operator_weights_[17], full_operator_weights_[18],
                full_operator_weights_[19]}));
        const auto accumulate_standard_outcome = [this, destroy, repair_operation](
            bool accepted,
            std::int64_t comparison,
            bool is_global_best,
            bool vehicle_reduction) noexcept {
            accumulate_full_stage04_outcome_noexcept(
                0, accepted, comparison, is_global_best, vehicle_reduction, true);
            accumulate_full_stage04_outcome_noexcept(
                static_cast<std::size_t>(14 + destroy), accepted, comparison,
                is_global_best, vehicle_reduction, true);
            accumulate_full_stage04_outcome_noexcept(
                static_cast<std::size_t>(17 + repair_operation), accepted,
                comparison, is_global_best, vehicle_reduction, true);
        };
        const std::unordered_set<std::int64_t> removed_set(
            removed.begin(), removed.end());
        const auto* boundaries = checked_data<std::int64_t>(legacy_offsets_);
        const auto* nodes = checked_data<std::int64_t>(legacy_indices_);
        std::vector<std::int64_t> partial_offsets{0};
        std::vector<std::int64_t> partial_indices;
        for (py::ssize_t route = 0; route + 1 < legacy_offsets_.size(); ++route) {
            const auto before = partial_indices.size();
            for (auto cursor = boundaries[route]; cursor < boundaries[route + 1]; ++cursor) {
                if (!removed_set.contains(nodes[cursor])) {
                    partial_indices.push_back(nodes[cursor]);
                }
            }
            if (partial_indices.size() != before) {
                partial_offsets.push_back(
                    static_cast<std::int64_t>(partial_indices.size()));
            }
        }
        py::array_t<std::int64_t> removed_array(removed.size());
        py::array_t<std::int64_t> partial_offsets_array(partial_offsets.size());
        py::array_t<std::int64_t> partial_indices_array(partial_indices.size());
        std::copy(removed.begin(), removed.end(), checked_data(removed_array));
        std::copy(
            partial_offsets.begin(), partial_offsets.end(),
            checked_data(partial_offsets_array));
        std::copy(
            partial_indices.begin(), partial_indices.end(),
            checked_data(partial_indices_array));
        auto repair = candidate_control_repair_v2(
            node_kind_, demand_, ready_time_, due_date_, service_time_,
            distance_, reachable_, vehicle_, lexical_rank_,
            partial_offsets_array, partial_indices_array, removed_array,
            screening_epsilon_, route_change_limit, true);
        auto repair_counters = py::cast<py::array_t<std::int64_t>>(repair[2]);
        py::array_t<std::int64_t> metadata(8);
        auto* values = checked_data(metadata);
        values[0] = iteration;
        values[1] = selected_main;
        values[2] = destroy;
        values[3] = repair_operation;
        values[4] = remove_count;
        values[5] = checked_data<std::int64_t>(repair_counters)[0];
        values[6] = -1;
        values[7] = 0;
        const auto record_total_exact_work = [&]() {
            values[7] = budget_.native_snapshot().started
                - entry_exact_started;
        };
        py::array_t<std::int64_t> insertion_context(3);
        checked_data(insertion_context)[0] = stable_int63("legacy");
        constexpr std::array<std::string_view, 3> repair_names{
            "greedy", "regret2", "energy"};
        checked_data(insertion_context)[1] = stable_int63(
            repair_names[static_cast<std::size_t>(repair_operation)]);
        checked_data(insertion_context)[2] = iteration;
        if (values[5] != 0) {
            accumulate_standard_outcome(false, 1, false, false);
            *rng_ = std::move(next_rng);
            return py::make_tuple(
                std::move(metadata), std::move(removed_array),
                std::move(partial_offsets_array),
                std::move(partial_indices_array), std::move(repair),
                py::none(), py::none(), lane_solution_state(0));
        }
        if (removed.size() != 1) {
            struct SequentialInsertionOption {
                double score;
                std::int64_t customer;
                std::int64_t route;
                std::int64_t position;
                std::vector<std::vector<std::int64_t>> routes;
            };
            const auto lexical_sequence_less = [this](
                const auto& left, const auto& right) {
                return std::lexicographical_compare(
                    left.begin(), left.end(), right.begin(), right.end(),
                    [this](std::int64_t lhs, std::int64_t rhs) {
                        return checked_data<std::int64_t>(lexical_rank_)[lhs]
                            < checked_data<std::int64_t>(lexical_rank_)[rhs];
                    });
            };
            const auto option_less = [&](
                const SequentialInsertionOption& left,
                const SequentialInsertionOption& right) {
                if (left.score != right.score) {
                    return left.score < right.score;
                }
                if (left.route != right.route) {
                    return left.route < right.route;
                }
                return lexical_sequence_less(
                    left.routes[static_cast<std::size_t>(left.route)],
                    right.routes[static_cast<std::size_t>(right.route)]);
            };
            std::vector<std::vector<std::int64_t>> sequential_routes;
            sequential_routes.reserve(partial_offsets.size() - 1);
            for (std::size_t route = 0; route + 1 < partial_offsets.size(); ++route) {
                sequential_routes.emplace_back(
                    partial_indices.begin() + partial_offsets[route],
                    partial_indices.begin() + partial_offsets[route + 1]);
            }
            std::vector<std::int64_t> pending(removed.begin(), removed.end());
            py::object last_transaction = py::none();
            std::int64_t exact_rows_seen = 0;
            const auto transaction_started = std::chrono::steady_clock::now();
            const auto make_routes_soa = [](const auto& routes) {
                std::vector<std::int64_t> offsets{0};
                std::vector<std::int64_t> indices;
                for (const auto& route : routes) {
                    indices.insert(indices.end(), route.begin(), route.end());
                    offsets.push_back(static_cast<std::int64_t>(indices.size()));
                }
                py::array_t<std::int64_t> offsets_array(offsets.size());
                py::array_t<std::int64_t> indices_array(indices.size());
                std::copy(offsets.begin(), offsets.end(), checked_data(offsets_array));
                std::copy(indices.begin(), indices.end(), checked_data(indices_array));
                return std::make_pair(
                    std::move(offsets_array), std::move(indices_array));
            };
            const auto make_expected = [this](const auto& routes) {
                std::vector<std::int64_t> expected;
                for (const auto& route : routes) {
                    expected.insert(expected.end(), route.begin(), route.end());
                }
                std::stable_sort(
                    expected.begin(), expected.end(),
                    [this](std::int64_t left, std::int64_t right) {
                        return checked_data<std::int64_t>(lexical_rank_)[left]
                            < checked_data<std::int64_t>(lexical_rank_)[right];
                    });
                py::array_t<std::int64_t> output(expected.size());
                std::copy(expected.begin(), expected.end(), checked_data(output));
                return output;
            };
            const auto remaining_deadline = [&]() {
                const auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - transaction_started).count();
                const auto remaining = checked_data<double>(deadline_array)[0] - elapsed;
                if (remaining <= 0.0) {
                    throw std::runtime_error(
                        "full native standard sequential repair exceeded its deadline");
                }
                py::array_t<double> output(1);
                checked_data(output)[0] = remaining;
                return output;
            };
            swap_active_with_lane_noexcept(0);
            ScopeRollback restore_sequential_lane([this]() noexcept {
                swap_active_with_lane_noexcept(0);
            });
            while (!pending.empty()) {
                std::vector<std::vector<SequentialInsertionOption>> options_by_customer(
                    pending.size());
                for (std::size_t pending_index = 0;
                     pending_index < pending.size(); ++pending_index) {
                    const auto [current_offsets, current_indices] =
                        make_routes_soa(sequential_routes);
                    auto pool = insertion_candidate_plans_v2_impl(
                        current_offsets, current_indices, pending[pending_index],
                        demand_, checked_data<double>(vehicle_)[1],
                        screening_epsilon_, false);
                    auto pool_plan_offsets =
                        py::cast<py::array_t<std::int64_t>>(pool[0]);
                    if (pool_plan_offsets.size() <= 1) {
                        continue;
                    }
                    auto pool_route_offsets =
                        py::cast<py::array_t<std::int64_t>>(pool[1]);
                    auto pool_route_indices =
                        py::cast<py::array_t<std::int64_t>>(pool[2]);
                    auto pool_metadata =
                        py::cast<py::array_t<std::int64_t>>(pool[3]);
                    auto expected_routes = sequential_routes;
                    expected_routes.push_back({pending[pending_index]});
                    auto expected = make_expected(expected_routes);
                    allow_partial_customer_coverage_ = true;
                    ScopeRollback restore_partial_coverage([this]() noexcept {
                        allow_partial_customer_coverage_ = false;
                    });
                    defer_composite_commit_ = true;
                    py::tuple transaction;
                    try {
                        transaction = evaluate_plans(
                            pool_plan_offsets, pool_route_offsets,
                            pool_route_indices, insertion_context,
                            remaining_deadline(), batch_array, expected);
                        auto feasible_order =
                            py::cast<py::array_t<std::int64_t>>(transaction[11]);
                        auto objectives =
                            py::cast<py::array_t<double>>(transaction[3]);
                        auto exact_metrics = py::cast<py::array_t<double>>(
                            pending_candidate_exact_payload_[4]);
                        const auto* plan_boundaries =
                            checked_data<std::int64_t>(pool_plan_offsets);
                        const auto* route_boundaries =
                            checked_data<std::int64_t>(pool_route_offsets);
                        const auto* route_nodes =
                            checked_data<std::int64_t>(pool_route_indices);
                        const auto* metadata_values =
                            checked_data<std::int64_t>(pool_metadata);
                        for (py::ssize_t order = 0;
                             order < feasible_order.size(); ++order) {
                            const auto plan =
                                checked_data<std::int64_t>(feasible_order)[order];
                            const auto first_route = plan_boundaries[plan];
                            const auto end_route = plan_boundaries[plan + 1];
                            const auto target = metadata_values[plan * 2];
                            const auto position = metadata_values[plan * 2 + 1];
                            std::vector<std::vector<std::int64_t>> candidate_routes;
                            candidate_routes.reserve(
                                static_cast<std::size_t>(end_route - first_route));
                            for (auto route = first_route; route < end_route; ++route) {
                                candidate_routes.emplace_back(
                                    route_nodes + route_boundaries[route],
                                    route_nodes + route_boundaries[route + 1]);
                            }
                            auto score = checked_data<double>(objectives)[plan * 2];
                            if (repair_operation == 2) {
                                score += 0.05 * checked_data<double>(exact_metrics)[
                                    (first_route + target) * 4 + 2];
                            }
                            options_by_customer[pending_index].push_back(
                                SequentialInsertionOption{
                                    score, pending[pending_index], target, position,
                                    std::move(candidate_routes)});
                        }
                        exact_rows_seen += py::cast<py::array_t<std::int64_t>>(
                            transaction[5]).size();
                        last_transaction = transaction;
                        defer_composite_commit_ = false;
                        commit_pending_composite_noexcept();
                        allow_partial_customer_coverage_ = false;
                        restore_partial_coverage.release();
                    } catch (...) {
                        defer_composite_commit_ = false;
                        if (pending_composite_active_) {
                            rollback_pending_composite();
                        }
                        throw;
                    }
                    std::stable_sort(
                        options_by_customer[pending_index].begin(),
                        options_by_customer[pending_index].end(), option_less);
                }
                std::vector<std::size_t> feasible_customers;
                for (std::size_t index = 0;
                     index < options_by_customer.size(); ++index) {
                    if (!options_by_customer[index].empty()) {
                        feasible_customers.push_back(index);
                    }
                }
                if (feasible_customers.empty()) {
                    record_total_exact_work();
                    accumulate_standard_outcome(false, 1, false, false);
                    *rng_ = std::move(next_rng);
                    restore_sequential_lane.rollback_now();
                    return py::make_tuple(
                        std::move(metadata), std::move(removed_array),
                        std::move(partial_offsets_array),
                        std::move(partial_indices_array), std::move(repair),
                        py::none(), std::move(last_transaction),
                        lane_solution_state(0));
                }
                std::size_t selected_customer = feasible_customers.front();
                if (repair_operation == 1) {
                    double selected_regret = -std::numeric_limits<double>::infinity();
                    double selected_draw = -1.0;
                    for (const auto index : feasible_customers) {
                        const auto& options = options_by_customer[index];
                        const auto regret = options.size() > 1
                            ? options[1].score - options[0].score
                            : 1.0e12;
                        const auto draw = next_rng.random();
                        if (regret > selected_regret
                            || (regret == selected_regret && draw > selected_draw)) {
                            selected_customer = index;
                            selected_regret = regret;
                            selected_draw = draw;
                        }
                    }
                } else {
                    for (const auto index : feasible_customers) {
                        const auto left_score =
                            options_by_customer[index].front().score;
                        const auto right_score =
                            options_by_customer[selected_customer].front().score;
                        if (left_score < right_score
                            || (left_score == right_score
                                && checked_data<std::int64_t>(lexical_rank_)[
                                       pending[index]]
                                    < checked_data<std::int64_t>(lexical_rank_)[
                                          pending[selected_customer]])) {
                            selected_customer = index;
                        }
                    }
                }
                auto selected = std::move(
                    options_by_customer[selected_customer].front());
                const auto [selected_offsets, selected_indices] =
                    make_routes_soa(selected.routes);
                py::array_t<std::int64_t> selected_plan_offsets(2);
                checked_data(selected_plan_offsets)[0] = 0;
                checked_data(selected_plan_offsets)[1] =
                    selected_offsets.size() - 1;
                auto selected_expected = make_expected(selected.routes);
                suppress_attempted_plan_journal_ = true;
                ScopeRollback restore_attempted([this]() noexcept {
                    suppress_attempted_plan_journal_ = false;
                });
                allow_partial_customer_coverage_ = true;
                ScopeRollback restore_selected_coverage([this]() noexcept {
                    allow_partial_customer_coverage_ = false;
                });
                defer_composite_commit_ = true;
                try {
                    auto selected_transaction = evaluate_plans(
                        selected_plan_offsets, selected_offsets, selected_indices,
                        insertion_context, remaining_deadline(), batch_array,
                        selected_expected);
                    const auto selected_plan = prepare_first_feasible_candidate(
                        selected_plan_offsets, selected_offsets, selected_indices,
                        selected_transaction);
                    if (!selected_plan.has_value()) {
                        throw std::logic_error(
                            "full native standard sequential selection lost feasibility");
                    }
                    defer_composite_commit_ = false;
                    commit_pending_composite_noexcept();
                    suppress_attempted_plan_journal_ = false;
                    restore_attempted.release();
                    allow_partial_customer_coverage_ = false;
                    restore_selected_coverage.release();
                    last_transaction = std::move(selected_transaction);
                } catch (...) {
                    defer_composite_commit_ = false;
                    if (pending_composite_active_) {
                        rollback_pending_composite();
                    }
                    throw;
                }
                sequential_routes = std::move(selected.routes);
                pending.erase(pending.begin()
                    + static_cast<std::ptrdiff_t>(selected_customer));
                if (!pending.empty()) {
                    last_candidate_ready_ = false;
                    last_candidate_offsets_ = py::array_t<std::int64_t>();
                    last_candidate_indices_ = py::array_t<std::int64_t>();
                    last_candidate_exact_payload_ = py::tuple();
                    last_candidate_objective_integer_ =
                        py::array_t<std::int64_t>();
                    last_candidate_objective_float_ = py::array_t<double>();
                }
            }
            values[6] = 0;
            record_total_exact_work();
            legacy_candidate_offsets_ = std::move(last_candidate_offsets_);
            legacy_candidate_indices_ = std::move(last_candidate_indices_);
            legacy_candidate_exact_payload_ =
                std::move(last_candidate_exact_payload_);
            legacy_candidate_objective_integer_ =
                std::move(last_candidate_objective_integer_);
            legacy_candidate_objective_float_ =
                std::move(last_candidate_objective_float_);
            last_candidate_ready_ = false;
            legacy_candidate_ready_ = true;
            legacy_candidate_operator_ = 0;
            legacy_candidate_destroy_operator_ = destroy;
            legacy_candidate_repair_operator_ = repair_operation;
            *rng_ = std::move(next_rng);
            restore_sequential_lane.rollback_now();
            auto repaired_soa = make_routes_soa(sequential_routes);
            auto selected_repair = py::make_tuple(
                std::move(repaired_soa.first), std::move(repaired_soa.second),
                repair_counters);
            return py::make_tuple(
                std::move(metadata), std::move(removed_array),
                std::move(partial_offsets_array),
                std::move(partial_indices_array), std::move(selected_repair),
                py::none(), std::move(last_transaction), lane_solution_state(0));
        }
        const auto customer = removed.front();
        auto insertion_pool = insertion_candidate_plans_v2_impl(
            partial_offsets_array, partial_indices_array, customer, demand_,
            checked_data<double>(vehicle_)[1], screening_epsilon_,
            partial_offsets.size() == 1);
        auto insertion_plan_offsets_array =
            py::cast<py::array_t<std::int64_t>>(insertion_pool[0]);
        auto insertion_route_offsets_array =
            py::cast<py::array_t<std::int64_t>>(insertion_pool[1]);
        auto insertion_route_indices_array =
            py::cast<py::array_t<std::int64_t>>(insertion_pool[2]);
        std::vector<std::int64_t> insertion_expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            insertion_expected.begin(), insertion_expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> insertion_expected_array(
            insertion_expected.size());
        std::copy(
            insertion_expected.begin(), insertion_expected.end(),
            checked_data(insertion_expected_array));
        swap_active_with_lane_noexcept(0);
        ScopeRollback restore_insertion_lane([this]() noexcept {
            swap_active_with_lane_noexcept(0);
        });
        defer_composite_commit_ = true;
        py::tuple insertion_transaction;
        std::optional<std::int64_t> selected_insertion;
        try {
            insertion_transaction = evaluate_plans(
                insertion_plan_offsets_array, insertion_route_offsets_array,
                insertion_route_indices_array, insertion_context,
                deadline_array, batch_array, insertion_expected_array);
            defer_composite_commit_ = false;
            selected_insertion = prepare_first_feasible_candidate(
                insertion_plan_offsets_array,
                insertion_route_offsets_array,
                insertion_route_indices_array,
                insertion_transaction);
            commit_pending_composite_noexcept();
            restore_insertion_lane.rollback_now();
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
        if ((repair_operation == 1 || repair_operation == 2)
            && selected_insertion.has_value()
            && partial_offsets.size() > 1) {
            // Python resolves Candidate Control base routes only after at
            // least one complete insertion plan is feasible.  Preserve the
            // prepared candidate while the partial-base cache transaction is
            // evaluated; an empty feasible set must consume no base exact
            // work.
            auto selected_offsets = std::move(last_candidate_offsets_);
            auto selected_indices = std::move(last_candidate_indices_);
            auto selected_exact = std::move(last_candidate_exact_payload_);
            auto selected_objective_integer =
                std::move(last_candidate_objective_integer_);
            auto selected_objective_float =
                std::move(last_candidate_objective_float_);
            last_candidate_ready_ = false;
            const auto restore_selected = [&]() {
                last_candidate_offsets_ = std::move(selected_offsets);
                last_candidate_indices_ = std::move(selected_indices);
                last_candidate_exact_payload_ = std::move(selected_exact);
                last_candidate_objective_integer_ =
                    std::move(selected_objective_integer);
                last_candidate_objective_float_ =
                    std::move(selected_objective_float);
                last_candidate_ready_ = true;
            };
            try {
                py::array_t<std::int64_t> base_plan_offsets(2);
                checked_data(base_plan_offsets)[0] = 0;
                checked_data(base_plan_offsets)[1] = partial_offsets.size() - 1;
                std::vector<std::int64_t> base_expected(partial_indices);
                std::stable_sort(
                    base_expected.begin(), base_expected.end(),
                    [&](std::int64_t left, std::int64_t right) {
                        return checked_data<std::int64_t>(lexical_rank_)[left]
                            < checked_data<std::int64_t>(lexical_rank_)[right];
                    });
                py::array_t<std::int64_t> base_expected_array(
                    base_expected.size());
                std::copy(
                    base_expected.begin(), base_expected.end(),
                    checked_data(base_expected_array));
                swap_active_with_lane_noexcept(0);
                ScopeRollback restore_base_lane([this]() noexcept {
                    swap_active_with_lane_noexcept(0);
                });
                suppress_attempted_plan_journal_ = true;
                ScopeRollback restore_base_attempted([this]() noexcept {
                    suppress_attempted_plan_journal_ = false;
                });
                allow_partial_customer_coverage_ = true;
                ScopeRollback restore_base_coverage([this]() noexcept {
                    allow_partial_customer_coverage_ = false;
                });
                defer_composite_commit_ = true;
                static_cast<void>(evaluate_plans(
                    base_plan_offsets, partial_offsets_array,
                    partial_indices_array, insertion_context,
                    deadline_array, batch_array, base_expected_array));
                defer_composite_commit_ = false;
                commit_pending_composite_noexcept();
                allow_partial_customer_coverage_ = false;
                restore_base_coverage.release();
                suppress_attempted_plan_journal_ = false;
                restore_base_attempted.release();
                restore_base_lane.rollback_now();
            } catch (...) {
                defer_composite_commit_ = false;
                if (pending_composite_active_) {
                    rollback_pending_composite();
                }
                restore_selected();
                throw;
            }
            restore_selected();
        }
        if (repair_operation == 2) {
            record_total_exact_work();
            if (!selected_insertion.has_value()) {
                accumulate_standard_outcome(false, 1, false, false);
                *rng_ = std::move(next_rng);
                return py::make_tuple(
                    std::move(metadata), std::move(removed_array),
                    std::move(partial_offsets_array),
                    std::move(partial_indices_array), std::move(repair),
                    py::none(), std::move(insertion_transaction),
                    lane_solution_state(0));
            }
            values[6] = *selected_insertion;
            auto selected_offsets = owned_array_copy<std::int64_t>(
                last_candidate_offsets_, "selected_energy_offsets", 1);
            auto selected_indices = owned_array_copy<std::int64_t>(
                last_candidate_indices_, "selected_energy_indices", 1);
            auto selected_repair = py::make_tuple(
                std::move(selected_offsets), std::move(selected_indices),
                repair_counters);
            legacy_candidate_offsets_ = std::move(last_candidate_offsets_);
            legacy_candidate_indices_ = std::move(last_candidate_indices_);
            legacy_candidate_exact_payload_ =
                std::move(last_candidate_exact_payload_);
            legacy_candidate_objective_integer_ =
                std::move(last_candidate_objective_integer_);
            legacy_candidate_objective_float_ =
                std::move(last_candidate_objective_float_);
            last_candidate_ready_ = false;
            legacy_candidate_ready_ = true;
            legacy_candidate_operator_ = 0;
            legacy_candidate_destroy_operator_ = destroy;
            legacy_candidate_repair_operator_ = repair_operation;
            *rng_ = std::move(next_rng);
            return py::make_tuple(
                std::move(metadata), std::move(removed_array),
                std::move(partial_offsets_array),
                std::move(partial_indices_array),
                std::move(selected_repair), py::none(),
                std::move(insertion_transaction), lane_solution_state(0));
        }
        const auto feasible_insertion_count = py::cast<py::array_t<std::int64_t>>(
            insertion_transaction[11]).size();
        if (repair_operation == 1 && feasible_insertion_count != 0) {
            // Python evaluates insertion feasibility before regret tie-breaking.
            // A failed regret repair therefore consumes no random draw.
            for (std::int64_t index = 0; index < remove_count; ++index) {
                static_cast<void>(next_rng.random());
            }
        }
        record_total_exact_work();
        if (!selected_insertion.has_value()) {
            accumulate_standard_outcome(false, 1, false, false);
            *rng_ = std::move(next_rng);
            return py::make_tuple(
                std::move(metadata), std::move(removed_array),
                std::move(partial_offsets_array),
                std::move(partial_indices_array), std::move(repair),
                py::none(), std::move(insertion_transaction),
                lane_solution_state(0));
        }
        values[6] = *selected_insertion;
        auto selected_offsets = owned_array_copy<std::int64_t>(
            last_candidate_offsets_, "selected_standard_offsets", 1);
        auto selected_indices = owned_array_copy<std::int64_t>(
            last_candidate_indices_, "selected_standard_indices", 1);
        auto selected_repair = py::make_tuple(
            std::move(selected_offsets), std::move(selected_indices),
            repair_counters);
        legacy_candidate_offsets_ = std::move(last_candidate_offsets_);
        legacy_candidate_indices_ = std::move(last_candidate_indices_);
        legacy_candidate_exact_payload_ =
            std::move(last_candidate_exact_payload_);
        legacy_candidate_objective_integer_ =
            std::move(last_candidate_objective_integer_);
        legacy_candidate_objective_float_ =
            std::move(last_candidate_objective_float_);
        last_candidate_ready_ = false;
        legacy_candidate_ready_ = true;
        legacy_candidate_operator_ = 0;
        legacy_candidate_destroy_operator_ = destroy;
        legacy_candidate_repair_operator_ = repair_operation;
        *rng_ = std::move(next_rng);
        return py::make_tuple(
            std::move(metadata), std::move(removed_array),
            std::move(partial_offsets_array),
            std::move(partial_indices_array), std::move(selected_repair),
            py::none(), std::move(insertion_transaction), lane_solution_state(0));
    }

    py::tuple quality_route_segment_probe(
        std::int64_t iteration,
        std::int64_t min_length,
        std::int64_t max_length,
        std::int64_t evaluation_budget,
        std::int64_t route_change_limit,
        py::handle deadline_remaining,
        py::handle batch_size) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || iteration < 0 || min_length <= 0
            || max_length < min_length || evaluation_budget <= 0
            || route_change_limit == 0 || route_change_limit < -1
            || last_candidate_ready_ || pending_composite_active_) {
            throw std::invalid_argument(
                "full native quality route-segment state/config is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "quality_segment_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "quality_segment_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native quality route-segment deadline/batch is invalid");
        }
        const auto* boundaries = checked_data<std::int64_t>(quality_offsets_);
        const auto* nodes = checked_data<std::int64_t>(quality_indices_);
        std::vector<std::array<std::int64_t, 8>> attempt_rows;
        std::vector<std::int64_t> removed_offsets{0};
        std::vector<std::int64_t> removed_indices;
        py::tuple selected_repair;
        py::array_t<std::int64_t> selected_offsets;
        py::array_t<std::int64_t> selected_indices;
        const auto candidate_limit = std::max<std::int64_t>(
            16, evaluation_budget * 4);
        std::int64_t considered = 0;
        bool found = false;
        for (py::ssize_t source = 0;
             source + 1 < quality_offsets_.size() && !found; ++source) {
            const auto source_size = boundaries[source + 1] - boundaries[source];
            if (source_size <= min_length) {
                continue;
            }
            const auto last_length = std::min(max_length, source_size - 1);
            for (auto length = min_length; length <= last_length && !found; ++length) {
                for (std::int64_t start = 0;
                     start + length <= source_size && !found; ++start) {
                    if (considered >= candidate_limit) {
                        break;
                    }
                    ++considered;
                    std::vector<std::int64_t> partial_offsets{0};
                    std::vector<std::int64_t> partial_indices;
                    std::vector<std::int64_t> removed;
                    for (py::ssize_t route = 0;
                         route + 1 < quality_offsets_.size(); ++route) {
                        for (auto cursor = boundaries[route];
                             cursor < boundaries[route + 1]; ++cursor) {
                            const auto relative = cursor - boundaries[route];
                            if (route == source && relative >= start
                                && relative < start + length) {
                                removed.push_back(nodes[cursor]);
                            } else {
                                partial_indices.push_back(nodes[cursor]);
                            }
                        }
                        partial_offsets.push_back(
                            static_cast<std::int64_t>(partial_indices.size()));
                    }
                    py::array_t<std::int64_t> partial_offsets_array(
                        partial_offsets.size());
                    py::array_t<std::int64_t> partial_indices_array(
                        partial_indices.size());
                    py::array_t<std::int64_t> removed_array(removed.size());
                    std::copy(
                        partial_offsets.begin(), partial_offsets.end(),
                        checked_data(partial_offsets_array));
                    std::copy(
                        partial_indices.begin(), partial_indices.end(),
                        checked_data(partial_indices_array));
                    std::copy(
                        removed.begin(), removed.end(), checked_data(removed_array));
                    auto repair = candidate_control_repair_v2(
                        node_kind_, demand_, ready_time_, due_date_, service_time_,
                        distance_, reachable_, vehicle_, lexical_rank_,
                        partial_offsets_array, partial_indices_array, removed_array,
                        screening_epsilon_, route_change_limit, false);
                    auto repaired_offsets =
                        py::cast<py::array_t<std::int64_t>>(repair[0]);
                    auto repaired_indices =
                        py::cast<py::array_t<std::int64_t>>(repair[1]);
                    auto counters =
                        py::cast<py::array_t<std::int64_t>>(repair[2]);
                    const auto failure = checked_data<std::int64_t>(counters)[0];
                    bool changed = false;
                    if (failure == 0
                        && repaired_offsets.size() == quality_offsets_.size()
                        && repaired_indices.size() == quality_indices_.size()) {
                        changed = !std::equal(
                            checked_data<std::int64_t>(repaired_offsets),
                            checked_data<std::int64_t>(repaired_offsets)
                                + repaired_offsets.size(),
                            checked_data<std::int64_t>(quality_offsets_))
                            || !std::equal(
                                checked_data<std::int64_t>(repaired_indices),
                                checked_data<std::int64_t>(repaired_indices)
                                    + repaired_indices.size(),
                                checked_data<std::int64_t>(quality_indices_));
                    }
                    attempt_rows.push_back({
                        static_cast<std::int64_t>(source), start, length,
                        failure, checked_data<std::int64_t>(counters)[1],
                        considered, changed ? 1 : 0, 0});
                    removed_indices.insert(
                        removed_indices.end(), removed.begin(), removed.end());
                    removed_offsets.push_back(
                        static_cast<std::int64_t>(removed_indices.size()));
                    if (changed) {
                        selected_repair = std::move(repair);
                        selected_offsets = std::move(repaired_offsets);
                        selected_indices = std::move(repaired_indices);
                        found = true;
                    }
                }
            }
        }
        py::array_t<std::int64_t> attempts({
            static_cast<py::ssize_t>(attempt_rows.size()), py::ssize_t(8)});
        for (std::size_t row = 0; row < attempt_rows.size(); ++row) {
            std::copy(
                attempt_rows[row].begin(), attempt_rows[row].end(),
                checked_data(attempts) + row * 8);
        }
        py::array_t<std::int64_t> removed_offsets_array(
            removed_offsets.size());
        py::array_t<std::int64_t> removed_indices_array(
            removed_indices.size());
        std::copy(
            removed_offsets.begin(), removed_offsets.end(),
            checked_data(removed_offsets_array));
        std::copy(
            removed_indices.begin(), removed_indices.end(),
            checked_data(removed_indices_array));
        py::array_t<std::int64_t> outcome(4);
        std::fill(checked_data(outcome), checked_data(outcome) + 4, 0);
        checked_data(outcome)[0] = -1;
        if (!found) {
            accumulate_full_stage04_outcome_noexcept(
                7, false, 1, false, false, true);
            return py::make_tuple(
                std::move(attempts), std::move(removed_offsets_array),
                std::move(removed_indices_array), py::none(), py::none(),
                std::move(outcome), lane_solution_state(1));
        }
        py::array_t<std::int64_t> plan_offsets(2);
        checked_data(plan_offsets)[0] = 0;
        checked_data(plan_offsets)[1] = selected_offsets.size() - 1;
        py::array_t<std::int64_t> context(3);
        checked_data(context)[0] = stable_int63("quality_shadow");
        checked_data(context)[1] = stable_int63("route_segment_destroy");
        checked_data(context)[2] = iteration;
        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        swap_active_with_lane_noexcept(1);
        ScopeRollback restore_lane([this]() noexcept {
            swap_active_with_lane_noexcept(1);
        });
        suppress_attempted_plan_journal_ = true;
        ScopeRollback restore_attempted([this]() noexcept {
            suppress_attempted_plan_journal_ = false;
        });
        defer_composite_commit_ = true;
        try {
            auto transaction = evaluate_plans(
                plan_offsets, selected_offsets, selected_indices, context,
                deadline_array, batch_array, expected_array);
            defer_composite_commit_ = false;
            const auto selected = prepare_first_feasible_candidate(
                plan_offsets, selected_offsets, selected_indices, transaction);
            if (selected.has_value()) {
                commit_pending_composite_noexcept();
                checked_data(outcome)[0] = *selected;
                const auto comparison = last_candidate_comparison();
                auto acceptance = apply_last_candidate(1.0, 1.0);
                checked_data(outcome)[1] = py::cast<std::int64_t>(acceptance[0]);
                checked_data(outcome)[2] = py::cast<std::int64_t>(acceptance[1]);
                checked_data(outcome)[3] = py::cast<std::int64_t>(acceptance[2]);
                accumulate_full_stage04_outcome_noexcept(
                    7, checked_data(outcome)[1] != 0, comparison,
                    checked_data(outcome)[2] != 0,
                    checked_data(outcome)[3] != 0, true);
            } else {
                commit_pending_composite_noexcept();
                accumulate_full_stage04_outcome_noexcept(
                    7, false, 1, false, false, true);
            }
            suppress_attempted_plan_journal_ = false;
            restore_attempted.release();
            restore_lane.rollback_now();
            return py::make_tuple(
                std::move(attempts), std::move(removed_offsets_array),
                std::move(removed_indices_array), std::move(selected_repair),
                std::move(transaction), std::move(outcome),
                lane_solution_state(1));
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
    }

    py::tuple quality_ejection_chain_probe(
        std::int64_t iteration,
        std::int64_t evaluation_budget,
        std::int64_t max_depth,
        std::int64_t beam_width,
        py::handle deadline_remaining,
        py::handle batch_size) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || iteration < 0 || evaluation_budget <= 0
            || max_depth <= 0 || beam_width <= 0 || last_candidate_ready_
            || pending_composite_active_) {
            throw std::invalid_argument(
                "full native quality ejection-chain state/config is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "quality_ejection_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "quality_ejection_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native quality ejection-chain deadline/batch is invalid");
        }
        const auto route_count = static_cast<std::size_t>(
            quality_offsets_.size() - 1);
        if (route_count <= 1) {
            py::array_t<std::int64_t> outcome(4);
            std::fill(
                checked_data(outcome), checked_data(outcome) + 4,
                std::int64_t{0});
            checked_data(outcome)[0] = -1;
            accumulate_full_stage04_outcome_noexcept(
                8, false, 1, false, false, true);
            return py::make_tuple(
                py::none(), py::none(), std::move(outcome),
                lane_solution_state(1));
        }
        const auto* boundaries = checked_data<std::int64_t>(quality_offsets_);
        const auto* nodes = checked_data<std::int64_t>(quality_indices_);
        std::vector<std::vector<std::int64_t>> base;
        base.reserve(route_count);
        for (std::size_t route = 0; route < route_count; ++route) {
            base.emplace_back(
                nodes + boundaries[route], nodes + boundaries[route + 1]);
        }
        struct State {
            std::vector<std::vector<std::int64_t>> routes;
            std::int64_t pending;
            std::int64_t depth;
        };
        std::vector<State> beam;
        for (std::size_t source = 0; source < base.size(); ++source) {
            if (base[source].size() <= 1) {
                continue;
            }
            for (std::size_t position = 0; position < base[source].size(); ++position) {
                auto routes = base;
                const auto pending = routes[source][position];
                routes[source].erase(
                    routes[source].begin()
                    + static_cast<std::ptrdiff_t>(position));
                beam.push_back(State{std::move(routes), pending, 0});
            }
        }
        const auto node_count = static_cast<std::size_t>(node_kind_.size());
        const auto* distances = checked_data<double>(distance_);
        const auto route_distance = [&](const auto& routes) {
            PythonFloatSum total;
            for (const auto& route : routes) {
                auto previous = depot_;
                for (const auto customer : route) {
                    total.add(distances[
                        static_cast<std::size_t>(previous) * node_count
                        + static_cast<std::size_t>(customer)]);
                    previous = customer;
                }
                total.add(distances[
                    static_cast<std::size_t>(previous) * node_count
                    + static_cast<std::size_t>(depot_)]);
            }
            return total.value();
        };
        const auto route_less = [&](const auto& left, const auto& right) {
            return std::lexicographical_compare(
                left.begin(), left.end(), right.begin(), right.end(),
                [&](std::int64_t lhs, std::int64_t rhs) {
                    return checked_data<std::int64_t>(lexical_rank_)[lhs]
                        < checked_data<std::int64_t>(lexical_rank_)[rhs];
                });
        };
        const auto plan_less = [&](const auto& left, const auto& right) {
            return std::lexicographical_compare(
                left.begin(), left.end(), right.begin(), right.end(), route_less);
        };
        std::stable_sort(
            beam.begin(), beam.end(), [&](const State& left, const State& right) {
                const auto left_distance = route_distance(left.routes);
                const auto right_distance = route_distance(right.routes);
                if (left_distance != right_distance) {
                    return left_distance < right_distance;
                }
                if (left.routes != right.routes) {
                    return plan_less(left.routes, right.routes);
                }
                return checked_data<std::int64_t>(lexical_rank_)[left.pending]
                    < checked_data<std::int64_t>(lexical_rank_)[right.pending];
            });
        if (beam.size() > static_cast<std::size_t>(beam_width)) {
            beam.resize(static_cast<std::size_t>(beam_width));
        }
        const auto candidate_limit = std::max<std::int64_t>(
            16, evaluation_budget * 4);
        std::vector<std::vector<std::vector<std::int64_t>>> candidates;
        std::vector<std::vector<std::int64_t>> changed_routes;
        std::vector<std::int64_t> candidate_groups;
        std::int64_t candidate_group = 0;
        for (const auto& state : beam) {
            const auto chain_depth = state.depth + 1;
            for (std::size_t target = 0; target < state.routes.size(); ++target) {
                for (std::size_t position = 0;
                     position <= state.routes[target].size(); ++position) {
                    if (static_cast<std::int64_t>(candidates.size())
                        >= candidate_limit) {
                        break;
                    }
                    auto candidate = state.routes;
                    candidate[target].insert(
                        candidate[target].begin()
                            + static_cast<std::ptrdiff_t>(position),
                        state.pending);
                    if (candidate == base) {
                        continue;
                    }
                    std::vector<std::int64_t> changes;
                    for (std::size_t route = 0; route < base.size(); ++route) {
                        if (candidate[route] != base[route]) {
                            changes.push_back(static_cast<std::int64_t>(route));
                        }
                    }
                    if (changes.empty()) {
                        continue;
                    }
                    candidates.push_back(std::move(candidate));
                    changes.push_back(chain_depth);
                    changed_routes.push_back(std::move(changes));
                    candidate_groups.push_back(candidate_group);
                }
                ++candidate_group;
            }
        }
        if (candidates.empty()) {
            py::array_t<std::int64_t> outcome(4);
            std::fill(
                checked_data(outcome), checked_data(outcome) + 4,
                std::int64_t{0});
            checked_data(outcome)[0] = -1;
            accumulate_full_stage04_outcome_noexcept(
                8, false, 1, false, false, true);
            return py::make_tuple(
                py::none(), py::none(), std::move(outcome),
                lane_solution_state(1));
        }
        py::array_t<std::int64_t> context(3);
        checked_data(context)[0] = stable_int63("quality_shadow");
        checked_data(context)[1] = stable_int63("ejection_chain");
        checked_data(context)[2] = iteration;
        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        const auto pack_single_plan = [](const auto& plan) {
            py::array_t<std::int64_t> plan_offsets(2);
            py::array_t<std::int64_t> route_offsets(plan.size() + 1);
            std::size_t index_count = 0;
            for (const auto& route : plan) {
                index_count += route.size();
            }
            py::array_t<std::int64_t> route_indices(index_count);
            checked_data(plan_offsets)[0] = 0;
            checked_data(plan_offsets)[1] =
                static_cast<std::int64_t>(plan.size());
            checked_data(route_offsets)[0] = 0;
            std::size_t cursor = 0;
            for (std::size_t route = 0; route < plan.size(); ++route) {
                std::copy(
                    plan[route].begin(), plan[route].end(),
                    checked_data(route_indices) + cursor);
                cursor += plan[route].size();
                checked_data(route_offsets)[route + 1] =
                    static_cast<std::int64_t>(cursor);
            }
            return py::make_tuple(
                std::move(plan_offsets), std::move(route_offsets),
                std::move(route_indices));
        };
        swap_active_with_lane_noexcept(1);
        ScopeRollback restore_lane([this]() noexcept {
            swap_active_with_lane_noexcept(1);
        });
        suppress_attempted_plan_journal_ = true;
        ScopeRollback restore_attempted([this]() noexcept {
            suppress_attempted_plan_journal_ = false;
        });
        std::vector<std::int64_t> statuses;
        std::vector<std::array<std::int64_t, 2>> objective_integers;
        std::vector<std::array<double, 2>> objective_floats;
        std::vector<std::int64_t> exact_deltas;
        std::optional<std::size_t> best_candidate;
        const auto exact_at_entry = budget_.native_snapshot().started;
        try {
            for (std::size_t candidate = 0; candidate < candidates.size(); ++candidate) {
                const auto exact_used =
                    budget_.native_snapshot().started - exact_at_entry;
                if (candidate > 0 && exact_used >= evaluation_budget
                    && candidate_groups[candidate]
                        != candidate_groups[candidate - 1]) {
                    break;
                }
                auto packed = pack_single_plan(candidates[candidate]);
                const auto exact_before = budget_.native_snapshot().started;
                defer_composite_commit_ = true;
                auto transaction = evaluate_plans(
                    packed[0], packed[1], packed[2], context,
                    deadline_array, batch_array, expected_array);
                defer_composite_commit_ = false;
                commit_pending_composite_noexcept();
                auto transaction_statuses =
                    py::cast<py::array_t<std::int64_t>>(transaction[1]);
                auto transaction_integers =
                    py::cast<py::array_t<std::int64_t>>(transaction[2]);
                auto transaction_floats =
                    py::cast<py::array_t<double>>(transaction[3]);
                const auto status = checked_data<std::int64_t>(
                    transaction_statuses)[0];
                statuses.push_back(status);
                objective_integers.push_back({
                    checked_data<std::int64_t>(transaction_integers)[0],
                    checked_data<std::int64_t>(transaction_integers)[1]});
                objective_floats.push_back({
                    checked_data<double>(transaction_floats)[0],
                    checked_data<double>(transaction_floats)[1]});
                exact_deltas.push_back(
                    budget_.native_snapshot().started - exact_before);
                if (status != 5) {
                    continue;
                }
                const auto key = std::make_tuple(
                    objective_integers.back()[0], objective_floats.back()[0],
                    objective_floats.back()[1], objective_integers.back()[1]);
                if (!best_candidate.has_value()) {
                    best_candidate = candidate;
                    continue;
                }
                const auto best = *best_candidate;
                const auto best_key = std::make_tuple(
                    objective_integers[best][0], objective_floats[best][0],
                    objective_floats[best][1], objective_integers[best][1]);
                if (key < best_key
                    || (key == best_key
                        && plan_less(candidates[candidate], candidates[best]))) {
                    best_candidate = candidate;
                }
            }
            candidates.resize(statuses.size());
            changed_routes.resize(statuses.size());
            candidate_groups.resize(statuses.size());
            py::array_t<std::int64_t> outcome(4);
            std::fill(
                checked_data(outcome), checked_data(outcome) + 4,
                std::int64_t{0});
            checked_data(outcome)[0] = -1;
            if (best_candidate.has_value()) {
                const auto selected_candidate = *best_candidate;
                auto selected_plan = pack_single_plan(
                    candidates[selected_candidate]);
                defer_composite_commit_ = true;
                auto selected_transaction = evaluate_plans(
                    selected_plan[0], selected_plan[1], selected_plan[2],
                    context, deadline_array, batch_array, expected_array);
                defer_composite_commit_ = false;
                auto selected_plan_offsets =
                    py::cast<py::array_t<std::int64_t>>(selected_plan[0]);
                auto selected_route_offsets =
                    py::cast<py::array_t<std::int64_t>>(selected_plan[1]);
                auto selected_route_indices =
                    py::cast<py::array_t<std::int64_t>>(selected_plan[2]);
                const auto selected = prepare_first_feasible_candidate(
                    selected_plan_offsets, selected_route_offsets,
                    selected_route_indices,
                    selected_transaction);
                if (!selected.has_value() || *selected != 0) {
                    throw std::logic_error(
                        "full native ejection-chain best candidate replay failed");
                }
                commit_pending_composite_noexcept();
                checked_data(outcome)[0] =
                    static_cast<std::int64_t>(selected_candidate);
                const auto comparison = last_candidate_comparison();
                auto acceptance = apply_last_candidate(1.0, 1.0);
                checked_data(outcome)[1] = py::cast<std::int64_t>(acceptance[0]);
                checked_data(outcome)[2] = py::cast<std::int64_t>(acceptance[1]);
                checked_data(outcome)[3] = py::cast<std::int64_t>(acceptance[2]);
                accumulate_full_stage04_outcome_noexcept(
                    8, checked_data(outcome)[1] != 0, comparison,
                    checked_data(outcome)[2] != 0,
                    checked_data(outcome)[3] != 0, true);
            } else {
                accumulate_full_stage04_outcome_noexcept(
                    8, false, 1, false, false, true);
            }
            std::vector<std::int64_t> plan_offsets{0};
            std::vector<std::int64_t> route_offsets{0};
            std::vector<std::int64_t> route_indices;
            for (const auto& plan : candidates) {
                for (const auto& route : plan) {
                    route_indices.insert(
                        route_indices.end(), route.begin(), route.end());
                    route_offsets.push_back(
                        static_cast<std::int64_t>(route_indices.size()));
                }
                plan_offsets.push_back(
                    static_cast<std::int64_t>(route_offsets.size() - 1));
            }
            std::vector<std::int64_t> change_offsets{0};
            std::vector<std::int64_t> change_indices;
            for (const auto& changes : changed_routes) {
                change_indices.insert(
                    change_indices.end(), changes.begin(), changes.end());
                change_offsets.push_back(
                    static_cast<std::int64_t>(change_indices.size()));
            }
            py::array_t<std::int64_t> plan_offsets_array(plan_offsets.size());
            py::array_t<std::int64_t> route_offsets_array(route_offsets.size());
            py::array_t<std::int64_t> route_indices_array(route_indices.size());
            py::array_t<std::int64_t> change_offsets_array(change_offsets.size());
            py::array_t<std::int64_t> change_indices_array(change_indices.size());
            py::array_t<std::int64_t> status_array(statuses.size());
            py::array_t<std::int64_t> objective_integer_array({
                static_cast<py::ssize_t>(statuses.size()), py::ssize_t(2)});
            py::array_t<double> objective_float_array({
                static_cast<py::ssize_t>(statuses.size()), py::ssize_t(2)});
            py::array_t<std::int64_t> exact_delta_array(exact_deltas.size());
            std::copy(
                plan_offsets.begin(), plan_offsets.end(),
                checked_data(plan_offsets_array));
            std::copy(
                route_offsets.begin(), route_offsets.end(),
                checked_data(route_offsets_array));
            std::copy(
                route_indices.begin(), route_indices.end(),
                checked_data(route_indices_array));
            std::copy(
                change_offsets.begin(), change_offsets.end(),
                checked_data(change_offsets_array));
            std::copy(
                change_indices.begin(), change_indices.end(),
                checked_data(change_indices_array));
            std::copy(statuses.begin(), statuses.end(), checked_data(status_array));
            std::copy(
                exact_deltas.begin(), exact_deltas.end(),
                checked_data(exact_delta_array));
            for (std::size_t row = 0; row < statuses.size(); ++row) {
                std::copy(
                    objective_integers[row].begin(),
                    objective_integers[row].end(),
                    checked_data(objective_integer_array) + row * 2);
                std::copy(
                    objective_floats[row].begin(), objective_floats[row].end(),
                    checked_data(objective_float_array) + row * 2);
            }
            auto pool = py::make_tuple(
                std::move(plan_offsets_array), std::move(route_offsets_array),
                std::move(route_indices_array), std::move(change_offsets_array),
                std::move(change_indices_array));
            auto journal = py::make_tuple(
                std::move(status_array), std::move(objective_integer_array),
                std::move(objective_float_array), std::move(exact_delta_array));
            suppress_attempted_plan_journal_ = false;
            restore_attempted.release();
            restore_lane.rollback_now();
            return py::make_tuple(
                std::move(pool), std::move(journal), std::move(outcome),
                lane_solution_state(1));
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
    }

    py::tuple run_three_lane_bootstrap(
        std::int64_t max_route_elimination_attempts,
        std::int64_t refinement_budget,
        std::int64_t route_change_limit,
        py::handle thresholds,
        py::handle fractions,
        py::handle deadline_remaining,
        py::handle batch_size) {
        const auto solve_started = std::chrono::steady_clock::now();
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !stage04_configured_
            || last_finished_stage04_iteration_ != -1
            || legacy_candidate_ready_ || last_candidate_ready_
            || pending_composite_active_) {
            throw std::runtime_error(
                "full native three-lane bootstrap state is invalid");
        }
        auto threshold_array = owned_array_copy<std::int64_t>(
            thresholds, "thresholds", 1);
        auto fraction_array = owned_array_copy<double>(fractions, "fractions", 1);
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "batch_size", 1);
        if (threshold_array.size() != 3 || fraction_array.size() != 6
            || deadline_array.size() != 1 || batch_array.size() != 1
            || max_route_elimination_attempts <= 0 || refinement_budget <= 0
            || route_change_limit == 0 || route_change_limit < -1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native three-lane bootstrap inputs are invalid");
        }
        const auto total_deadline = checked_data<double>(deadline_array)[0];
        const auto remaining_deadline = [&]() {
            return total_deadline - std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_started).count();
        };
        const auto next_deadline = [&]() {
            const auto remaining = remaining_deadline();
            if (remaining <= 0.0) {
                throw std::runtime_error(
                    "full native three-lane bootstrap reached its deadline");
            }
            py::array_t<double> output(1);
            checked_data(output)[0] = remaining;
            return output;
        };
        ScopeRollback discard_pending_legacy([this]() noexcept {
            legacy_candidate_ready_ = false;
            legacy_candidate_operator_ = -1;
            legacy_candidate_destroy_operator_ = -1;
            legacy_candidate_repair_operator_ = -1;
        });
        const auto make_termination = [this](
            std::int64_t reason, std::int64_t completed_iterations) {
            const auto budget = budget_.native_snapshot();
            py::array_t<std::int64_t> output(6);
            checked_data(output)[0] = reason;
            checked_data(output)[1] = budget_.exact_budget_;
            checked_data(output)[2] = budget.started;
            checked_data(output)[3] = budget.completed;
            checked_data(output)[4] = budget.interrupted;
            checked_data(output)[5] = completed_iterations;
            return output;
        };

        py::object stage04_initialization = py::none();
        if (!stage04_search_initialized_) {
            stage04_initialization = initialize_stage04_search(
                next_deadline(), batch_array);
        }
        if (budget_.budget_reached()) {
            auto legacy_state = lane_solution_state(0);
            auto quality_state = lane_solution_state(1);
            auto constraint_state = lane_solution_state(2);
            auto best = best_solution_payload();
            auto termination = make_termination(1, 0);
            auto full_stage04 = full_stage04_state();
            auto semantic_payload = py::make_tuple(
                py::none(), py::none(), py::none(), py::none(), py::none(),
                py::none(), legacy_state, quality_state, constraint_state, best,
                stage04_initialization, termination, full_stage04);
            std::string evidence(
                "stage05.2-native-three-lane-semantic-stream-v2");
            append_nested_evidence(evidence, semantic_payload);
            return py::make_tuple(
                py::none(), py::none(), py::none(), py::none(), py::none(),
                py::none(), std::move(legacy_state), std::move(quality_state),
                std::move(constraint_state), std::move(best),
                std::move(stage04_initialization), std::move(termination),
                std::move(full_stage04), native_sha256_hex(evidence));
        }
        auto legacy = legacy_route_elimination_probe(
            0, max_route_elimination_attempts, route_change_limit,
            next_deadline(), batch_array, true);
        py::object refinement = py::none();
        py::object quality = py::none();
        py::object constraint = py::none();
        py::object legacy_acceptance = py::none();
        py::object stage_boundary = py::none();
        std::optional<std::array<std::int64_t, 8>> refinement_totals_before;
        const auto reduced_vehicle_threshold = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(
                std::ceil(static_cast<double>(all_customers_.size()) / 5.0)) - 1);
        if (legacy_candidate_ready_ && all_customers_.size() > 1
            && legacy_candidate_offsets_.size() - 1
                < legacy_offsets_.size() - 1
            && legacy_candidate_offsets_.size() - 1
                <= reduced_vehicle_threshold
            && !budget_.budget_reached()) {
            refinement_totals_before = full_operator_totals_[13];
            refinement = legacy_vehicle_reduction_refinement(
                0, refinement_budget, next_deadline(), batch_array);
        }
        if (!budget_.budget_reached()) {
            quality = quality_changed_probe(
                0, 0, next_deadline(), batch_array);
        }
        if (!budget_.budget_reached()) {
            try {
                constraint = constraint_iteration(
                    0, 0, false, threshold_array, fraction_array,
                    next_deadline(), batch_array, route_change_limit);
            } catch (const std::runtime_error& error) {
                if (std::string_view(error.what()).find("deadline")
                    == std::string_view::npos) {
                    throw;
                }
                if (legacy_candidate_ready_) {
                    accumulate_full_stage04_outcome_noexcept(
                        2, false, 1, false, false, true);
                }
                legacy_candidate_ready_ = false;
                legacy_candidate_operator_ = -1;
                legacy_candidate_destroy_operator_ = -1;
                legacy_candidate_repair_operator_ = -1;
                discard_pending_legacy.release();
                auto legacy_state = lane_solution_state(0);
                auto quality_state = lane_solution_state(1);
                auto constraint_state = lane_solution_state(2);
                auto best = best_solution_payload();
                auto termination = make_termination(2, 1);
                auto full_stage04 = full_stage04_state();
                auto semantic_payload = py::make_tuple(
                    legacy, refinement, quality, py::none(),
                    legacy_acceptance, stage_boundary, legacy_state,
                    quality_state, constraint_state, best,
                    stage04_initialization, termination, full_stage04);
                std::string evidence(
                    "stage05.2-native-three-lane-semantic-stream-v2");
                append_nested_evidence(evidence, semantic_payload);
                return py::make_tuple(
                    std::move(legacy), std::move(refinement),
                    std::move(quality), py::none(),
                    std::move(legacy_acceptance), std::move(stage_boundary),
                    std::move(legacy_state), std::move(quality_state),
                    std::move(constraint_state), std::move(best),
                    std::move(stage04_initialization), std::move(termination),
                    std::move(full_stage04), native_sha256_hex(evidence));
            }
        }
        if (budget_.budget_reached()) {
            if (legacy_candidate_ready_) {
                accumulate_full_stage04_outcome_noexcept(
                    2, false, 1, false, false, true);
            }
            if (!refinement.is_none() && refinement_totals_before.has_value()) {
                auto refinement_payload = py::cast<py::tuple>(refinement);
                auto refinement_metadata =
                    py::cast<py::array_t<std::int64_t>>(refinement_payload[0]);
                if (checked_data<std::int64_t>(refinement_metadata)[1] != 0) {
                    full_operator_totals_[13] = *refinement_totals_before;
                    ++full_operator_totals_[13][0];
                }
            }
            legacy_candidate_ready_ = false;
            legacy_candidate_operator_ = -1;
            legacy_candidate_destroy_operator_ = -1;
            legacy_candidate_repair_operator_ = -1;
            discard_pending_legacy.release();
            auto legacy_state = lane_solution_state(0);
            auto quality_state = lane_solution_state(1);
            auto constraint_state = lane_solution_state(2);
            auto best = best_solution_payload();
            auto termination = make_termination(1, 1);
            auto full_stage04 = full_stage04_state();
            auto semantic_payload = py::make_tuple(
                legacy, refinement, quality, constraint,
                legacy_acceptance, stage_boundary, legacy_state,
                quality_state, constraint_state, best,
                stage04_initialization, termination, full_stage04);
            std::string evidence(
                "stage05.2-native-three-lane-semantic-stream-v2");
            append_nested_evidence(evidence, semantic_payload);
            return py::make_tuple(
                std::move(legacy), std::move(refinement),
                std::move(quality), std::move(constraint),
                std::move(legacy_acceptance), std::move(stage_boundary),
                std::move(legacy_state), std::move(quality_state),
                std::move(constraint_state), std::move(best),
                std::move(stage04_initialization), std::move(termination),
                std::move(full_stage04), native_sha256_hex(evidence));
        }
        if (legacy_candidate_ready_) {
            if (!rng_.has_value()) {
                throw std::logic_error(
                    "full native legacy acceptance lost its RNG state");
            }
            legacy_acceptance = apply_legacy_candidate(
                stage04_initial_temperature_, rng_->random());
        }
        const auto quality_global_best = !quality.is_none()
            && checked_data<std::int64_t>(
                py::cast<py::array_t<std::int64_t>>(
                    py::cast<py::tuple>(quality)[2]))[2] != 0;
        const auto constraint_global_best = !constraint.is_none()
            && checked_data<std::int64_t>(
                py::cast<py::array_t<std::int64_t>>(
                    py::cast<py::tuple>(constraint)[2]))[4] != 0;
        const auto legacy_global_best = !legacy_acceptance.is_none()
            && py::cast<std::int64_t>(
                py::cast<py::tuple>(legacy_acceptance)[1]) != 0;
        last_iteration_global_best_improved_ = quality_global_best
            || constraint_global_best || legacy_global_best;
        main_stagnation_iterations_ = last_iteration_global_best_improved_
            ? 0 : main_stagnation_iterations_ + 1;
        stage_boundary = finish_stage04_iteration(0, false);
        discard_pending_legacy.release();
        auto legacy_state = lane_solution_state(0);
        auto quality_state = lane_solution_state(1);
        auto constraint_state = lane_solution_state(2);
        auto best = best_solution_payload();
        auto termination = make_termination(0, 1);
        auto full_stage04 = full_stage04_state();
        auto semantic_payload = py::make_tuple(
            legacy, refinement, quality, constraint,
            legacy_acceptance, stage_boundary, legacy_state,
            quality_state, constraint_state, best,
            stage04_initialization, termination, full_stage04);
        std::string evidence(
            "stage05.2-native-three-lane-semantic-stream-v2");
        append_nested_evidence(evidence, semantic_payload);
        return py::make_tuple(
            std::move(legacy), std::move(refinement),
            std::move(quality), std::move(constraint),
            std::move(legacy_acceptance), std::move(stage_boundary),
            std::move(legacy_state), std::move(quality_state),
            std::move(constraint_state), std::move(best),
            std::move(stage04_initialization), std::move(termination),
            std::move(full_stage04), native_sha256_hex(evidence));
    }

    py::tuple run_three_lane_followup(
        std::int64_t iteration,
        std::int64_t max_iterations,
        double removal_fraction,
        std::int64_t route_elimination_max_attempts,
        std::int64_t refinement_budget,
        std::int64_t route_segment_min_length,
        std::int64_t route_segment_max_length,
        std::int64_t route_segment_budget,
        std::int64_t ejection_chain_budget,
        std::int64_t ejection_chain_max_depth,
        std::int64_t ejection_chain_beam_width,
        std::int64_t route_change_limit,
        py::handle thresholds,
        py::handle fractions,
        py::handle deadline_remaining,
        py::handle batch_size) {
        const auto solve_started = std::chrono::steady_clock::now();
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !stage04_configured_ || !stage04_search_initialized_
            || iteration <= 0 || max_iterations <= iteration
            || last_finished_stage04_iteration_ != iteration - 1
            || legacy_candidate_ready_ || last_candidate_ready_
            || pending_composite_active_) {
            throw std::runtime_error(
                "full native three-lane follow-up state is invalid");
        }
        auto threshold_array = owned_array_copy<std::int64_t>(
            thresholds, "thresholds", 1);
        auto fraction_array = owned_array_copy<double>(fractions, "fractions", 1);
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "batch_size", 1);
        if (threshold_array.size() != 3 || fraction_array.size() != 6
            || deadline_array.size() != 1 || batch_array.size() != 1
            || route_elimination_max_attempts <= 0
            || refinement_budget <= 0
            || route_segment_min_length <= 0
            || route_segment_max_length < route_segment_min_length
            || route_segment_budget <= 0
            || ejection_chain_budget <= 0
            || ejection_chain_max_depth <= 0
            || ejection_chain_beam_width <= 0
            || route_change_limit == 0 || route_change_limit < -1
            || !std::isfinite(removal_fraction) || removal_fraction <= 0.0
            || removal_fraction > 1.0
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native three-lane follow-up inputs are invalid");
        }
        const auto total_deadline = checked_data<double>(deadline_array)[0];
        const auto remaining_deadline = [&]() {
            return total_deadline - std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_started).count();
        };
        const auto next_deadline = [&]() {
            const auto remaining = remaining_deadline();
            if (remaining <= 0.0) {
                throw std::runtime_error(
                    "full native three-lane follow-up reached its deadline");
            }
            py::array_t<double> output(1);
            checked_data(output)[0] = remaining;
            return output;
        };
        const auto make_termination = [this](
            std::int64_t reason, std::int64_t completed_iterations) {
            const auto budget = budget_.native_snapshot();
            py::array_t<std::int64_t> output(6);
            checked_data(output)[0] = reason;
            checked_data(output)[1] = budget_.exact_budget_;
            checked_data(output)[2] = budget.started;
            checked_data(output)[3] = budget.completed;
            checked_data(output)[4] = budget.interrupted;
            checked_data(output)[5] = completed_iterations;
            return output;
        };
        ScopeRollback discard_pending_legacy([this]() noexcept {
            legacy_candidate_ready_ = false;
            legacy_candidate_operator_ = -1;
            legacy_candidate_destroy_operator_ = -1;
            legacy_candidate_repair_operator_ = -1;
        });
        py::object legacy = py::none();
        py::object quality = py::none();
        py::object constraint = py::none();
        py::object refinement = py::none();
        py::object legacy_acceptance = py::none();
        py::object stage_boundary = py::none();
        std::optional<std::array<std::int64_t, 8>> refinement_totals_before;
        const auto make_deadline_result = [&]() {
            if (legacy_candidate_ready_) {
                accumulate_full_stage04_outcome_noexcept(
                    static_cast<std::size_t>(legacy_candidate_operator_),
                    false, 1, false, false, true);
                if (legacy_candidate_operator_ == 0
                    || legacy_candidate_operator_ == 1) {
                    accumulate_full_stage04_outcome_noexcept(
                        static_cast<std::size_t>(
                            14 + legacy_candidate_destroy_operator_),
                        false, 1, false, false, true);
                    if (legacy_candidate_operator_ == 0) {
                        accumulate_full_stage04_outcome_noexcept(
                            static_cast<std::size_t>(
                                17 + legacy_candidate_repair_operator_),
                            false, 1, false, false, true);
                    }
                }
            }
            legacy_candidate_ready_ = false;
            legacy_candidate_operator_ = -1;
            legacy_candidate_destroy_operator_ = -1;
            legacy_candidate_repair_operator_ = -1;
            discard_pending_legacy.release();
            auto legacy_state = lane_solution_state(0);
            auto quality_state = lane_solution_state(1);
            auto constraint_state = lane_solution_state(2);
            auto best = best_solution_payload();
            auto termination = make_termination(2, iteration);
            auto full_stage04 = full_stage04_state();
            auto semantic_payload = py::make_tuple(
                legacy, refinement, quality, constraint,
                legacy_acceptance, stage_boundary, legacy_state,
                quality_state, constraint_state, best, py::none(),
                termination, full_stage04);
            std::string evidence(
                "stage05.2-native-three-lane-semantic-stream-v2");
            append_nested_evidence(evidence, semantic_payload);
            return py::make_tuple(
                std::move(legacy), std::move(refinement), std::move(quality),
                std::move(constraint), std::move(legacy_acceptance),
                std::move(stage_boundary), std::move(legacy_state),
                std::move(quality_state), std::move(constraint_state),
                std::move(best), py::none(), std::move(termination),
                std::move(full_stage04), native_sha256_hex(evidence));
        };
        try {
        if (iteration == 1) {
            legacy = legacy_vehicle_count_aware_probe(
                iteration, removal_fraction, route_change_limit,
                next_deadline(), batch_array, true);
        } else if (iteration == 2) {
            legacy = legacy_route_merge_probe(
                iteration, next_deadline(), batch_array, true);
        } else if (iteration >= 3) {
            if (!rng_.has_value()) {
                throw std::logic_error(
                    "full native main dispatcher lost its RNG state");
            }
            auto selector = *rng_;
            const auto selected_main = static_cast<std::int64_t>(
                selector.weighted_index({
                    full_operator_weights_[0], full_operator_weights_[1],
                    full_operator_weights_[2], full_operator_weights_[3]}));
            if (selected_main == 0) {
                legacy = legacy_standard_probe(
                    iteration, removal_fraction, route_change_limit,
                    next_deadline(), batch_array);
            } else if (selected_main == 1) {
                legacy = legacy_vehicle_count_aware_probe(
                    iteration, removal_fraction, route_change_limit,
                    next_deadline(), batch_array, true, true);
            } else if (selected_main == 2 && legacy_offsets_.size() == 2) {
                *rng_ = std::move(selector);
                py::array_t<std::int64_t> metadata(4);
                checked_data(metadata)[0] = iteration;
                checked_data(metadata)[1] = 2;
                checked_data(metadata)[2] = legacy_offsets_.size() - 1;
                checked_data(metadata)[3] = 0;
                accumulate_full_stage04_outcome_noexcept(
                    2, false, 1, false, false, true);
                legacy = py::make_tuple(
                    std::move(metadata), lane_solution_state(0));
            } else if (selected_main == 2) {
                // Route elimination has no random choices after the weighted
                // operator draw, so commit that draw before building plans.
                *rng_ = std::move(selector);
                legacy = legacy_route_elimination_probe(
                    iteration, route_elimination_max_attempts,
                    route_change_limit, next_deadline(), batch_array, true);
            } else if (selected_main == 3) {
                // route_merge itself has no random choices. Commit the
                // weighted dispatcher draw before constructing its pool.
                *rng_ = std::move(selector);
                legacy = legacy_route_merge_probe(
                    iteration, next_deadline(), batch_array, true);
            } else {
                throw std::runtime_error(
                    "full native weighted main dispatcher selected an unimplemented operator");
            }
        } else {
            throw std::runtime_error(
                "full native three-lane follow-up operator is not implemented");
        }
        const auto reduced_vehicle_threshold = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(
                std::ceil(static_cast<double>(all_customers_.size()) / 5.0)) - 1);
        if (legacy_candidate_ready_ && all_customers_.size() > 1
            && legacy_candidate_offsets_.size() - 1
                < legacy_offsets_.size() - 1
            && legacy_candidate_offsets_.size() - 1
                <= reduced_vehicle_threshold
            && !budget_.budget_reached()) {
            refinement_totals_before = full_operator_totals_[13];
            refinement = legacy_vehicle_reduction_refinement(
                iteration, refinement_budget, next_deadline(), batch_array);
        }
        if (!budget_.budget_reached()) {
            if (iteration < 3) {
                quality = quality_changed_probe(
                    iteration, iteration, next_deadline(), batch_array);
            } else if (iteration == 3) {
                quality = quality_route_segment_probe(
                    iteration, route_segment_min_length,
                    route_segment_max_length, route_segment_budget,
                    route_change_limit, next_deadline(), batch_array);
            } else if (iteration == 4) {
                quality = quality_ejection_chain_probe(
                    iteration, ejection_chain_budget,
                    ejection_chain_max_depth, ejection_chain_beam_width,
                    next_deadline(), batch_array);
            }
        }
        if (!budget_.budget_reached()
            && (iteration < 4
                || iteration % checked_data<std::int64_t>(threshold_array)[2]
                    == 0)) {
            constraint = constraint_iteration(
                iteration, main_stagnation_iterations_,
                last_iteration_global_best_improved_, threshold_array,
                fraction_array, next_deadline(), batch_array,
                route_change_limit);
        }
        } catch (const NativeExactDeadlineInterruption&) {
            return make_deadline_result();
        } catch (const std::runtime_error& error) {
            if (std::string_view(error.what()).find("deadline")
                == std::string_view::npos) {
                throw;
            }
            return make_deadline_result();
        }
        if (budget_.budget_reached()) {
            // Fixed-work exhaustion discards the enclosing candidate
            // transaction atomically.  Python still charges the operator and
            // adaptive-role attempt that produced the incomplete candidate;
            // preserve those audit counters without committing solver state.
            if (legacy_candidate_ready_) {
                accumulate_full_stage04_outcome_noexcept(
                    static_cast<std::size_t>(legacy_candidate_operator_),
                    false, 1, false, false, true);
                if (legacy_candidate_operator_ == 0
                    || legacy_candidate_operator_ == 1) {
                    accumulate_full_stage04_outcome_noexcept(
                        static_cast<std::size_t>(
                            14 + legacy_candidate_destroy_operator_),
                        false, 1, false, false, true);
                    if (legacy_candidate_operator_ == 0) {
                        accumulate_full_stage04_outcome_noexcept(
                            static_cast<std::size_t>(
                                17 + legacy_candidate_repair_operator_),
                            false, 1, false, false, true);
                    }
                }
            }
            if (!refinement.is_none() && refinement_totals_before.has_value()) {
                auto refinement_payload = py::cast<py::tuple>(refinement);
                auto refinement_metadata =
                    py::cast<py::array_t<std::int64_t>>(refinement_payload[0]);
                if (checked_data<std::int64_t>(refinement_metadata)[1] != 0) {
                    full_operator_totals_[13] = *refinement_totals_before;
                    ++full_operator_totals_[13][0];
                }
            }
            legacy_candidate_ready_ = false;
            legacy_candidate_operator_ = -1;
            legacy_candidate_destroy_operator_ = -1;
            legacy_candidate_repair_operator_ = -1;
            discard_pending_legacy.release();
            auto legacy_state = lane_solution_state(0);
            auto quality_state = lane_solution_state(1);
            auto constraint_state = lane_solution_state(2);
            auto best = best_solution_payload();
            auto termination = make_termination(1, iteration);
            auto full_stage04 = full_stage04_state();
            auto semantic_payload = py::make_tuple(
                legacy, refinement, quality, constraint, legacy_acceptance,
                stage_boundary, legacy_state, quality_state, constraint_state,
                best, py::none(), termination, full_stage04);
            std::string evidence(
                "stage05.2-native-three-lane-semantic-stream-v2");
            append_nested_evidence(evidence, semantic_payload);
            return py::make_tuple(
                std::move(legacy), std::move(refinement), std::move(quality),
                std::move(constraint), std::move(legacy_acceptance),
                std::move(stage_boundary), std::move(legacy_state),
                std::move(quality_state), std::move(constraint_state),
                std::move(best), py::none(), std::move(termination),
                std::move(full_stage04), native_sha256_hex(evidence));
        }
        stage04_reheat_floor_ *= 0.99;
        bool legacy_candidate_was_accepted = false;
        if (legacy_candidate_ready_) {
            const auto cooling = std::max(
                0.001, 1.0 - static_cast<double>(iteration)
                    / static_cast<double>(max_iterations));
            legacy_acceptance = apply_legacy_candidate(
                std::max(
                    stage04_initial_temperature_ * cooling,
                    stage04_reheat_floor_),
                rng_->random());
            legacy_candidate_was_accepted = py::cast<std::int64_t>(
                py::cast<py::tuple>(legacy_acceptance)[0]) != 0;
        }
        const auto quality_global_best = [&]() {
            if (quality.is_none()) {
                return false;
            }
            const auto quality_payload = py::cast<py::tuple>(quality);
            // The route-segment payload carries its acceptance outcome at
            // slot 5; the changed-route and ejection-chain payloads carry it
            // at slot 2.  Reading the route-segment removed-customer vector
            // as an outcome silently reset stagnation after iteration 3.
            const auto outcome_index = iteration == 3 ? 5 : 2;
            return checked_data<std::int64_t>(
                py::cast<py::array_t<std::int64_t>>(
                    quality_payload[outcome_index]))[2] != 0;
        }();
        const auto constraint_global_best = !constraint.is_none()
            && checked_data<std::int64_t>(
                py::cast<py::array_t<std::int64_t>>(
                    py::cast<py::tuple>(constraint)[2]))[4] != 0;
        const auto legacy_global_best = !legacy_acceptance.is_none()
            && py::cast<std::int64_t>(
                py::cast<py::tuple>(legacy_acceptance)[1]) != 0;
        last_iteration_global_best_improved_ = quality_global_best
            || constraint_global_best || legacy_global_best;
        main_stagnation_iterations_ = last_iteration_global_best_improved_
            ? 0 : main_stagnation_iterations_ + 1;
        stage_boundary = finish_stage04_iteration_impl(
            iteration, false, legacy_candidate_was_accepted);
        discard_pending_legacy.release();
        auto legacy_state = lane_solution_state(0);
        auto quality_state = lane_solution_state(1);
        auto constraint_state = lane_solution_state(2);
        auto best = best_solution_payload();
        auto termination = make_termination(0, iteration + 1);
        auto full_stage04 = full_stage04_state();
        auto semantic_payload = py::make_tuple(
            legacy, refinement, quality, constraint, legacy_acceptance,
            stage_boundary, legacy_state, quality_state, constraint_state,
            best, py::none(), termination, full_stage04);
        std::string evidence(
            "stage05.2-native-three-lane-semantic-stream-v2");
        append_nested_evidence(evidence, semantic_payload);
        return py::make_tuple(
            std::move(legacy), std::move(refinement), std::move(quality),
            std::move(constraint), std::move(legacy_acceptance),
            std::move(stage_boundary), std::move(legacy_state),
            std::move(quality_state), std::move(constraint_state),
            std::move(best), py::none(), std::move(termination),
            std::move(full_stage04), native_sha256_hex(evidence));
    }

    py::array_t<std::int64_t> three_lane_termination_state(
        std::int64_t reason,
        std::int64_t completed_iterations) const {
        if (!initialized_ || reason < 0 || reason > 2
            || completed_iterations < 0) {
            throw std::invalid_argument(
                "full native three-lane termination state is invalid");
        }
        const auto budget = budget_.native_snapshot();
        py::array_t<std::int64_t> output(6);
        checked_data(output)[0] = reason;
        checked_data(output)[1] = budget_.exact_budget_;
        checked_data(output)[2] = budget.started;
        checked_data(output)[3] = budget.completed;
        checked_data(output)[4] = budget.interrupted;
        checked_data(output)[5] = completed_iterations;
        return output;
    }

    std::int64_t main_stagnation_iterations() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        return main_stagnation_iterations_;
    }

    py::tuple constraint_probe(
        std::int64_t operation,
        std::int64_t requested_count,
        std::uint64_t seed,
        py::handle context_ids,
        py::handle deadline_remaining,
        py::handle batch_size,
        std::int64_t route_change_limit) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_) {
            throw std::runtime_error(
                "full native search engine must be initialized before a constraint probe");
        }
        if (last_candidate_ready_) {
            throw std::runtime_error(
                "full native search engine has an unapplied candidate");
        }
        const auto probe_started = std::chrono::steady_clock::now();
        auto context_array = owned_array_copy<std::int64_t>(
            context_ids, "context_ids", 1);
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "batch_size", 1);
        if (context_array.size() != 3 || deadline_array.size() != 1
            || batch_array.size() != 1
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native constraint probe scalar-array shape is invalid");
        }
        const auto* context = checked_data<std::int64_t>(context_array);
        if (context[0] < 0 || context[1] < 0 || context[2] < 0) {
            throw std::invalid_argument(
                "full native constraint probe context IDs must be non-negative");
        }
        const auto deadline_seconds = checked_data<double>(deadline_array)[0];
        if (!std::isfinite(deadline_seconds) || deadline_seconds <= 0.0) {
            throw std::invalid_argument(
                "full native constraint probe deadline must be finite and positive");
        }
        if (operation < 0 || operation > 3 || requested_count <= 0
            || route_change_limit == 0 || route_change_limit < -1) {
            throw std::invalid_argument(
                "full native constraint probe configuration is invalid");
        }
        const auto remaining_at_boundary = [&]() {
            return deadline_seconds - std::chrono::duration<double>(
                std::chrono::steady_clock::now() - probe_started).count();
        };
        const auto require_deadline = [&]() {
            if (remaining_at_boundary() <= 0.0) {
                throw std::runtime_error(
                    "full native constraint probe reached its deadline");
            }
        };
        auto removal = constraint_removal_v2(
            operation,
            node_kind_,
            demand_,
            ready_time_,
            due_date_,
            service_time_,
            distance_,
            reachable_,
            vehicle_,
            lexical_rank_,
            current_offsets_,
            current_indices_,
            current_exact_payload_[0],
            current_exact_payload_[1],
            current_exact_payload_[4],
            requested_count,
            seed);
        auto removal_metadata = py::cast<py::array_t<std::int64_t>>(removal[6]);
        if (checked_data<std::int64_t>(removal_metadata)[0] != 0) {
            auto result = py::make_tuple(
                std::move(removal), py::none(), py::none());
            require_deadline();
            return result;
        }
        auto repair = candidate_control_repair_v2(
            node_kind_,
            demand_,
            ready_time_,
            due_date_,
            service_time_,
            distance_,
            reachable_,
            vehicle_,
            lexical_rank_,
            removal[0],
            removal[1],
            removal[2],
            screening_epsilon_,
            route_change_limit,
            false);
        auto repair_metadata = py::cast<py::array_t<std::int64_t>>(repair[2]);
        if (checked_data<std::int64_t>(repair_metadata)[0] != 0) {
            auto result = py::make_tuple(
                std::move(removal), std::move(repair), py::none());
            require_deadline();
            return result;
        }
        auto repaired_offsets = py::cast<py::array_t<std::int64_t>>(repair[0]);
        auto repaired_indices = py::cast<py::array_t<std::int64_t>>(repair[1]);
        py::array_t<std::int64_t> plan_offsets(2);
        checked_data(plan_offsets)[0] = 0;
        checked_data(plan_offsets)[1] = repaired_offsets.size() - 1;
        std::vector<std::int64_t> expected(
            all_customers_.begin(), all_customers_.end());
        std::stable_sort(
            expected.begin(), expected.end(),
            [&](std::int64_t left, std::int64_t right) {
                return checked_data<std::int64_t>(lexical_rank_)[left]
                    < checked_data<std::int64_t>(lexical_rank_)[right];
            });
        py::array_t<std::int64_t> expected_array(expected.size());
        std::copy(expected.begin(), expected.end(), checked_data(expected_array));
        require_deadline();
        py::array_t<double> adjusted_deadline(1);
        checked_data(adjusted_deadline)[0] = remaining_at_boundary();
        suppress_attempted_plan_journal_ = true;
        ScopeRollback restore_attempted_plan_policy([this]() noexcept {
            suppress_attempted_plan_journal_ = false;
        });
        defer_composite_commit_ = true;
        try {
            auto transaction = evaluate_plans(
                plan_offsets,
                repair[0],
                repair[1],
                context_array,
                adjusted_deadline,
                batch_array,
                expected_array);
            defer_composite_commit_ = false;
            auto feasible_order = py::cast<py::array_t<std::int64_t>>(transaction[11]);
            const auto candidate_ready = feasible_order.size() > 0;
            py::tuple candidate_exact;
            py::array_t<std::int64_t> candidate_offsets;
            py::array_t<std::int64_t> candidate_indices;
            py::array_t<std::int64_t> candidate_objective_integer;
            py::array_t<double> candidate_objective_float;
            if (candidate_ready) {
                const auto plan_id = checked_data<std::int64_t>(feasible_order)[0];
                if (plan_id != 0) {
                    throw std::logic_error(
                        "full native constraint probe returned an unknown plan identity");
                }
                if (!pending_candidate_exact_ready_) {
                    throw std::logic_error(
                        "full native constraint probe lost its exact candidate payload");
                }
                candidate_exact = pending_candidate_exact_payload_;
                candidate_offsets = owned_array_copy<std::int64_t>(
                    repaired_offsets, "prepared_candidate_offsets", 1);
                candidate_indices = owned_array_copy<std::int64_t>(
                    repaired_indices, "prepared_candidate_indices", 1);
                auto objective_integer_matrix =
                    py::cast<py::array_t<std::int64_t>>(transaction[2]);
                auto objective_float_matrix =
                    py::cast<py::array_t<double>>(transaction[3]);
                candidate_objective_integer = py::array_t<std::int64_t>(2);
                candidate_objective_float = py::array_t<double>(2);
                std::copy(
                    checked_data<std::int64_t>(objective_integer_matrix),
                    checked_data<std::int64_t>(objective_integer_matrix) + 2,
                    checked_data(candidate_objective_integer));
                std::copy(
                    checked_data<double>(objective_float_matrix),
                    checked_data<double>(objective_float_matrix) + 2,
                    checked_data(candidate_objective_float));
            }
            auto result = py::make_tuple(
                std::move(removal), std::move(repair), std::move(transaction));
            require_deadline();
            if (probe_envelope_failure_injection_) {
                probe_envelope_failure_injection_ = false;
                throw std::runtime_error(
                    "injected full native constraint-probe envelope failure");
            }
            if (!defer_iteration_commit_) {
                commit_pending_composite_noexcept();
            }
            if (candidate_ready) {
                last_candidate_offsets_ = std::move(candidate_offsets);
                last_candidate_indices_ = std::move(candidate_indices);
                last_candidate_exact_payload_ = std::move(candidate_exact);
                last_candidate_objective_integer_ =
                    std::move(candidate_objective_integer);
                last_candidate_objective_float_ =
                    std::move(candidate_objective_float);
            }
            last_candidate_ready_ = candidate_ready;
            suppress_attempted_plan_journal_ = false;
            restore_attempted_plan_policy.release();
            return result;
        } catch (...) {
            defer_composite_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            throw;
        }
    }

    py::tuple apply_last_candidate(double temperature, double random_draw) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!last_candidate_ready_) {
            throw std::runtime_error(
                "full native search engine has no prepared candidate to apply");
        }
        if (!std::isfinite(temperature) || temperature <= 0.0
            || !std::isfinite(random_draw) || random_draw < 0.0
            || random_draw > 1.0) {
            throw std::invalid_argument(
                "full native candidate acceptance inputs are invalid");
        }
        const auto current_key = evrptw::formal_objective::key_from_arrays(
            current_objective_integer_, current_objective_float_);
        const auto candidate_key = evrptw::formal_objective::key_from_arrays(
            last_candidate_objective_integer_, last_candidate_objective_float_);
        const auto best_key = evrptw::formal_objective::key_from_arrays(
            best_objective_integer_, best_objective_float_);
        const auto current_vehicles = std::get<0>(current_key);
        const auto candidate_vehicles = std::get<0>(candidate_key);
        bool accepted = false;
        if (candidate_vehicles < current_vehicles) {
            accepted = true;
        } else if (candidate_vehicles == current_vehicles
                   && candidate_key <= current_key) {
            accepted = true;
        } else if (candidate_vehicles == current_vehicles
                   && std::get<1>(candidate_key) != std::get<1>(current_key)) {
            const auto distance_delta =
                checked_data<double>(last_candidate_objective_float_)[0]
                - checked_data<double>(current_objective_float_)[0];
            accepted = random_draw < std::exp(-distance_delta / temperature);
        }
        const auto improved_best = accepted && candidate_key < best_key;
        const auto vehicle_reduction = accepted
            && candidate_vehicles < current_vehicles;
        py::array_t<std::int64_t> next_current_offsets;
        py::array_t<std::int64_t> next_current_indices;
        py::tuple next_current_exact;
        py::array_t<std::int64_t> next_current_objective_integer;
        py::array_t<double> next_current_objective_float;
        py::array_t<std::int64_t> next_best_offsets;
        py::array_t<std::int64_t> next_best_indices;
        py::tuple next_best_exact;
        py::array_t<std::int64_t> next_best_objective_integer;
        py::array_t<double> next_best_objective_float;
        if (accepted) {
            next_current_offsets = last_candidate_offsets_;
            next_current_indices = last_candidate_indices_;
            next_current_exact = last_candidate_exact_payload_;
            next_current_objective_integer = last_candidate_objective_integer_;
            next_current_objective_float = last_candidate_objective_float_;
        }
        if (improved_best) {
            next_best_offsets = last_candidate_offsets_;
            next_best_indices = last_candidate_indices_;
            next_best_exact = last_candidate_exact_payload_;
            next_best_objective_integer = last_candidate_objective_integer_;
            next_best_objective_float = last_candidate_objective_float_;
        }
        auto result = py::make_tuple(
            accepted ? 1 : 0,
            improved_best ? 1 : 0,
            vehicle_reduction ? 1 : 0);
        if (accepted) {
            current_offsets_ = std::move(next_current_offsets);
            current_indices_ = std::move(next_current_indices);
            current_exact_payload_ = std::move(next_current_exact);
            current_objective_integer_ = std::move(next_current_objective_integer);
            current_objective_float_ = std::move(next_current_objective_float);
        }
        if (improved_best) {
            best_offsets_ = std::move(next_best_offsets);
            best_indices_ = std::move(next_best_indices);
            best_exact_payload_ = std::move(next_best_exact);
            best_objective_integer_ = std::move(next_best_objective_integer);
            best_objective_float_ = std::move(next_best_objective_float);
        }
        last_candidate_ready_ = false;
        return result;
    }

    void configure_stage04(
        py::handle integer_config,
        py::handle float_config) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_) {
            throw std::runtime_error(
                "full native search engine must be initialized before Stage 4");
        }
        if (stage04_configured_) {
            throw std::runtime_error(
                "full native Stage 4 is already configured for this solve");
        }
        if (last_candidate_ready_ || pending_composite_active_) {
            throw std::runtime_error(
                "full native Stage 4 cannot be configured during a candidate transaction");
        }
        auto integers = owned_array_copy<std::int64_t>(
            integer_config, "stage04_integer", 1);
        auto floats = owned_array_copy<double>(
            float_config, "stage04_float", 1);
        if (integers.size() != 15 || floats.size() != 15) {
            throw std::invalid_argument(
                "full native Stage 4 configuration arrays have an invalid shape");
        }
        const auto* integer_values = checked_data<std::int64_t>(integers);
        const auto* float_values = checked_data<double>(floats);
        if (integer_values[0] != 1 || integer_values[1] <= 0
            || integer_values[2] <= 0
            || (integer_values[3] != 0 && integer_values[3] != 1)
            || (integer_values[4] != 0 && integer_values[4] != 1)
            || integer_values[5] <= 0
            || (integer_values[6] != 0 && integer_values[6] != 1)
            || integer_values[7] <= 0 || integer_values[8] < 0
            || (integer_values[9] != 0 && integer_values[9] != 1)
            || integer_values[10] <= 0 || integer_values[11] < 0
            || (integer_values[12] != 0 && integer_values[12] != 1)
            || integer_values[13] <= 0 || integer_values[14] <= 0) {
            throw std::invalid_argument(
                "full native Stage 4 integer configuration is invalid");
        }
        for (py::ssize_t index = 0; index < floats.size(); ++index) {
            if (!std::isfinite(float_values[index])) {
                throw std::invalid_argument(
                    "full native Stage 4 float configuration is not finite");
            }
        }
        if (!(float_values[0] > 0.0 && float_values[0] <= 1.0)
            || !(float_values[1] > 0.0 && float_values[1] < 1.0)
            || !(float_values[2] >= 0.0 && float_values[2] <= 1.0)
            || !(float_values[11] > 0.0 && float_values[11] < 1.0)
            || float_values[12] <= 0.0
            || float_values[13] <= 0.0
            || !(float_values[14] > 0.0 && float_values[14] <= 1.0)) {
            throw std::invalid_argument(
                "full native Stage 4 weight configuration is invalid");
        }
        std::array<double, 7> rewards{};
        for (std::size_t index = 0; index < rewards.size(); ++index) {
            rewards[index] = float_values[index + 4];
            if (rewards[index] < 0.0 || rewards[index] > 64.0) {
                throw std::invalid_argument(
                    "full native Stage 4 reward configuration is invalid");
            }
        }
        stage04_segment_length_ = integer_values[1];
        stage04_min_calls_ = integer_values[2];
        stage04_fixed_weights_ = integer_values[3] != 0;
        stage04_auto_temperature_ = integer_values[4] != 0;
        stage04_temperature_sample_size_ = integer_values[5];
        stage04_reheat_enabled_ = integer_values[6] != 0;
        stage04_reheat_stagnation_threshold_ = integer_values[7];
        stage04_max_reheats_ = integer_values[8];
        stage04_restart_enabled_ = integer_values[9] != 0;
        stage04_restart_stagnation_threshold_ = integer_values[10];
        stage04_max_restarts_ = integer_values[11];
        stage04_intensification_enabled_ = integer_values[12] != 0;
        stage04_intensification_iterations_ = integer_values[13];
        stage04_weight_reaction_ = float_values[0];
        stage04_weight_floor_ = float_values[1];
        stage04_weight_smoothing_ = float_values[2];
        stage04_temperature_target_ = float_values[11];
        stage04_temperature_fallback_fraction_ = float_values[12];
        stage04_reheat_factor_ = float_values[13];
        stage04_intensification_removal_fraction_ = float_values[14];
        stage04_rewards_ = rewards;
        full_operator_weights_.fill(1.0);
        full_operator_segment_rewards_.fill(0.0);
        full_operator_segment_calls_.fill(0);
        for (auto& totals : full_operator_totals_) {
            totals.fill(0);
        }
        constraint_weights_.fill(1.0);
        constraint_segment_rewards_.fill(0.0);
        constraint_segment_calls_.fill(0);
        for (auto& totals : constraint_totals_) {
            totals.fill(0);
        }
        last_finished_stage04_iteration_ = -1;
        last_completed_constraint_iteration_ = -1;
        stage04_reheat_count_ = 0;
        stage04_restart_count_ = 0;
        stage04_reheat_floor_ = 0.0;
        stage04_intensification_active_ = false;
        stage04_intensification_remaining_ = 0;
        stage04_configured_ = true;
    }

    py::tuple initialize_stage04_search(
        py::handle deadline_remaining,
        py::handle batch_size) {
        const auto initialization_started = std::chrono::steady_clock::now();
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !stage04_configured_ || stage04_search_initialized_
            || !rng_.has_value() || last_candidate_ready_
            || legacy_candidate_ready_ || pending_composite_active_) {
            throw std::runtime_error(
                "full native Stage 4 search initialization state is invalid");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "stage04_deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "stage04_batch_size", 1);
        if (deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "full native Stage 4 search initialization inputs are invalid");
        }
        const auto total_deadline = checked_data<double>(deadline_array)[0];
        const auto next_deadline = [&]() {
            const auto remaining = total_deadline - std::chrono::duration<double>(
                std::chrono::steady_clock::now() - initialization_started).count();
            if (remaining <= 0.0) {
                throw std::runtime_error(
                    "full native Stage 4 initialization reached its deadline");
            }
            py::array_t<double> output(1);
            checked_data(output)[0] = remaining;
            return output;
        };
        const auto current_distance =
            checked_data<double>(current_objective_float_)[0];
        const auto fallback = std::max(
            1.0, current_distance * stage04_temperature_fallback_fraction_);
        std::vector<double> positive_deltas;
        std::vector<std::int64_t> evaluated_plan_offsets{0};
        std::vector<std::int64_t> evaluated_route_offsets{0};
        std::vector<std::int64_t> evaluated_route_indices;
        auto next_rng = *rng_;
        const auto route_count = current_offsets_.size() - 1;
        if (stage04_auto_temperature_ && route_count > 1) {
            const auto* route_boundaries =
                checked_data<std::int64_t>(current_offsets_);
            const auto* route_nodes =
                checked_data<std::int64_t>(current_indices_);
            std::vector<std::vector<std::int64_t>> base_routes;
            for (py::ssize_t route = 0; route < route_count; ++route) {
                base_routes.emplace_back(
                    route_nodes + route_boundaries[route],
                    route_nodes + route_boundaries[route + 1]);
            }
            const auto sample_count = std::min<std::int64_t>(
                stage04_temperature_sample_size_,
                std::max<std::int64_t>(
                    5, static_cast<std::int64_t>(all_customers_.size())));
            std::vector<std::int64_t> expected(
                all_customers_.begin(), all_customers_.end());
            std::stable_sort(
                expected.begin(), expected.end(),
                [&](std::int64_t left, std::int64_t right) {
                    return checked_data<std::int64_t>(lexical_rank_)[left]
                        < checked_data<std::int64_t>(lexical_rank_)[right];
                });
            py::array_t<std::int64_t> expected_array(expected.size());
            std::copy(expected.begin(), expected.end(), checked_data(expected_array));
            py::array_t<std::int64_t> context(3);
            checked_data(context)[0] = stable_int63("legacy");
            checked_data(context)[1] = stable_int63("initialization");
            checked_data(context)[2] = -1;
            suppress_attempted_plan_journal_ = true;
            suppress_round_budget_ = true;
            ScopeRollback restore_protocol_flags([this]() noexcept {
                suppress_attempted_plan_journal_ = false;
                suppress_round_budget_ = false;
            });
            for (std::int64_t sample = 0; sample < sample_count; ++sample) {
                std::vector<std::int64_t> order(
                    static_cast<std::size_t>(route_count));
                std::iota(order.begin(), order.end(), std::int64_t{0});
                next_rng.shuffle(order);
                const auto first = static_cast<std::size_t>(
                    next_rng.randbelow(static_cast<std::uint64_t>(route_count)));
                const auto second = static_cast<std::size_t>(
                    next_rng.randbelow(static_cast<std::uint64_t>(route_count)));
                if (first == second) {
                    continue;
                }
                std::vector<std::vector<std::int64_t>> candidate;
                for (std::size_t index = 0; index < order.size(); ++index) {
                    if (index != first && index != second) {
                        candidate.push_back(
                            base_routes[static_cast<std::size_t>(order[index])]);
                    }
                }
                auto merged = base_routes[static_cast<std::size_t>(order[first])];
                const auto& tail =
                    base_routes[static_cast<std::size_t>(order[second])];
                merged.insert(merged.end(), tail.begin(), tail.end());
                candidate.push_back(std::move(merged));
                py::array_t<std::int64_t> plan_offsets(2);
                py::array_t<std::int64_t> route_offsets(candidate.size() + 1);
                checked_data(plan_offsets)[0] = 0;
                checked_data(plan_offsets)[1] =
                    static_cast<std::int64_t>(candidate.size());
                checked_data(route_offsets)[0] = 0;
                std::vector<std::int64_t> packed_indices;
                for (std::size_t route = 0; route < candidate.size(); ++route) {
                    packed_indices.insert(
                        packed_indices.end(),
                        candidate[route].begin(), candidate[route].end());
                    checked_data(route_offsets)[route + 1] =
                        static_cast<std::int64_t>(packed_indices.size());
                    evaluated_route_indices.insert(
                        evaluated_route_indices.end(),
                        candidate[route].begin(), candidate[route].end());
                    evaluated_route_offsets.push_back(
                        static_cast<std::int64_t>(evaluated_route_indices.size()));
                }
                evaluated_plan_offsets.push_back(
                    static_cast<std::int64_t>(evaluated_route_offsets.size() - 1));
                py::array_t<std::int64_t> route_indices(packed_indices.size());
                std::copy(
                    packed_indices.begin(), packed_indices.end(),
                    checked_data(route_indices));
                auto transaction = evaluate_plans(
                    plan_offsets, route_offsets, route_indices, context,
                    next_deadline(), batch_array, expected_array);
                auto feasible = py::cast<py::array_t<std::int64_t>>(
                    transaction[11]);
                if (feasible.size() == 0) {
                    continue;
                }
                auto objective =
                    py::cast<py::array_t<double>>(transaction[3]);
                const auto delta = checked_data<double>(objective)[0]
                    - current_distance;
                if (delta > 0.0) {
                    positive_deltas.push_back(delta);
                }
            }
            suppress_attempted_plan_journal_ = false;
            suppress_round_budget_ = false;
            restore_protocol_flags.release();
        }
        auto temperature = fallback;
        if (positive_deltas.size() >= 3) {
            PythonFloatSum total;
            for (const auto delta : positive_deltas) {
                total.add(delta);
            }
            const auto mean = total.value()
                / static_cast<double>(positive_deltas.size());
            if (mean > 0.0) {
                temperature = std::max(
                    1.0, -mean / std::log(stage04_temperature_target_));
            }
        }
        *rng_ = std::move(next_rng);
        stage04_initial_temperature_ = temperature;
        stage04_search_initialized_ = true;
        py::array_t<double> delta_array(positive_deltas.size());
        std::copy(
            positive_deltas.begin(), positive_deltas.end(),
            checked_data(delta_array));
        py::array_t<std::int64_t> plan_offsets_array(
            evaluated_plan_offsets.size());
        py::array_t<std::int64_t> route_offsets_array(
            evaluated_route_offsets.size());
        py::array_t<std::int64_t> route_indices_array(
            evaluated_route_indices.size());
        std::copy(
            evaluated_plan_offsets.begin(), evaluated_plan_offsets.end(),
            checked_data(plan_offsets_array));
        std::copy(
            evaluated_route_offsets.begin(), evaluated_route_offsets.end(),
            checked_data(route_offsets_array));
        std::copy(
            evaluated_route_indices.begin(), evaluated_route_indices.end(),
            checked_data(route_indices_array));
        return py::make_tuple(
            temperature, std::move(delta_array), std::move(plan_offsets_array),
            std::move(route_offsets_array), std::move(route_indices_array));
    }

    void record_constraint_stage04_outcome(
        std::int64_t iteration,
        std::int64_t operation,
        bool accepted,
        std::int64_t comparison,
        bool is_global_best,
        bool vehicle_reduction) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!stage04_configured_) {
            throw std::runtime_error(
                "full native Stage 4 must be configured before recording an outcome");
        }
        if (iteration < 0 || operation < 0 || operation >= 4
            || comparison < -1 || comparison > 1) {
            throw std::invalid_argument(
                "full native Stage 4 constraint outcome is invalid");
        }
        if (iteration <= last_finished_stage04_iteration_) {
            throw std::invalid_argument(
                "full native Stage 4 iteration is already finished");
        }
        if (iteration != last_finished_stage04_iteration_ + 1) {
            throw std::invalid_argument(
                "full native Stage 4 outcome belongs to a future iteration");
        }
        if ((!accepted && (is_global_best || vehicle_reduction))
            || (accepted && is_global_best && comparison >= 0)
            || (accepted && vehicle_reduction && comparison >= 0)) {
            throw std::invalid_argument(
                "full native Stage 4 constraint outcome flags are inconsistent");
        }
        if (last_candidate_ready_ || pending_composite_active_) {
            throw std::runtime_error(
                "full native Stage 4 outcome cannot cross a candidate transaction");
        }
        auto next_rewards = constraint_segment_rewards_;
        auto next_calls = constraint_segment_calls_;
        auto next_totals = constraint_totals_;
        accumulate_constraint_stage04_outcome_noexcept(
            static_cast<std::size_t>(operation), accepted, comparison,
            is_global_best, vehicle_reduction,
            next_rewards, next_calls, next_totals);
        constraint_segment_rewards_ = next_rewards;
        constraint_segment_calls_ = next_calls;
        constraint_totals_ = next_totals;
    }

    py::tuple constraint_stage04_state() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!stage04_configured_) {
            throw std::runtime_error(
                "full native Stage 4 must be configured before state inspection");
        }
        py::array_t<double> weights(4);
        py::array_t<double> reward_sums(4);
        py::array_t<std::int64_t> segment_calls(4);
        py::array_t<std::int64_t> totals({py::ssize_t(4), py::ssize_t(8)});
        std::copy(
            constraint_weights_.begin(), constraint_weights_.end(),
            checked_data(weights));
        std::copy(
            constraint_segment_rewards_.begin(),
            constraint_segment_rewards_.end(),
            checked_data(reward_sums));
        std::copy(
            constraint_segment_calls_.begin(), constraint_segment_calls_.end(),
            checked_data(segment_calls));
        for (std::size_t operation = 0; operation < 4; ++operation) {
            std::copy(
                constraint_totals_[operation].begin(),
                constraint_totals_[operation].end(),
                checked_data(totals) + operation * 8);
        }
        return py::make_tuple(
            std::move(weights), std::move(reward_sums),
            std::move(segment_calls), std::move(totals));
    }

    py::tuple full_stage04_state() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!stage04_configured_) {
            throw std::runtime_error(
                "full native Stage 4 must be configured before state inspection");
        }
        py::array_t<double> weights(full_operator_weights_.size());
        py::array_t<double> reward_sums(full_operator_segment_rewards_.size());
        py::array_t<std::int64_t> segment_calls(full_operator_segment_calls_.size());
        py::array_t<std::int64_t> totals({
            static_cast<py::ssize_t>(full_operator_totals_.size()),
            py::ssize_t(8)});
        std::copy(
            full_operator_weights_.begin(), full_operator_weights_.end(),
            checked_data(weights));
        std::copy(
            full_operator_segment_rewards_.begin(),
            full_operator_segment_rewards_.end(), checked_data(reward_sums));
        std::copy(
            full_operator_segment_calls_.begin(),
            full_operator_segment_calls_.end(), checked_data(segment_calls));
        for (std::size_t operation = 0;
             operation < full_operator_totals_.size(); ++operation) {
            std::copy(
                full_operator_totals_[operation].begin(),
                full_operator_totals_[operation].end(),
                checked_data(totals) + operation * 8);
        }
        return py::make_tuple(
            std::move(weights), std::move(reward_sums),
            std::move(segment_calls), std::move(totals));
    }

    py::tuple finish_stage04_iteration(
        std::int64_t iteration,
        bool budget_boundary) {
        return finish_stage04_iteration_impl(iteration, budget_boundary, true);
    }

    py::tuple finish_stage04_iteration_impl(
        std::int64_t iteration,
        bool budget_boundary,
        bool advance_intensification) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!stage04_configured_ || iteration < 0) {
            throw std::invalid_argument(
                "full native Stage 4 boundary state is invalid");
        }
        if (iteration != last_finished_stage04_iteration_ + 1) {
            throw std::invalid_argument(
                "full native Stage 4 iterations must finish exactly once in order");
        }
        if (last_candidate_ready_
            || (pending_composite_active_ && !defer_global_commit_)) {
            throw std::runtime_error(
                "full native Stage 4 boundary cannot cross a candidate transaction");
        }
        py::array_t<std::int64_t> statuses(4);
        py::array_t<double> old_new_weights(
            {py::ssize_t(4), py::ssize_t(2)});
        py::array_t<std::int64_t> calls_at_boundary(4);
        py::array_t<double> rewards_at_boundary(4);
        auto* status_values = checked_data(statuses);
        auto* weight_values = checked_data(old_new_weights);
        std::copy(
            constraint_segment_calls_.begin(), constraint_segment_calls_.end(),
            checked_data(calls_at_boundary));
        std::copy(
            constraint_segment_rewards_.begin(),
            constraint_segment_rewards_.end(),
            checked_data(rewards_at_boundary));
        auto next_weights = constraint_weights_;
        auto next_rewards = constraint_segment_rewards_;
        auto next_calls = constraint_segment_calls_;
        auto next_full_weights = full_operator_weights_;
        auto next_full_rewards = full_operator_segment_rewards_;
        auto next_full_calls = full_operator_segment_calls_;
        const auto is_boundary =
            (iteration + 1) % stage04_segment_length_ == 0;
        for (std::size_t index = 0; index < 4; ++index) {
            status_values[index] = -1;
            weight_values[index * 2] = constraint_weights_[index];
            weight_values[index * 2 + 1] = constraint_weights_[index];
        }
        const auto effective_budget_boundary =
            budget_boundary || budget_.budget_reached();
        if (is_boundary && !effective_budget_boundary
            && !stage04_fixed_weights_) {
            for (std::size_t index = 0; index < 4; ++index) {
                if (next_calls[index] >= stage04_min_calls_) {
                    const auto average = next_rewards[index]
                        / static_cast<double>(next_calls[index]);
                    const auto reacted = std::max(
                        stage04_weight_floor_,
                        (1.0 - stage04_weight_reaction_) * next_weights[index]
                            + stage04_weight_reaction_ * average);
                    next_weights[index] = stage04_weight_smoothing_
                            * next_weights[index]
                        + (1.0 - stage04_weight_smoothing_) * reacted;
                    status_values[index] = 1;
                    weight_values[index * 2 + 1] = next_weights[index];
                } else {
                    status_values[index] = 0;
                }
                next_rewards[index] = 0.0;
                next_calls[index] = 0;
            }
            for (std::size_t index = 0; index < next_full_weights.size(); ++index) {
                if (next_full_calls[index] >= stage04_min_calls_) {
                    const auto average = next_full_rewards[index]
                        / static_cast<double>(next_full_calls[index]);
                    const auto reacted = std::max(
                        stage04_weight_floor_,
                        (1.0 - stage04_weight_reaction_) * next_full_weights[index]
                            + stage04_weight_reaction_ * average);
                    next_full_weights[index] = stage04_weight_smoothing_
                            * next_full_weights[index]
                        + (1.0 - stage04_weight_smoothing_) * reacted;
                }
                next_full_rewards[index] = 0.0;
                next_full_calls[index] = 0;
            }
        }
        bool reheat_triggered = false;
        if (stage04_reheat_enabled_
            && stage04_reheat_count_ < stage04_max_reheats_
            && main_stagnation_iterations_
                >= stage04_reheat_stagnation_threshold_) {
            stage04_reheat_floor_ =
                stage04_initial_temperature_ * stage04_reheat_factor_;
            ++stage04_reheat_count_;
            reheat_triggered = true;
        }
        bool restart_triggered = false;
        if (stage04_restart_enabled_
            && stage04_restart_count_ < stage04_max_restarts_
            && main_stagnation_iterations_
                >= stage04_restart_stagnation_threshold_) {
            legacy_offsets_ = owned_array_copy<std::int64_t>(
                best_offsets_, "restart_best_offsets", 1);
            legacy_indices_ = owned_array_copy<std::int64_t>(
                best_indices_, "restart_best_indices", 1);
            legacy_exact_payload_ = owned_exact_state_copy(best_exact_payload_);
            legacy_objective_integer_ = owned_array_copy<std::int64_t>(
                best_objective_integer_, "restart_best_objective_integer", 1);
            legacy_objective_float_ = owned_array_copy<double>(
                best_objective_float_, "restart_best_objective_float", 1);
            main_stagnation_iterations_ = 0;
            ++stage04_restart_count_;
            stage04_reheat_floor_ =
                stage04_initial_temperature_ * stage04_reheat_factor_;
            if (stage04_intensification_enabled_
                && !stage04_intensification_active_) {
                stage04_intensification_active_ = true;
                stage04_intensification_remaining_ =
                    stage04_intensification_iterations_;
            }
            restart_triggered = true;
        }
        if (stage04_intensification_active_ && advance_intensification) {
            if (stage04_intensification_remaining_ > 0) {
                --stage04_intensification_remaining_;
            } else {
                stage04_intensification_active_ = false;
            }
        }
        py::array_t<std::int64_t> control_status(7);
        checked_data(control_status)[0] = reheat_triggered ? 1 : 0;
        checked_data(control_status)[1] = stage04_reheat_count_;
        checked_data(control_status)[2] = restart_triggered ? 1 : 0;
        checked_data(control_status)[3] = stage04_restart_count_;
        checked_data(control_status)[4] =
            stage04_intensification_active_ ? 1 : 0;
        checked_data(control_status)[5] = stage04_intensification_remaining_;
        checked_data(control_status)[6] = main_stagnation_iterations_;
        py::array_t<double> control_float(1);
        checked_data(control_float)[0] = stage04_reheat_floor_;
        auto result = py::make_tuple(
            std::move(statuses), std::move(old_new_weights),
            std::move(calls_at_boundary), std::move(rewards_at_boundary),
            std::move(control_status), std::move(control_float));
        constraint_weights_ = next_weights;
        constraint_segment_rewards_ = next_rewards;
        constraint_segment_calls_ = next_calls;
        full_operator_weights_ = next_full_weights;
        full_operator_segment_rewards_ = next_full_rewards;
        full_operator_segment_calls_ = next_full_calls;
        last_finished_stage04_iteration_ = iteration;
        return result;
    }

    py::tuple constraint_iteration(
        std::int64_t iteration,
        std::int64_t stagnation_iterations,
        bool global_best_reset,
        py::handle thresholds,
        py::handle fractions,
        py::handle deadline_remaining,
        py::handle batch_size,
        std::int64_t route_change_limit) {
        const auto iteration_started = std::chrono::steady_clock::now();
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !constraint_rng_.has_value()) {
            throw std::runtime_error(
                "full native search engine must be initialized before an iteration");
        }
        if (!stage04_configured_) {
            throw std::runtime_error(
                "full native Stage 4 must be configured before an iteration");
        }
        if (last_candidate_ready_) {
            throw std::runtime_error(
                "full native search engine has an unapplied candidate");
        }
        if (iteration < 0 || stagnation_iterations < 0) {
            throw std::invalid_argument(
                "full native constraint iteration values cannot be negative");
        }
        if (iteration <= last_finished_stage04_iteration_) {
            throw std::invalid_argument(
                "full native Stage 4 iteration is already finished");
        }
        if (iteration != last_finished_stage04_iteration_ + 1) {
            throw std::invalid_argument(
                "full native constraint iteration is not the active global iteration");
        }
        if (iteration == last_completed_constraint_iteration_) {
            throw std::invalid_argument(
                "full native constraint iteration already completed");
        }
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        if (deadline_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0) {
            throw std::invalid_argument(
                "full native constraint iteration deadline is invalid");
        }
        const auto deadline_seconds = checked_data<double>(deadline_array)[0];
        const auto remaining_at_boundary = [&]() {
            return deadline_seconds - std::chrono::duration<double>(
                std::chrono::steady_clock::now() - iteration_started).count();
        };
        const auto require_deadline = [&]() {
            if (remaining_at_boundary() <= 0.0) {
                throw std::runtime_error(
                    "full native constraint iteration reached its deadline");
            }
        };
        auto next_constraint_rng = *constraint_rng_;
        auto next_constraint_weights = constraint_weights_;
        auto next_segment_rewards = constraint_segment_rewards_;
        auto next_segment_calls = constraint_segment_calls_;
        auto next_totals = constraint_totals_;
        const auto operation = iteration < 4
            ? iteration
            : static_cast<std::int64_t>(next_constraint_rng.weighted_index(
                std::vector<double>(next_constraint_weights.begin(),
                                    next_constraint_weights.end())));
        constexpr std::array<std::string_view, 4> operation_names{
            "station_pressure",
            "time_window_conflict",
            "worst_energy_detour",
            "shaw_related",
        };
        auto selection = dynamic_removal_selection_v2(
            static_cast<std::int64_t>(all_customers_.size()),
            stagnation_iterations,
            iteration,
            thresholds,
            fractions,
            global_best_reset);
        // Python reserves a per-probe seed immediately after selecting the
        // constraint operator, even when the dynamic removal count is zero.
        // Consume it on every path so later weighted operator draws remain
        // byte-for-byte aligned with random.Random.
        const auto probe_seed = next_constraint_rng.randbelow(1ULL << 32U);
        const auto requested_count = checked_data<std::int64_t>(selection)[1];
        if (requested_count <= 0) {
            auto removal = constraint_removal_v2(
                operation,
                node_kind_,
                demand_,
                ready_time_,
                due_date_,
                service_time_,
                distance_,
                reachable_,
                vehicle_,
                lexical_rank_,
                current_offsets_,
                current_indices_,
                current_exact_payload_[0],
                current_exact_payload_[1],
                current_exact_payload_[4],
                requested_count,
                0);
            py::array_t<std::int64_t> outcome(6);
            std::fill(
                checked_data(outcome), checked_data(outcome) + 6,
                std::int64_t{0});
            checked_data(outcome)[0] = operation;
            auto probe = py::make_tuple(
                std::move(removal), py::none(), py::none());
            causal_context_ = {
                stable_int63("constraint_lane"),
                stable_int63(
                    operation_names[static_cast<std::size_t>(operation)]),
                iteration};
            causal_context_transaction_id_ = allocate_causal_transaction();
            accumulate_constraint_stage04_outcome_noexcept(
                static_cast<std::size_t>(operation), false, 1, false, false,
                next_segment_rewards, next_segment_calls, next_totals);
            *constraint_rng_ = std::move(next_constraint_rng);
            constraint_weights_ = next_constraint_weights;
            constraint_segment_rewards_ = next_segment_rewards;
            constraint_segment_calls_ = next_segment_calls;
            constraint_totals_ = next_totals;
            accumulate_full_stage04_outcome_noexcept(
                static_cast<std::size_t>(operation + 9), false, 1, false,
                false, true);
            last_completed_constraint_iteration_ = iteration;
            return py::make_tuple(
                std::move(selection), std::move(probe), std::move(outcome));
        }
        py::array_t<std::int64_t> outcome(6);
        auto* outcome_values = checked_data(outcome);
        outcome_values[0] = operation;
        outcome_values[1] = static_cast<std::int64_t>(probe_seed);
        outcome_values[2] = 0;
        outcome_values[3] = 0;
        outcome_values[4] = 0;
        outcome_values[5] = 0;
        py::tuple result(3);
        result[0] = selection;
        result[2] = outcome;
        py::array_t<std::int64_t> context_ids(3);
        auto* context = checked_data(context_ids);
        context[0] = stable_int63("constraint_lane");
        context[1] = stable_int63(
            operation_names[static_cast<std::size_t>(operation)]);
        context[2] = iteration;
        py::array_t<double> adjusted_deadline(1);
        require_deadline();
        checked_data(adjusted_deadline)[0] = remaining_at_boundary();
        defer_iteration_commit_ = true;
        bool iteration_candidate_ready = false;
        try {
            auto probe = constraint_probe(
                operation,
                requested_count,
                probe_seed,
                context_ids,
                adjusted_deadline,
                batch_size,
                route_change_limit);
            result[1] = probe;
            iteration_candidate_ready = last_candidate_ready_;
            if (iteration_candidate_ready) {
                const auto candidate_is_incumbent =
                    last_candidate_offsets_.size() == current_offsets_.size()
                    && last_candidate_indices_.size() == current_indices_.size()
                    && std::equal(
                        checked_data<std::int64_t>(last_candidate_offsets_),
                        checked_data<std::int64_t>(last_candidate_offsets_)
                            + last_candidate_offsets_.size(),
                        checked_data<std::int64_t>(current_offsets_))
                    && std::equal(
                        checked_data<std::int64_t>(last_candidate_indices_),
                        checked_data<std::int64_t>(last_candidate_indices_)
                            + last_candidate_indices_.size(),
                        checked_data<std::int64_t>(current_indices_));
                if (candidate_is_incumbent) {
                    last_candidate_ready_ = false;
                    iteration_candidate_ready = false;
                }
            }
            outcome_values[2] = iteration_candidate_ready ? 1 : 0;
            if (constraint_iteration_deadline_injection_) {
                constraint_iteration_deadline_injection_ = false;
                throw std::runtime_error(
                    "injected full native constraint-iteration deadline before commit");
            }
            require_deadline();
            std::int64_t comparison = 1;
            if (iteration_candidate_ready) {
                const auto candidate_key =
                    evrptw::formal_objective::key_from_arrays(
                    last_candidate_objective_integer_,
                    last_candidate_objective_float_);
                const auto current_key =
                    evrptw::formal_objective::key_from_arrays(
                    current_objective_integer_, current_objective_float_);
                comparison = candidate_key < current_key
                    ? -1
                    : candidate_key == current_key ? 0 : 1;
            }
            if (iteration_candidate_ready && !budget_.budget_reached()) {
                const auto current_distance =
                    checked_data<double>(current_objective_float_)[0];
                const auto applied = apply_last_candidate(
                    std::max(1.0, current_distance * 0.05), 0.0);
                outcome_values[2] = 1;
                outcome_values[3] = PyLong_AS_LONG(applied[0].ptr());
                outcome_values[4] = PyLong_AS_LONG(applied[1].ptr());
                outcome_values[5] = PyLong_AS_LONG(applied[2].ptr());
            } else if (iteration_candidate_ready) {
                last_candidate_ready_ = false;
            }
            const auto accepted = outcome_values[3] != 0;
            const auto improved_best = outcome_values[4] != 0;
            const auto vehicle_reduction = outcome_values[5] != 0;
            accumulate_constraint_stage04_outcome_noexcept(
                static_cast<std::size_t>(operation), accepted, comparison,
                improved_best, vehicle_reduction,
                next_segment_rewards, next_segment_calls, next_totals);
            if (!defer_global_commit_) {
                commit_pending_composite_noexcept();
            }
            defer_iteration_commit_ = false;
            *constraint_rng_ = std::move(next_constraint_rng);
            constraint_weights_ = next_constraint_weights;
            constraint_segment_rewards_ = next_segment_rewards;
            constraint_segment_calls_ = next_segment_calls;
            constraint_totals_ = next_totals;
            accumulate_full_stage04_outcome_noexcept(
                static_cast<std::size_t>(operation + 9), accepted, comparison,
                improved_best, vehicle_reduction, true);
            last_completed_constraint_iteration_ = iteration;
            return result;
        } catch (...) {
            defer_iteration_commit_ = false;
            if (pending_composite_active_) {
                rollback_pending_composite();
            }
            if (iteration_candidate_ready) {
                last_candidate_ready_ = false;
            }
            throw;
        }
    }

    py::tuple run_constraint_search(
        std::int64_t start_iteration,
        std::int64_t iteration_count,
        std::int64_t initial_stagnation_iterations,
        py::handle thresholds,
        py::handle fractions,
        py::handle deadline_remaining,
        py::handle batch_size,
        std::int64_t route_change_limit) {
        const auto search_started = std::chrono::steady_clock::now();
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !stage04_configured_) {
            throw std::runtime_error(
                "full native search engine must be initialized and configured");
        }
        if (start_iteration < 0 || iteration_count <= 0
            || initial_stagnation_iterations < 0) {
            throw std::invalid_argument(
                "full native constraint-search iteration values are invalid");
        }
        if (start_iteration != last_finished_stage04_iteration_ + 1) {
            throw std::invalid_argument(
                "full native constraint search does not start at the active iteration");
        }
        if (start_iteration + iteration_count > 4) {
            throw std::invalid_argument(
                "constraint-only search beyond the bootstrap rounds requires the native global controller");
        }
        auto threshold_array = owned_array_copy<std::int64_t>(
            thresholds, "thresholds", 1);
        auto fraction_array = owned_array_copy<double>(fractions, "fractions", 1);
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "batch_size", 1);
        if (threshold_array.size() != 3 || fraction_array.size() != 6
            || deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0
            || route_change_limit == 0 || route_change_limit < -1) {
            throw std::invalid_argument(
                "full native constraint-search arrays are invalid");
        }

        const auto total_deadline = checked_data<double>(deadline_array)[0];
        const auto entry_budget = budget_.native_snapshot();
        py::array_t<std::int64_t> event_integer(
            {static_cast<py::ssize_t>(iteration_count), py::ssize_t(16)});
        py::array_t<std::int64_t> event_objective_integer(
            {static_cast<py::ssize_t>(iteration_count), py::ssize_t(6)});
        py::array_t<double> event_objective(
            {static_cast<py::ssize_t>(iteration_count), py::ssize_t(6)});
        py::array_t<std::uint8_t> candidate_hashes(
            {static_cast<py::ssize_t>(iteration_count), py::ssize_t(32)});
        py::array_t<std::int64_t> stage04_status(
            {static_cast<py::ssize_t>(iteration_count), py::ssize_t(4)});
        py::array_t<double> stage04_weights(
            {static_cast<py::ssize_t>(iteration_count),
             py::ssize_t(4), py::ssize_t(2)});
        py::array_t<std::int64_t> stage04_calls(
            {static_cast<py::ssize_t>(iteration_count), py::ssize_t(4)});
        py::array_t<double> stage04_rewards(
            {static_cast<py::ssize_t>(iteration_count), py::ssize_t(4)});
        std::fill(
            checked_data(candidate_hashes),
            checked_data(candidate_hashes) + iteration_count * 32,
            std::uint8_t{0});
        std::vector<std::int64_t> plan_offsets{0};
        std::vector<std::int64_t> route_offsets{0};
        std::vector<std::int64_t> route_indices;
        plan_offsets.reserve(static_cast<std::size_t>(iteration_count) + 1);

        auto stagnation_iterations = initial_stagnation_iterations;
        std::int64_t completed_iterations = 0;
        std::int64_t termination_reason = 0;
        const auto injected_deadline_after =
            constraint_search_deadline_after_completed_injection_;
        constraint_search_deadline_after_completed_injection_ = -1;
        for (std::int64_t ordinal = 0; ordinal < iteration_count; ++ordinal) {
            if (budget_.budget_reached()) {
                termination_reason = 1;
                break;
            }
            if (injected_deadline_after == completed_iterations) {
                termination_reason = 2;
                break;
            }
            const auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - search_started).count();
            const auto remaining = total_deadline - elapsed;
            if (remaining <= 0.0) {
                termination_reason = 2;
                break;
            }
            py::array_t<double> adjusted_deadline(1);
            checked_data(adjusted_deadline)[0] = remaining;
            const auto before_budget = budget_.native_snapshot();
            const auto iteration = start_iteration + ordinal;
            py::tuple iteration_payload;
            try {
                iteration_payload = constraint_iteration(
                    iteration,
                    stagnation_iterations,
                    false,
                    threshold_array,
                    fraction_array,
                    adjusted_deadline,
                    batch_array,
                    route_change_limit);
            } catch (const std::runtime_error& error) {
                if (std::string_view(error.what()).find("deadline")
                    == std::string_view::npos) {
                    throw;
                }
                termination_reason = 2;
                break;
            }
            auto selection = py::cast<py::array_t<std::int64_t>>(
                iteration_payload[0]);
            auto probe = py::cast<py::tuple>(iteration_payload[1]);
            auto outcome = py::cast<py::array_t<std::int64_t>>(
                iteration_payload[2]);
            const auto* selection_values = checked_data<std::int64_t>(selection);
            const auto* outcome_values = checked_data<std::int64_t>(outcome);
            const auto after_budget = budget_.native_snapshot();

            auto* event = checked_data(event_integer) + ordinal * 16;
            event[0] = 0;  // candidate_state
            event[1] = 2;  // constraint_lane
            event[2] = iteration;
            event[3] = 9 + outcome_values[0];
            event[4] = selection_values[1];
            event[5] = selection_values[2];
            event[6] = outcome_values[2];
            event[7] = outcome_values[3];
            event[8] = outcome_values[4];
            event[9] = outcome_values[5];
            event[10] = after_budget.started - before_budget.started;
            event[11] = after_budget.completed - before_budget.completed;
            event[12] = after_budget.interrupted - before_budget.interrupted;
            event[13] = 3;
            event[14] = current_offsets_.size() - 1;
            event[15] = best_offsets_.size() - 1;

            std::int64_t appended_routes = 0;
            if (!probe[1].is_none()) {
                auto repair = py::cast<py::tuple>(probe[1]);
                auto repair_offsets = py::cast<py::array_t<std::int64_t>>(repair[0]);
                auto repair_indices = py::cast<py::array_t<std::int64_t>>(repair[1]);
                auto repair_metadata = py::cast<py::array_t<std::int64_t>>(repair[2]);
                const auto* repair_boundaries =
                    checked_data<std::int64_t>(repair_offsets);
                const auto* repair_nodes = checked_data<std::int64_t>(repair_indices);
                if (checked_data<std::int64_t>(repair_metadata)[0] == 0) {
                    appended_routes = repair_offsets.size() - 1;
                    for (py::ssize_t route = 0; route < appended_routes; ++route) {
                        route_indices.insert(
                            route_indices.end(),
                            repair_nodes + repair_boundaries[route],
                            repair_nodes + repair_boundaries[route + 1]);
                        route_offsets.push_back(
                            static_cast<std::int64_t>(route_indices.size()));
                    }
                    event[13] = outcome_values[2] != 0 ? 0 : 3;
                } else {
                    event[13] = 2;
                }
            } else {
                auto removal = py::cast<py::tuple>(probe[0]);
                auto removal_metadata =
                    py::cast<py::array_t<std::int64_t>>(removal[6]);
                event[13] = checked_data<std::int64_t>(removal_metadata)[0] != 0
                    ? 1 : 3;
            }
            plan_offsets.push_back(plan_offsets.back() + appended_routes);

            std::string candidate_evidence(
                "stage05.2-native-candidate-route-identity-v2");
            const auto first_route = plan_offsets[
                static_cast<std::size_t>(ordinal)];
            const auto last_route = plan_offsets[
                static_cast<std::size_t>(ordinal + 1)];
            for (auto route = first_route; route < last_route; ++route) {
                const auto begin = route_offsets[static_cast<std::size_t>(route)];
                const auto end = route_offsets[static_cast<std::size_t>(route + 1)];
                append_evidence_values(
                    candidate_evidence,
                    route_indices.data() + begin,
                    static_cast<std::size_t>(end - begin));
            }
            const auto candidate_digest = native_sha256_digest(candidate_evidence);
            std::copy(
                candidate_digest.begin(), candidate_digest.end(),
                checked_data(candidate_hashes) + ordinal * 32);

            auto* objective_integers =
                checked_data(event_objective_integer) + ordinal * 6;
            auto* objectives = checked_data(event_objective) + ordinal * 6;
            objective_integers[0] = -1;
            objective_integers[1] = -1;
            objectives[0] = std::numeric_limits<double>::quiet_NaN();
            objectives[1] = std::numeric_limits<double>::quiet_NaN();
            if (!probe[2].is_none() && outcome_values[2] != 0) {
                auto transaction = py::cast<py::tuple>(probe[2]);
                auto candidate_integers =
                    py::cast<py::array_t<std::int64_t>>(transaction[2]);
                auto candidate_floats =
                    py::cast<py::array_t<double>>(transaction[3]);
                objective_integers[0] =
                    checked_data<std::int64_t>(candidate_integers)[0];
                objective_integers[1] =
                    checked_data<std::int64_t>(candidate_integers)[1];
                objectives[0] = checked_data<double>(candidate_floats)[0];
                objectives[1] = checked_data<double>(candidate_floats)[1];
            }
            objective_integers[2] =
                checked_data<std::int64_t>(current_objective_integer_)[0];
            objective_integers[3] =
                checked_data<std::int64_t>(current_objective_integer_)[1];
            objective_integers[4] =
                checked_data<std::int64_t>(best_objective_integer_)[0];
            objective_integers[5] =
                checked_data<std::int64_t>(best_objective_integer_)[1];
            objectives[2] = checked_data<double>(current_objective_float_)[0];
            objectives[3] = checked_data<double>(current_objective_float_)[1];
            objectives[4] = checked_data<double>(best_objective_float_)[0];
            objectives[5] = checked_data<double>(best_objective_float_)[1];

            auto stage_boundary = finish_stage04_iteration(
                iteration, budget_.budget_reached());
            auto statuses = py::cast<py::array_t<std::int64_t>>(stage_boundary[0]);
            auto weights = py::cast<py::array_t<double>>(stage_boundary[1]);
            auto calls = py::cast<py::array_t<std::int64_t>>(stage_boundary[2]);
            auto rewards = py::cast<py::array_t<double>>(stage_boundary[3]);
            std::copy(
                checked_data<std::int64_t>(statuses),
                checked_data<std::int64_t>(statuses) + 4,
                checked_data(stage04_status) + ordinal * 4);
            std::copy(
                checked_data<double>(weights),
                checked_data<double>(weights) + 8,
                checked_data(stage04_weights) + ordinal * 8);
            std::copy(
                checked_data<std::int64_t>(calls),
                checked_data<std::int64_t>(calls) + 4,
                checked_data(stage04_calls) + ordinal * 4);
            std::copy(
                checked_data<double>(rewards),
                checked_data<double>(rewards) + 4,
                checked_data(stage04_rewards) + ordinal * 4);
            stagnation_iterations = outcome_values[4] != 0
                ? 0 : stagnation_iterations + 1;
            ++completed_iterations;
            if (budget_.budget_reached()) {
                termination_reason = 1;
                break;
            }
        }

        if (completed_iterations != iteration_count) {
            py::array_t<std::int64_t> trimmed_event_integer(
                {static_cast<py::ssize_t>(completed_iterations), py::ssize_t(16)});
            py::array_t<std::int64_t> trimmed_event_objective_integer(
                {static_cast<py::ssize_t>(completed_iterations), py::ssize_t(6)});
            py::array_t<double> trimmed_event_objective(
                {static_cast<py::ssize_t>(completed_iterations), py::ssize_t(6)});
            py::array_t<std::uint8_t> trimmed_candidate_hashes(
                {static_cast<py::ssize_t>(completed_iterations), py::ssize_t(32)});
            py::array_t<std::int64_t> trimmed_stage04_status(
                {static_cast<py::ssize_t>(completed_iterations), py::ssize_t(4)});
            py::array_t<double> trimmed_stage04_weights(
                {static_cast<py::ssize_t>(completed_iterations),
                 py::ssize_t(4), py::ssize_t(2)});
            py::array_t<std::int64_t> trimmed_stage04_calls(
                {static_cast<py::ssize_t>(completed_iterations), py::ssize_t(4)});
            py::array_t<double> trimmed_stage04_rewards(
                {static_cast<py::ssize_t>(completed_iterations), py::ssize_t(4)});
            std::copy(
                checked_data(event_integer),
                checked_data(event_integer) + completed_iterations * 16,
                checked_data(trimmed_event_integer));
            std::copy(
                checked_data(event_objective_integer),
                checked_data(event_objective_integer) + completed_iterations * 6,
                checked_data(trimmed_event_objective_integer));
            std::copy(
                checked_data(event_objective),
                checked_data(event_objective) + completed_iterations * 6,
                checked_data(trimmed_event_objective));
            std::copy(
                checked_data(candidate_hashes),
                checked_data(candidate_hashes) + completed_iterations * 32,
                checked_data(trimmed_candidate_hashes));
            std::copy(
                checked_data(stage04_status),
                checked_data(stage04_status) + completed_iterations * 4,
                checked_data(trimmed_stage04_status));
            std::copy(
                checked_data(stage04_weights),
                checked_data(stage04_weights) + completed_iterations * 8,
                checked_data(trimmed_stage04_weights));
            std::copy(
                checked_data(stage04_calls),
                checked_data(stage04_calls) + completed_iterations * 4,
                checked_data(trimmed_stage04_calls));
            std::copy(
                checked_data(stage04_rewards),
                checked_data(stage04_rewards) + completed_iterations * 4,
                checked_data(trimmed_stage04_rewards));
            event_integer = std::move(trimmed_event_integer);
            event_objective_integer =
                std::move(trimmed_event_objective_integer);
            event_objective = std::move(trimmed_event_objective);
            candidate_hashes = std::move(trimmed_candidate_hashes);
            stage04_status = std::move(trimmed_stage04_status);
            stage04_weights = std::move(trimmed_stage04_weights);
            stage04_calls = std::move(trimmed_stage04_calls);
            stage04_rewards = std::move(trimmed_stage04_rewards);
        }

        py::array_t<std::int64_t> plan_offsets_array(plan_offsets.size());
        py::array_t<std::int64_t> route_offsets_array(route_offsets.size());
        py::array_t<std::int64_t> route_indices_array(route_indices.size());
        std::copy(
            plan_offsets.begin(), plan_offsets.end(),
            checked_data(plan_offsets_array));
        std::copy(
            route_offsets.begin(), route_offsets.end(),
            checked_data(route_offsets_array));
        std::copy(
            route_indices.begin(), route_indices.end(),
            checked_data(route_indices_array));
        py::array_t<std::int64_t> termination(13);
        const auto terminal_budget = budget_.native_snapshot();
        auto* terminal_values = checked_data(termination);
        terminal_values[0] = termination_reason;
        terminal_values[1] = completed_iterations;
        terminal_values[2] = iteration_count;
        terminal_values[3] = budget_.exact_budget_;
        terminal_values[4] = entry_budget.started;
        terminal_values[5] = entry_budget.completed;
        terminal_values[6] = entry_budget.interrupted;
        terminal_values[7] = terminal_budget.started;
        terminal_values[8] = terminal_budget.completed;
        terminal_values[9] = terminal_budget.interrupted;
        std::array<std::int64_t, 3> event_exact_deltas{};
        for (std::int64_t event = 0; event < completed_iterations; ++event) {
            for (std::int64_t counter = 0; counter < 3; ++counter) {
                event_exact_deltas[static_cast<std::size_t>(counter)] +=
                    checked_data<std::int64_t>(event_integer)[
                        event * 16 + 10 + counter];
            }
        }
        terminal_values[10] = terminal_budget.started - entry_budget.started
            - event_exact_deltas[0];
        terminal_values[11] = terminal_budget.completed - entry_budget.completed
            - event_exact_deltas[1];
        terminal_values[12] = terminal_budget.interrupted - entry_budget.interrupted
            - event_exact_deltas[2];
        std::string evidence("stage05.2-native-constraint-semantic-stream-v2");
        append_evidence_array(evidence, event_integer);
        append_evidence_array(evidence, event_objective_integer);
        append_evidence_array(evidence, event_objective);
        append_evidence_array(evidence, plan_offsets_array);
        append_evidence_array(evidence, route_offsets_array);
        append_evidence_array(evidence, route_indices_array);
        append_evidence_array(evidence, candidate_hashes);
        append_evidence_array(evidence, stage04_status);
        append_evidence_array(evidence, stage04_weights);
        append_evidence_array(evidence, stage04_calls);
        append_evidence_array(evidence, stage04_rewards);
        append_evidence_array(evidence, termination);
        return py::make_tuple(
            std::move(event_integer), std::move(event_objective_integer),
            std::move(event_objective),
            std::move(plan_offsets_array), std::move(route_offsets_array),
            std::move(route_indices_array), std::move(candidate_hashes),
            std::move(stage04_status), std::move(stage04_weights),
            std::move(stage04_calls), std::move(stage04_rewards),
            std::move(termination),
            native_sha256_hex(evidence));
    }

    py::tuple run_global_search(
        std::int64_t start_iteration,
        std::int64_t iteration_count,
        std::int64_t initial_stagnation_iterations,
        py::handle thresholds,
        py::handle fractions,
        py::handle deadline_remaining,
        py::handle batch_size,
        std::int64_t route_change_limit) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || !stage04_configured_) {
            throw std::runtime_error(
                "full native search engine must be initialized and configured");
        }
        if (start_iteration < 0 || iteration_count != 1
            || (start_iteration == 0
                && (last_finished_stage04_iteration_ != -1
                    || current_offsets_.size() != 2))
            || (start_iteration > 0
                && last_finished_stage04_iteration_ != start_iteration - 1)) {
            throw std::invalid_argument(
                "native global controller iteration is not the active one-route round");
        }
        if (initial_stagnation_iterations < 0) {
            throw std::invalid_argument(
                "native global controller stagnation cannot be negative");
        }
        auto threshold_array = owned_array_copy<std::int64_t>(
            thresholds, "thresholds", 1);
        auto fraction_array = owned_array_copy<double>(fractions, "fractions", 1);
        auto deadline_array = owned_array_copy<double>(
            deadline_remaining, "deadline_remaining", 1);
        auto batch_array = owned_array_copy<std::int64_t>(
            batch_size, "batch_size", 1);
        if (threshold_array.size() != 3 || fraction_array.size() != 6
            || deadline_array.size() != 1 || batch_array.size() != 1
            || !std::isfinite(checked_data<double>(deadline_array)[0])
            || checked_data<double>(deadline_array)[0] <= 0.0
            || checked_data<std::int64_t>(batch_array)[0] <= 0) {
            throw std::invalid_argument(
                "native global controller arrays are invalid");
        }
        const auto entry_budget = budget_.native_snapshot();
        const auto make_empty_terminal = [this, start_iteration](
            std::int64_t reason,
            const NativeBudgetStateV2::NativeSnapshot& entry,
            const NativeBudgetStateV2::NativeSnapshot& final) {
            py::array_t<std::int64_t> events(
                py::array::ShapeContainer{0, 26});
            py::array_t<double> ranking(0);
            py::array_t<std::int64_t> removed_offsets(1);
            py::array_t<std::int64_t> removed_indices(0);
            py::array_t<std::int64_t> plan_offsets(1);
            py::array_t<std::int64_t> route_offsets(1);
            py::array_t<std::int64_t> route_indices(0);
            py::array_t<std::int64_t> objective_integer(
                py::array::ShapeContainer{0, 2});
            py::array_t<double> objective_float(
                py::array::ShapeContainer{0, 2});
            py::array_t<std::int64_t> stage_status(
                py::array::ShapeContainer{0, 4});
            py::array_t<double> stage_weights(
                py::array::ShapeContainer{0, 4, 2});
            py::array_t<std::int64_t> stage_calls(
                py::array::ShapeContainer{0, 4});
            py::array_t<double> stage_rewards(
                py::array::ShapeContainer{0, 4});
            checked_data(removed_offsets)[0] = 0;
            checked_data(plan_offsets)[0] = 0;
            checked_data(route_offsets)[0] = 0;
            py::array_t<std::int64_t> termination(11);
            auto* terminal_values = checked_data(termination);
            terminal_values[0] = reason;
            terminal_values[1] = start_iteration;
            terminal_values[2] = 0;
            terminal_values[3] = start_iteration;
            terminal_values[4] = budget_.exact_budget_;
            terminal_values[5] = entry.started;
            terminal_values[6] = entry.completed;
            terminal_values[7] = entry.interrupted;
            terminal_values[8] = final.started;
            terminal_values[9] = final.completed;
            terminal_values[10] = final.interrupted;
            std::string evidence(
                "stage05.2-native-global-semantic-stream-v2");
            append_evidence_array(evidence, events);
            append_evidence_array(evidence, ranking);
            append_evidence_array(evidence, removed_offsets);
            append_evidence_array(evidence, removed_indices);
            append_evidence_array(evidence, plan_offsets);
            append_evidence_array(evidence, route_offsets);
            append_evidence_array(evidence, route_indices);
            append_evidence_array(evidence, objective_integer);
            append_evidence_array(evidence, objective_float);
            append_evidence_array(evidence, stage_status);
            append_evidence_array(evidence, stage_weights);
            append_evidence_array(evidence, stage_calls);
            append_evidence_array(evidence, stage_rewards);
            append_evidence_array(evidence, termination);
            return py::make_tuple(
                std::move(events), std::move(ranking),
                std::move(removed_offsets), std::move(removed_indices),
                std::move(plan_offsets), std::move(route_offsets),
                std::move(route_indices), std::move(objective_integer),
                std::move(objective_float), std::move(stage_status),
                std::move(stage_weights), std::move(stage_calls),
                std::move(stage_rewards), std::move(termination),
                native_sha256_hex(evidence));
        };
        if (budget_.budget_reached()) {
            return make_empty_terminal(1, entry_budget, entry_budget);
        }
        auto previous_indices = current_indices_;
        struct GlobalSearchSnapshot {
            NativeBudgetStateV2::NativeSnapshot budget;
            py::array_t<std::int64_t> current_offsets;
            py::array_t<std::int64_t> current_indices;
            py::tuple current_exact;
            py::array_t<std::int64_t> current_objective_integer;
            py::array_t<double> current_objective_float;
            py::array_t<std::int64_t> best_offsets;
            py::array_t<std::int64_t> best_indices;
            py::tuple best_exact;
            py::array_t<std::int64_t> best_objective_integer;
            py::array_t<double> best_objective_float;
            bool last_candidate_ready;
            py::array_t<std::int64_t> last_candidate_offsets;
            py::array_t<std::int64_t> last_candidate_indices;
            py::tuple last_candidate_exact;
            py::array_t<std::int64_t> last_candidate_objective_integer;
            py::array_t<double> last_candidate_objective_float;
            std::optional<PythonRandom> constraint_rng;
            std::array<double, 4> constraint_weights;
            std::array<double, 4> constraint_segment_rewards;
            std::array<std::int64_t, 4> constraint_segment_calls;
            std::array<std::array<std::int64_t, 8>, 4> constraint_totals;
            std::int64_t last_finished_stage04_iteration;
            std::int64_t last_completed_constraint_iteration;
            std::int64_t main_stagnation_iterations;
            bool last_iteration_global_best_improved;
            NativeCausalJournalV2::Snapshot causal;
        };
        GlobalSearchSnapshot snapshot{
            entry_budget,
            current_offsets_, current_indices_, current_exact_payload_,
            current_objective_integer_, current_objective_float_,
            best_offsets_, best_indices_, best_exact_payload_,
            best_objective_integer_, best_objective_float_,
            last_candidate_ready_, last_candidate_offsets_,
            last_candidate_indices_, last_candidate_exact_payload_,
            last_candidate_objective_integer_, last_candidate_objective_float_,
            constraint_rng_, constraint_weights_, constraint_segment_rewards_,
            constraint_segment_calls_, constraint_totals_,
            last_finished_stage04_iteration_,
            last_completed_constraint_iteration_,
            main_stagnation_iterations_, last_iteration_global_best_improved_,
            causal_journal_.snapshot()};
        defer_global_commit_ = true;
        ScopeRollback rollback(
            [this, snapshot = std::move(snapshot)]() mutable noexcept {
                if (pending_composite_active_) {
                    rollback_pending_composite();
                }
                budget_.rollback_outer_preserving_exact_noexcept(snapshot.budget);
                causal_journal_.rollback_noexcept(snapshot.causal);
                defer_composite_commit_ = false;
                defer_iteration_commit_ = false;
                defer_global_commit_ = false;
                current_offsets_ = std::move(snapshot.current_offsets);
                current_indices_ = std::move(snapshot.current_indices);
                current_exact_payload_ = std::move(snapshot.current_exact);
                current_objective_integer_ =
                    std::move(snapshot.current_objective_integer);
                current_objective_float_ =
                    std::move(snapshot.current_objective_float);
                best_offsets_ = std::move(snapshot.best_offsets);
                best_indices_ = std::move(snapshot.best_indices);
                best_exact_payload_ = std::move(snapshot.best_exact);
                best_objective_integer_ =
                    std::move(snapshot.best_objective_integer);
                best_objective_float_ =
                    std::move(snapshot.best_objective_float);
                last_candidate_ready_ = snapshot.last_candidate_ready;
                last_candidate_offsets_ =
                    std::move(snapshot.last_candidate_offsets);
                last_candidate_indices_ =
                    std::move(snapshot.last_candidate_indices);
                last_candidate_exact_payload_ =
                    std::move(snapshot.last_candidate_exact);
                last_candidate_objective_integer_ =
                    std::move(snapshot.last_candidate_objective_integer);
                last_candidate_objective_float_ =
                    std::move(snapshot.last_candidate_objective_float);
                constraint_rng_ = std::move(snapshot.constraint_rng);
                constraint_weights_ = snapshot.constraint_weights;
                constraint_segment_rewards_ =
                    snapshot.constraint_segment_rewards;
                constraint_segment_calls_ = snapshot.constraint_segment_calls;
                constraint_totals_ = snapshot.constraint_totals;
                last_finished_stage04_iteration_ =
                    snapshot.last_finished_stage04_iteration;
                last_completed_constraint_iteration_ =
                    snapshot.last_completed_constraint_iteration;
                main_stagnation_iterations_ = snapshot.main_stagnation_iterations;
                last_iteration_global_best_improved_ =
                    snapshot.last_iteration_global_best_improved;
            });
        const auto previous_distance =
            checked_data<double>(current_objective_float_)[0];
        py::tuple constraint;
        try {
            constraint = constraint_iteration(
                start_iteration,
                initial_stagnation_iterations,
                false,
                threshold_array,
                fraction_array,
                deadline_array,
                batch_array,
                route_change_limit);
        } catch (const NativeExactDeadlineInterruption&) {
            rollback.rollback_now();
            const auto terminal_budget = budget_.native_snapshot();
            return make_empty_terminal(2, entry_budget, terminal_budget);
        } catch (const std::runtime_error& error) {
            if (std::string_view(error.what()).find("deadline")
                == std::string_view::npos) {
                throw;
            }
            rollback.rollback_now();
            const auto terminal_budget = budget_.native_snapshot();
            return make_empty_terminal(2, entry_budget, terminal_budget);
        }
        if (global_search_envelope_failure_injection_) {
            global_search_envelope_failure_injection_ = false;
            throw std::runtime_error(
                "injected full native global-search envelope failure");
        }
        auto selection = py::cast<py::array_t<std::int64_t>>(constraint[0]);
        auto probe = py::cast<py::tuple>(constraint[1]);
        auto outcome = py::cast<py::array_t<std::int64_t>>(constraint[2]);
        auto removal = py::cast<py::tuple>(probe[0]);
        const auto* selection_values = checked_data<std::int64_t>(selection);
        const auto* outcome_values = checked_data<std::int64_t>(outcome);
        if (probe[1].is_none()) {
            auto removal_metadata =
                py::cast<py::array_t<std::int64_t>>(removal[6]);
            if (!probe[2].is_none()
                || checked_data<std::int64_t>(removal_metadata)[0] != 2
                || selection_values[1] != 0
                || outcome_values[0] < 0 || outcome_values[0] >= 4
                || outcome_values[2] != 0) {
                throw std::logic_error(
                    "native global no-removable bootstrap is inconsistent");
            }
            last_iteration_global_best_improved_ = false;
            main_stagnation_iterations_ = initial_stagnation_iterations + 1;
            auto stage_boundary = finish_stage04_iteration(start_iteration, false);
            py::array_t<std::int64_t> events(
                {py::ssize_t(3), py::ssize_t(26)});
            std::fill(
                checked_data(events), checked_data(events) + 3 * 26,
                std::int64_t{0});
            const auto initialize_event = [&](py::ssize_t row) {
                auto* event = checked_data(events) + row * 26;
                event[2] = start_iteration;
                event[16] = -1;
                event[17] = -1;
                event[20] = -2;
                event[24] = -1;
                event[25] = -1;
                return event;
            };
            auto* quality = initialize_event(0);
            quality[1] = 1;
            quality[3] = 4;
            auto* constraint_event = initialize_event(1);
            constraint_event[1] = 2;
            constraint_event[3] = 9 + outcome_values[0];
            constraint_event[5] = 4;
            constraint_event[15] = 2;
            constraint_event[24] = 0;
            constraint_event[25] = 1;
            auto* legacy = initialize_event(2);
            legacy[3] = 2;

            py::array_t<double> ranking(3);
            std::fill(checked_data(ranking), checked_data(ranking) + 3, 0.0);
            py::array_t<std::int64_t> removed_offsets(4);
            std::fill(
                checked_data(removed_offsets), checked_data(removed_offsets) + 4,
                std::int64_t{0});
            py::array_t<std::int64_t> removed_indices(0);
            py::array_t<std::int64_t> plan_offsets(4);
            std::fill(
                checked_data(plan_offsets), checked_data(plan_offsets) + 4,
                std::int64_t{0});
            py::array_t<std::int64_t> route_offsets(1);
            checked_data(route_offsets)[0] = 0;
            py::array_t<std::int64_t> route_indices(0);
            py::array_t<std::int64_t> objective_integer(
                {py::ssize_t(3), py::ssize_t(2)});
            std::fill(
                checked_data(objective_integer),
                checked_data(objective_integer) + 6,
                std::int64_t{-1});
            py::array_t<double> objective_float(
                {py::ssize_t(3), py::ssize_t(2)});
            std::fill(
                checked_data(objective_float),
                checked_data(objective_float) + 6,
                std::numeric_limits<double>::quiet_NaN());
            auto boundary_status =
                py::cast<py::array_t<std::int64_t>>(stage_boundary[0]);
            auto boundary_weights =
                py::cast<py::array_t<double>>(stage_boundary[1]);
            auto boundary_calls =
                py::cast<py::array_t<std::int64_t>>(stage_boundary[2]);
            auto boundary_rewards =
                py::cast<py::array_t<double>>(stage_boundary[3]);
            py::array_t<std::int64_t> stage_status(
                {py::ssize_t(1), py::ssize_t(4)});
            py::array_t<double> stage_weights(
                {py::ssize_t(1), py::ssize_t(4), py::ssize_t(2)});
            py::array_t<std::int64_t> stage_calls(
                {py::ssize_t(1), py::ssize_t(4)});
            py::array_t<double> stage_rewards(
                {py::ssize_t(1), py::ssize_t(4)});
            std::copy(
                checked_data<std::int64_t>(boundary_status),
                checked_data<std::int64_t>(boundary_status) + 4,
                checked_data(stage_status));
            std::copy(
                checked_data<double>(boundary_weights),
                checked_data<double>(boundary_weights) + 8,
                checked_data(stage_weights));
            std::copy(
                checked_data<std::int64_t>(boundary_calls),
                checked_data<std::int64_t>(boundary_calls) + 4,
                checked_data(stage_calls));
            std::copy(
                checked_data<double>(boundary_rewards),
                checked_data<double>(boundary_rewards) + 4,
                checked_data(stage_rewards));
            const auto terminal_budget = budget_.native_snapshot();
            py::array_t<std::int64_t> termination(11);
            auto* terminal_values = checked_data(termination);
            terminal_values[0] = 0;
            terminal_values[1] = start_iteration;
            terminal_values[2] = 1;
            terminal_values[3] = start_iteration + 1;
            terminal_values[4] = budget_.exact_budget_;
            terminal_values[5] = entry_budget.started;
            terminal_values[6] = entry_budget.completed;
            terminal_values[7] = entry_budget.interrupted;
            terminal_values[8] = terminal_budget.started;
            terminal_values[9] = terminal_budget.completed;
            terminal_values[10] = terminal_budget.interrupted;
            std::string evidence(
                "stage05.2-native-global-semantic-stream-v2");
            append_evidence_array(evidence, events);
            append_evidence_array(evidence, ranking);
            append_evidence_array(evidence, removed_offsets);
            append_evidence_array(evidence, removed_indices);
            append_evidence_array(evidence, plan_offsets);
            append_evidence_array(evidence, route_offsets);
            append_evidence_array(evidence, route_indices);
            append_evidence_array(evidence, objective_integer);
            append_evidence_array(evidence, objective_float);
            append_evidence_array(evidence, stage_status);
            append_evidence_array(evidence, stage_weights);
            append_evidence_array(evidence, stage_calls);
            append_evidence_array(evidence, stage_rewards);
            append_evidence_array(evidence, termination);
            auto result = py::make_tuple(
                std::move(events), std::move(ranking),
                std::move(removed_offsets), std::move(removed_indices),
                std::move(plan_offsets), std::move(route_offsets),
                std::move(route_indices), std::move(objective_integer),
                std::move(objective_float), std::move(stage_status),
                std::move(stage_weights), std::move(stage_calls),
                std::move(stage_rewards), std::move(termination),
                native_sha256_hex(evidence));
            defer_global_commit_ = false;
            rollback.release();
            return result;
        }
        auto repair = py::cast<py::tuple>(probe[1]);
        auto transaction = py::cast<py::tuple>(probe[2]);
        if (outcome_values[0] != 0
            || py::cast<py::array_t<std::int64_t>>(repair[0]).size() != 2) {
            throw std::logic_error(
                "native global bootstrap did not produce the expected constraint transaction");
        }
        const auto candidate_feasible = outcome_values[2] != 0;
        const auto budget_boundary = budget_.budget_reached();
        last_iteration_global_best_improved_ =
            candidate_feasible && outcome_values[4] != 0;
        main_stagnation_iterations_ = last_iteration_global_best_improved_
            ? 0 : initial_stagnation_iterations + 1;
        auto stage_boundary = finish_stage04_iteration(
            start_iteration, budget_boundary);

        const py::ssize_t event_count = budget_boundary ? 3 : 4;
        py::array_t<std::int64_t> events({event_count, py::ssize_t(26)});
        std::fill(
            checked_data(events), checked_data(events) + event_count * 26,
            std::int64_t{0});
        const auto initialize_event = [&](py::ssize_t row) {
            auto* event = checked_data(events) + row * 26;
            event[2] = start_iteration;
            event[16] = -1;
            event[17] = -1;
            event[20] = -2;
            event[24] = -1;
            event[25] = -1;
            return event;
        };
        auto* quality = initialize_event(0);
        quality[1] = 1;
        quality[3] = 4;
        quality[4] = 0;
        quality[5] = 0;

        auto* ranked_removal = initialize_event(1);
        ranked_removal[1] = 2;
        ranked_removal[3] = 9;
        ranked_removal[4] = 1;
        ranked_removal[5] = 1;
        ranked_removal[9] = 1;
        ranked_removal[12] = 1;
        ranked_removal[13] = selection_values[1];
        ranked_removal[14] = selection_values[2];
        ranked_removal[15] = 2;
        ranked_removal[16] = 0;
        ranked_removal[17] = 0;
        ranked_removal[22] = initial_stagnation_iterations;
        ranked_removal[24] = selection_values[0];
        ranked_removal[25] = 0;

        auto* repaired = initialize_event(2);
        repaired[1] = 2;
        repaired[3] = 9;
        repaired[4] = candidate_feasible ? 1 : 2;
        repaired[5] = candidate_feasible ? 2 : 3;
        repaired[6] = outcome_values[3];
        repaired[7] = outcome_values[5];
        repaired[8] = candidate_feasible
            && checked_data<double>(
                py::cast<py::array_t<double>>(transaction[3]))[0]
                < previous_distance - 1e-9;
        repaired[9] = 1;
        repaired[10] = outcome_values[2];
        repaired[11] = py::cast<py::array_t<std::int64_t>>(transaction[5]).size();
        repaired[13] = selection_values[1];
        repaired[14] = selection_values[2];
        repaired[15] = 2;
        repaired[17] = 0;
        repaired[20] = candidate_feasible ? 0 : -2;
        repaired[22] = initial_stagnation_iterations;
        repaired[24] = selection_values[0];
        repaired[25] = 0;

        if (!budget_boundary) {
            auto* legacy = initialize_event(3);
            legacy[3] = 2;
            legacy[4] = 0;
            legacy[5] = 0;
        }

        py::array_t<double> ranking(event_count);
        std::fill(
            checked_data(ranking), checked_data(ranking) + event_count, 0.0);
        auto removal_scores = py::cast<py::array_t<double>>(removal[4]);
        checked_data(ranking)[1] = checked_data<double>(removal_scores)[0];

        auto removed = py::cast<py::array_t<std::int64_t>>(removal[2]);
        if (removed.size() <= 0) {
            throw std::logic_error(
                "native global bootstrap lost the removed-customer set");
        }
        py::array_t<std::int64_t> removed_offsets(event_count + 1);
        std::vector<std::int64_t> removed_boundaries{
            0, 0, removed.size(), removed.size() * 2};
        if (!budget_boundary) {
            removed_boundaries.push_back(removed.size() * 2);
        }
        std::copy(
            removed_boundaries.begin(), removed_boundaries.end(),
            checked_data(removed_offsets));
        py::array_t<std::int64_t> removed_indices(removed.size() * 2);
        std::copy(
            checked_data<std::int64_t>(removed),
            checked_data<std::int64_t>(removed) + removed.size(),
            checked_data(removed_indices));
        std::copy(
            checked_data<std::int64_t>(removed),
            checked_data<std::int64_t>(removed) + removed.size(),
            checked_data(removed_indices) + removed.size());

        auto remaining_offsets = py::cast<py::array_t<std::int64_t>>(removal[0]);
        auto remaining_indices = py::cast<py::array_t<std::int64_t>>(removal[1]);
        auto repaired_offsets = py::cast<py::array_t<std::int64_t>>(repair[0]);
        auto repaired_indices = py::cast<py::array_t<std::int64_t>>(repair[1]);
        const auto repaired_route_changed =
            previous_indices.size() != repaired_indices.size()
            || !std::equal(
                checked_data<std::int64_t>(previous_indices),
                checked_data<std::int64_t>(previous_indices)
                    + previous_indices.size(),
                checked_data<std::int64_t>(repaired_indices));
        repaired[17] = repaired_route_changed ? 0 : -1;
        py::array_t<std::int64_t> plan_offsets(event_count + 1);
        std::vector<std::int64_t> plan_boundaries{0, 0, 1, 2};
        if (!budget_boundary) {
            plan_boundaries.push_back(2);
        }
        std::copy(
            plan_boundaries.begin(), plan_boundaries.end(),
            checked_data(plan_offsets));
        py::array_t<std::int64_t> route_offsets(3);
        checked_data(route_offsets)[0] = 0;
        checked_data(route_offsets)[1] = remaining_indices.size();
        checked_data(route_offsets)[2] =
            remaining_indices.size() + repaired_indices.size();
        py::array_t<std::int64_t> route_indices(
            remaining_indices.size() + repaired_indices.size());
        std::copy(
            checked_data<std::int64_t>(remaining_indices),
            checked_data<std::int64_t>(remaining_indices) + remaining_indices.size(),
            checked_data(route_indices));
        std::copy(
            checked_data<std::int64_t>(repaired_indices),
            checked_data<std::int64_t>(repaired_indices) + repaired_indices.size(),
            checked_data(route_indices) + remaining_indices.size());
        static_cast<void>(remaining_offsets);
        static_cast<void>(repaired_offsets);

        py::array_t<std::int64_t> objective_integer(
            {event_count, py::ssize_t(2)});
        py::array_t<double> objective_float({event_count, py::ssize_t(2)});
        std::fill(
            checked_data(objective_integer),
            checked_data(objective_integer) + event_count * 2,
            std::int64_t{-1});
        std::fill(
            checked_data(objective_float),
            checked_data(objective_float) + event_count * 2,
            std::numeric_limits<double>::quiet_NaN());
        auto transaction_integer =
            py::cast<py::array_t<std::int64_t>>(transaction[2]);
        auto transaction_float = py::cast<py::array_t<double>>(transaction[3]);
        for (py::ssize_t row : {py::ssize_t(1), py::ssize_t(2)}) {
            std::copy(
                checked_data<std::int64_t>(transaction_integer),
                checked_data<std::int64_t>(transaction_integer) + 2,
                checked_data(objective_integer) + row * 2);
            std::copy(
                checked_data<double>(transaction_float),
                checked_data<double>(transaction_float) + 2,
                checked_data(objective_float) + row * 2);
        }
        auto stage_status = py::cast<py::array_t<std::int64_t>>(stage_boundary[0]);
        auto stage_weights = py::cast<py::array_t<double>>(stage_boundary[1]);
        auto stage_calls = py::cast<py::array_t<std::int64_t>>(stage_boundary[2]);
        auto stage_rewards = py::cast<py::array_t<double>>(stage_boundary[3]);
        py::array_t<std::int64_t> stage_status_matrix(
            {py::ssize_t(1), py::ssize_t(4)});
        py::array_t<double> stage_weights_matrix(
            {py::ssize_t(1), py::ssize_t(4), py::ssize_t(2)});
        py::array_t<std::int64_t> stage_calls_matrix(
            {py::ssize_t(1), py::ssize_t(4)});
        py::array_t<double> stage_rewards_matrix(
            {py::ssize_t(1), py::ssize_t(4)});
        std::copy(
            checked_data<std::int64_t>(stage_status),
            checked_data<std::int64_t>(stage_status) + 4,
            checked_data(stage_status_matrix));
        std::copy(
            checked_data<double>(stage_weights),
            checked_data<double>(stage_weights) + 8,
            checked_data(stage_weights_matrix));
        std::copy(
            checked_data<std::int64_t>(stage_calls),
            checked_data<std::int64_t>(stage_calls) + 4,
            checked_data(stage_calls_matrix));
        std::copy(
            checked_data<double>(stage_rewards),
            checked_data<double>(stage_rewards) + 4,
            checked_data(stage_rewards_matrix));
        const auto terminal_budget = budget_.native_snapshot();
        py::array_t<std::int64_t> termination(11);
        auto* terminal_values = checked_data(termination);
        terminal_values[0] = budget_boundary ? 1 : 0;
        terminal_values[1] = start_iteration;
        terminal_values[2] = 1;
        terminal_values[3] = start_iteration + 1;
        terminal_values[4] = budget_.exact_budget_;
        terminal_values[5] = entry_budget.started;
        terminal_values[6] = entry_budget.completed;
        terminal_values[7] = entry_budget.interrupted;
        terminal_values[8] = terminal_budget.started;
        terminal_values[9] = terminal_budget.completed;
        terminal_values[10] = terminal_budget.interrupted;
        std::string evidence("stage05.2-native-global-semantic-stream-v2");
        append_evidence_array(evidence, events);
        append_evidence_array(evidence, ranking);
        append_evidence_array(evidence, removed_offsets);
        append_evidence_array(evidence, removed_indices);
        append_evidence_array(evidence, plan_offsets);
        append_evidence_array(evidence, route_offsets);
        append_evidence_array(evidence, route_indices);
        append_evidence_array(evidence, objective_integer);
        append_evidence_array(evidence, objective_float);
        append_evidence_array(evidence, stage_status_matrix);
        append_evidence_array(evidence, stage_weights_matrix);
        append_evidence_array(evidence, stage_calls_matrix);
        append_evidence_array(evidence, stage_rewards_matrix);
        append_evidence_array(evidence, termination);
        auto result = py::make_tuple(
            std::move(events), std::move(ranking),
            std::move(removed_offsets), std::move(removed_indices),
            std::move(plan_offsets), std::move(route_offsets),
            std::move(route_indices), std::move(objective_integer),
            std::move(objective_float), std::move(stage_status_matrix),
            std::move(stage_weights_matrix), std::move(stage_calls_matrix),
            std::move(stage_rewards_matrix), std::move(termination),
            native_sha256_hex(evidence));
        commit_pending_composite_noexcept();
        defer_global_commit_ = false;
        rollback.release();
        return result;
    }

    void inject_global_search_envelope_failure_once() {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        global_search_envelope_failure_injection_ = true;
    }

    void inject_constraint_search_deadline_after_completed_once(
        std::int64_t completed_iterations) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (completed_iterations < 0
            || constraint_search_deadline_after_completed_injection_ >= 0) {
            throw std::invalid_argument(
                "full native constraint-search deadline injection is invalid");
        }
        constraint_search_deadline_after_completed_injection_ =
            completed_iterations;
    }

    void inject_constraint_iteration_deadline_before_commit_once() {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        constraint_iteration_deadline_injection_ = true;
    }

    void inject_exact_kernel_deadline_once() {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        exact_kernel_deadline_injection_ = true;
    }

    void inject_constraint_probe_envelope_failure_once() {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        probe_envelope_failure_injection_ = true;
    }

    void inject_commit_failure_once(std::int64_t step) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (step < 1 || step > 3) {
            throw std::invalid_argument(
                "full native commit failure injection step must be 1, 2, or 3");
        }
        commit_failure_injection_ = step;
    }

    [[nodiscard]] bool initialized() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        return initialized_;
    }

    [[nodiscard]] std::size_t work_pool_active_tasks() const noexcept {
        return work_pool_->active_task_count();
    }

    [[nodiscard]] std::size_t work_pool_peak_active_tasks() const noexcept {
        return work_pool_->peak_active_task_count();
    }

    [[nodiscard]] std::size_t work_pool_thread_count() const noexcept {
        return static_cast<std::size_t>(work_pool_->thread_count());
    }

    void suppress_plan_screening_negative_cache(bool suppress) noexcept {
        suppress_plan_screening_negative_cache_ = suppress;
    }

    py::tuple state() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        auto cache = route_cache_.snapshot();
        auto negative = negative_cache_.snapshot();
        return py::make_tuple(
            py::cast<py::array_t<std::int64_t>>(cache[4]),
            budget_.state(),
            attempted_plans_.size(),
            py::cast<py::array_t<std::int64_t>>(negative[3]));
    }

    py::tuple solution_state() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_) {
            throw std::runtime_error(
                "full native search engine must be initialized before solution inspection");
        }
        return py::make_tuple(
            owned_array_copy<std::int64_t>(
                current_offsets_, "current_route_offsets", 1),
            owned_array_copy<std::int64_t>(
                current_indices_, "current_route_indices", 1),
            owned_array_copy<std::int64_t>(
                current_objective_integer_, "current_objective_integer", 1),
            owned_array_copy<double>(
                current_objective_float_, "current_objective_float", 1),
            owned_array_copy<std::int64_t>(
                best_offsets_, "best_route_offsets", 1),
            owned_array_copy<std::int64_t>(
                best_indices_, "best_route_indices", 1),
            owned_array_copy<std::int64_t>(
                best_objective_integer_, "best_objective_integer", 1),
            owned_array_copy<double>(
                best_objective_float_, "best_objective_float", 1));
    }

    py::tuple lane_solution_state(std::int64_t lane) const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || lane < 0 || lane > 2) {
            throw std::invalid_argument(
                "full native lane solution inspection is invalid");
        }
        const auto* offsets = lane == 0 ? &legacy_offsets_
            : lane == 1 ? &quality_offsets_ : &current_offsets_;
        const auto* indices = lane == 0 ? &legacy_indices_
            : lane == 1 ? &quality_indices_ : &current_indices_;
        const auto* objective_integer = lane == 0 ? &legacy_objective_integer_
            : lane == 1 ? &quality_objective_integer_ : &current_objective_integer_;
        const auto* objective_float = lane == 0 ? &legacy_objective_float_
            : lane == 1 ? &quality_objective_float_ : &current_objective_float_;
        return py::make_tuple(
            owned_array_copy<std::int64_t>(*offsets, "lane_route_offsets", 1),
            owned_array_copy<std::int64_t>(*indices, "lane_route_indices", 1),
            owned_array_copy<std::int64_t>(
                *objective_integer, "lane_objective_integer", 1),
            owned_array_copy<double>(*objective_float, "lane_objective_float", 1));
    }

    py::tuple best_solution_payload() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_) {
            throw std::runtime_error(
                "full native search engine must be initialized before best inspection");
        }
        return py::make_tuple(
            owned_array_copy<std::int64_t>(
                best_offsets_, "best_route_offsets", 1),
            owned_array_copy<std::int64_t>(
                best_indices_, "best_route_indices", 1),
            owned_exact_state_copy(best_exact_payload_),
            owned_array_copy<std::int64_t>(
                best_objective_integer_, "best_objective_integer", 1),
            owned_array_copy<double>(
                best_objective_float_, "best_objective_float", 1));
    }

    py::tuple exact_backend_metrics_payload() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_) {
            throw std::runtime_error(
                "full native search engine must be initialized before metrics inspection");
        }
        py::array_t<std::int64_t> counters(10);
        std::copy(
            exact_backend_totals_.begin(), exact_backend_totals_.end(),
            checked_data(counters));
        py::array_t<std::int64_t> occupancies(exact_launch_occupancies_.size());
        std::copy(
            exact_launch_occupancies_.begin(), exact_launch_occupancies_.end(),
            checked_data(occupancies));
        py::array_t<double> timings(1);
        checked_data(timings)[0] = exact_backend_seconds_;
        return py::make_tuple(
            std::move(counters), std::move(occupancies), std::move(timings));
    }

    py::tuple exact_journal_payload() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_) {
            throw std::runtime_error(
                "full native search engine must be initialized before journal inspection");
        }
        py::tuple batches(exact_journal_.size());
        std::string evidence("stage05.2-native-exact-journal-v2");
        for (std::size_t ordinal = 0; ordinal < exact_journal_.size(); ++ordinal) {
            const auto& batch = exact_journal_[ordinal];
            py::array_t<std::int64_t> context(3);
            std::copy(batch.context.begin(), batch.context.end(), checked_data(context));
            py::array_t<std::int64_t> route_offsets(batch.route_offsets.size());
            py::array_t<std::int64_t> route_indices(batch.route_indices.size());
            std::copy(
                batch.route_offsets.begin(), batch.route_offsets.end(),
                checked_data(route_offsets));
            std::copy(
                batch.route_indices.begin(), batch.route_indices.end(),
                checked_data(route_indices));
            std::vector<std::int64_t> path_offsets{0};
            std::vector<std::int64_t> path_indices;
            py::array_t<std::int64_t> statuses(batch.results.size());
            py::array_t<std::int64_t> reasons(batch.results.size());
            py::array_t<double> metrics(
                {static_cast<py::ssize_t>(batch.results.size()), py::ssize_t(4)});
            py::array_t<std::int64_t> labels(
                {static_cast<py::ssize_t>(batch.results.size()), py::ssize_t(3)});
            for (std::size_t route = 0; route < batch.results.size(); ++route) {
                const auto& result = batch.results[route];
                path_indices.insert(
                    path_indices.end(), result.path.begin(), result.path.end());
                path_offsets.push_back(
                    static_cast<std::int64_t>(path_indices.size()));
                checked_data(statuses)[route] = result.status;
                checked_data(reasons)[route] = result.reason;
                std::copy(
                    result.metrics.begin(), result.metrics.end(),
                    checked_data(metrics) + route * 4);
                std::copy(
                    result.label_counters.begin(), result.label_counters.end(),
                    checked_data(labels) + route * 3);
            }
            py::array_t<std::int64_t> path_offsets_array(path_offsets.size());
            py::array_t<std::int64_t> path_indices_array(path_indices.size());
            std::copy(
                path_offsets.begin(), path_offsets.end(),
                checked_data(path_offsets_array));
            std::copy(
                path_indices.begin(), path_indices.end(),
                checked_data(path_indices_array));
            auto batch_payload = py::make_tuple(
                std::move(context), std::move(route_offsets),
                std::move(route_indices), std::move(path_offsets_array),
                std::move(path_indices_array), std::move(statuses),
                std::move(reasons), std::move(metrics), std::move(labels));
            append_nested_evidence(evidence, batch_payload);
            batches[static_cast<py::ssize_t>(ordinal)] = std::move(batch_payload);
        }
        return py::make_tuple(
            std::move(batches), native_sha256_hex(evidence));
    }

    py::tuple control_journal_payload() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || pending_composite_active_) {
            throw std::runtime_error(
                "full native control journal is not at a committed boundary");
        }
        py::tuple batches(control_journal_.size());
        std::string evidence("stage05.2-native-control-journal-v2");
        py::ssize_t ordinal = 0;
        for (const auto& batch : control_journal_) {
            const auto integer_array = [](const auto& values) {
                py::array_t<std::int64_t> output(values.size());
                std::copy(values.begin(), values.end(), checked_data(output));
                return output;
            };
            py::array_t<std::int64_t> context(3);
            std::copy(batch.context.begin(), batch.context.end(), checked_data(context));
            auto plan_offsets = integer_array(batch.plan_offsets);
            auto route_offsets = integer_array(batch.route_offsets);
            auto route_indices = integer_array(batch.route_indices);
            auto ranked = integer_array(batch.ranked);
            auto selected = integer_array(batch.selected);
            auto decisions = integer_array(batch.decision_codes);
            auto statuses = integer_array(batch.statuses);
            py::array_t<std::int64_t> ranking_integer(
                {static_cast<py::ssize_t>(batch.decision_codes.size()),
                 py::ssize_t(2)});
            std::copy(
                batch.ranking_integer.begin(), batch.ranking_integer.end(),
                checked_data(ranking_integer));
            py::array_t<double> ranking_float(batch.ranking_float.size());
            std::copy(
                batch.ranking_float.begin(), batch.ranking_float.end(),
                checked_data(ranking_float));
            auto resolutions = integer_array(batch.route_resolutions);
            auto cache_statistics = integer_array(batch.cache_statistics);
            auto budget_state = integer_array(batch.budget_state);
            auto protocol_flags = integer_array(batch.protocol_flags);
            auto batch_payload = py::make_tuple(
                std::move(context), std::move(plan_offsets),
                std::move(route_offsets), std::move(route_indices),
                std::move(ranked), std::move(selected),
                std::move(decisions), std::move(statuses),
                std::move(ranking_integer), std::move(ranking_float),
                std::move(resolutions), std::move(cache_statistics),
                std::move(budget_state), std::move(protocol_flags));
            append_nested_evidence(evidence, batch_payload);
            batches[ordinal++] = std::move(batch_payload);
        }
        py::array_t<std::int64_t> screening_statistics(
            screening_statistics_.size());
        std::copy(
            screening_statistics_.begin(), screening_statistics_.end(),
            checked_data(screening_statistics));
        std::vector<std::pair<std::int64_t, std::int64_t>> ordered_reasons(
            screening_reason_counts_.begin(), screening_reason_counts_.end());
        std::sort(ordered_reasons.begin(), ordered_reasons.end());
        py::array_t<std::int64_t> reason_codes(ordered_reasons.size());
        py::array_t<std::int64_t> reason_counts(ordered_reasons.size());
        for (std::size_t index = 0; index < ordered_reasons.size(); ++index) {
            checked_data(reason_codes)[index] = ordered_reasons[index].first;
            checked_data(reason_counts)[index] = ordered_reasons[index].second;
        }
        py::array_t<double> screening_timing(1);
        checked_data(screening_timing)[0] = screening_seconds_;
        py::array_t<std::int64_t> screening_occupancies(
            screening_occupancies_.size());
        std::copy(
            screening_occupancies_.begin(), screening_occupancies_.end(),
            checked_data(screening_occupancies));
        py::array_t<std::int64_t> screening_contexts({
            static_cast<py::ssize_t>(screening_journal_.size()), py::ssize_t{3}});
        py::array_t<std::int64_t> screening_route_offsets(
            screening_journal_.size() + 1);
        std::size_t screening_route_index_count = 0;
        checked_data(screening_route_offsets)[0] = 0;
        for (std::size_t row = 0; row < screening_journal_.size(); ++row) {
            screening_route_index_count += screening_journal_[row].route.size();
            checked_data(screening_route_offsets)[row + 1] =
                static_cast<std::int64_t>(screening_route_index_count);
        }
        py::array_t<std::int64_t> screening_route_indices(
            screening_route_index_count);
        py::array_t<std::int64_t> screening_codes({
            static_cast<py::ssize_t>(screening_journal_.size()), py::ssize_t{16}});
        py::array_t<double> screening_metrics({
            static_cast<py::ssize_t>(screening_journal_.size()), py::ssize_t{15}});
        py::array_t<std::int64_t> screening_flags({
            static_cast<py::ssize_t>(screening_journal_.size()), py::ssize_t{4}});
        std::size_t screening_route_cursor = 0;
        for (std::size_t row = 0; row < screening_journal_.size(); ++row) {
            const auto& journal = screening_journal_[row];
            std::copy(
                journal.context.begin(), journal.context.end(),
                checked_data(screening_contexts) + row * 3);
            std::copy(
                journal.route.begin(), journal.route.end(),
                checked_data(screening_route_indices) + screening_route_cursor);
            screening_route_cursor += journal.route.size();
            std::copy(
                journal.codes.begin(), journal.codes.end(),
                checked_data(screening_codes) + row * 16);
            std::copy(
                journal.metrics.begin(), journal.metrics.end(),
                checked_data(screening_metrics) + row * 15);
            std::copy(
                journal.flags.begin(), journal.flags.end(),
                checked_data(screening_flags) + row * 4);
        }
        auto screening_payload = py::make_tuple(
            std::move(screening_statistics), std::move(reason_codes),
            std::move(reason_counts), std::move(screening_timing),
            std::move(screening_contexts), std::move(screening_route_offsets),
            std::move(screening_route_indices), std::move(screening_codes),
            std::move(screening_metrics), std::move(screening_flags),
            std::move(screening_occupancies));
        append_nested_evidence(evidence, screening_payload);
        return py::make_tuple(
            std::move(batches), std::move(screening_payload),
            native_sha256_hex(evidence));
    }

private:
    struct ExactJournalBatch {
        std::array<std::int64_t, 3> context{};
        std::vector<std::int64_t> route_offsets;
        std::vector<std::int64_t> route_indices;
        std::vector<NativeRouteCacheV2::ExactPayload> results;
    };

    struct ControlJournalBatch {
        std::array<std::int64_t, 3> context{};
        std::int64_t transaction_id = -1;
        std::vector<std::int64_t> plan_offsets;
        std::vector<std::int64_t> route_offsets;
        std::vector<std::int64_t> route_indices;
        std::vector<std::int64_t> ranked;
        std::vector<std::int64_t> selected;
        // 0=ineligible, 1=already attempted, 2=selected, 3=not selected.
        std::vector<std::int64_t> decision_codes;
        std::vector<std::int64_t> statuses;
        std::vector<std::int64_t> ranking_integer;
        std::vector<double> ranking_float;
        std::vector<std::int64_t> route_resolutions;
        std::vector<std::int64_t> cache_statistics;
        std::vector<std::int64_t> budget_state;
        std::vector<std::int64_t> protocol_flags;
    };

    struct ScreeningJournalRow {
        std::array<std::int64_t, 3> context{};
        std::vector<std::int64_t> route;
        std::array<std::int64_t, 16> codes{};
        std::array<double, 15> metrics{};
        // physical_evaluated, negative_cache_hit, physical_owner,
        // optimistic-reachability matrix queries.
        std::array<std::int64_t, 4> flags{};
    };

    mutable std::recursive_mutex state_mutex_;
    NativeRouteCacheV2 route_cache_;
    NativeNegativeRouteCacheV2 negative_cache_;
    NativeBudgetStateV2 budget_;
    NativeAttemptedPlanSetV2 attempted_plans_;
    std::int64_t proposal_top_k_;
    double screening_epsilon_;
    std::int64_t worker_count_;
    std::shared_ptr<NativeWorkPool> work_pool_;
    std::int64_t batch_size_ = 0;
    std::int64_t commit_failure_injection_ = 0;
    bool probe_envelope_failure_injection_ = false;
    bool constraint_iteration_deadline_injection_ = false;
    bool exact_kernel_deadline_injection_ = false;
    bool global_search_envelope_failure_injection_ = false;
    std::int64_t constraint_search_deadline_after_completed_injection_ = -1;
    bool defer_composite_commit_ = false;
    bool defer_iteration_commit_ = false;
    bool defer_global_commit_ = false;
    bool suppress_attempted_plan_journal_ = false;
    bool suppress_plan_screening_negative_cache_ = false;
    bool allow_partial_customer_coverage_ = false;
    bool suppress_round_budget_ = false;
    bool pending_composite_active_ = false;
    bool pending_round_protocol_ = false;
    bool pending_negative_store_ = false;
    bool pending_attempted_mark_ = false;
    std::optional<NativeBudgetStateV2::NativeSnapshot> pending_budget_snapshot_;
    std::list<ControlJournalBatch> pending_control_journal_;
    std::optional<NativeCausalJournalV2::Snapshot> pending_causal_snapshot_;
    bool pending_candidate_exact_ready_ = false;
    py::tuple pending_candidate_exact_payload_;
    std::int64_t depot_ = -1;
    std::vector<std::int64_t> recharge_nodes_;
    std::unordered_set<std::int64_t> all_customers_;
    std::vector<std::string> node_names_;
    bool initialized_ = false;
    py::array_t<std::int64_t> node_kind_;
    py::array_t<double> demand_;
    py::array_t<double> ready_time_;
    py::array_t<double> due_date_;
    py::array_t<double> service_time_;
    py::array_t<double> distance_;
    py::array_t<std::uint8_t> reachable_;
    py::array_t<double> vehicle_;
    py::array_t<std::int64_t> lexical_rank_;
    py::array_t<std::int64_t> current_offsets_;
    py::array_t<std::int64_t> current_indices_;
    py::tuple current_exact_payload_;
    py::array_t<std::int64_t> current_objective_integer_;
    py::array_t<double> current_objective_float_;
    py::array_t<std::int64_t> legacy_offsets_;
    py::array_t<std::int64_t> legacy_indices_;
    py::tuple legacy_exact_payload_;
    py::array_t<std::int64_t> legacy_objective_integer_;
    py::array_t<double> legacy_objective_float_;
    py::array_t<std::int64_t> quality_offsets_;
    py::array_t<std::int64_t> quality_indices_;
    py::tuple quality_exact_payload_;
    py::array_t<std::int64_t> quality_objective_integer_;
    py::array_t<double> quality_objective_float_;
    py::array_t<std::int64_t> best_offsets_;
    py::array_t<std::int64_t> best_indices_;
    py::tuple best_exact_payload_;
    std::array<std::int64_t, 10> exact_backend_totals_{};
    std::vector<std::int64_t> exact_launch_occupancies_;
    std::vector<ExactJournalBatch> exact_journal_;
    std::list<ControlJournalBatch> control_journal_;
    NativeCausalJournalV2 causal_journal_;
    std::int64_t next_causal_transaction_id_ = 0;
    std::array<std::int64_t, 3> causal_context_{-1, -1, -1};
    std::int64_t causal_context_transaction_id_ = -1;
    std::vector<ScreeningJournalRow> screening_journal_;
    // Semantic route decisions and physical kernel calls are distinct because
    // a native transaction may deduplicate repeated route sequences.
    std::array<std::int64_t, 8> screening_statistics_{};
    std::unordered_map<std::int64_t, std::int64_t> screening_reason_counts_;
    std::vector<std::int64_t> screening_occupancies_;
    double screening_seconds_ = 0.0;
    double exact_backend_seconds_ = 0.0;
    py::array_t<std::int64_t> best_objective_integer_;
    py::array_t<double> best_objective_float_;
    bool last_candidate_ready_ = false;
    py::array_t<std::int64_t> last_candidate_offsets_;
    py::array_t<std::int64_t> last_candidate_indices_;
    py::tuple last_candidate_exact_payload_;
    py::array_t<std::int64_t> last_candidate_objective_integer_;
    py::array_t<double> last_candidate_objective_float_;
    bool legacy_candidate_ready_ = false;
    std::int64_t legacy_candidate_operator_ = -1;
    std::int64_t legacy_candidate_destroy_operator_ = -1;
    std::int64_t legacy_candidate_repair_operator_ = -1;
    py::array_t<std::int64_t> legacy_candidate_offsets_;
    py::array_t<std::int64_t> legacy_candidate_indices_;
    py::tuple legacy_candidate_exact_payload_;
    py::array_t<std::int64_t> legacy_candidate_objective_integer_;
    py::array_t<double> legacy_candidate_objective_float_;
    std::optional<PythonRandom> rng_;
    std::optional<PythonRandom> constraint_rng_;
    bool stage04_configured_ = false;
    bool stage04_search_initialized_ = false;
    bool stage04_fixed_weights_ = false;
    bool stage04_auto_temperature_ = false;
    std::int64_t stage04_temperature_sample_size_ = 0;
    double stage04_temperature_target_ = 0.5;
    double stage04_temperature_fallback_fraction_ = 0.05;
    double stage04_initial_temperature_ = 1.0;
    bool stage04_reheat_enabled_ = false;
    std::int64_t stage04_reheat_stagnation_threshold_ = 0;
    std::int64_t stage04_max_reheats_ = 0;
    bool stage04_restart_enabled_ = false;
    std::int64_t stage04_restart_stagnation_threshold_ = 0;
    std::int64_t stage04_max_restarts_ = 0;
    bool stage04_intensification_enabled_ = false;
    std::int64_t stage04_intensification_iterations_ = 0;
    double stage04_reheat_factor_ = 0.0;
    double stage04_intensification_removal_fraction_ = 0.0;
    std::int64_t stage04_reheat_count_ = 0;
    std::int64_t stage04_restart_count_ = 0;
    double stage04_reheat_floor_ = 0.0;
    bool stage04_intensification_active_ = false;
    std::int64_t stage04_intensification_remaining_ = 0;
    std::int64_t stage04_segment_length_ = 0;
    std::int64_t stage04_min_calls_ = 0;
    double stage04_weight_reaction_ = 0.0;
    double stage04_weight_floor_ = 0.0;
    double stage04_weight_smoothing_ = 0.0;
    std::array<double, 7> stage04_rewards_{};
    std::array<double, 20> full_operator_weights_{};
    std::array<double, 20> full_operator_segment_rewards_{};
    std::array<std::int64_t, 20> full_operator_segment_calls_{};
    std::array<std::array<std::int64_t, 8>, 20> full_operator_totals_{};
    std::array<double, 4> constraint_weights_{1.0, 1.0, 1.0, 1.0};
    std::array<double, 4> constraint_segment_rewards_{};
    std::array<std::int64_t, 4> constraint_segment_calls_{};
    std::array<std::array<std::int64_t, 8>, 4> constraint_totals_{};
    std::int64_t last_finished_stage04_iteration_ = -1;
    std::int64_t last_completed_constraint_iteration_ = -1;
    std::int64_t main_stagnation_iterations_ = 0;
    bool last_iteration_global_best_improved_ = false;

    std::optional<std::int64_t> prepare_first_feasible_candidate(
        const py::array_t<std::int64_t>& plan_offsets,
        const py::array_t<std::int64_t>& route_offsets,
        const py::array_t<std::int64_t>& route_indices,
        const py::tuple& transaction) {
        auto feasible_order =
            py::cast<py::array_t<std::int64_t>>(transaction[11]);
        if (feasible_order.size() == 0) {
            return std::nullopt;
        }
        const auto plan_id = checked_data<std::int64_t>(feasible_order)[0];
        if (plan_id < 0 || plan_id + 1 >= plan_offsets.size()
            || !pending_candidate_exact_ready_) {
            throw std::logic_error(
                "full native selected plan lost its exact payload");
        }
        const auto* plan_boundaries =
            checked_data<std::int64_t>(plan_offsets);
        const auto first_route = plan_boundaries[plan_id];
        const auto end_route = plan_boundaries[plan_id + 1];
        if (first_route < 0 || end_route <= first_route
            || end_route >= route_offsets.size()) {
            throw std::logic_error(
                "full native selected plan has invalid route boundaries");
        }
        const auto* route_boundaries =
            checked_data<std::int64_t>(route_offsets);
        const auto first_index = route_boundaries[first_route];
        const auto end_index = route_boundaries[end_route];
        const auto selected_route_count = end_route - first_route;
        py::array_t<std::int64_t> candidate_offsets(selected_route_count + 1);
        for (std::int64_t route = 0; route <= selected_route_count; ++route) {
            checked_data(candidate_offsets)[route] =
                route_boundaries[first_route + route] - first_index;
        }
        py::array_t<std::int64_t> candidate_indices(end_index - first_index);
        std::copy(
            checked_data<std::int64_t>(route_indices) + first_index,
            checked_data<std::int64_t>(route_indices) + end_index,
            checked_data(candidate_indices));

        auto all_path_offsets = py::cast<py::array_t<std::int64_t>>(
            pending_candidate_exact_payload_[0]);
        auto all_path_indices = py::cast<py::array_t<std::int64_t>>(
            pending_candidate_exact_payload_[1]);
        auto all_statuses = py::cast<py::array_t<std::int64_t>>(
            pending_candidate_exact_payload_[2]);
        auto all_reasons = py::cast<py::array_t<std::int64_t>>(
            pending_candidate_exact_payload_[3]);
        auto all_metrics = py::cast<py::array_t<double>>(
            pending_candidate_exact_payload_[4]);
        auto all_labels = py::cast<py::array_t<std::int64_t>>(
            pending_candidate_exact_payload_[5]);
        const auto* all_path_boundaries =
            checked_data<std::int64_t>(all_path_offsets);
        const auto first_path = all_path_boundaries[first_route];
        const auto end_path = all_path_boundaries[end_route];
        py::array_t<std::int64_t> candidate_path_offsets(
            selected_route_count + 1);
        for (std::int64_t route = 0; route <= selected_route_count; ++route) {
            checked_data(candidate_path_offsets)[route] =
                all_path_boundaries[first_route + route] - first_path;
        }
        py::array_t<std::int64_t> candidate_path_indices(end_path - first_path);
        std::copy(
            checked_data<std::int64_t>(all_path_indices) + first_path,
            checked_data<std::int64_t>(all_path_indices) + end_path,
            checked_data(candidate_path_indices));
        py::array_t<std::int64_t> candidate_statuses(selected_route_count);
        py::array_t<std::int64_t> candidate_reasons(selected_route_count);
        py::array_t<double> candidate_metrics(
            {static_cast<py::ssize_t>(selected_route_count), py::ssize_t(4)});
        py::array_t<std::int64_t> candidate_labels(
            {static_cast<py::ssize_t>(selected_route_count), py::ssize_t(3)});
        std::copy(
            checked_data<std::int64_t>(all_statuses) + first_route,
            checked_data<std::int64_t>(all_statuses) + end_route,
            checked_data(candidate_statuses));
        std::copy(
            checked_data<std::int64_t>(all_reasons) + first_route,
            checked_data<std::int64_t>(all_reasons) + end_route,
            checked_data(candidate_reasons));
        std::copy(
            checked_data<double>(all_metrics) + first_route * 4,
            checked_data<double>(all_metrics) + end_route * 4,
            checked_data(candidate_metrics));
        std::copy(
            checked_data<std::int64_t>(all_labels) + first_route * 3,
            checked_data<std::int64_t>(all_labels) + end_route * 3,
            checked_data(candidate_labels));
        auto candidate_exact = py::make_tuple(
            std::move(candidate_path_offsets),
            std::move(candidate_path_indices),
            std::move(candidate_statuses),
            std::move(candidate_reasons),
            std::move(candidate_metrics),
            std::move(candidate_labels));

        auto objective_integer_matrix =
            py::cast<py::array_t<std::int64_t>>(transaction[2]);
        auto objective_float_matrix =
            py::cast<py::array_t<double>>(transaction[3]);
        py::array_t<std::int64_t> candidate_objective_integer(2);
        py::array_t<double> candidate_objective_float(2);
        std::copy(
            checked_data<std::int64_t>(objective_integer_matrix) + plan_id * 2,
            checked_data<std::int64_t>(objective_integer_matrix) + plan_id * 2 + 2,
            checked_data(candidate_objective_integer));
        std::copy(
            checked_data<double>(objective_float_matrix) + plan_id * 2,
            checked_data<double>(objective_float_matrix) + plan_id * 2 + 2,
            checked_data(candidate_objective_float));
        last_candidate_offsets_ = std::move(candidate_offsets);
        last_candidate_indices_ = std::move(candidate_indices);
        last_candidate_exact_payload_ = std::move(candidate_exact);
        last_candidate_objective_integer_ =
            std::move(candidate_objective_integer);
        last_candidate_objective_float_ = std::move(candidate_objective_float);
        last_candidate_ready_ = true;
        return plan_id;
    }

    void accumulate_constraint_stage04_outcome_noexcept(
        std::size_t operation,
        bool accepted,
        std::int64_t comparison,
        bool is_global_best,
        bool vehicle_reduction,
        std::array<double, 4>& segment_rewards,
        std::array<std::int64_t, 4>& segment_calls,
        std::array<std::array<std::int64_t, 8>, 4>& totals) const noexcept {
        auto& operation_totals = totals[operation];
        ++operation_totals[0];
        auto reward = stage04_rewards_[0];
        if (accepted) {
            ++operation_totals[1];
            if (comparison < 0) {
                ++operation_totals[2];
            } else if (comparison == 0) {
                ++operation_totals[3];
            } else {
                ++operation_totals[4];
            }
            reward = is_global_best
                ? (vehicle_reduction ? stage04_rewards_[6] : stage04_rewards_[5])
                : comparison < 0
                ? (vehicle_reduction ? stage04_rewards_[4] : stage04_rewards_[3])
                : comparison == 0 ? stage04_rewards_[2] : stage04_rewards_[1];
            if (is_global_best) {
                ++operation_totals[6];
            }
            if (vehicle_reduction) {
                ++operation_totals[7];
            }
        } else {
            ++operation_totals[5];
        }
        segment_rewards[operation] += reward;
        ++segment_calls[operation];
    }

    [[nodiscard]] std::int64_t last_candidate_comparison() const {
        if (!last_candidate_ready_) {
            throw std::logic_error(
                "full native candidate comparison has no prepared candidate");
        }
        const auto candidate = evrptw::formal_objective::key_from_arrays(
            last_candidate_objective_integer_, last_candidate_objective_float_);
        const auto current = evrptw::formal_objective::key_from_arrays(
            current_objective_integer_, current_objective_float_);
        return candidate < current ? -1 : candidate == current ? 0 : 1;
    }

    void accumulate_full_stage04_outcome_noexcept(
        std::size_t operation,
        bool accepted,
        std::int64_t comparison,
        bool is_global_best,
        bool vehicle_reduction,
        bool adaptive) noexcept {
        const auto outcome_flags = (comparison < 0 ? std::int64_t{1}
            : comparison == 0 ? std::int64_t{2} : std::int64_t{4})
            | (is_global_best ? std::int64_t{8} : std::int64_t{0})
            | (vehicle_reduction ? std::int64_t{16} : std::int64_t{0})
            | (adaptive ? std::int64_t{32} : std::int64_t{0});
        causal_journal_.append(
            NativeCausalStreamCode::operator_event,
            NativeCausalEventCode::operator_outcome,
            causal_context_[0], static_cast<std::int64_t>(operation),
            causal_context_[2], causal_context_transaction_id_,
            static_cast<std::int64_t>(operation), accepted ? 1 : 0,
            outcome_flags);
        causal_journal_.append(
            NativeCausalStreamCode::stage04,
            NativeCausalEventCode::stage04_outcome,
            causal_context_[0], static_cast<std::int64_t>(operation),
            causal_context_[2], causal_context_transaction_id_,
            static_cast<std::int64_t>(operation), accepted ? 1 : 0,
            outcome_flags);
        auto& totals = full_operator_totals_[operation];
        ++totals[0];
        auto reward = stage04_rewards_[0];
        if (accepted) {
            ++totals[1];
            if (comparison < 0) {
                ++totals[2];
            } else if (comparison == 0) {
                ++totals[3];
            } else {
                ++totals[4];
            }
            reward = is_global_best
                ? (vehicle_reduction ? stage04_rewards_[6] : stage04_rewards_[5])
                : comparison < 0
                ? (vehicle_reduction ? stage04_rewards_[4] : stage04_rewards_[3])
                : comparison == 0 ? stage04_rewards_[2] : stage04_rewards_[1];
            if (is_global_best) {
                ++totals[6];
            }
            if (vehicle_reduction) {
                ++totals[7];
            }
        } else {
            ++totals[5];
        }
        if (adaptive) {
            full_operator_segment_rewards_[operation] += reward;
            ++full_operator_segment_calls_[operation];
        }
    }

    static py::tuple owned_exact_state_copy(const py::tuple& payload) {
        if (payload.size() < 6) {
            throw std::logic_error(
                "full native exact state lost its typed schema");
        }
        auto copied = py::make_tuple(
            owned_array_copy<std::int64_t>(payload[0], "path_offsets", 1),
            owned_array_copy<std::int64_t>(payload[1], "path_indices", 1),
            owned_array_copy<std::int64_t>(payload[2], "result_statuses", 1),
            owned_array_copy<std::int64_t>(payload[3], "reason_codes", 1),
            owned_array_copy<double>(payload[4], "result_metrics", 2),
            owned_array_copy<std::int64_t>(payload[5], "label_counters", 2));
        if (payload.size() == 6) {
            return copied;
        }
        if (payload.size() != 7) {
            throw std::logic_error(
                "full native exact state has an unknown typed schema");
        }
        return py::make_tuple(
            copied[0], copied[1], copied[2], copied[3], copied[4], copied[5],
            owned_array_copy<std::int64_t>(payload[6], "batch_counters", 1));
    }

    void swap_active_with_lane_noexcept(std::int64_t lane) noexcept {
        if (lane == 0) {
            std::swap(current_offsets_, legacy_offsets_);
            std::swap(current_indices_, legacy_indices_);
            std::swap(current_exact_payload_, legacy_exact_payload_);
            std::swap(current_objective_integer_, legacy_objective_integer_);
            std::swap(current_objective_float_, legacy_objective_float_);
            return;
        }
        std::swap(current_offsets_, quality_offsets_);
        std::swap(current_indices_, quality_indices_);
        std::swap(current_exact_payload_, quality_exact_payload_);
        std::swap(current_objective_integer_, quality_objective_integer_);
        std::swap(current_objective_float_, quality_objective_float_);
    }

    void record_exact_backend_metrics(
        const py::tuple& payload,
        double elapsed_seconds) {
        if (payload.size() != 7) {
            throw std::logic_error(
                "full native exact backend metrics require the complete payload");
        }
        if (!std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0) {
            throw std::logic_error(
                "full native exact backend elapsed time is invalid");
        }
        auto counters = py::cast<py::array_t<std::int64_t>>(payload[6]);
        if (counters.ndim() != 1 || counters.size() != 10
            || (counters.flags() & py::array::c_style) == 0) {
            throw std::logic_error(
                "full native exact backend counters lost their typed schema");
        }
        const auto* values = checked_data<std::int64_t>(counters);
        if (values[0] < 0 || values[1] < 0 || values[2] < 0
            || values[3] < 0 || values[4] < 0 || values[5] < 0
            || values[6] < 0 || values[7] < 0 || values[8] < 0
            || values[9] <= 0 || values[0] != values[1]
            || values[1] != values[2] + values[3]
            || values[4] != values[8] || values[8] > 1) {
            throw std::logic_error(
                "full native exact backend counters are inconsistent");
        }
        if (exact_backend_totals_[9] != 0
            && exact_backend_totals_[9] != values[9]) {
            throw std::logic_error(
                "full native exact backend batch size changed during solve");
        }
        for (std::size_t index = 0; index < 9; ++index) {
            exact_backend_totals_[index] += values[index];
        }
        exact_backend_totals_[9] = values[9];
        if (values[8] == 1) {
            if (values[1] <= 0) {
                throw std::logic_error(
                    "full native exact launch has no started work");
            }
            exact_launch_occupancies_.push_back(values[1]);
        }
        exact_backend_seconds_ += elapsed_seconds;
    }

    void record_exact_journal_batch(
        const std::array<std::int64_t, 3>& context,
        const py::array_t<std::int64_t>& route_offsets,
        const py::array_t<std::int64_t>& route_indices,
        const py::tuple& exact_payload,
        const std::int64_t transaction_id) {
        if (exact_payload.size() != 7 || route_offsets.size() < 2) {
            throw std::logic_error(
                "full native exact journal received an invalid batch");
        }
        const auto route_count = static_cast<std::size_t>(route_offsets.size() - 1);
        auto path_offsets = py::cast<py::array_t<std::int64_t>>(exact_payload[0]);
        auto path_indices = py::cast<py::array_t<std::int64_t>>(exact_payload[1]);
        auto statuses = py::cast<py::array_t<std::int64_t>>(exact_payload[2]);
        auto reasons = py::cast<py::array_t<std::int64_t>>(exact_payload[3]);
        auto metrics = py::cast<py::array_t<double>>(exact_payload[4]);
        auto labels = py::cast<py::array_t<std::int64_t>>(exact_payload[5]);
        if (
            path_offsets.size() != static_cast<py::ssize_t>(route_count + 1)
            || statuses.size() != static_cast<py::ssize_t>(route_count)
            || reasons.size() != static_cast<py::ssize_t>(route_count)
            || metrics.ndim() != 2
            || metrics.shape(0) != static_cast<py::ssize_t>(route_count)
            || metrics.shape(1) != 4
            || labels.ndim() != 2
            || labels.shape(0) != static_cast<py::ssize_t>(route_count)
            || labels.shape(1) != 3) {
            throw std::logic_error(
                "full native exact journal batch shapes do not reconcile");
        }
        ExactJournalBatch batch;
        batch.context = context;
        batch.route_offsets.assign(
            checked_data<std::int64_t>(route_offsets),
            checked_data<std::int64_t>(route_offsets) + route_offsets.size());
        batch.route_indices.assign(
            checked_data<std::int64_t>(route_indices),
            checked_data<std::int64_t>(route_indices) + route_indices.size());
        batch.results.reserve(route_count);
        const auto* path_boundaries = checked_data<std::int64_t>(path_offsets);
        const auto* paths = checked_data<std::int64_t>(path_indices);
        for (std::size_t route = 0; route < route_count; ++route) {
            NativeRouteCacheV2::ExactPayload result;
            result.path.assign(
                paths + path_boundaries[route],
                paths + path_boundaries[route + 1]);
            result.status = checked_data<std::int64_t>(statuses)[route];
            result.reason = checked_data<std::int64_t>(reasons)[route];
            std::copy(
                checked_data<double>(metrics) + route * 4,
                checked_data<double>(metrics) + (route + 1) * 4,
                result.metrics.begin());
            std::copy(
                checked_data<std::int64_t>(labels) + route * 3,
                checked_data<std::int64_t>(labels) + (route + 1) * 3,
                result.label_counters.begin());
            const auto flags = std::int64_t{1} | std::int64_t{2}
                | (result.status == 0 ? std::int64_t{4} : std::int64_t{0});
            causal_journal_.append(
                NativeCausalStreamCode::exact,
                NativeCausalEventCode::exact_route_result,
                context[0], context[1], context[2], transaction_id,
                static_cast<std::int64_t>(route), result.status, flags);
            batch.results.push_back(std::move(result));
        }
        exact_journal_.push_back(std::move(batch));
    }

    void record_screening_outputs(
        const std::vector<evrptw::native_kernels::ScreenOutput>& outputs,
        const std::vector<std::size_t>& evaluated_rows,
        const std::vector<std::size_t>& semantic_rows,
        const std::array<std::int64_t, 3>& context,
        const std::int64_t* route_offsets,
        const std::int64_t* route_indices,
        const std::int64_t* negative_hits,
        std::int64_t negative_cache_hits,
        double elapsed_seconds,
        const std::int64_t transaction_id) {
        screening_statistics_[0] += static_cast<std::int64_t>(
            semantic_rows.size());
        screening_statistics_[3] += negative_cache_hits;
        screening_statistics_[5] += static_cast<std::int64_t>(
            evaluated_rows.size());
        for (const auto row : evaluated_rows) {
            if (outputs[row].codes[0] == 1) {
                ++screening_statistics_[6];
            } else {
                ++screening_statistics_[7];
            }
        }
        for (const auto row : semantic_rows) {
            const auto& output = outputs[row];
            if (output.codes[0] == 1) {
                ++screening_statistics_[1];
            } else {
                ++screening_statistics_[2];
                ++screening_reason_counts_[output.codes[1]];
            }
        }
        const std::unordered_set<std::size_t> physically_evaluated(
            evaluated_rows.begin(), evaluated_rows.end());
        std::unordered_set<std::size_t> physical_owner_recorded;
        for (std::size_t semantic = 0; semantic < semantic_rows.size(); ++semantic) {
            const auto unique_row = semantic_rows[semantic];
            const auto& output = outputs[unique_row];
            ScreeningJournalRow journal;
            journal.context = context;
            journal.route.assign(
                route_indices + route_offsets[semantic],
                route_indices + route_offsets[semantic + 1]);
            std::copy(output.codes.begin(), output.codes.end(), journal.codes.begin());
            std::copy(
                output.metrics.begin(), output.metrics.end(), journal.metrics.begin());
            journal.flags[0] = physically_evaluated.contains(unique_row) ? 1 : 0;
            journal.flags[1] = negative_hits[unique_row] != 0 ? 1 : 0;
            journal.flags[2] = (
                physically_evaluated.contains(unique_row)
                && physical_owner_recorded.insert(unique_row).second)
                ? 1
                : 0;
            journal.flags[3] = journal.flags[2] != 0
                ? output.reachability_queries : 0;
            const auto flags = journal.flags[0]
                | (journal.flags[1] << 1)
                | (journal.flags[2] << 2)
                | (journal.flags[3] << 8);
            causal_journal_.append(
                NativeCausalStreamCode::screening,
                NativeCausalEventCode::screening_decision,
                context[0], context[1], context[2], transaction_id,
                static_cast<std::int64_t>(semantic), output.codes[0], flags);
            screening_journal_.push_back(std::move(journal));
        }
        screening_seconds_ += elapsed_seconds;
    }

    void append_control_journal_causal(
        const std::list<ControlJournalBatch>& batches) {
        const auto before = causal_journal_.snapshot();
        try {
            for (const auto& batch : batches) {
                const auto plan_count = batch.plan_offsets.size() - 1;
                if (batch.decision_codes.size() != plan_count
                    || batch.statuses.size() != plan_count) {
                    throw std::logic_error(
                        "full native causal control batch shape is invalid");
                }
                const auto selected = [&batch](const std::size_t plan) {
                    return std::find(
                        batch.selected.begin(), batch.selected.end(),
                        static_cast<std::int64_t>(plan)) != batch.selected.end();
                };
                for (std::size_t plan = 0; plan < plan_count; ++plan) {
                    const auto decision = batch.decision_codes[plan];
                    const auto flags = (selected(plan) ? std::int64_t{1} : 0)
                        | (batch.statuses[plan] == 5 ? std::int64_t{2} : 0);
                    causal_journal_.append(
                        NativeCausalStreamCode::candidate_control,
                        NativeCausalEventCode::candidate_plan,
                        batch.context[0], batch.context[1], batch.context[2],
                        batch.transaction_id,
                        static_cast<std::int64_t>(plan), decision, flags);
                }
            }
        } catch (...) {
            causal_journal_.rollback_noexcept(before);
            throw;
        }
    }

    void append_causal_cache_events(
        const std::array<std::int64_t, 3>& context,
        const std::int64_t transaction_id,
        const py::array_t<std::int64_t>& hit_flags,
        const bool store) {
        const auto* hits = checked_data<std::int64_t>(hit_flags);
        const auto before = causal_journal_.snapshot();
        try {
            for (py::ssize_t route = 0; route < hit_flags.size(); ++route) {
                // A store receipt is emitted only for an actual cache miss;
                // cache lookups retain both hit and miss decisions.
                if (store && hits[route] != 0) {
                    continue;
                }
                causal_journal_.append(
                    NativeCausalStreamCode::cache,
                    store ? NativeCausalEventCode::cache_store
                          : NativeCausalEventCode::cache_lookup,
                    context[0], context[1], context[2], transaction_id,
                    static_cast<std::int64_t>(route), hits[route],
                    store ? std::int64_t{2} : std::int64_t{1});
            }
        } catch (...) {
            causal_journal_.rollback_noexcept(before);
            throw;
        }
    }

    void append_causal_exact_work(
        const std::array<std::int64_t, 3>& context,
        const std::int64_t transaction_id,
        const std::int64_t route_count) {
        causal_journal_.append(
            NativeCausalStreamCode::exact,
            NativeCausalEventCode::exact_work,
            context[0], context[1], context[2], transaction_id,
            route_count, route_count, 1);
    }

    [[nodiscard]] std::int64_t allocate_causal_transaction() noexcept {
        return next_causal_transaction_id_++;
    }

public:
    void append_causal_termination(
        const std::int64_t reason,
        const std::int64_t completed_iterations) {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || pending_composite_active_) {
            throw std::runtime_error(
                "full native causal termination is not at a committed boundary");
        }
        if (reason == 1 || reason == 2 || reason == 3) {
            causal_journal_.append(
                NativeCausalStreamCode::deadline,
                reason == 2 ? NativeCausalEventCode::deadline_boundary
                             : NativeCausalEventCode::budget_boundary,
                -1, -1, completed_iterations, -1, -1, reason, 1);
        }
        causal_journal_.append(
            NativeCausalStreamCode::termination,
            NativeCausalEventCode::termination,
            -1, -1, completed_iterations, -1, -1, reason, 0);
    }

    [[nodiscard]] py::tuple causal_journal_payload() const {
        std::unique_lock state_lock(state_mutex_, std::try_to_lock);
        if (!state_lock.owns_lock()) {
            throw std::runtime_error(
                "full native search engine already has an active operation");
        }
        if (!initialized_ || pending_composite_active_) {
            throw std::runtime_error(
                "full native causal journal is not at a committed boundary");
        }
        return causal_journal_.payload();
    }

private:
    void clear_pending_composite_noexcept() noexcept {
        pending_round_protocol_ = false;
        pending_negative_store_ = false;
        pending_attempted_mark_ = false;
        pending_budget_snapshot_.reset();
        pending_candidate_exact_ready_ = false;
        pending_causal_snapshot_.reset();
        pending_control_journal_.clear();
        pending_composite_active_ = false;
    }

    void commit_pending_composite_noexcept() noexcept {
        if (!pending_composite_active_) {
            return;
        }
        if (pending_round_protocol_) {
            route_cache_.commit_protocol_transaction_noexcept();
        }
        if (pending_negative_store_) {
            negative_cache_.commit_store_batch_noexcept();
        }
        if (pending_attempted_mark_) {
            attempted_plans_.commit_mark_batch_noexcept();
        }
        control_journal_.splice(
            control_journal_.end(), pending_control_journal_);
        clear_pending_composite_noexcept();
    }

    void rollback_pending_composite() noexcept {
        if (!pending_composite_active_ || !pending_budget_snapshot_.has_value()) {
            std::terminate();
        }
        if (pending_attempted_mark_) {
            attempted_plans_.rollback_mark_batch_noexcept();
        }
        if (pending_negative_store_) {
            negative_cache_.rollback_store_batch_noexcept();
        }
        if (pending_round_protocol_) {
            route_cache_.rollback_protocol_transaction_noexcept();
        }
        if (pending_causal_snapshot_.has_value()) {
            causal_journal_.rollback_noexcept(*pending_causal_snapshot_);
        }
        const auto budget_snapshot = *pending_budget_snapshot_;
        clear_pending_composite_noexcept();
        rollback_round_budget_preserving_exact(budget_snapshot);
    }

    void rollback_round_budget_preserving_exact(
        const NativeBudgetStateV2::NativeSnapshot& snapshot) noexcept {
        budget_.rollback_preserving_exact_noexcept(snapshot);
    }

    py::array_t<std::uint8_t> exact_semantic_hashes(
        const py::array_t<std::int64_t>& route_offsets,
        const py::array_t<std::int64_t>& route_indices,
        const py::tuple& exact_payload) const {
        const auto route_count = route_offsets.size() - 1;
        auto path_offsets = py::cast<py::array_t<std::int64_t>>(exact_payload[0]);
        auto path_indices = py::cast<py::array_t<std::int64_t>>(exact_payload[1]);
        auto statuses = py::cast<py::array_t<std::int64_t>>(exact_payload[2]);
        auto reasons = py::cast<py::array_t<std::int64_t>>(exact_payload[3]);
        auto metrics = py::cast<py::array_t<double>>(exact_payload[4]);
        auto labels = py::cast<py::array_t<std::int64_t>>(exact_payload[5]);
        py::array_t<std::uint8_t> hashes(
            {route_count, py::ssize_t(32)});
        const auto* route_boundaries = checked_data<std::int64_t>(route_offsets);
        const auto* routes = checked_data<std::int64_t>(route_indices);
        const auto* path_boundaries = checked_data<std::int64_t>(path_offsets);
        const auto* paths = checked_data<std::int64_t>(path_indices);
        for (py::ssize_t route = 0; route < route_count; ++route) {
            std::string evidence("stage05.2-native-route-result-v2");
            append_evidence_values(
                evidence,
                routes + route_boundaries[route],
                static_cast<std::size_t>(
                    route_boundaries[route + 1] - route_boundaries[route]));
            append_evidence_values(
                evidence, checked_data<std::int64_t>(statuses) + route, 1);
            append_evidence_values(
                evidence, checked_data<std::int64_t>(reasons) + route, 1);
            append_evidence_values(
                evidence, checked_data<double>(metrics) + route * 4, 4);
            append_evidence_values(
                evidence, checked_data<std::int64_t>(labels) + route * 3, 3);
            append_evidence_values(
                evidence,
                paths + path_boundaries[route],
                static_cast<std::size_t>(
                    path_boundaries[route + 1] - path_boundaries[route]));
            const auto digest = native_sha256_digest(evidence);
            std::copy(
                digest.begin(), digest.end(), checked_data(hashes) + route * 32);
        }
        return hashes;
    }

    static void append_json_hex_escape(std::string& output, std::uint16_t value) {
        constexpr char hexadecimal[] = "0123456789abcdef";
        output += "\\u";
        output.push_back(hexadecimal[(value >> 12U) & 0xFU]);
        output.push_back(hexadecimal[(value >> 8U) & 0xFU]);
        output.push_back(hexadecimal[(value >> 4U) & 0xFU]);
        output.push_back(hexadecimal[value & 0xFU]);
    }

    static void append_json_string(std::string& output, const std::string& value) {
        output.push_back('"');
        for (std::size_t offset = 0; offset < value.size();) {
            const auto first = static_cast<unsigned char>(value[offset]);
            if (first < 0x80U) {
                ++offset;
                switch (first) {
                case '"': output += "\\\""; break;
                case '\\': output += "\\\\"; break;
                case '\b': output += "\\b"; break;
                case '\f': output += "\\f"; break;
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                default:
                    if (first < 0x20U) {
                        append_json_hex_escape(
                            output, static_cast<std::uint16_t>(first));
                    } else {
                        output.push_back(static_cast<char>(first));
                    }
                    break;
                }
                continue;
            }
            std::uint32_t codepoint = 0;
            std::size_t width = 0;
            if ((first & 0xE0U) == 0xC0U) {
                codepoint = first & 0x1FU;
                width = 2;
            } else if ((first & 0xF0U) == 0xE0U) {
                codepoint = first & 0x0FU;
                width = 3;
            } else if ((first & 0xF8U) == 0xF0U) {
                codepoint = first & 0x07U;
                width = 4;
            } else {
                throw std::invalid_argument("full native node name is not valid UTF-8");
            }
            if (offset + width > value.size()) {
                throw std::invalid_argument("full native node name is truncated UTF-8");
            }
            for (std::size_t continuation = 1; continuation < width; ++continuation) {
                const auto byte = static_cast<unsigned char>(value[offset + continuation]);
                if ((byte & 0xC0U) != 0x80U) {
                    throw std::invalid_argument("full native node name is not valid UTF-8");
                }
                codepoint = (codepoint << 6U) | (byte & 0x3FU);
            }
            const auto minimum = width == 2 ? 0x80U : width == 3 ? 0x800U : 0x10000U;
            if (codepoint < minimum || codepoint > 0x10FFFFU
                || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
                throw std::invalid_argument("full native node name is non-canonical UTF-8");
            }
            offset += width;
            if (codepoint <= 0xFFFFU) {
                append_json_hex_escape(
                    output, static_cast<std::uint16_t>(codepoint));
            } else {
                const auto adjusted = codepoint - 0x10000U;
                append_json_hex_escape(
                    output,
                    static_cast<std::uint16_t>(0xD800U + (adjusted >> 10U)));
                append_json_hex_escape(
                    output,
                    static_cast<std::uint16_t>(0xDC00U + (adjusted & 0x3FFU)));
            }
        }
        output.push_back('"');
    }

    static void append_json_integer(std::string& output, std::int64_t value) {
        char buffer[32];
        const auto converted = std::to_chars(std::begin(buffer), std::end(buffer), value);
        if (converted.ec != std::errc{}) {
            throw std::runtime_error("full native JSON integer conversion failed");
        }
        output.append(buffer, converted.ptr);
    }

    static void append_json_float(std::string& output, double value) {
        if (std::isnan(value)) {
            output += "NaN";
            return;
        }
        if (std::isinf(value)) {
            output += std::signbit(value) ? "-Infinity" : "Infinity";
            return;
        }
        char buffer[64];
        const auto converted = std::to_chars(
            std::begin(buffer), std::end(buffer), value, std::chars_format::general);
        if (converted.ec != std::errc{}) {
            throw std::runtime_error("full native JSON float conversion failed");
        }
        output.append(buffer, converted.ptr);
        const auto first = output.size() - static_cast<std::size_t>(converted.ptr - buffer);
        if (output.find_first_of(".eE", first) == std::string::npos) {
            output += ".0";
        }
    }

    py::array_t<std::int64_t> exact_entry_bytes(
        const py::array_t<std::int64_t>& path_offsets,
        const py::array_t<std::int64_t>& path_indices,
        const py::array_t<std::int64_t>& statuses,
        const py::array_t<std::int64_t>& reasons,
        const py::array_t<double>& metrics,
        const py::array_t<std::int64_t>& labels) const {
        const auto route_count = path_offsets.size() - 1;
        py::array_t<std::int64_t> output(route_count);
        const auto* paths = checked_data<std::int64_t>(path_offsets);
        const auto* path_nodes = checked_data<std::int64_t>(path_indices);
        const auto* status_values = checked_data<std::int64_t>(statuses);
        const auto* reason_values = checked_data<std::int64_t>(reasons);
        const auto* metric_values = checked_data<double>(metrics);
        const auto* label_values = checked_data<std::int64_t>(labels);
        for (py::ssize_t route = 0; route < route_count; ++route) {
            const bool feasible = status_values[route] == 0;
            std::string payload;
            payload.reserve(256);
            payload += "{\"charged_energy\":";
            append_json_float(payload, metric_values[route * 4 + 2]);
            payload += ",\"charging_time\":";
            append_json_float(payload, metric_values[route * 4 + 3]);
            payload += ",\"distance\":";
            append_json_float(payload, metric_values[route * 4]);
            payload += ",\"failure_reason\":";
            if (feasible) {
                append_json_string(payload, "");
            } else if (reason_values[route] == 1) {
                append_json_string(
                    payload,
                    "no feasible station-insertion pattern for fixed customer order");
            } else {
                throw std::logic_error(
                    "full native cache cannot serialize an interrupted exact result");
            }
            payload += ",\"feasible\":";
            payload += feasible ? "true" : "false";
            payload += ",\"labels_expanded\":";
            append_json_integer(payload, label_values[route * 3 + 1]);
            payload += ",\"labels_generated\":";
            append_json_integer(payload, label_values[route * 3]);
            payload += ",\"labels_pruned\":";
            append_json_integer(payload, label_values[route * 3 + 2]);
            payload += ",\"route\":[";
            for (std::int64_t path = paths[route]; path < paths[route + 1]; ++path) {
                if (path != paths[route]) {
                    payload.push_back(',');
                }
                const auto node = path_nodes[path];
                if (node < 0 || node >= static_cast<std::int64_t>(node_names_.size())) {
                    throw std::logic_error(
                        "full native exact path references an unknown node name");
                }
                append_json_string(payload, node_names_[node]);
            }
            payload += "],\"total_energy\":";
            append_json_float(payload, metric_values[route * 4 + 1]);
            payload.push_back('}');
            checked_data(output)[route] = static_cast<std::int64_t>(128 + payload.size());
        }
        return output;
    }
};

thread_local std::shared_ptr<NativeWorkPool> scheduler_work_pool_context;
thread_local double scheduler_queue_wait_seconds_context = 0.0;
thread_local std::size_t scheduler_queue_depth_context = 0;

template <typename T>
std::vector<T> owned_search_vector(
    py::handle value,
    const char* name,
    int dimensions) {
    auto array = checked_array<T>(value, name, dimensions);
    return std::vector<T>(
        checked_data<T>(array), checked_data<T>(array) + array.size());
}

template <typename T, std::size_t Size>
std::array<T, Size> owned_search_array(
    py::handle value,
    const char* name) {
    auto array = checked_array<T>(value, name, 1);
    if (array.size() != static_cast<py::ssize_t>(Size)) {
        throw std::invalid_argument(
            std::string(name) + " has an invalid fixed shape");
    }
    std::array<T, Size> output{};
    std::copy(
        checked_data<T>(array), checked_data<T>(array) + Size,
        output.begin());
    return output;
}

evrptw::native_search::RequestV2 owned_native_search_request_v2(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle node_name_offsets,
    py::handle node_name_bytes,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining,
    py::handle protocol_control,
    py::handle protocol_options,
    py::handle stage04_integer,
    py::handle stage04_float,
    py::handle operator_integer,
    py::handle operator_float) {
    evrptw::native_search::RequestV2 request;
    auto& problem = request.problem;
    problem.node_kind = owned_search_vector<std::int64_t>(
        node_kind, "node_kind", 1);
    problem.demand = owned_search_vector<double>(demand, "demand", 1);
    problem.ready_time = owned_search_vector<double>(
        ready_time, "ready_time", 1);
    problem.due_date = owned_search_vector<double>(due_date, "due_date", 1);
    problem.service_time = owned_search_vector<double>(
        service_time, "service_time", 1);
    problem.distance = owned_search_vector<double>(distance, "distance", 2);
    problem.reachable = owned_search_vector<std::uint8_t>(
        reachable, "reachable", 2);
    problem.vehicle = owned_search_array<double, 5>(vehicle, "vehicle");
    problem.lexical_rank = owned_search_vector<std::int64_t>(
        lexical_rank, "lexical_rank", 1);
    problem.node_name_offsets = owned_search_vector<std::int64_t>(
        node_name_offsets, "node_name_offsets", 1);
    problem.node_name_bytes = owned_search_vector<std::uint8_t>(
        node_name_bytes, "node_name_bytes", 1);
    problem.initial_route_offsets = owned_search_vector<std::int64_t>(
        initial_route_offsets, "initial_route_offsets", 1);
    problem.initial_route_indices = owned_search_vector<std::int64_t>(
        initial_route_indices, "initial_route_indices", 1);
    auto& config = request.config;
    config.search_control = owned_search_array<std::int64_t, 5>(
        control, "control");
    auto deadline_array = checked_array<double>(
        deadline_remaining, "deadline_remaining", 1);
    if (deadline_array.size() != 1) {
        throw std::invalid_argument(
            "deadline_remaining has an invalid fixed shape");
    }
    config.deadline_remaining = checked_data<double>(deadline_array)[0];
    config.protocol_control = owned_search_array<std::int64_t, 13>(
        protocol_control, "protocol_control");
    config.protocol_options = owned_search_array<double, 2>(
        protocol_options, "protocol_options");
    config.stage04_integer = owned_search_array<std::int64_t, 15>(
        stage04_integer, "stage04_integer");
    config.stage04_float = owned_search_array<double, 15>(
        stage04_float, "stage04_float");
    config.operator_integer = owned_search_array<std::int64_t, 24>(
        operator_integer, "operator_integer");
    config.operator_float = owned_search_array<double, 7>(
        operator_float, "operator_float");
    request.validate();
    return request;
}

py::tuple native_search_request_receipt_v2(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle node_name_offsets,
    py::handle node_name_bytes,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining,
    py::handle protocol_control,
    py::handle protocol_options,
    py::handle stage04_integer,
    py::handle stage04_float,
    py::handle operator_integer,
    py::handle operator_float) {
    const auto request = owned_native_search_request_v2(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, lexical_rank, node_name_offsets, node_name_bytes,
        initial_route_offsets, initial_route_indices, control,
        deadline_remaining, protocol_control, protocol_options,
        stage04_integer, stage04_float, operator_integer, operator_float);
    py::array_t<std::int64_t> counts(8);
    auto* values = checked_data(counts);
    values[0] = static_cast<std::int64_t>(request.problem.node_count());
    values[1] = static_cast<std::int64_t>(request.problem.route_count());
    values[2] = static_cast<std::int64_t>(
        request.problem.initial_route_indices.size());
    values[3] = static_cast<std::int64_t>(
        request.problem.node_name_bytes.size());
    values[4] = static_cast<std::int64_t>(request.problem.distance.size());
    values[5] = request.config.search_control[1];
    values[6] = request.config.protocol_control[2];
    values[7] = request.config.search_control[3];
    return py::make_tuple(std::move(counts), request.sha256());
}

py::tuple full_native_alns_v2(
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle node_name_offsets,
    py::handle node_name_bytes,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining,
    py::handle protocol_control,
    py::handle protocol_options,
    py::handle stage04_integer,
    py::handle stage04_float,
    py::handle operator_integer,
    py::handle operator_float) {
    const auto owned_request = owned_native_search_request_v2(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, lexical_rank, node_name_offsets, node_name_bytes,
        initial_route_offsets, initial_route_indices, control,
        deadline_remaining, protocol_control, protocol_options,
        stage04_integer, stage04_float, operator_integer, operator_float);
    auto initial_offsets = checked_array<std::int64_t>(
        initial_route_offsets, "initial_route_offsets", 1);
    auto initial_indices = checked_array<std::int64_t>(
        initial_route_indices, "initial_route_indices", 1);
    auto base_control = checked_array<std::int64_t>(control, "control", 1);
    auto deadline = checked_array<double>(
        deadline_remaining, "deadline_remaining", 1);
    auto protocol_control_array = checked_array<std::int64_t>(
        protocol_control, "protocol_control", 1);
    auto protocol_options_array = checked_array<double>(
        protocol_options, "protocol_options", 1);
    auto stage04_integer_array = checked_array<std::int64_t>(
        stage04_integer, "stage04_integer", 1);
    auto stage04_float_array = checked_array<double>(
        stage04_float, "stage04_float", 1);
    auto operator_integer_array = checked_array<std::int64_t>(
        operator_integer, "operator_integer", 1);
    auto operator_float_array = checked_array<double>(
        operator_float, "operator_float", 1);
    if (initial_offsets.size() < 2 || initial_indices.size() == 0) {
        throw std::invalid_argument(
            "full native v2 requires a non-empty warm-start SoA");
    }
    if (base_control.size() != 5 || deadline.size() != 1
        || protocol_control_array.size() != 13
        || protocol_options_array.size() != 2
        || stage04_integer_array.size() != 15
        || stage04_float_array.size() != 15
        || operator_integer_array.size() != 24
        || operator_float_array.size() != 7) {
        throw std::invalid_argument(
            "full native v2 configuration arrays have an invalid shape");
    }
    const auto* protocol_values = owned_request.config.protocol_control.data();
    if ((protocol_values[0] != 0 && protocol_values[0] != 1)
        || protocol_values[1] <= 0 || protocol_values[2] <= 0
        || (protocol_values[3] != 1 && protocol_values[3] != 4)
        || protocol_values[4] < 10 || protocol_values[5] < 0) {
        throw std::invalid_argument(
            "full native v2 Candidate Control values are invalid");
    }
    const auto* options = owned_request.config.protocol_options.data();
    if (!(options[0] > 0.0 && options[0] <= 1.0)
        || !std::isfinite(options[1]) || options[1] <= 0.0) {
        throw std::invalid_argument("full native v2 protocol options are invalid");
    }
    const auto* base_control_values = owned_request.config.search_control.data();
    const auto* operator_integer_values =
        owned_request.config.operator_integer.data();
    const auto* operator_float_values =
        owned_request.config.operator_float.data();
    if (base_control_values[1] < 1) {
        throw std::invalid_argument(
            "full native v2 iteration count must be positive");
    }
    const auto solve_started = std::chrono::steady_clock::now();
    const auto client_dispatch_threads =
#ifdef __linux__
        native_kernel_scheduler_endpoint.empty()
            ? base_control_values[3]
            : std::int64_t{0};
#else
        base_control_values[3];
#endif
    NativeSearchEngineV2 engine(
        base_control_values[4],
        protocol_values[2],
        protocol_values[7],
        protocol_values[8],
        protocol_values[7],
        protocol_values[1],
        options[1],
        client_dispatch_threads,
        scheduler_work_pool_context);
    engine.configure_node_names_owned(owned_request.problem);
    engine.suppress_plan_screening_negative_cache(true);
    static_cast<void>(engine.initialize(
        node_kind,
        demand,
        ready_time,
        due_date,
        service_time,
        distance,
        reachable,
        vehicle,
        lexical_rank,
        initial_route_offsets,
        initial_route_indices,
        control,
        deadline_remaining));
    engine.configure_stage04(stage04_integer_array, stage04_float_array);
    if (!engine.initialized()) {
        throw std::logic_error("full native v2 search engine lost initialization state");
    }
    py::array_t<std::int64_t> thresholds(3);
    std::copy(
        operator_integer_values + 21,
        operator_integer_values + 24,
        checked_data(thresholds));
    py::array_t<double> fractions(6);
    std::copy(
        operator_float_values + 1,
        operator_float_values + 7,
        checked_data(fractions));
    py::array_t<std::int64_t> batch(1);
    checked_data(batch)[0] = base_control_values[2];
    // Every Stage 2.3 solve owns the same legacy, quality-shadow, and
    // constraint lanes, including a warm start that currently contains only
    // one route.  Route count is search state, not an execution-mode switch:
    // routing one-route solves through the old global prototype would replace
    // the real operator/RNG/Stage 4 trajectory with a synthetic event window.
    const auto three_lane = true;
    py::tuple global;
    py::tuple terminal_global;
    std::optional<py::array_t<std::int64_t>> terminal_override;
    if (three_lane) {
        const auto counts_toward_fixed_work_exhaustion = [](
            const py::tuple& semantic) {
            if (semantic.size() != 14 || semantic[4].is_none()) {
                return false;
            }
            const auto acceptance = py::cast<py::tuple>(semantic[4]);
            return acceptance.size() >= 1
                && py::cast<std::int64_t>(acceptance[0]) != 0;
        };
        const auto mark_three_lane_termination = [](
            const py::tuple& semantic,
            std::int64_t reason) {
            if (semantic.size() != 14 || reason < 1 || reason > 3) {
                throw std::invalid_argument(
                    "full native three-lane terminal semantic payload is invalid");
            }
            auto source_termination = py::cast<py::array_t<std::int64_t>>(
                semantic[11]);
            if (source_termination.size() != 6) {
                throw std::invalid_argument(
                    "full native three-lane terminal state is invalid");
            }
            py::array_t<std::int64_t> termination(6);
            std::copy(
                checked_data<std::int64_t>(source_termination),
                checked_data<std::int64_t>(source_termination) + 6,
                checked_data(termination));
            checked_data(termination)[0] = reason;
            py::tuple output(14);
            for (py::ssize_t index = 0; index < 13; ++index) {
                output[index] = index == 11
                    ? py::object(termination) : py::object(semantic[index]);
            }
            py::tuple semantic_payload(13);
            for (py::ssize_t index = 0; index < 13; ++index) {
                semantic_payload[index] = output[index];
            }
            std::string evidence(
                "stage05.2-native-three-lane-semantic-stream-v2");
            append_nested_evidence(evidence, semantic_payload);
            output[13] = native_sha256_hex(evidence);
            return output;
        };
        const auto pre_bootstrap_termination =
            engine.three_lane_termination_state(0, 0);
        const auto pre_bootstrap_started =
            checked_data<std::int64_t>(pre_bootstrap_termination)[2];
        auto bootstrap = engine.run_three_lane_bootstrap(
            operator_integer_values[0],
            operator_integer_values[4],
            -1, thresholds, fractions, deadline, batch);
        terminal_global = bootstrap;
        if (base_control_values[1] == 1
            || checked_data<std::int64_t>(
                py::cast<py::array_t<std::int64_t>>(bootstrap[11]))[0] != 0) {
            global = std::move(bootstrap);
        } else {
            py::list iteration_list;
            iteration_list.append(bootstrap);
            auto bootstrap_termination = py::cast<py::array_t<std::int64_t>>(
                bootstrap[11]);
            std::int64_t previous_started =
                checked_data<std::int64_t>(bootstrap_termination)[2];
            std::int64_t no_exact_rounds =
                counts_toward_fixed_work_exhaustion(bootstrap)
                && previous_started == pre_bootstrap_started
                ? 1 : 0;
            for (std::int64_t iteration = 1;
                 iteration < base_control_values[1]; ++iteration) {
                const auto elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - solve_started).count();
                const auto remaining = checked_data<double>(deadline)[0] - elapsed;
                if (remaining <= 0.0) {
                    terminal_override = engine.three_lane_termination_state(
                        2, iteration);
                    break;
                }
                py::array_t<double> followup_deadline(1);
                checked_data(followup_deadline)[0] = remaining;
                auto followup = engine.run_three_lane_followup(
                    iteration, base_control_values[1], options[0],
                    operator_integer_values[0],
                    operator_integer_values[4],
                    operator_integer_values[12],
                    operator_integer_values[13],
                    std::min(
                        operator_integer_values[8],
                        operator_integer_values[11]),
                    std::min(
                        operator_integer_values[9],
                        operator_integer_values[10]),
                    operator_integer_values[14],
                    operator_integer_values[15],
                    -1,
                    thresholds, fractions, followup_deadline, batch);
                auto followup_termination =
                    py::cast<py::array_t<std::int64_t>>(followup[11]);
                auto* followup_terminal_values =
                    checked_data<std::int64_t>(followup_termination);
                const auto started = followup_terminal_values[2];
                const auto exhaustion_eligible =
                    counts_toward_fixed_work_exhaustion(followup);
                if (exhaustion_eligible) {
                    no_exact_rounds = started == previous_started
                        ? no_exact_rounds + 1 : 0;
                }
                previous_started = started;
                if (exhaustion_eligible
                    && protocol_values[0] == 1
                    && followup_terminal_values[5] >= protocol_values[5]
                    && no_exact_rounds >= protocol_values[4]) {
                    followup = mark_three_lane_termination(followup, 3);
                    terminal_global = followup;
                    iteration_list.append(followup);
                    break;
                }
                terminal_global = followup;
                iteration_list.append(followup);
                if (followup_terminal_values[0] != 0) {
                    break;
                }
            }
            auto semantic_iterations = py::tuple(iteration_list);
            std::string semantic_evidence(
                "stage05.2-native-three-lane-search-stream-v2");
            append_nested_evidence(semantic_evidence, semantic_iterations);
            global = py::make_tuple(
                std::move(semantic_iterations), native_sha256_hex(semantic_evidence));
        }
    } else {
        // The global semantic controller emits one canonical round at a time.
        // Keep that transaction boundary intact while aggregating the rounds
        // into the flat payload consumed by the full-native result envelope.
        // This is the one-route counterpart of the multi-route bootstrap /
        // follow-up stream; no Python-side or scalar fallback is involved.
        py::list round_payloads;
        std::int64_t stagnation_iterations = 0;
        std::int64_t completed_rounds = 0;
        std::int64_t terminal_reason = 0;
        py::array_t<std::int64_t> first_termination;
        py::array_t<std::int64_t> last_termination;
        for (std::int64_t iteration = 0;
             iteration < base_control_values[1]; ++iteration) {
            const auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_started).count();
            const auto remaining = checked_data<double>(deadline)[0] - elapsed;
            if (remaining <= 0.0) {
                terminal_reason = 2;
                break;
            }
            py::array_t<double> round_deadline(1);
            checked_data(round_deadline)[0] = remaining;
            auto round = engine.run_global_search(
                iteration, 1, stagnation_iterations, thresholds, fractions,
                round_deadline, batch, -1);
            auto round_termination = py::cast<py::array_t<std::int64_t>>(
                round[13]);
            if (iteration == 0) {
                first_termination = round_termination;
            }
            last_termination = round_termination;
            const auto* round_terminal = checked_data<std::int64_t>(
                round_termination);
            const auto round_reason = round_terminal[0];
            const auto round_completed = round_terminal[2];
            if (round_reason == 0 || round_reason == 1) {
                if (round_reason == 1 && round_completed == 0) {
                    terminal_reason = 1;
                    break;
                }
                if (round_completed != 1) {
                    throw std::logic_error(
                        "native global round completion count is invalid");
                }
                ++completed_rounds;
                stagnation_iterations = engine.main_stagnation_iterations();
                round_payloads.append(round);
                terminal_reason = round_reason;
            } else if (round_reason == 2) {
                terminal_reason = 2;
                if (round_completed != 0) {
                    throw std::logic_error(
                        "native global interrupted round is not empty");
                }
                break;
            } else {
                throw std::logic_error(
                    "native global round returned an unknown termination reason");
            }
            if (round_reason != 0) {
                break;
            }
        }
        if (round_payloads.size() == 0) {
            if (!first_termination || terminal_reason != 1) {
                throw std::logic_error(
                    "native global controller produced no terminal payload");
            }
            py::array_t<std::int64_t> events(
                py::array::ShapeContainer{0, 26});
            py::array_t<double> ranking(0);
            py::array_t<std::int64_t> removed_offsets(1);
            py::array_t<std::int64_t> removed_indices(0);
            py::array_t<std::int64_t> plan_offsets(1);
            py::array_t<std::int64_t> route_offsets(1);
            py::array_t<std::int64_t> route_indices(0);
            py::array_t<std::int64_t> objective_integer(
                py::array::ShapeContainer{0, 2});
            py::array_t<double> objective_float(
                py::array::ShapeContainer{0, 2});
            py::array_t<std::int64_t> stage_status(
                py::array::ShapeContainer{0, 4});
            py::array_t<double> stage_weights(
                py::array::ShapeContainer{0, 4, 2});
            py::array_t<std::int64_t> stage_calls(
                py::array::ShapeContainer{0, 4});
            py::array_t<double> stage_rewards(
                py::array::ShapeContainer{0, 4});
            checked_data(removed_offsets)[0] = 0;
            checked_data(plan_offsets)[0] = 0;
            checked_data(route_offsets)[0] = 0;
            py::array_t<std::int64_t> termination(11);
            auto* terminal_values = checked_data(termination);
            const auto* first_values = checked_data<std::int64_t>(first_termination);
            terminal_values[0] = 1;
            terminal_values[1] = 0;
            terminal_values[2] = 0;
            terminal_values[3] = 0;
            std::copy(first_values + 4, first_values + 11, terminal_values + 4);
            std::string evidence(
                "stage05.2-native-global-semantic-stream-v2");
            append_evidence_array(evidence, events);
            append_evidence_array(evidence, ranking);
            append_evidence_array(evidence, removed_offsets);
            append_evidence_array(evidence, removed_indices);
            append_evidence_array(evidence, plan_offsets);
            append_evidence_array(evidence, route_offsets);
            append_evidence_array(evidence, route_indices);
            append_evidence_array(evidence, objective_integer);
            append_evidence_array(evidence, objective_float);
            append_evidence_array(evidence, stage_status);
            append_evidence_array(evidence, stage_weights);
            append_evidence_array(evidence, stage_calls);
            append_evidence_array(evidence, stage_rewards);
            append_evidence_array(evidence, termination);
            global = py::make_tuple(
                std::move(events), std::move(ranking),
                std::move(removed_offsets), std::move(removed_indices),
                std::move(plan_offsets), std::move(route_offsets),
                std::move(route_indices), std::move(objective_integer),
                std::move(objective_float), std::move(stage_status),
                std::move(stage_weights), std::move(stage_calls),
                std::move(stage_rewards), std::move(termination),
                native_sha256_hex(evidence));
            terminal_global = global;
        } else {
            std::vector<std::int64_t> event_values;
            std::vector<double> ranking_values;
            std::vector<std::int64_t> removed_offsets{0};
            std::vector<std::int64_t> removed_indices;
            std::vector<std::int64_t> plan_offsets{0};
            std::vector<std::int64_t> route_offsets{0};
            std::vector<std::int64_t> route_indices;
            std::vector<std::int64_t> objective_integer_values;
            std::vector<double> objective_float_values;
            std::vector<std::int64_t> stage_status_values;
            std::vector<double> stage_weight_values;
            std::vector<std::int64_t> stage_call_values;
            std::vector<double> stage_reward_values;
            std::int64_t route_count_base = 0;
            for (const auto& item : round_payloads) {
                auto round = py::cast<py::tuple>(item);
                auto round_events = py::cast<py::array_t<std::int64_t>>(round[0]);
                auto round_ranking = py::cast<py::array_t<double>>(round[1]);
                auto round_removed_offsets =
                    py::cast<py::array_t<std::int64_t>>(round[2]);
                auto round_removed_indices =
                    py::cast<py::array_t<std::int64_t>>(round[3]);
                auto round_plan_offsets =
                    py::cast<py::array_t<std::int64_t>>(round[4]);
                auto round_route_offsets =
                    py::cast<py::array_t<std::int64_t>>(round[5]);
                auto round_route_indices =
                    py::cast<py::array_t<std::int64_t>>(round[6]);
                auto round_objective_integer =
                    py::cast<py::array_t<std::int64_t>>(round[7]);
                auto round_objective_float =
                    py::cast<py::array_t<double>>(round[8]);
                auto round_stage_status =
                    py::cast<py::array_t<std::int64_t>>(round[9]);
                auto round_stage_weights =
                    py::cast<py::array_t<double>>(round[10]);
                auto round_stage_calls =
                    py::cast<py::array_t<std::int64_t>>(round[11]);
                auto round_stage_rewards =
                    py::cast<py::array_t<double>>(round[12]);
                const auto event_count = round_events.shape(0);
                event_values.insert(
                    event_values.end(), checked_data<std::int64_t>(round_events),
                    checked_data<std::int64_t>(round_events)
                        + event_count * 26);
                ranking_values.insert(
                    ranking_values.end(), checked_data<double>(round_ranking),
                    checked_data<double>(round_ranking) + event_count);
                const auto removed_base = removed_indices.size();
                for (py::ssize_t index = 1;
                     index < round_removed_offsets.size(); ++index) {
                    removed_offsets.push_back(
                        static_cast<std::int64_t>(removed_base)
                        + checked_data<std::int64_t>(round_removed_offsets)[index]);
                }
                removed_indices.insert(
                    removed_indices.end(),
                    checked_data<std::int64_t>(round_removed_indices),
                    checked_data<std::int64_t>(round_removed_indices)
                        + round_removed_indices.size());
                const auto plan_base = route_count_base;
                for (py::ssize_t index = 1;
                     index < round_plan_offsets.size(); ++index) {
                    plan_offsets.push_back(
                        plan_base
                        + checked_data<std::int64_t>(round_plan_offsets)[index]);
                }
                const auto route_base = route_indices.size();
                for (py::ssize_t index = 1;
                     index < round_route_offsets.size(); ++index) {
                    route_offsets.push_back(
                        static_cast<std::int64_t>(route_base)
                        + checked_data<std::int64_t>(round_route_offsets)[index]);
                }
                route_count_base += round_route_offsets.size() - 1;
                route_indices.insert(
                    route_indices.end(),
                    checked_data<std::int64_t>(round_route_indices),
                    checked_data<std::int64_t>(round_route_indices)
                        + round_route_indices.size());
                objective_integer_values.insert(
                    objective_integer_values.end(),
                    checked_data<std::int64_t>(round_objective_integer),
                    checked_data<std::int64_t>(round_objective_integer)
                        + event_count * 2);
                objective_float_values.insert(
                    objective_float_values.end(),
                    checked_data<double>(round_objective_float),
                    checked_data<double>(round_objective_float)
                        + event_count * 2);
                stage_status_values.insert(
                    stage_status_values.end(),
                    checked_data<std::int64_t>(round_stage_status),
                    checked_data<std::int64_t>(round_stage_status)
                        + round_stage_status.size());
                stage_weight_values.insert(
                    stage_weight_values.end(),
                    checked_data<double>(round_stage_weights),
                    checked_data<double>(round_stage_weights)
                        + round_stage_weights.size());
                stage_call_values.insert(
                    stage_call_values.end(),
                    checked_data<std::int64_t>(round_stage_calls),
                    checked_data<std::int64_t>(round_stage_calls)
                        + round_stage_calls.size());
                stage_reward_values.insert(
                    stage_reward_values.end(),
                    checked_data<double>(round_stage_rewards),
                    checked_data<double>(round_stage_rewards)
                        + round_stage_rewards.size());
            }
            const auto event_count = ranking_values.size();
            py::array_t<std::int64_t> events({
                static_cast<py::ssize_t>(event_count), py::ssize_t(26)});
            std::copy(event_values.begin(), event_values.end(), checked_data(events));
            py::array_t<double> ranking(event_count);
            std::copy(ranking_values.begin(), ranking_values.end(), checked_data(ranking));
            py::array_t<std::int64_t> removed_offsets_array(removed_offsets.size());
            std::copy(
                removed_offsets.begin(), removed_offsets.end(),
                checked_data(removed_offsets_array));
            py::array_t<std::int64_t> removed_indices_array(removed_indices.size());
            std::copy(
                removed_indices.begin(), removed_indices.end(),
                checked_data(removed_indices_array));
            py::array_t<std::int64_t> plan_offsets_array(plan_offsets.size());
            std::copy(
                plan_offsets.begin(), plan_offsets.end(),
                checked_data(plan_offsets_array));
            py::array_t<std::int64_t> route_offsets_array(route_offsets.size());
            std::copy(
                route_offsets.begin(), route_offsets.end(),
                checked_data(route_offsets_array));
            py::array_t<std::int64_t> route_indices_array(route_indices.size());
            std::copy(
                route_indices.begin(), route_indices.end(),
                checked_data(route_indices_array));
            py::array_t<std::int64_t> objective_integer({
                static_cast<py::ssize_t>(event_count), py::ssize_t(2)});
            std::copy(
                objective_integer_values.begin(), objective_integer_values.end(),
                checked_data(objective_integer));
            py::array_t<double> objective_float({
                static_cast<py::ssize_t>(event_count), py::ssize_t(2)});
            std::copy(
                objective_float_values.begin(), objective_float_values.end(),
                checked_data(objective_float));
            py::array_t<std::int64_t> stage_status({
                static_cast<py::ssize_t>(completed_rounds), py::ssize_t(4)});
            std::copy(
                stage_status_values.begin(), stage_status_values.end(),
                checked_data(stage_status));
            py::array_t<double> stage_weights({
                static_cast<py::ssize_t>(completed_rounds), py::ssize_t(4),
                py::ssize_t(2)});
            std::copy(
                stage_weight_values.begin(), stage_weight_values.end(),
                checked_data(stage_weights));
            py::array_t<std::int64_t> stage_calls({
                static_cast<py::ssize_t>(completed_rounds), py::ssize_t(4)});
            std::copy(
                stage_call_values.begin(), stage_call_values.end(),
                checked_data(stage_calls));
            py::array_t<double> stage_rewards({
                static_cast<py::ssize_t>(completed_rounds), py::ssize_t(4)});
            std::copy(
                stage_reward_values.begin(), stage_reward_values.end(),
                checked_data(stage_rewards));
            py::array_t<std::int64_t> termination(11);
            auto* terminal_values = checked_data(termination);
            terminal_values[0] = terminal_reason;
            terminal_values[1] = 0;
            terminal_values[2] = completed_rounds;
            terminal_values[3] = completed_rounds;
            const auto* first_values = checked_data<std::int64_t>(first_termination);
            const auto* last_values = checked_data<std::int64_t>(last_termination);
            terminal_values[4] = last_values[4];
            std::copy(first_values + 5, first_values + 8, terminal_values + 5);
            std::copy(last_values + 8, last_values + 11, terminal_values + 8);
            std::string evidence(
                "stage05.2-native-global-semantic-stream-v2");
            append_evidence_array(evidence, events);
            append_evidence_array(evidence, ranking);
            append_evidence_array(evidence, removed_offsets_array);
            append_evidence_array(evidence, removed_indices_array);
            append_evidence_array(evidence, plan_offsets_array);
            append_evidence_array(evidence, route_offsets_array);
            append_evidence_array(evidence, route_indices_array);
            append_evidence_array(evidence, objective_integer);
            append_evidence_array(evidence, objective_float);
            append_evidence_array(evidence, stage_status);
            append_evidence_array(evidence, stage_weights);
            append_evidence_array(evidence, stage_calls);
            append_evidence_array(evidence, stage_rewards);
            append_evidence_array(evidence, termination);
            global = py::make_tuple(
                std::move(events), std::move(ranking),
                std::move(removed_offsets_array), std::move(removed_indices_array),
                std::move(plan_offsets_array), std::move(route_offsets_array),
                std::move(route_indices_array), std::move(objective_integer),
                std::move(objective_float), std::move(stage_status),
                std::move(stage_weights), std::move(stage_calls),
                std::move(stage_rewards), std::move(termination),
                native_sha256_hex(evidence));
            terminal_global = global;
        }
    }
    auto best = engine.best_solution_payload();
    auto backend_metrics = engine.exact_backend_metrics_payload();
    auto exact_journal = engine.exact_journal_payload();
    auto control_journal = engine.control_journal_payload();
    auto route_offsets = py::cast<py::array_t<std::int64_t>>(best[0]);
    auto route_indices = py::cast<py::array_t<std::int64_t>>(best[1]);
    auto exact_state = py::cast<py::tuple>(best[2]);
    py::tuple exact_payload(7);
    for (py::ssize_t index = 0; index < 6; ++index) {
        exact_payload[index] = exact_state[index];
    }
    const auto best_route_count = route_offsets.size() - 1;
    if (exact_state.size() >= 7) {
        exact_payload[6] = exact_state[6];
    } else {
        py::array_t<std::int64_t> exact_batch_counters(10);
        auto* exact_counter_values = checked_data(exact_batch_counters);
        exact_counter_values[0] = best_route_count;
        exact_counter_values[1] = best_route_count;
        exact_counter_values[2] = best_route_count;
        exact_counter_values[3] = 0;
        exact_counter_values[4] = best_route_count > 0 ? 1 : 0;
        exact_counter_values[5] = 0;
        exact_counter_values[6] = 0;
        exact_counter_values[7] = 0;
        exact_counter_values[8] = best_route_count > 0 ? 1 : 0;
        exact_counter_values[9] = base_control_values[2];
        exact_payload[6] = std::move(exact_batch_counters);
    }
    py::array_t<std::int64_t> events;
    py::array_t<std::int64_t> termination;
    if (three_lane) {
        events = py::array_t<std::int64_t>(
            py::array::ShapeContainer{0, 26});
        termination = terminal_override.has_value()
            ? std::move(*terminal_override)
            : py::cast<py::array_t<std::int64_t>>(terminal_global[11]);
    } else {
        events = py::cast<py::array_t<std::int64_t>>(global[0]);
        termination = py::cast<py::array_t<std::int64_t>>(global[13]);
    }
    const auto* terminal_values = checked_data<std::int64_t>(termination);
    py::array_t<std::int64_t> counters(8);
    auto* counter_values = checked_data(counters);
    counter_values[0] = three_lane
        ? (terminal_values[0] == 0 ? 1 : 0) : terminal_values[2];
    counter_values[1] = three_lane ? terminal_values[2] : terminal_values[8];
    counter_values[2] = three_lane ? terminal_values[3] : terminal_values[9];
    std::vector<py::tuple> legacy_acceptances;
    if (three_lane) {
        if (global.size() == 2) {
            auto iteration_payloads = py::cast<py::tuple>(global[0]);
            for (const auto& item : iteration_payloads) {
                auto iteration_payload = py::cast<py::tuple>(item);
                if (!iteration_payload[4].is_none()) {
                    legacy_acceptances.push_back(
                        py::cast<py::tuple>(iteration_payload[4]));
                }
            }
        } else if (!global[4].is_none()) {
            legacy_acceptances.push_back(py::cast<py::tuple>(global[4]));
        }
    }
    counter_values[0] = three_lane ? terminal_values[5] : terminal_values[2];
    counter_values[3] = 0;
    counter_values[4] = 0;
    for (const auto& acceptance : legacy_acceptances) {
        counter_values[3] += py::cast<std::int64_t>(acceptance[0]);
        counter_values[4] += py::cast<std::int64_t>(acceptance[1]);
    }
    counter_values[5] = three_lane
        && (terminal_values[0] == 1 || terminal_values[0] == 2)
        ? 0 : counter_values[0] - counter_values[3];
    counter_values[6] = three_lane ? terminal_values[4] : terminal_values[10];
    counter_values[7] = 0;
    py::array_t<std::int64_t> trajectory(
        {static_cast<py::ssize_t>(counter_values[0]), py::ssize_t(7)});
    if (counter_values[0] > 0) {
        for (std::int64_t iteration = 0;
             iteration < counter_values[0]; ++iteration) {
            auto* row = checked_data(trajectory) + iteration * 7;
            row[0] = iteration;
            row[1] = three_lane
                ? (iteration == 0 ? 2
                    : iteration == 1 ? 1
                    : iteration == 2 ? 3 : 0)
                : 9;
            row[2] = route_offsets.size() - 1;
            row[3] = counter_values[1];
            row[4] = three_lane
                ? 1
                : events.shape(0) >= 3
                ? checked_data<std::int64_t>(events)[2 * 26 + 4]
                : 0;
            row[5] = three_lane
                && iteration < static_cast<std::int64_t>(
                    legacy_acceptances.size())
                ? py::cast<std::int64_t>(
                    legacy_acceptances[static_cast<std::size_t>(iteration)][0])
                : !three_lane && events.shape(0) >= 3
                ? checked_data<std::int64_t>(events)[2 * 26 + 6]
                : 0;
            row[6] = three_lane
                && iteration < static_cast<std::int64_t>(
                    legacy_acceptances.size())
                ? py::cast<std::int64_t>(
                    legacy_acceptances[static_cast<std::size_t>(iteration)][2])
                : !three_lane && events.shape(0) >= 3
                ? checked_data<std::int64_t>(events)[2 * 26 + 7]
                : 0;
        }
    }
    engine.append_causal_termination(terminal_values[0], terminal_values[5]);
    auto causal_journal = engine.causal_journal_payload();
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_started).count();
    auto backend_timings = py::cast<py::array_t<double>>(backend_metrics[2]);
    if (backend_timings.ndim() != 1 || backend_timings.size() != 1
        || (backend_timings.flags() & py::array::c_style) == 0) {
        throw std::logic_error(
            "full native v2 backend timings lost their typed schema");
    }
    const auto exact_seconds = checked_data<double>(backend_timings)[0];
    if (!std::isfinite(exact_seconds) || exact_seconds < 0.0
        || exact_seconds > elapsed) {
        throw std::logic_error(
            "full native v2 exact timing is outside the solve interval");
    }
#ifdef __linux__
    const auto remote_telemetry =
        evrptw::native_client::telemetry_snapshot();
#endif
    py::array_t<double> timings(12);
    checked_data(timings)[0] = elapsed - exact_seconds;
    checked_data(timings)[1] = exact_seconds;
    checked_data(timings)[2] = elapsed;
    checked_data(timings)[3] =
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? remote_telemetry.queue_wait_seconds
        :
#endif
          scheduler_queue_wait_seconds_context;
    checked_data(timings)[4] = static_cast<double>(
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? remote_telemetry.peak_active_tasks
        :
#endif
        engine.work_pool_peak_active_tasks());
    checked_data(timings)[5] = static_cast<double>(
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? 0
        :
#endif
        engine.work_pool_active_tasks());
    checked_data(timings)[6] = static_cast<double>(
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? remote_telemetry.peak_queue_depth
        :
#endif
          scheduler_queue_depth_context);
    checked_data(timings)[7] =
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ||
#endif
        scheduler_work_pool_context ? 1.0 : 0.0;
    checked_data(timings)[8] = static_cast<double>(
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? remote_telemetry.pool_thread_count
        :
#endif
        engine.work_pool_thread_count());
    checked_data(timings)[9] = static_cast<double>(
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? engine.work_pool_thread_count()
        : 0
#else
        0
#endif
    );
    checked_data(timings)[10] = static_cast<double>(
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? remote_telemetry.request_count
        : 0
#else
        0
#endif
    );
    checked_data(timings)[11] = static_cast<double>(
#ifdef __linux__
        !native_kernel_scheduler_endpoint.empty()
        ? remote_telemetry.screening_batch_request_count
        : 0
#else
        0
#endif
    );
    std::string evidence("stage05.2-full-native-alns-v2");
    const auto append_raw_array = [&](const auto& array) {
        evidence.append(
            reinterpret_cast<const char*>(array.data()),
            static_cast<std::size_t>(array.nbytes()));
    };
    append_raw_array(route_offsets);
    append_raw_array(route_indices);
    for (const auto item : exact_payload) {
        const auto array = py::cast<py::array>(item);
        if ((array.flags() & py::array::c_style) == 0) {
            throw std::logic_error(
                "full native v2 exact output is not contiguous");
        }
        evidence.append(
            reinterpret_cast<const char*>(array.data()),
            static_cast<std::size_t>(array.nbytes()));
    }
    append_raw_array(counters);
    append_raw_array(trajectory);
    const auto semantic_sha256 = py::cast<std::string>(
        global[three_lane ? (global.size() == 2 ? 1 : 13) : 14]);
    evidence.append(semantic_sha256);
    for (py::ssize_t index = 0; index < 2; ++index) {
        const auto item = backend_metrics[index];
        const auto array = py::cast<py::array>(item);
        if ((array.flags() & py::array::c_style) == 0) {
            throw std::logic_error(
                "full native v2 backend metrics output is not contiguous");
        }
        evidence.append(
            reinterpret_cast<const char*>(array.data()),
            static_cast<std::size_t>(array.nbytes()));
    }
    evidence.append(py::cast<std::string>(exact_journal[1]));
    evidence.append(py::cast<std::string>(control_journal[2]));
    evidence.append(py::cast<std::string>(causal_journal[11]));
    return py::make_tuple(
        std::move(route_offsets),
        std::move(route_indices),
        std::move(exact_payload),
        std::move(counters),
        std::move(timings),
        std::move(trajectory),
        native_sha256_hex(evidence),
        std::move(global),
        std::move(backend_metrics),
        std::move(exact_journal),
        std::move(control_journal),
        std::move(causal_journal));
}

#ifdef __linux__
py::tuple native_search_request_host_receipt_v2(
    const std::string& socket_path,
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle node_name_offsets,
    py::handle node_name_bytes,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining,
    py::handle protocol_control,
    py::handle protocol_options,
    py::handle stage04_integer,
    py::handle stage04_float,
    py::handle operator_integer,
    py::handle operator_float) {
    if (socket_path.empty()) {
        throw std::invalid_argument(
            "native search-request scheduler endpoint is empty");
    }
    const auto request = owned_native_search_request_v2(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, lexical_rank, node_name_offsets, node_name_bytes,
        initial_route_offsets, initial_route_indices, control,
        deadline_remaining, protocol_control, protocol_options,
        stage04_integer, stage04_float, operator_integer, operator_float);
    evrptw::native_client::SearchRequestReceipt receipt;
    {
        py::gil_scoped_release release;
        receipt = evrptw::native_client::search_request_receipt(
            socket_path, request);
    }
    py::array_t<std::int64_t> counts(receipt.counts.size());
    std::copy(
        receipt.counts.begin(), receipt.counts.end(), checked_data(counts));
    return py::make_tuple(std::move(counts), std::move(receipt.sha256));
}

py::tuple full_native_alns_host_v2(
    const std::string& socket_path,
    py::handle node_kind,
    py::handle demand,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle reachable,
    py::handle vehicle,
    py::handle lexical_rank,
    py::handle node_name_offsets,
    py::handle node_name_bytes,
    py::handle initial_route_offsets,
    py::handle initial_route_indices,
    py::handle control,
    py::handle deadline_remaining,
    py::handle protocol_control,
    py::handle protocol_options,
    py::handle stage04_integer,
    py::handle stage04_float,
    py::handle operator_integer,
    py::handle operator_float) {
    if (socket_path.empty() || !native_kernel_scheduler_endpoint.empty()) {
        throw std::invalid_argument(
            "full native host kernel scheduler endpoint is invalid");
    }
    evrptw::native_client::KernelClientTelemetryCollector telemetry_collector;
    NativeSchedulerThreadContext scheduler_context(
        socket_path, true, &telemetry_collector);
    return full_native_alns_v2(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, lexical_rank, node_name_offsets, node_name_bytes,
        initial_route_offsets, initial_route_indices, control,
        deadline_remaining, protocol_control, protocol_options,
        stage04_integer, stage04_float, operator_integer, operator_float);
}

void test_native_kernel_fault_v2(
    const std::string& socket_path,
    const std::string& fault) {
    py::gil_scoped_release release;
    evrptw::native_client::test_fault(socket_path, fault);
}
#endif

py::tuple propagate_routes_numeric(
    py::handle node_kind,
    py::handle ready_time,
    py::handle due_date,
    py::handle service_time,
    py::handle distance,
    py::handle vehicle,
    py::handle base_chain,
    py::handle candidate_chain,
    py::handle base_edge_distances,
    py::handle base_earliest_arrivals,
    py::handle base_latest_departures,
    py::handle epsilon) {
    auto kind_array = checked_array<std::int64_t>(node_kind, "node_kind", 1);
    auto ready_array = checked_array<double>(ready_time, "ready_time", 1);
    auto due_array = checked_array<double>(due_date, "due_date", 1);
    auto service_array = checked_array<double>(service_time, "service_time", 1);
    auto distance_array = checked_array<double>(distance, "distance", 2);
    auto vehicle_array = checked_array<double>(vehicle, "vehicle", 1);
    auto base_array = checked_array<std::int64_t>(base_chain, "base_chain", 1);
    auto candidate_array = checked_array<std::int64_t>(candidate_chain, "candidate_chain", 1);
    auto edges_array = checked_array<double>(base_edge_distances, "base_edge_distances", 1);
    auto earliest_array = checked_array<double>(
        base_earliest_arrivals, "base_earliest_arrivals", 1);
    auto latest_array = checked_array<double>(
        base_latest_departures, "base_latest_departures", 1);
    auto epsilon_array = checked_array<double>(epsilon, "epsilon", 1);
    const auto node_count = static_cast<std::size_t>(kind_array.request().shape[0]);
    const auto base_size = static_cast<std::size_t>(base_array.request().shape[0]);
    const auto candidate_size = static_cast<std::size_t>(candidate_array.request().shape[0]);
    if (node_count == 0 || ready_array.request().shape[0] != kind_array.request().shape[0]
        || due_array.request().shape[0] != kind_array.request().shape[0]
        || service_array.request().shape[0] != kind_array.request().shape[0]) {
        throw std::invalid_argument("propagation node arrays must share one non-zero length");
    }
    if (distance_array.request().shape[0] != kind_array.request().shape[0]
        || distance_array.request().shape[1] != kind_array.request().shape[0]) {
        throw std::invalid_argument("distance must have shape (n, n)");
    }
    if (vehicle_array.request().shape[0] != 5) {
        throw std::invalid_argument("vehicle must have shape (5,)");
    }
    if (base_size < 2 || edges_array.request().shape[0] != static_cast<py::ssize_t>(base_size - 1)
        || earliest_array.request().shape[0] != static_cast<py::ssize_t>(base_size)
        || latest_array.request().shape[0] != static_cast<py::ssize_t>(base_size)) {
        throw std::invalid_argument("base snapshot arrays have inconsistent lengths");
    }
    if (candidate_size < 2) {
        throw std::invalid_argument("candidate_chain must contain depot endpoints");
    }
    if (epsilon_array.request().shape[0] != 1
        || checked_data<double>(epsilon_array)[0] <= 0.0
        || !std::isfinite(checked_data<double>(epsilon_array)[0])) {
        throw std::invalid_argument("epsilon must have shape (1,) and be finite and positive");
    }
    const auto* kinds = checked_data<std::int64_t>(kind_array);
    const auto* base = checked_data<std::int64_t>(base_array);
    for (std::size_t position = 0; position < base_size; ++position) {
        if (base[position] < 0 || static_cast<std::size_t>(base[position]) >= node_count
            || ((position == 0 || position + 1 == base_size)
                    ? kinds[base[position]] != depot_kind
                    : kinds[base[position]] != customer_kind)) {
            throw std::invalid_argument("base_chain is not a canonical customer route chain");
        }
    }
    const auto* ready = checked_data<double>(ready_array);
    const auto* due = checked_data<double>(due_array);
    const auto* service = checked_data<double>(service_array);
    const auto* distances = checked_data<double>(distance_array);
    const auto* vehicle_values = checked_data<double>(vehicle_array);
    const auto* candidate = checked_data<std::int64_t>(candidate_array);
    const auto* edges = checked_data<double>(edges_array);
    const auto* earliest = checked_data<double>(earliest_array);
    const auto* latest = checked_data<double>(latest_array);
    const auto epsilon_value = checked_data<double>(epsilon_array)[0];
    PropagationOutput result;
    {
        py::gil_scoped_release release;
        result = run_incremental_propagation(
            kinds,
            ready,
            due,
            service,
            distances,
            vehicle_values,
            base,
            base_size,
            candidate,
            candidate_size,
            edges,
            earliest,
            latest,
            node_count,
            epsilon_value);
    }
    py::array_t<std::int64_t> codes(result.codes.size());
    py::array_t<double> metrics(result.metrics.size());
    std::copy(result.codes.begin(), result.codes.end(), checked_data(codes));
    std::copy(result.metrics.begin(), result.metrics.end(), checked_data(metrics));
    return py::make_tuple(std::move(codes), std::move(metrics));
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native kernels for EVRP-TW route evaluation";
    module.attr("__build_git_revision__") = EVRPTW_BUILD_GIT_REVISION;
    module.def("stage052_native_architecture_capabilities_v2", []() {
        // These values are production gates, not aspirational feature flags.
        // Flip a field only together with its end-to-end differential and
        // process-topology evidence.
        py::array_t<std::int64_t> capabilities(4);
        auto* values = checked_data(capabilities);
        values[0] = 0;  // host scheduler owns the complete candidate transaction
        values[1] = 0;  // full native search executes without the Python GIL
        values[2] = 0;  // host wave has one exclusive 24-thread compute pool
        values[3] = 0;  // every canonical event has a runtime causal ID
        return capabilities;
    });
    module.def(
        "python_random_golden_v1",
        &python_random_golden_v1,
        py::arg("seed"),
        py::arg("random_count"),
        py::arg("randbelow_bounds"),
        py::arg("sample_population"),
        py::arg("sample_size"),
        py::arg("weights"),
        py::arg("shuffle_size"));
    module.def(
        "legacy_destroy_v2",
        &legacy_destroy_v2,
        py::arg("seed"),
        py::arg("operation"),
        py::arg("remove_count"),
        py::arg("route_offsets"),
        py::arg("route_indices"),
        py::arg("distance"),
        py::arg("lexical_rank"),
        py::arg("depot"));
    module.def("native_sha256_v1", &native_sha256_v1, py::arg("payload"));
    module.def(
        "native_objective_acceptance_v1",
        &native_objective_acceptance_v1,
        py::arg("current_integer"),
        py::arg("current_float"),
        py::arg("candidate_integer"),
        py::arg("candidate_float"),
        py::arg("temperatures"),
        py::arg("random_draws"));
    module.def(
        "stage04_segment_update_v1",
        &stage04_segment_update_v1,
        py::arg("weights"),
        py::arg("reward_sums"),
        py::arg("calls"),
        py::arg("options"));
    module.def(
        "dynamic_removal_selection_v2",
        &dynamic_removal_selection_v2,
        py::arg("customer_count"),
        py::arg("stagnation_iterations"),
        py::arg("iteration"),
        py::arg("thresholds"),
        py::arg("fractions"),
        py::arg("global_best_reset"));
    module.def(
        "changed_candidate_pool_v1",
        &changed_candidate_pool_v1,
        py::arg("operation"),
        py::arg("route_offsets"),
        py::arg("route_indices"));
    module.def(
        "insertion_candidate_plans_v2",
        &insertion_candidate_plans_v2,
        py::arg("current_route_offsets"),
        py::arg("current_route_indices"),
        py::arg("customer"),
        py::arg("demand"),
        py::arg("load_capacity"),
        py::arg("epsilon"));
    module.def(
        "route_merge_candidate_pool_v2",
        &route_merge_candidate_pool_v2,
        py::arg("route_offsets"),
        py::arg("route_indices"),
        py::arg("route_objective_metrics"),
        py::arg("demand"),
        py::arg("load_capacity"),
        py::arg("epsilon"),
        py::arg("pair_pruning"),
        py::arg("preserve_duplicates"));
    module.def(
        "assemble_changed_candidate_plans_v1",
        &assemble_changed_candidate_plans_v1,
        py::arg("current_route_offsets"),
        py::arg("current_route_indices"),
        py::arg("changed_route_indices"),
        py::arg("change_offsets"),
        py::arg("change_indices"));
    module.def(
        "rank_candidate_plans_v1",
        &rank_candidate_plans_v1,
        py::arg("plan_offsets"),
        py::arg("route_offsets"),
        py::arg("route_indices"),
        py::arg("route_distance_lower_bounds"),
        py::arg("current_route_offsets"),
        py::arg("current_route_indices"),
        py::arg("lexical_rank"),
        py::arg("attempted_flags"),
        py::arg("top_k"));
    module.def(
        "prepare_candidate_plans_v2",
        &prepare_candidate_plans_v2,
        py::arg("plan_offsets"), py::arg("route_offsets"),
        py::arg("route_indices"), py::arg("expected_customer_indices"),
        py::arg("node_kind"), py::arg("lexical_rank"),
        py::arg("complete_customer_indices"), py::arg("customer_kind"),
        py::arg("allow_partial_customer_coverage"));
    module.def(
        "decide_candidate_plans_v2",
        &decide_candidate_plans_v2,
        py::arg("plan_offsets"), py::arg("coverage_eligible"),
        py::arg("screening_passed"), py::arg("attempted_flags"),
        py::arg("current_route_count"));
    module.def(
        "order_feasible_candidate_plans_v2",
        &order_feasible_candidate_plans_v2,
        py::arg("plan_offsets"), py::arg("route_offsets"),
        py::arg("route_indices"), py::arg("objective_integer"),
        py::arg("objective_float"), py::arg("lexical_rank"),
        py::arg("feasible_plan_ids"));
    module.def(
        "changed_candidate_plan_selection_v1",
        &changed_candidate_plan_selection_v1,
        py::arg("operation"), py::arg("node_kind"), py::arg("demand"),
        py::arg("ready_time"), py::arg("due_date"), py::arg("service_time"),
        py::arg("distance"), py::arg("reachable"), py::arg("vehicle"),
        py::arg("lexical_rank"), py::arg("current_route_offsets"),
        py::arg("current_route_indices"), py::arg("screening_options"),
        py::arg("negative_offsets"), py::arg("negative_indices"),
        py::arg("negative_reason_codes"), py::arg("attempted_flags"),
        py::arg("top_k"), py::arg("worker_count"));
    py::class_<NativeRouteCacheV2>(module, "NativeRouteCacheV2")
        .def(
            py::init<std::int64_t, std::int64_t>(),
            py::arg("max_entries"), py::arg("max_memory_bytes"))
        .def(
            "lookup_many", &NativeRouteCacheV2::lookup_many,
            py::arg("route_offsets"), py::arg("route_indices"))
        .def(
            "begin_store_many_atomic",
            &NativeRouteCacheV2::begin_store_many_atomic,
            py::arg("route_offsets"), py::arg("route_indices"),
            py::arg("semantic_hashes"), py::arg("entry_bytes"))
        .def(
            "begin_store_exact_many_atomic",
            &NativeRouteCacheV2::begin_store_exact_many_atomic,
            py::arg("route_offsets"), py::arg("route_indices"),
            py::arg("path_offsets"), py::arg("path_indices"),
            py::arg("result_statuses"), py::arg("reason_codes"),
            py::arg("result_metrics"), py::arg("label_counters"),
            py::arg("semantic_hashes"), py::arg("entry_bytes"))
        .def(
            "lookup_exact_many", &NativeRouteCacheV2::lookup_exact_many,
            py::arg("route_offsets"), py::arg("route_indices"))
        .def(
            "begin_protocol_transaction",
            &NativeRouteCacheV2::begin_protocol_transaction)
        .def(
            "commit_protocol_transaction",
            &NativeRouteCacheV2::commit_protocol_transaction)
        .def(
            "rollback_protocol_transaction",
            &NativeRouteCacheV2::rollback_protocol_transaction)
        .def(
            "inject_protocol_journal_failure_once",
            &NativeRouteCacheV2::inject_protocol_journal_failure_once)
        .def("commit_store_batch", &NativeRouteCacheV2::commit_store_batch)
        .def("rollback_store_batch", &NativeRouteCacheV2::rollback_store_batch)
        .def("snapshot", &NativeRouteCacheV2::snapshot);
    py::class_<NativeNegativeRouteCacheV2>(module, "NativeNegativeRouteCacheV2")
        .def(py::init<std::int64_t>(), py::arg("capacity"))
        .def(
            "lookup_many", &NativeNegativeRouteCacheV2::lookup_many,
            py::arg("route_offsets"), py::arg("route_indices"))
        .def(
            "begin_store_many_atomic",
            &NativeNegativeRouteCacheV2::begin_store_many_atomic,
            py::arg("route_offsets"), py::arg("route_indices"),
            py::arg("reason_codes"))
        .def("commit_store_batch", &NativeNegativeRouteCacheV2::commit_store_batch)
        .def(
            "rollback_store_batch", &NativeNegativeRouteCacheV2::rollback_store_batch)
        .def("snapshot", &NativeNegativeRouteCacheV2::snapshot);
    py::class_<NativeBudgetStateV2>(module, "NativeBudgetStateV2")
        .def(
            py::init<std::int64_t, std::int64_t>(),
            py::arg("exact_budget"), py::arg("round_budget"))
        .def(
            "begin_round", &NativeBudgetStateV2::begin_round,
            py::arg("lane_id"), py::arg("iteration"))
        .def("finish_round", &NativeBudgetStateV2::finish_round)
        .def(
            "reserve_round", &NativeBudgetStateV2::reserve_round,
            py::arg("requested"), py::arg("atomic"))
        .def(
            "reserve_exact", &NativeBudgetStateV2::reserve_exact,
            py::arg("requested"))
        .def("exact_remaining", &NativeBudgetStateV2::exact_remaining)
        .def(
            "candidate_round_remaining",
            &NativeBudgetStateV2::candidate_round_remaining)
        .def(
            "complete_exact", &NativeBudgetStateV2::complete_exact,
            py::arg("count"))
        .def(
            "interrupt_exact", &NativeBudgetStateV2::interrupt_exact,
            py::arg("count"))
        .def("snapshot", &NativeBudgetStateV2::snapshot)
        .def("restore", &NativeBudgetStateV2::restore, py::arg("snapshot"))
        .def("state", &NativeBudgetStateV2::state);
    py::class_<NativeAttemptedPlanSetV2>(module, "NativeAttemptedPlanSetV2")
        .def(py::init<>())
        .def(
            "lookup", &NativeAttemptedPlanSetV2::lookup,
            py::arg("plan_offsets"), py::arg("route_offsets"),
            py::arg("route_indices"))
        .def(
            "begin_mark_many_atomic",
            &NativeAttemptedPlanSetV2::begin_mark_many_atomic,
            py::arg("plan_offsets"), py::arg("route_offsets"),
            py::arg("route_indices"), py::arg("plan_ids"))
        .def("commit_mark_batch", &NativeAttemptedPlanSetV2::commit_mark_batch)
        .def("rollback_mark_batch", &NativeAttemptedPlanSetV2::rollback_mark_batch)
        .def("size", &NativeAttemptedPlanSetV2::size);
    py::class_<NativeSearchEngineV2>(module, "NativeSearchEngineV2")
        .def(
            py::init<
                std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                std::int64_t, std::int64_t, double, std::int64_t>(),
            py::arg("exact_budget"),
            py::arg("round_budget"),
            py::arg("cache_entries"),
            py::arg("cache_memory_bytes"),
            py::arg("negative_cache_entries"),
            py::arg("proposal_top_k"),
            py::arg("screening_epsilon"),
            py::arg("worker_count"))
        .def(
            "configure_node_names", &NativeSearchEngineV2::configure_node_names,
            py::arg("name_offsets"), py::arg("name_bytes"))
        .def(
            "initialize", &NativeSearchEngineV2::initialize,
            py::arg("node_kind"), py::arg("demand"),
            py::arg("ready_time"), py::arg("due_date"),
            py::arg("service_time"), py::arg("distance"),
            py::arg("reachable"), py::arg("vehicle"),
            py::arg("lexical_rank"), py::arg("initial_route_offsets"),
            py::arg("initial_route_indices"), py::arg("control"),
            py::arg("deadline_remaining"))
        .def(
            "evaluate_plans", &NativeSearchEngineV2::evaluate_plans,
            py::arg("plan_offsets"), py::arg("route_offsets"),
            py::arg("route_indices"), py::arg("context_ids"),
            py::arg("deadline_remaining"), py::arg("batch_size"),
            py::arg("expected_customer_indices"))
        .def(
            "legacy_route_elimination_probe",
            &NativeSearchEngineV2::legacy_route_elimination_probe,
            py::arg("iteration"), py::arg("max_attempts"),
            py::arg("route_change_limit"),
            py::arg("deadline_remaining"), py::arg("batch_size"),
            py::arg("defer_acceptance") = false)
        .def(
            "apply_legacy_candidate",
            &NativeSearchEngineV2::apply_legacy_candidate,
            py::arg("temperature"), py::arg("random_draw"))
        .def(
            "legacy_vehicle_reduction_refinement",
            &NativeSearchEngineV2::legacy_vehicle_reduction_refinement,
            py::arg("iteration"), py::arg("evaluation_budget"),
            py::arg("deadline_remaining"), py::arg("batch_size"))
        .def(
            "quality_changed_probe",
            &NativeSearchEngineV2::quality_changed_probe,
            py::arg("operation"), py::arg("iteration"),
            py::arg("deadline_remaining"), py::arg("batch_size"))
        .def(
            "run_three_lane_bootstrap",
            &NativeSearchEngineV2::run_three_lane_bootstrap,
            py::arg("max_route_elimination_attempts"),
            py::arg("refinement_budget"),
            py::arg("route_change_limit"), py::arg("thresholds"),
            py::arg("fractions"), py::arg("deadline_remaining"),
            py::arg("batch_size"))
        .def(
            "run_three_lane_followup",
            &NativeSearchEngineV2::run_three_lane_followup,
            py::arg("iteration"), py::arg("max_iterations"),
            py::arg("removal_fraction"),
            py::arg("route_elimination_max_attempts"),
            py::arg("refinement_budget"),
            py::arg("route_segment_min_length"),
            py::arg("route_segment_max_length"),
            py::arg("route_segment_budget"),
            py::arg("ejection_chain_budget"),
            py::arg("ejection_chain_max_depth"),
            py::arg("ejection_chain_beam_width"),
            py::arg("route_change_limit"), py::arg("thresholds"),
            py::arg("fractions"), py::arg("deadline_remaining"),
            py::arg("batch_size"))
        .def(
            "constraint_probe", &NativeSearchEngineV2::constraint_probe,
            py::arg("operation"), py::arg("requested_count"),
            py::arg("seed"), py::arg("context_ids"),
            py::arg("deadline_remaining"), py::arg("batch_size"),
            py::arg("route_change_limit"))
        .def(
            "apply_last_candidate", &NativeSearchEngineV2::apply_last_candidate,
            py::arg("temperature"), py::arg("random_draw"))
        .def(
            "configure_stage04", &NativeSearchEngineV2::configure_stage04,
            py::arg("integer_config"), py::arg("float_config"))
        .def(
            "initialize_stage04_search",
            &NativeSearchEngineV2::initialize_stage04_search,
            py::arg("deadline_remaining"), py::arg("batch_size"))
        .def(
            "constraint_stage04_state",
            &NativeSearchEngineV2::constraint_stage04_state)
        .def(
            "full_stage04_state",
            &NativeSearchEngineV2::full_stage04_state)
        .def(
            "record_constraint_stage04_outcome",
            &NativeSearchEngineV2::record_constraint_stage04_outcome,
            py::arg("iteration"), py::arg("operation"),
            py::arg("accepted"), py::arg("comparison"),
            py::arg("is_global_best"), py::arg("vehicle_reduction"))
        .def(
            "finish_stage04_iteration",
            &NativeSearchEngineV2::finish_stage04_iteration,
            py::arg("iteration"), py::arg("budget_boundary"))
        .def(
            "constraint_iteration", &NativeSearchEngineV2::constraint_iteration,
            py::arg("iteration"), py::arg("stagnation_iterations"),
            py::arg("global_best_reset"), py::arg("thresholds"),
            py::arg("fractions"), py::arg("deadline_remaining"),
            py::arg("batch_size"),
            py::arg("route_change_limit"))
        .def(
            "run_constraint_search", &NativeSearchEngineV2::run_constraint_search,
            py::arg("start_iteration"), py::arg("iteration_count"),
            py::arg("initial_stagnation_iterations"),
            py::arg("thresholds"), py::arg("fractions"),
            py::arg("deadline_remaining"), py::arg("batch_size"),
            py::arg("route_change_limit"))
        .def(
            "run_global_search", &NativeSearchEngineV2::run_global_search,
            py::arg("start_iteration"), py::arg("iteration_count"),
            py::arg("initial_stagnation_iterations"),
            py::arg("thresholds"), py::arg("fractions"),
            py::arg("deadline_remaining"), py::arg("batch_size"),
            py::arg("route_change_limit"))
        .def("initialized", &NativeSearchEngineV2::initialized)
        .def(
            "inject_commit_failure_once",
            &NativeSearchEngineV2::inject_commit_failure_once,
            py::arg("step"))
        .def(
            "inject_constraint_probe_envelope_failure_once",
            &NativeSearchEngineV2::inject_constraint_probe_envelope_failure_once)
        .def(
            "inject_global_search_envelope_failure_once",
            &NativeSearchEngineV2::inject_global_search_envelope_failure_once)
        .def(
            "inject_constraint_iteration_deadline_before_commit_once",
            &NativeSearchEngineV2::
                inject_constraint_iteration_deadline_before_commit_once)
        .def(
            "inject_exact_kernel_deadline_once",
            &NativeSearchEngineV2::inject_exact_kernel_deadline_once)
        .def(
            "inject_constraint_search_deadline_after_completed_once",
            &NativeSearchEngineV2::
                inject_constraint_search_deadline_after_completed_once,
            py::arg("completed_iterations"))
        .def("state", &NativeSearchEngineV2::state)
        .def("solution_state", &NativeSearchEngineV2::solution_state)
        .def(
            "lane_solution_state",
            &NativeSearchEngineV2::lane_solution_state,
            py::arg("lane"))
        .def(
            "best_solution_payload",
            &NativeSearchEngineV2::best_solution_payload);
    py::class_<Stage052ReplayState>(module, "Stage052ReplayState")
        .def(
            py::init<const py::dict&, const py::dict&>(),
            py::arg("axis_budgets"),
            py::arg("persistence_ledgers") = py::dict())
        .def("consume", &Stage052ReplayState::consume, py::arg("encoded_columns"))
        .def("finish", &Stage052ReplayState::finish);
    module.def(
        "pack_stage052_screening_occurrences",
        &pack_stage052_screening_occurrences,
        py::arg("events"),
        py::arg("definition_cache"),
        py::arg("negative_evidence_cache"),
        py::arg("first_event_id"));
    module.def(
        "create_stage052_screening_definition_cache",
        &create_stage052_screening_definition_cache,
        py::arg("capacity") = 262144);
    module.def(
        "stage052_screening_definition_cache_size",
        &stage052_screening_definition_cache_size,
        py::arg("definition_cache"));
    module.def(
        "stage052_screening_definition_cache_capacity",
        &stage052_screening_definition_cache_capacity,
        py::arg("definition_cache"));
    module.def(
        "create_stage052_definition_identity_store",
        &create_stage052_definition_identity_store,
        py::arg("capacity"));
    module.def(
        "register_stage052_definition_identities",
        &register_stage052_definition_identities,
        py::arg("identity_store"),
        py::arg("definitions"));
    module.def(
        "stage052_definition_identity_store_size",
        &stage052_definition_identity_store_size,
        py::arg("identity_store"));
    module.def(
        "pack_stage052_screening_transactions",
        &pack_stage052_screening_transactions,
        py::arg("events"),
        py::arg("definition_cache"),
        py::arg("negative_evidence_cache"),
        py::arg("lane_ids"),
        py::arg("operator_ids"),
        py::arg("route_ids"),
        py::arg("resolve_route_id"),
        py::arg("stable_dictionary_id"),
        py::arg("definition_identity"),
        py::arg("first_event_id"));
    module.def(
        "pack_stage052_neighborhood_events",
        &pack_stage052_neighborhood_events,
        py::arg("events"),
        py::arg("lane_ids"),
        py::arg("operator_ids"),
        py::arg("extras_cache"),
        py::arg("allowed_fields"),
        py::arg("missing_extra"),
        py::arg("stable_dictionary_id"),
        py::arg("json_text"),
        py::arg("first_event_id"));
    module.def(
        "pack_stage052_deferred_sparse_events",
        &pack_stage052_deferred_sparse_events,
        py::arg("events"),
        py::arg("route_ids"),
        py::arg("lane_ids"),
        py::arg("operator_ids"),
        py::arg("route_evaluation_extras_cache"),
        py::arg("cache_event_extras_cache"),
        py::arg("resolve_route_id"),
        py::arg("stable_dictionary_id"),
        py::arg("json_text"),
        py::arg("first_event_id"));
    module.def("route_distance", &route_distance, py::arg("points"), py::arg("route"));
    module.def("distance_matrix", &distance_matrix, py::arg("points"));
    module.def(
        "two_opt_delta", &two_opt_delta, py::arg("points"), py::arg("route"),
        py::arg("first"), py::arg("second"));
    module.def(
        "exact_charging_batch_numeric",
        &exact_charging_batch_numeric,
        py::arg("node_kind"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("vehicle"),
        py::arg("order_offsets"),
        py::arg("order_indices"),
        py::arg("deadline_remaining"),
        py::arg("batch_size"));
    module.def(
        "screen_routes_numeric",
        &screen_routes_numeric,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("route_indices"),
        py::arg("options"),
        py::arg("incremental"));
    module.def(
        "candidate_control_repair_v2",
        &candidate_control_repair_v2,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("lexical_rank"),
        py::arg("partial_route_offsets"),
        py::arg("partial_route_indices"),
        py::arg("removed_customer_indices"),
        py::arg("epsilon"),
        py::arg("route_change_limit"),
        py::arg("allow_new_routes"));
    module.def(
        "constraint_removal_v2",
        &constraint_removal_v2,
        py::arg("operation"),
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("lexical_rank"),
        py::arg("route_offsets"),
        py::arg("route_indices"),
        py::arg("path_offsets"),
        py::arg("path_indices"),
        py::arg("result_metrics"),
        py::arg("requested_count"),
        py::arg("seed"));
    module.def(
        "screen_route_batch_transaction_v2",
        &screen_route_batch_transaction_v2,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("route_offsets"),
        py::arg("route_indices"),
        py::arg("candidate_ids"),
        py::arg("options"),
        py::arg("incremental"),
        py::arg("negative_offsets"),
        py::arg("negative_indices"),
        py::arg("negative_reason_codes"));
    module.def(
        "candidate_round_transaction_v1",
        &candidate_round_transaction_v1,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("route_offsets"),
        py::arg("route_indices"),
        py::arg("candidate_ids"),
        py::arg("lexical_rank"),
        py::arg("options"),
        py::arg("incremental"),
        py::arg("negative_offsets"),
        py::arg("negative_indices"),
        py::arg("negative_reason_codes"),
        py::arg("cache_hit_flags"),
        py::arg("control"),
        py::arg("deadline_remaining"),
        py::arg("batch_size"),
        py::arg("context_ids"));
    module.def(
        "candidate_round_transaction_v2",
        &candidate_round_transaction_v2,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("route_offsets"),
        py::arg("route_indices"),
        py::arg("candidate_ids"),
        py::arg("lexical_rank"),
        py::arg("options"),
        py::arg("incremental"),
        py::arg("negative_offsets"),
        py::arg("negative_indices"),
        py::arg("negative_reason_codes"),
        py::arg("cache_hit_flags"),
        py::arg("control"),
        py::arg("deadline_remaining"),
        py::arg("batch_size"),
        py::arg("context_ids"),
        py::arg("resource_receipt"));
    module.def(
        "full_native_alns_v1",
        &full_native_alns_v1,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("vehicle"),
        py::arg("lexical_rank"),
        py::arg("initial_route_offsets"),
        py::arg("initial_route_indices"),
        py::arg("control"),
        py::arg("deadline_remaining"));
    module.def(
        "native_search_request_receipt_v2",
        &native_search_request_receipt_v2,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("lexical_rank"),
        py::arg("node_name_offsets"),
        py::arg("node_name_bytes"),
        py::arg("initial_route_offsets"),
        py::arg("initial_route_indices"),
        py::arg("control"),
        py::arg("deadline_remaining"),
        py::arg("protocol_control"),
        py::arg("protocol_options"),
        py::arg("stage04_integer"),
        py::arg("stage04_float"),
        py::arg("operator_integer"),
        py::arg("operator_float"));
    module.def(
        "full_native_alns_v2",
        &full_native_alns_v2,
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("lexical_rank"),
        py::arg("node_name_offsets"),
        py::arg("node_name_bytes"),
        py::arg("initial_route_offsets"),
        py::arg("initial_route_indices"),
        py::arg("control"),
        py::arg("deadline_remaining"),
        py::arg("protocol_control"),
        py::arg("protocol_options"),
        py::arg("stage04_integer"),
        py::arg("stage04_float"),
        py::arg("operator_integer"),
        py::arg("operator_float"));
#ifdef __linux__
    module.def(
        "native_search_request_host_receipt_v2",
        &native_search_request_host_receipt_v2,
        py::arg("socket_path"),
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("lexical_rank"),
        py::arg("node_name_offsets"),
        py::arg("node_name_bytes"),
        py::arg("initial_route_offsets"),
        py::arg("initial_route_indices"),
        py::arg("control"),
        py::arg("deadline_remaining"),
        py::arg("protocol_control"),
        py::arg("protocol_options"),
        py::arg("stage04_integer"),
        py::arg("stage04_float"),
        py::arg("operator_integer"),
        py::arg("operator_float"));
    module.def(
        "full_native_alns_host_v2",
        &full_native_alns_host_v2,
        py::arg("socket_path"),
        py::arg("node_kind"),
        py::arg("demand"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("reachable"),
        py::arg("vehicle"),
        py::arg("lexical_rank"),
        py::arg("node_name_offsets"),
        py::arg("node_name_bytes"),
        py::arg("initial_route_offsets"),
        py::arg("initial_route_indices"),
        py::arg("control"),
        py::arg("deadline_remaining"),
        py::arg("protocol_control"),
        py::arg("protocol_options"),
        py::arg("stage04_integer"),
        py::arg("stage04_float"),
        py::arg("operator_integer"),
        py::arg("operator_float"));
    module.def(
        "_test_native_kernel_fault_v2",
        &test_native_kernel_fault_v2,
        py::arg("socket_path"),
        py::arg("fault"));
#endif
    module.def(
        "full_native_initialize_v2",
        &full_native_initialize_v2,
        py::arg("node_kind"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("vehicle"),
        py::arg("initial_route_offsets"),
        py::arg("initial_route_indices"),
        py::arg("control"),
        py::arg("deadline_remaining"));
    module.def(
        "propagate_routes_numeric",
        &propagate_routes_numeric,
        py::arg("node_kind"),
        py::arg("ready_time"),
        py::arg("due_date"),
        py::arg("service_time"),
        py::arg("distance"),
        py::arg("vehicle"),
        py::arg("base_chain"),
        py::arg("candidate_chain"),
        py::arg("base_edge_distances"),
        py::arg("base_earliest_arrivals"),
        py::arg("base_latest_departures"),
        py::arg("epsilon"));
}
