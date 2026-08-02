#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <descrobject.h>

namespace py = pybind11;

using Point = std::pair<double, double>;

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
    const auto round_objective = [](double value) {
        constexpr auto scale = 1'000'000'000.0;
        return std::nearbyint(value * scale) / scale;
    };
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
        const auto current_key = std::make_tuple(
            current_vehicles,
            round_objective(current_distance),
            round_objective(current_charging_time),
            current_charging_count);
        const auto candidate_key = std::make_tuple(
            candidate_vehicles,
            round_objective(candidate_distance),
            round_objective(candidate_charging_time),
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

py::tuple insertion_candidate_plans_v2(
    py::handle current_route_offsets,
    py::handle current_route_indices,
    std::int64_t customer,
    py::handle demand,
    double load_capacity,
    double epsilon) {
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
    for (std::size_t target = 0; target <= route_count; ++target) {
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
        for (const auto [source_index, target_index] : {
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
    if (plan_offsets_array.size() < 1 || route_offsets_array.size() < 1
        || current_offsets_array.size() < 1 || lexical_array.size() == 0
        || top_k <= 0) {
        throw std::invalid_argument("candidate-plan ranking shape/config is invalid");
    }
    const auto plan_count = static_cast<std::size_t>(
        plan_offsets_array.size() - 1);
    const auto route_count = static_cast<std::size_t>(
        route_offsets_array.size() - 1);
    const auto current_count = static_cast<std::size_t>(
        current_offsets_array.size() - 1);
    if (lower_bounds_array.size() != static_cast<py::ssize_t>(route_count)
        || attempted_array.size() != static_cast<py::ssize_t>(plan_count)) {
        throw std::invalid_argument(
            "candidate-plan route metrics/flags do not align");
    }
    const auto* plans = checked_data<std::int64_t>(plan_offsets_array);
    const auto* routes = checked_data<std::int64_t>(route_offsets_array);
    const auto* indices = checked_data<std::int64_t>(route_indices_array);
    const auto* lower_bounds = checked_data<double>(lower_bounds_array);
    const auto* current_offsets = checked_data<std::int64_t>(current_offsets_array);
    const auto* current_indices = checked_data<std::int64_t>(current_indices_array);
    const auto* lexical = checked_data<std::int64_t>(lexical_array);
    const auto* attempted = checked_data<std::int64_t>(attempted_array);
    const auto validate_offsets = [](
        const std::int64_t* values,
        std::size_t count,
        std::int64_t terminal,
        const char* name) {
        if (values[0] != 0 || values[count] != terminal) {
            throw std::invalid_argument(std::string(name) + " boundary is invalid");
        }
        for (std::size_t index = 0; index < count; ++index) {
            if (values[index] < 0 || values[index] > values[index + 1]) {
                throw std::invalid_argument(std::string(name) + " must be monotonic");
            }
        }
    };
    validate_offsets(
        plans, plan_count, static_cast<std::int64_t>(route_count), "plan_offsets");
    validate_offsets(
        routes, route_count, static_cast<std::int64_t>(route_indices_array.size()),
        "route_offsets");
    validate_offsets(
        current_offsets, current_count,
        static_cast<std::int64_t>(current_indices_array.size()),
        "current_route_offsets");
    std::unordered_set<std::int64_t> lexical_values;
    for (py::ssize_t node = 0; node < lexical_array.size(); ++node) {
        if (lexical[node] < 0 || lexical[node] >= lexical_array.size()
            || !lexical_values.insert(lexical[node]).second) {
            throw std::invalid_argument("lexical_rank must be a permutation");
        }
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (!std::isfinite(lower_bounds[route]) || lower_bounds[route] < 0.0) {
            throw std::invalid_argument(
                "route_distance_lower_bounds must be finite and non-negative");
        }
        for (auto cursor = routes[route]; cursor < routes[route + 1]; ++cursor) {
            if (indices[cursor] < 0 || indices[cursor] >= lexical_array.size()) {
                throw std::invalid_argument("candidate plan contains an unknown node");
            }
        }
    }
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        if (attempted[plan] != 0 && attempted[plan] != 1) {
            throw std::invalid_argument("attempted_flags must contain only zero or one");
        }
    }

    const auto route_equal = [](
        const std::int64_t* left,
        std::int64_t left_begin,
        std::int64_t left_end,
        const std::int64_t* right,
        std::int64_t right_begin,
        std::int64_t right_end) {
        return left_end - left_begin == right_end - right_begin
            && std::equal(left + left_begin, left + left_end, right + right_begin);
    };
    std::vector<std::int64_t> vehicle_counts(plan_count, 0);
    std::vector<std::int64_t> changed_counts(plan_count, 0);
    std::vector<double> optimistic_distances(plan_count, 0.0);
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        vehicle_counts[plan] = plans[plan + 1] - plans[plan];
        PythonFloatSum distance_sum;
        for (auto route = plans[plan]; route < plans[plan + 1]; ++route) {
            distance_sum.add(lower_bounds[route]);
            bool unchanged = false;
            for (std::size_t current = 0; current < current_count; ++current) {
                if (route_equal(
                        indices, routes[route], routes[route + 1], current_indices,
                        current_offsets[current], current_offsets[current + 1])) {
                    unchanged = true;
                    break;
                }
            }
            changed_counts[plan] += unchanged ? 0 : 1;
        }
        optimistic_distances[plan] = distance_sum.value();
    }
    const auto route_less = [&](std::int64_t left, std::int64_t right) {
        auto left_cursor = routes[left];
        auto right_cursor = routes[right];
        while (left_cursor < routes[left + 1] && right_cursor < routes[right + 1]) {
            const auto left_rank = lexical[indices[left_cursor]];
            const auto right_rank = lexical[indices[right_cursor]];
            if (left_rank != right_rank) {
                return left_rank < right_rank;
            }
            ++left_cursor;
            ++right_cursor;
        }
        return routes[left + 1] - routes[left]
            < routes[right + 1] - routes[right];
    };
    const auto plan_routes_less = [&](std::size_t left, std::size_t right) {
        auto left_route = plans[left];
        auto right_route = plans[right];
        while (left_route < plans[left + 1] && right_route < plans[right + 1]) {
            if (route_less(left_route, right_route)) {
                return true;
            }
            if (route_less(right_route, left_route)) {
                return false;
            }
            ++left_route;
            ++right_route;
        }
        return vehicle_counts[left] < vehicle_counts[right];
    };
    std::vector<std::int64_t> ranked(plan_count);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(
        ranked.begin(), ranked.end(), [&](std::int64_t left_id, std::int64_t right_id) {
            const auto left = static_cast<std::size_t>(left_id);
            const auto right = static_cast<std::size_t>(right_id);
            if (vehicle_counts[left] != vehicle_counts[right]) {
                return vehicle_counts[left] < vehicle_counts[right];
            }
            if (optimistic_distances[left] != optimistic_distances[right]) {
                return optimistic_distances[left] < optimistic_distances[right];
            }
            if (changed_counts[left] != changed_counts[right]) {
                return changed_counts[left] < changed_counts[right];
            }
            if (plan_routes_less(left, right)) {
                return true;
            }
            if (plan_routes_less(right, left)) {
                return false;
            }
            return left < right;
        });
    std::vector<std::int64_t> selected;
    selected.reserve(std::min<std::size_t>(plan_count, static_cast<std::size_t>(top_k)));
    for (const auto plan_id : ranked) {
        if (attempted[plan_id] == 0) {
            selected.push_back(plan_id);
            if (selected.size() == static_cast<std::size_t>(top_k)) {
                break;
            }
        }
    }
    py::array_t<std::int64_t> ranked_array(ranked.size());
    py::array_t<std::int64_t> selected_array(selected.size());
    py::array_t<std::int64_t> integer_metrics(
        {static_cast<py::ssize_t>(plan_count), py::ssize_t(2)});
    py::array_t<double> float_metrics(plan_count);
    std::copy(ranked.begin(), ranked.end(), checked_data(ranked_array));
    std::copy(selected.begin(), selected.end(), checked_data(selected_array));
    auto* integer_values = checked_data(integer_metrics);
    for (std::size_t plan = 0; plan < plan_count; ++plan) {
        integer_values[plan * 2] = vehicle_counts[plan];
        integer_values[plan * 2 + 1] = changed_counts[plan];
    }
    std::copy(
        optimistic_distances.begin(), optimistic_distances.end(),
        checked_data(float_metrics));
    return py::make_tuple(
        std::move(ranked_array), std::move(selected_array),
        std::move(integer_metrics), std::move(float_metrics));
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
            const auto [_, newly_seen] = seen_keys_.insert(key);
            if (newly_seen) {
                record_protocol_seen(key);
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
                    Entry evicted = std::move(entries_.front());
                    record_protocol_eviction(evicted, next_key(entries_.begin()));
                    index_.erase(evicted.key);
                    entries_.pop_front();
                    --statistics_[6];
                    statistics_[8] -= evicted.entry_bytes;
                    ++statistics_[4];
                    ++eviction_counts[index];
                    if (!inserted.contains(evicted.key)) {
                        journal.evicted_entries.push_back(std::move(evicted));
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
                index_.emplace(stored_entry->key, stored_entry);
                record_protocol_insertion(stored_entry->key);
                ++statistics_[3];
                ++statistics_[6];
                statistics_[8] += bytes[index];
                statistics_[7] = std::max(statistics_[7], statistics_[6]);
                statistics_[9] = std::max(statistics_[9], statistics_[8]);
            }
        } catch (...) {
            rollback_journal(journal);
            throw;
        }
        active_batch_ = std::move(journal);
        py::array_t<std::int64_t> status_array(statuses.size());
        py::array_t<std::int64_t> eviction_array(eviction_counts.size());
        std::copy(statuses.begin(), statuses.end(), checked_data(status_array));
        std::copy(
            eviction_counts.begin(), eviction_counts.end(),
            checked_data(eviction_array));
        return py::make_tuple(
            std::move(status_array), std::move(eviction_array), statistics_array());
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
            const auto [_, newly_seen] = seen_keys_.insert(key);
            if (newly_seen) {
                record_protocol_seen(key);
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
        protocol_snapshot_ = ProtocolSnapshot{statistics_, {}};
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
        rollback_protocol_operations(protocol_snapshot_->operations);
        statistics_ = protocol_snapshot_->statistics;
        protocol_snapshot_.reset();
        return statistics_array();
    }

    py::array_t<std::int64_t> commit_store_batch() {
        require_active_batch("commit_store_batch");
        active_batch_.reset();
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
        std::vector<Entry> evicted_entries;
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
        std::optional<Entry> entry;
    };
    struct ProtocolSnapshot {
        std::array<std::int64_t, 11> statistics;
        std::vector<ProtocolOperation> operations;
    };

    std::int64_t max_entries_;
    std::int64_t max_memory_bytes_;
    std::list<Entry> entries_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    std::unordered_set<std::string> seen_keys_;
    std::array<std::int64_t, 11> statistics_{};
    std::optional<BatchJournal> active_batch_;
    std::optional<ProtocolSnapshot> protocol_snapshot_;

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
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::seen, key, std::nullopt, std::nullopt});
        }
    }

    void record_protocol_move(std::list<Entry>::iterator entry) {
        if (protocol_snapshot_.has_value() && std::next(entry) != entries_.end()) {
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::move,
                entry->key,
                next_key(entry),
                std::nullopt,
            });
        }
    }

    void record_protocol_insertion(const std::string& key) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::insertion, key, std::nullopt, std::nullopt});
        }
    }

    void record_protocol_eviction(
        const Entry& entry,
        std::optional<std::string> following_key) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.push_back(ProtocolOperation{
                ProtocolOperationKind::eviction,
                entry.key,
                std::move(following_key),
                entry,
            });
        }
    }

    void rollback_protocol_operations(
        const std::vector<ProtocolOperation>& operations) {
        for (auto operation = operations.rbegin(); operation != operations.rend(); ++operation) {
            if (operation->kind == ProtocolOperationKind::seen) {
                seen_keys_.erase(operation->key);
                continue;
            }
            if (operation->kind == ProtocolOperationKind::insertion) {
                const auto found = find_entry(operation->key);
                if (found == entries_.end()) {
                    throw std::runtime_error(
                        "native route-cache protocol rollback lost an insertion");
                }
                index_.erase(operation->key);
                entries_.erase(found);
                continue;
            }
            if (operation->kind == ProtocolOperationKind::eviction) {
                if (!operation->entry.has_value() || index_.contains(operation->key)) {
                    throw std::runtime_error(
                        "native route-cache protocol eviction journal is invalid");
                }
                auto position = entries_.end();
                if (operation->next_key.has_value()) {
                    const auto following = index_.find(*operation->next_key);
                    if (following == index_.end()) {
                        throw std::runtime_error(
                            "native route-cache protocol rollback lost an eviction anchor");
                    }
                    position = following->second;
                }
                auto restored = entries_.insert(position, *operation->entry);
                index_.emplace(restored->key, restored);
                continue;
            }
            const auto found = find_entry(operation->key);
            if (found == entries_.end()) {
                throw std::runtime_error(
                    "native route-cache protocol rollback lost an LRU entry");
            }
            auto position = entries_.end();
            if (operation->next_key.has_value()) {
                const auto following = index_.find(*operation->next_key);
                if (following == index_.end()) {
                    throw std::runtime_error(
                        "native route-cache protocol rollback lost an LRU anchor");
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

    void rollback_journal(const BatchJournal& journal) {
        if (protocol_snapshot_.has_value()) {
            protocol_snapshot_->operations.resize(
                journal.protocol_operation_count_before);
        }
        const std::unordered_set<std::string> inserted(
            journal.inserted_keys.begin(), journal.inserted_keys.end());
        entries_.remove_if([&](const Entry& entry) {
            if (!inserted.contains(entry.key)) {
                return false;
            }
            index_.erase(entry.key);
            return true;
        });
        for (auto entry = journal.evicted_entries.rbegin();
             entry != journal.evicted_entries.rend(); ++entry) {
            entries_.push_front(*entry);
            index_[entries_.front().key] = entries_.begin();
        }
        statistics_ = journal.statistics_before;
    }

    py::array_t<std::int64_t> statistics_array() const {
        py::array_t<std::int64_t> output(statistics_.size());
        std::copy(statistics_.begin(), statistics_.end(), checked_data(output));
        return output;
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

private:
    friend class NativeSearchEngineV2;
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
    py::tuple sequence(py::len(tokens));
    for (py::ssize_t index = 0; index < py::len(tokens); ++index) {
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
        for (py::ssize_t check_index = 0; check_index < py::len(checks); ++check_index) {
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
        for (py::ssize_t index = 0; index < extra_count; ++index) {
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
        for (py::ssize_t index = 0; index < extra_count; ++index) {
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

namespace {

constexpr double exact_epsilon = 1e-9;
constexpr std::int64_t depot_kind = 0;
constexpr std::int64_t customer_kind = 1;
constexpr std::int64_t station_kind = 2;
constexpr std::int64_t feasible_status = 0;
constexpr std::int64_t infeasible_status = 1;
constexpr std::int64_t interrupted_status = 2;
constexpr std::int64_t no_failure_reason = 0;
constexpr std::int64_t no_feasible_pattern_reason = 1;
constexpr std::int64_t deadline_reason = 2;

struct ExactLabel {
    std::int64_t progress;
    std::int64_t node;
    double elapsed_time;
    double battery;
    double distance;
    double total_energy;
    double charged_energy;
    double charging_time;
    std::int64_t parent;
    bool live;
};

struct ExactQueueEntry {
    double distance;
    double elapsed_time;
    std::int64_t negative_progress;
    std::int64_t serial;
    std::size_t label_index;
};

struct ExactQueueLater {
    bool operator()(const ExactQueueEntry& left, const ExactQueueEntry& right) const {
        return std::tie(left.distance, left.elapsed_time, left.negative_progress, left.serial)
            > std::tie(right.distance, right.elapsed_time, right.negative_progress, right.serial);
    }
};

struct ExactSearchState {
    std::vector<std::int64_t> order;
    std::vector<ExactLabel> labels;
    std::vector<std::vector<std::size_t>> state_labels;
    std::priority_queue<ExactQueueEntry, std::vector<ExactQueueEntry>, ExactQueueLater> queue;
    std::optional<ExactLabel> best;
    std::int64_t generated = 1;
    std::int64_t expanded = 0;
    std::int64_t pruned = 0;
    std::int64_t serial = 1;
    bool completed = false;
    bool interrupted = false;
};

struct ExactRequest {
    std::size_t route;
    std::size_t label_index;
    std::int64_t destination;
    std::int64_t progress;
};

struct ExactBatchOutput {
    std::vector<std::int64_t> path_offsets;
    std::vector<std::int64_t> path_indices;
    std::vector<std::int64_t> statuses;
    std::vector<std::int64_t> reasons;
    std::vector<double> metrics;
    std::vector<std::int64_t> label_counters;
    std::vector<std::int64_t> batch_counters;
};

bool exact_dominates(const ExactLabel& left, const ExactLabel& right) {
    const bool no_worse = left.elapsed_time <= right.elapsed_time + exact_epsilon
        && left.battery + exact_epsilon >= right.battery
        && left.distance <= right.distance + exact_epsilon;
    const bool strictly_better = left.elapsed_time < right.elapsed_time - exact_epsilon
        || left.battery > right.battery + exact_epsilon
        || left.distance < right.distance - exact_epsilon;
    return no_worse && strictly_better;
}

bool exact_better_terminal(const ExactLabel& candidate, const ExactLabel& incumbent) {
    return std::tie(candidate.distance, candidate.elapsed_time)
        < std::tie(incumbent.distance, incumbent.elapsed_time);
}

std::optional<std::size_t> pop_live_exact_label(ExactSearchState& state) {
    while (!state.queue.empty()) {
        const auto entry = state.queue.top();
        state.queue.pop();
        if (state.labels[entry.label_index].live) {
            return entry.label_index;
        }
    }
    return std::nullopt;
}

std::vector<std::int64_t> reconstruct_exact_path(
    const ExactSearchState& state,
    const ExactLabel& terminal) {
    std::vector<std::int64_t> reversed;
    reversed.push_back(terminal.node);
    auto parent = terminal.parent;
    while (parent >= 0) {
        const auto& label = state.labels[static_cast<std::size_t>(parent)];
        reversed.push_back(label.node);
        parent = label.parent;
    }
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

ExactBatchOutput run_exact_charging_batch(
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
    std::int64_t depot,
    const std::vector<std::int64_t>& stations,
    double deadline_remaining,
    std::int64_t batch_size) {
    const auto started = std::chrono::steady_clock::now();
    ExactBatchOutput output;
    output.path_offsets.assign(route_count + 1, 0);
    output.statuses.assign(route_count, interrupted_status);
    output.reasons.assign(route_count, deadline_reason);
    output.metrics.assign(route_count * 4, 0.0);
    output.label_counters.assign(route_count * 3, 0);
    output.batch_counters.assign(10, 0);
    output.batch_counters[0] = static_cast<std::int64_t>(route_count);
    output.batch_counters[1] = static_cast<std::int64_t>(route_count);
    output.batch_counters[4] = route_count == 0 ? 0 : 1;
    output.batch_counters[8] = route_count == 0 ? 0 : 1;
    output.batch_counters[9] = batch_size;

    const auto deadline_expired = [&]() {
        ++output.batch_counters[7];
        if (std::isinf(deadline_remaining) && deadline_remaining > 0.0) {
            return false;
        }
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return elapsed >= std::max(0.0, deadline_remaining);
    };
    if (deadline_expired()) {
        output.batch_counters[3] = static_cast<std::int64_t>(route_count);
        return output;
    }

    std::vector<ExactSearchState> states(route_count);
    for (std::size_t route = 0; route < route_count; ++route) {
        if (deadline_expired()) {
            output.batch_counters[3] = static_cast<std::int64_t>(route_count);
            return output;
        }
        auto& state = states[route];
        const auto begin = order_offsets[route];
        const auto end = order_offsets[route + 1];
        if (end > begin) {
            state.order.assign(order_indices + begin, order_indices + end);
        }
        state.state_labels.resize((state.order.size() + 1) * node_count);
        state.labels.push_back(ExactLabel{
            0,
            depot,
            std::max(0.0, ready[depot]),
            vehicle[0],
            0.0,
            0.0,
            0.0,
            0.0,
            -1,
            true,
        });
        state.state_labels[static_cast<std::size_t>(depot)].push_back(0);
        state.queue.push(ExactQueueEntry{0.0, std::max(0.0, ready[depot]), 0, 0, 0});
    }

    bool deadline_hit = deadline_expired();
    while (!deadline_hit) {
        std::vector<ExactRequest> requests;
        bool progressed = false;
        for (std::size_t route = 0; route < route_count; ++route) {
            auto& state = states[route];
            if (state.completed) {
                continue;
            }
            const auto live_index = pop_live_exact_label(state);
            if (!live_index.has_value()) {
                state.completed = true;
                continue;
            }
            const auto& label = state.labels[*live_index];
            if (state.best.has_value()
                && label.distance >= state.best->distance - exact_epsilon) {
                ++state.pruned;
                progressed = true;
                continue;
            }
            ++state.expanded;
            progressed = true;
            if (label.progress < static_cast<std::int64_t>(state.order.size())) {
                requests.push_back(ExactRequest{
                    route,
                    *live_index,
                    state.order[static_cast<std::size_t>(label.progress)],
                    label.progress + 1,
                });
            } else {
                requests.push_back(ExactRequest{route, *live_index, depot, label.progress});
            }
            for (const auto station : stations) {
                if (station != label.node) {
                    requests.push_back(ExactRequest{
                        route,
                        *live_index,
                        station,
                        label.progress,
                    });
                }
            }
        }
        deadline_hit = deadline_expired();
        if (deadline_hit) {
            break;
        }
        if (requests.empty()) {
            if (!progressed) {
                break;
            }
            continue;
        }

        for (std::size_t offset = 0; offset < requests.size();
             offset += static_cast<std::size_t>(batch_size)) {
            deadline_hit = deadline_expired();
            if (deadline_hit) {
                break;
            }
            const auto chunk_end = std::min(
                requests.size(), offset + static_cast<std::size_t>(batch_size));
            ++output.batch_counters[5];
            output.batch_counters[6] += static_cast<std::int64_t>(chunk_end - offset);
            for (std::size_t request_index = offset; request_index < chunk_end; ++request_index) {
                const auto& request = requests[request_index];
                auto& state = states[request.route];
                const auto& label = state.labels[request.label_index];
                const auto destination = request.destination;
                ++state.generated;
                const auto leg_distance = distances[
                    static_cast<std::size_t>(label.node) * node_count
                    + static_cast<std::size_t>(destination)];
                const auto energy = leg_distance * vehicle[2];
                if (energy > label.battery + exact_epsilon) {
                    ++state.pruned;
                    continue;
                }
                auto battery = std::max(0.0, label.battery - energy);
                auto elapsed = std::max(
                    label.elapsed_time + leg_distance / vehicle[4], ready[destination]);
                if (elapsed > due[destination] + exact_epsilon) {
                    ++state.pruned;
                    continue;
                }
                double charged = 0.0;
                double charging_time = 0.0;
                if (node_kinds[destination] == customer_kind) {
                    elapsed += service[destination];
                } else if (node_kinds[destination] == station_kind) {
                    charged = vehicle[0] - battery;
                    charging_time = charged * vehicle[3];
                    elapsed += charging_time;
                    if (elapsed > due[destination] + exact_epsilon) {
                        ++state.pruned;
                        continue;
                    }
                    battery = vehicle[0];
                }
                ExactLabel candidate{
                    request.progress,
                    destination,
                    elapsed,
                    battery,
                    label.distance + leg_distance,
                    label.total_energy + energy,
                    label.charged_energy + charged,
                    label.charging_time + charging_time,
                    static_cast<std::int64_t>(request.label_index),
                    true,
                };
                if (request.progress == static_cast<std::int64_t>(state.order.size())
                    && node_kinds[destination] == depot_kind) {
                    if (!state.best.has_value()
                        || exact_better_terminal(candidate, *state.best)) {
                        state.best = candidate;
                    }
                    continue;
                }

                const auto group_index = static_cast<std::size_t>(request.progress) * node_count
                    + static_cast<std::size_t>(destination);
                auto& current = state.state_labels[group_index];
                bool dominated = false;
                for (const auto existing_index : current) {
                    if (exact_dominates(state.labels[existing_index], candidate)) {
                        dominated = true;
                        break;
                    }
                }
                if (dominated) {
                    ++state.pruned;
                    continue;
                }
                std::vector<std::size_t> survivors;
                survivors.reserve(current.size() + 1);
                for (const auto existing_index : current) {
                    if (exact_dominates(candidate, state.labels[existing_index])) {
                        state.labels[existing_index].live = false;
                        ++state.pruned;
                    } else {
                        survivors.push_back(existing_index);
                    }
                }
                const auto candidate_index = state.labels.size();
                state.labels.push_back(candidate);
                survivors.push_back(candidate_index);
                current = std::move(survivors);
                ++state.serial;
                state.queue.push(ExactQueueEntry{
                    candidate.distance,
                    candidate.elapsed_time,
                    -candidate.progress,
                    state.serial,
                    candidate_index,
                });
            }
        }
    }

    for (std::size_t route = 0; route < route_count; ++route) {
        auto& state = states[route];
        if (!state.completed && deadline_hit) {
            state.interrupted = true;
        } else if (!state.completed) {
            state.completed = true;
        }
        output.label_counters[route * 3] = state.generated;
        output.label_counters[route * 3 + 1] = state.expanded;
        output.label_counters[route * 3 + 2] = state.pruned;
        if (state.interrupted) {
            continue;
        }
        ++output.batch_counters[2];
        if (!state.best.has_value()) {
            output.statuses[route] = infeasible_status;
            output.reasons[route] = no_feasible_pattern_reason;
            output.metrics[route * 4] = std::numeric_limits<double>::infinity();
            continue;
        }
        output.statuses[route] = feasible_status;
        output.reasons[route] = no_failure_reason;
        output.metrics[route * 4] = state.best->distance;
        output.metrics[route * 4 + 1] = state.best->total_energy;
        output.metrics[route * 4 + 2] = state.best->charged_energy;
        output.metrics[route * 4 + 3] = state.best->charging_time;
        const auto path = reconstruct_exact_path(state, *state.best);
        output.path_indices.insert(output.path_indices.end(), path.begin(), path.end());
        output.path_offsets[route + 1] = static_cast<std::int64_t>(output.path_indices.size());
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (output.path_offsets[route + 1] == 0) {
            output.path_offsets[route + 1] = output.path_offsets[route];
        }
    }
    output.batch_counters[3] = static_cast<std::int64_t>(route_count) - output.batch_counters[2];
    return output;
}

}  // namespace

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

    ExactBatchOutput result;
    {
        py::gil_scoped_release release;
        result = run_exact_charging_batch(
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

constexpr std::int64_t screen_reason_none = 0;
constexpr std::int64_t screen_reason_structure = 1;
constexpr std::int64_t screen_reason_capacity = 2;
constexpr std::int64_t screen_reason_forward = 3;
constexpr std::int64_t screen_reason_backward = 4;
constexpr std::int64_t screen_reason_slack = 5;
constexpr std::int64_t screen_reason_energy = 6;
constexpr std::int64_t screen_reason_structural_energy = 7;
constexpr std::int64_t screen_reason_legacy_time = 8;
constexpr std::int64_t screen_reason_legacy_energy = 9;
constexpr std::int64_t check_structure = 1;
constexpr std::int64_t check_capacity = 2;
constexpr std::int64_t check_forward = 3;
constexpr std::int64_t check_backward = 4;
constexpr std::int64_t check_slack = 5;
constexpr std::int64_t check_distance = 6;
constexpr std::int64_t check_energy = 7;
constexpr std::int64_t check_structural_energy = 8;
constexpr std::int64_t check_fail = 0;
constexpr std::int64_t check_pass = 1;
constexpr std::int64_t check_recorded = 2;

struct ScreenOutput {
    std::vector<std::int64_t> codes = std::vector<std::int64_t>(16, 0);
    std::vector<double> metrics = std::vector<double>(15, 0.0);
};

void append_screen_event(
    ScreenOutput& output,
    std::int64_t check,
    std::int64_t status,
    double value) {
    const auto count = static_cast<std::size_t>(output.codes[7]);
    if (count >= 8) {
        throw std::logic_error("screening emitted more than eight canonical check events");
    }
    output.codes[8 + count] = check * 10 + status;
    output.metrics[7 + count] = value;
    output.codes[7] += 1;
}

void reject_screen(
    ScreenOutput& output,
    std::int64_t reason,
    std::int64_t failed_check,
    double value) {
    output.codes[0] = 0;
    output.codes[1] = reason;
    output.codes[2] = failed_check;
    append_screen_event(output, failed_check, check_fail, value);
}

ScreenOutput run_screen_route(
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
    ScreenOutput output;
    const bool full = options[0] >= 0.5;
    const auto epsilon = options[1];
    const bool has_reference = options[3] >= 0.5;
    const bool use_incremental = incremental[0] >= 0.5;
    output.codes[4] = 0;
    output.codes[5] = full ? 1 : 0;
    output.codes[6] = use_incremental ? 1 : 0;
    output.codes[3] = full ? 0 : 1;
    output.metrics[4] = has_reference
        ? 0.0
        : std::numeric_limits<double>::quiet_NaN();

    bool known_sequence = true;
    bool customer_sequence = true;
    bool duplicate_free = true;
    std::vector<bool> seen(node_count, false);
    for (std::size_t position = 0; position < route_size; ++position) {
        const auto node = route[position];
        if (node < 0 || static_cast<std::size_t>(node) >= node_count) {
            known_sequence = false;
            customer_sequence = false;
            continue;
        }
        if (kinds[node] != customer_kind) {
            customer_sequence = false;
        }
        if (seen[static_cast<std::size_t>(node)]) {
            duplicate_free = false;
        }
        seen[static_cast<std::size_t>(node)] = true;
    }

    double distance_lower_bound = 0.0;
    if (use_incremental) {
        distance_lower_bound = incremental[1];
    } else if (known_sequence) {
        PythonFloatSum distance_sum;
        auto origin = depot;
        for (std::size_t position = 0; position < route_size; ++position) {
            const auto destination = route[position];
            distance_sum.add(distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)]);
            origin = destination;
        }
        distance_sum.add(distances[
            static_cast<std::size_t>(origin) * node_count
            + static_cast<std::size_t>(depot)]);
        distance_lower_bound = distance_sum.value();
    }
    output.metrics[3] = distance_lower_bound;
    if (has_reference) {
        output.metrics[4] = distance_lower_bound - options[2];
    }
    if (!full) {
        output.metrics[3] = 0.0;
        output.metrics[4] = std::numeric_limits<double>::quiet_NaN();
    }
    if (!known_sequence || !customer_sequence) {
        output.codes[3] = 0;
        output.metrics[0] = 0.0;
        output.metrics[1] = 0.0;
        if (full) {
            reject_screen(output, screen_reason_structure, check_structure, 0.0);
        } else {
            output.codes[1] = screen_reason_structure;
            output.codes[2] = check_structure;
        }
        return output;
    }

    PythonFloatSum demand_sum;
    for (std::size_t position = 0; position < route_size; ++position) {
        demand_sum.add(demands[route[position]]);
    }
    const auto demand = demand_sum.value();
    output.metrics[0] = demand;
    if (demand > vehicle[1] + epsilon) {
        output.codes[3] = 0;
        if (full) {
            reject_screen(output, screen_reason_capacity, check_capacity, demand);
        } else {
            output.codes[1] = screen_reason_capacity;
            output.codes[2] = check_capacity;
        }
        return output;
    }

    auto current_time = std::max(0.0, ready[depot]);
    if (!full) {
        auto origin = depot;
        for (std::size_t position = 0; position <= route_size; ++position) {
            const auto destination = position < route_size ? route[position] : depot;
            current_time += distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] / vehicle[4];
            current_time = std::max(current_time, ready[destination]);
            if (kinds[destination] == customer_kind) {
                if (current_time > due[destination] + epsilon) {
                    output.codes[1] = screen_reason_legacy_time;
                    output.codes[2] = check_forward;
                    output.metrics[1] = current_time;
                    return output;
                }
                current_time += service[destination];
            }
            origin = destination;
        }
        output.metrics[1] = current_time;
        origin = depot;
        for (std::size_t position = 0; position <= route_size; ++position) {
            const auto destination = position < route_size ? route[position] : depot;
            if (reachable[
                    static_cast<std::size_t>(origin) * node_count
                    + static_cast<std::size_t>(destination)] == 0U) {
                output.codes[1] = screen_reason_legacy_energy;
                output.codes[2] = check_energy;
                return output;
            }
            origin = destination;
        }
        output.codes[0] = 1;
        output.codes[3] = 1;
        output.codes[4] = 1;
        output.metrics[6] = 1.0;
        return output;
    }

    output.codes[3] = 1;
    output.codes[4] = 1;
    output.metrics[6] = 1.0;
    append_screen_event(
        output, check_structure, check_pass, duplicate_free ? 1.0 : 0.0);
    if (!duplicate_free) {
        reject_screen(output, screen_reason_structure, check_structure, 0.0);
        return output;
    }
    append_screen_event(output, check_capacity, check_pass, demand);

    auto min_slack = std::numeric_limits<double>::infinity();
    std::vector<double> earliest(node_count, 0.0);
    std::vector<bool> has_earliest(node_count, false);
    if (use_incremental) {
        current_time = incremental[3];
        min_slack = incremental[2];
        if (incremental[4] < 0.5) {
            output.metrics[1] = current_time;
            output.metrics[2] = min_slack;
            reject_screen(output, screen_reason_forward, check_forward, min_slack);
            return output;
        }
    } else {
        auto origin = depot;
        for (std::size_t position = 0; position <= route_size; ++position) {
            const auto destination = position < route_size ? route[position] : depot;
            current_time += distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] / vehicle[4];
            current_time = std::max(current_time, ready[destination]);
            if (kinds[destination] == customer_kind) {
                earliest[static_cast<std::size_t>(destination)] = current_time;
                has_earliest[static_cast<std::size_t>(destination)] = true;
                const auto slack = due[destination] - current_time;
                min_slack = std::min(min_slack, slack);
                if (slack < -epsilon) {
                    output.metrics[1] = current_time;
                    output.metrics[2] = slack;
                    reject_screen(output, screen_reason_forward, check_forward, slack);
                    return output;
                }
                current_time += service[destination];
            } else if (kinds[destination] == depot_kind) {
                const auto slack = due[destination] - current_time;
                min_slack = std::min(min_slack, slack);
                if (slack < -epsilon) {
                    output.metrics[1] = current_time;
                    output.metrics[2] = slack;
                    reject_screen(output, screen_reason_forward, check_forward, slack);
                    return output;
                }
            }
            origin = destination;
        }
    }
    append_screen_event(output, check_forward, check_pass, current_time);

    if (use_incremental) {
        if (incremental[5] < 0.5) {
            output.metrics[1] = current_time;
            output.metrics[2] = min_slack;
            reject_screen(output, screen_reason_backward, check_backward, min_slack);
            return output;
        }
    } else {
        auto latest_departure = due[depot];
        std::vector<double> latest(node_count, 0.0);
        std::vector<bool> has_latest(node_count, false);
        for (std::size_t reverse = route_size + 1; reverse-- > 0;) {
            const auto origin = reverse == 0 ? depot : route[reverse - 1];
            const auto destination = reverse < route_size ? route[reverse] : depot;
            double latest_arrival = 0.0;
            if (kinds[destination] == customer_kind) {
                latest_arrival = std::min(
                    due[destination], latest_departure - service[destination]);
                latest[static_cast<std::size_t>(destination)] = latest_arrival;
                has_latest[static_cast<std::size_t>(destination)] = true;
            } else {
                latest_arrival = std::min(due[destination], latest_departure);
            }
            latest_departure = latest_arrival - distances[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] / vehicle[4];
        }
        for (std::size_t node = 0; node < node_count; ++node) {
            if (has_earliest[node] && has_latest[node]) {
                const auto slack = latest[node] - earliest[node];
                min_slack = std::min(min_slack, slack);
                if (slack < -epsilon) {
                    output.metrics[1] = current_time;
                    output.metrics[2] = slack;
                    reject_screen(output, screen_reason_backward, check_backward, slack);
                    return output;
                }
            }
        }
    }
    const auto reported_slack = std::isfinite(min_slack) ? min_slack : 0.0;
    append_screen_event(output, check_backward, check_pass, reported_slack);
    if (min_slack < -epsilon) {
        output.metrics[1] = current_time;
        output.metrics[2] = min_slack;
        reject_screen(output, screen_reason_slack, check_slack, min_slack);
        return output;
    }
    append_screen_event(output, check_slack, check_pass, reported_slack);
    append_screen_event(output, check_distance, check_recorded, distance_lower_bound);

    auto origin = depot;
    for (std::size_t position = 0; position <= route_size; ++position) {
        const auto destination = position < route_size ? route[position] : depot;
        if (reachable[
                static_cast<std::size_t>(origin) * node_count
                + static_cast<std::size_t>(destination)] == 0U) {
            output.codes[3] = 0;
            output.codes[4] = 0;
            output.metrics[1] = current_time;
            output.metrics[2] = reported_slack;
            output.metrics[6] = 0.0;
            reject_screen(output, screen_reason_energy, check_energy, 0.0);
            return output;
        }
        origin = destination;
    }
    output.codes[3] = 1;
    output.codes[4] = 1;
    append_screen_event(output, check_energy, check_pass, 1.0);

    double structural_energy = 0.0;
    for (std::size_t position = 0; position < route_size; ++position) {
        const auto customer = route[position];
        auto to_customer = std::numeric_limits<double>::infinity();
        auto from_customer = std::numeric_limits<double>::infinity();
        for (const auto recharge : recharge_nodes) {
            to_customer = std::min(
                to_customer,
                distances[static_cast<std::size_t>(recharge) * node_count
                    + static_cast<std::size_t>(customer)]);
            from_customer = std::min(
                from_customer,
                distances[static_cast<std::size_t>(customer) * node_count
                    + static_cast<std::size_t>(recharge)]);
        }
        structural_energy = std::max(
            structural_energy, (to_customer + from_customer) * vehicle[2]);
    }
    output.metrics[5] = structural_energy;
    if (structural_energy > vehicle[0] + epsilon) {
        output.metrics[1] = current_time;
        output.metrics[2] = reported_slack;
        reject_screen(
            output,
            screen_reason_structural_energy,
            check_structural_energy,
            structural_energy);
        return output;
    }
    append_screen_event(output, check_structural_energy, check_pass, structural_energy);
    output.codes[0] = 1;
    output.codes[1] = screen_reason_none;
    output.codes[2] = 0;
    output.metrics[1] = current_time;
    output.metrics[2] = reported_slack;
    output.metrics[6] = 1.0;
    return output;
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
    ScreenOutput result;
    {
        py::gil_scoped_release release;
        result = run_screen_route(
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
                    const auto screened = run_screen_route(
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
        const auto screened = run_screen_route(
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

    std::vector<ScreenOutput> outputs(candidate_count);
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
                    outputs[index] = run_screen_route(
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
        || context_array.request().shape[0] != 3) {
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
    py::tuple exact_payload = exact_charging_batch_numeric(
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
    return candidate_round_transaction_impl(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, route_offsets, route_indices, candidate_ids,
        lexical_rank, options, incremental, negative_offsets, negative_indices,
        negative_reason_codes, cache_hit_flags, control, deadline_remaining,
        batch_size, context_ids,
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
    py::handle batch_size, py::handle context_ids) {
    return candidate_round_transaction_impl(
        node_kind, demand, ready_time, due_date, service_time, distance,
        reachable, vehicle, route_offsets, route_indices, candidate_ids,
        lexical_rank, options, incremental, negative_offsets, negative_indices,
        negative_reason_codes, cache_hit_flags, control, deadline_remaining,
        batch_size, context_ids,
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
        const auto round_objective = [](double value) {
            constexpr auto scale = 1'000'000'000.0;
            return std::nearbyint(value * scale) / scale;
        };
        return std::make_tuple(
            static_cast<std::int64_t>(vehicle_count),
            round_objective(total_distance),
            round_objective(total_charging_time),
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
        if (statuses[index] != 0) {
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

class NativeSearchEngineV2 {
public:
    NativeSearchEngineV2(
        std::int64_t exact_budget,
        std::int64_t round_budget,
        std::int64_t cache_entries,
        std::int64_t cache_memory_bytes)
        : route_cache_(cache_entries, cache_memory_bytes),
          budget_(exact_budget, round_budget) {}

    py::tuple initialize(
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
        if (initialized_) {
            throw std::runtime_error(
                "full native v2 search engine cannot be initialized twice");
        }
        auto initialized = full_native_initialize_v2(
            node_kind,
            ready_time,
            due_date,
            service_time,
            distance,
            vehicle,
            initial_route_offsets,
            initial_route_indices,
            control,
            deadline_remaining);
        auto offsets_array = checked_array<std::int64_t>(
            initial_route_offsets, "initial_route_offsets", 1);
        auto indices_array = checked_array<std::int64_t>(
            initial_route_indices, "initial_route_indices", 1);
        auto exact_payload = py::cast<py::tuple>(initialized[0]);
        const auto route_count = static_cast<std::int64_t>(offsets_array.size() - 1);
        const auto reservation = budget_.reserve_exact(route_count);
        const auto* reserved = checked_data<std::int64_t>(reservation);
        if (reserved[1] != route_count) {
            throw std::runtime_error(
                "full native warm start does not fit the exact-call budget");
        }
        auto exact_counters = py::cast<py::array_t<std::int64_t>>(exact_payload[6]);
        const auto* counter_values = checked_data<std::int64_t>(exact_counters);
        if (counter_values[2] != route_count || counter_values[3] != 0) {
            budget_.interrupt_exact(route_count);
            throw std::runtime_error(
                "full native warm-start exact transaction did not complete atomically");
        }
        budget_.complete_exact(route_count);

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
        py::array_t<std::uint8_t> semantic_hashes(
            {static_cast<py::ssize_t>(route_count), py::ssize_t(32)});
        py::array_t<std::int64_t> entry_bytes(route_count);
        for (std::int64_t route = 0; route < route_count; ++route) {
            std::string evidence("stage05.2-native-route-result-v2");
            const auto append = [&evidence](const auto* values, std::size_t count) {
                evidence.append(
                    reinterpret_cast<const char*>(values), count * sizeof(*values));
            };
            append(
                route_indices + route_offsets[route],
                static_cast<std::size_t>(route_offsets[route + 1] - route_offsets[route]));
            append(statuses + route, 1);
            append(reasons + route, 1);
            append(metrics + route * 4, 4);
            append(labels + route * 3, 3);
            append(
                path_indices + path_offsets[route],
                static_cast<std::size_t>(path_offsets[route + 1] - path_offsets[route]));
            const auto digest = native_sha256_digest(evidence);
            std::copy(
                digest.begin(), digest.end(),
                checked_data(semantic_hashes) + route * 32);
            checked_data(entry_bytes)[route] = static_cast<std::int64_t>(
                256
                + (route_offsets[route + 1] - route_offsets[route])
                    * static_cast<std::int64_t>(sizeof(std::int64_t))
                + (path_offsets[route + 1] - path_offsets[route])
                    * static_cast<std::int64_t>(sizeof(std::int64_t)));
        }
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
        route_cache_.commit_store_batch();
        current_offsets_ = py::cast<py::array_t<std::int64_t>>(offsets_array);
        current_indices_ = py::cast<py::array_t<std::int64_t>>(indices_array);
        current_exact_payload_ = exact_payload;
        current_objective_integer_ = py::cast<py::array_t<std::int64_t>>(initialized[1]);
        current_objective_float_ = py::cast<py::array_t<double>>(initialized[2]);
        initialized_ = true;
        return initialized;
    }

    [[nodiscard]] bool initialized() const noexcept {
        return initialized_;
    }

private:
    NativeRouteCacheV2 route_cache_;
    NativeBudgetStateV2 budget_;
    NativeAttemptedPlanSetV2 attempted_plans_;
    bool initialized_ = false;
    py::array_t<std::int64_t> current_offsets_;
    py::array_t<std::int64_t> current_indices_;
    py::tuple current_exact_payload_;
    py::array_t<std::int64_t> current_objective_integer_;
    py::array_t<double> current_objective_float_;
};

py::tuple full_native_alns_v2(
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
    py::handle deadline_remaining,
    py::handle protocol_control,
    py::handle protocol_options,
    py::handle stage04_integer,
    py::handle stage04_float,
    py::handle operator_integer,
    py::handle operator_float) {
    static_cast<void>(node_kind);
    static_cast<void>(demand);
    static_cast<void>(ready_time);
    static_cast<void>(due_date);
    static_cast<void>(service_time);
    static_cast<void>(distance);
    static_cast<void>(vehicle);
    static_cast<void>(lexical_rank);
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
    const auto* protocol_values = checked_data<std::int64_t>(
        protocol_control_array);
    if ((protocol_values[0] != 0 && protocol_values[0] != 1)
        || protocol_values[1] <= 0 || protocol_values[2] <= 0
        || (protocol_values[3] != 1 && protocol_values[3] != 4)
        || protocol_values[4] < 10 || protocol_values[5] < 0) {
        throw std::invalid_argument(
            "full native v2 Candidate Control values are invalid");
    }
    const auto* options = checked_data<double>(protocol_options_array);
    if (!(options[0] > 0.0 && options[0] <= 1.0)
        || !std::isfinite(options[1]) || options[1] <= 0.0) {
        throw std::invalid_argument("full native v2 protocol options are invalid");
    }
    const auto* base_control_values = checked_data<std::int64_t>(base_control);
    NativeSearchEngineV2 engine(
        base_control_values[4],
        protocol_values[2],
        protocol_values[7],
        protocol_values[8]);
    static_cast<void>(engine.initialize(
        node_kind,
        ready_time,
        due_date,
        service_time,
        distance,
        vehicle,
        initial_route_offsets,
        initial_route_indices,
        control,
        deadline_remaining));
    if (!engine.initialized()) {
        throw std::logic_error("full native v2 search engine lost initialization state");
    }
    throw std::runtime_error(
        "full native v2 semantic engine is incomplete; refusing prototype fallback");
}

#ifdef __linux__
namespace {

constexpr std::uint64_t scheduler_max_control_bytes = 1ULL << 20;

bool scheduler_read_exact(int descriptor, char* destination, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
        const auto received = ::recv(
            descriptor, destination + offset, size - offset, 0);
        if (received <= 0) {
            return false;
        }
        offset += static_cast<std::size_t>(received);
    }
    return true;
}

void scheduler_send_all(int descriptor, const char* source, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
        const auto sent = ::send(
            descriptor, source + offset, size - offset, MSG_NOSIGNAL);
        if (sent <= 0) {
            throw std::runtime_error("host scheduler could not send a complete IPC frame");
        }
        offset += static_cast<std::size_t>(sent);
    }
}

std::string scheduler_read_frame(int descriptor) {
    std::array<unsigned char, 8> header{};
    if (!scheduler_read_exact(
            descriptor,
            reinterpret_cast<char*>(header.data()),
            header.size())) {
        throw std::runtime_error("host scheduler received a partial IPC frame");
    }
    std::uint64_t size = 0;
    for (const auto byte : header) {
        size = (size << 8) | static_cast<std::uint64_t>(byte);
    }
    if (size == 0 || size > scheduler_max_control_bytes) {
        throw std::runtime_error("host scheduler control frame size is invalid");
    }
    std::string payload(static_cast<std::size_t>(size), '\0');
    if (!scheduler_read_exact(descriptor, payload.data(), payload.size())) {
        throw std::runtime_error("host scheduler received a partial IPC frame");
    }
    return payload;
}

void scheduler_send_frame(int descriptor, const std::string& payload) {
    std::array<unsigned char, 8> header{};
    auto size = static_cast<std::uint64_t>(payload.size());
    for (std::size_t index = header.size(); index-- > 0;) {
        header[index] = static_cast<unsigned char>(size & 0xffU);
        size >>= 8;
    }
    scheduler_send_all(
        descriptor,
        reinterpret_cast<const char*>(header.data()),
        header.size());
    scheduler_send_all(descriptor, payload.data(), payload.size());
}

void scheduler_handle_connection(int descriptor) {
    std::string output_name;
    try {
        const auto request_text = scheduler_read_frame(descriptor);
        py::gil_scoped_acquire acquire;
        const auto json = py::module_::import("json");
        const auto shared_memory = py::module_::import("multiprocessing.shared_memory");
        const auto numpy = py::module_::import("numpy");
        const auto pickle = py::module_::import("pickle");
        const auto request = py::cast<py::dict>(
            json.attr("loads")(request_text));
        if (!request.contains("operation")
            || py::cast<std::string>(request["operation"]) != "full_native_alns_v1") {
            throw std::runtime_error("host scheduler received an unknown operation");
        }
        const auto descriptors = py::cast<py::list>(request["arrays"]);
        if (descriptors.size() != 12) {
            throw std::runtime_error(
                "host scheduler received an invalid SoA descriptor set");
        }
        py::list segments;
        py::list arrays;
        for (const auto item : descriptors) {
            const auto raw = py::cast<py::dict>(item);
            const auto segment = shared_memory.attr("SharedMemory")(
                py::arg("name") = raw["name"],
                py::arg("track") = false);
            segments.append(segment);
            const auto array = numpy.attr("ndarray")(
                raw["shape"],
                py::arg("dtype") = raw["dtype"],
                py::arg("buffer") = segment.attr("buf"));
            if (!py::cast<bool>(array.attr("flags").attr("c_contiguous"))) {
                throw std::runtime_error(
                    "host scheduler shared-memory array is not contiguous");
            }
            if (py::cast<py::ssize_t>(array.attr("nbytes"))
                != py::cast<py::ssize_t>(raw["nbytes"])) {
                throw std::runtime_error(
                    "host scheduler shared-memory array does not reconcile");
            }
            arrays.append(array);
        }
        const auto result = full_native_alns_v1(
            arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5],
            arrays[6], arrays[7], arrays[8], arrays[9], arrays[10], arrays[11]);
        const auto encoded_object = pickle.attr("dumps")(
            result, py::arg("protocol") = 5);
        const auto encoded = py::cast<std::string>(encoded_object);
        const auto output = shared_memory.attr("SharedMemory")(
            py::arg("create") = true,
            py::arg("size") = std::max<std::size_t>(1, encoded.size()));
        output.attr("buf").attr("__setitem__")(
            py::slice(0, static_cast<py::ssize_t>(encoded.size()), 1),
            py::bytes(encoded));
        py::dict response;
        response["ok"] = true;
        output_name = py::cast<std::string>(output.attr("name"));
        response["output_name"] = output_name;
        response["output_size"] = encoded.size();
        const auto response_text = py::cast<std::string>(json.attr("dumps")(
            response,
            py::arg("sort_keys") = true,
            py::arg("separators") = py::make_tuple(",", ":")));
        for (const auto segment : segments) {
            py::reinterpret_borrow<py::object>(segment).attr("close")();
        }
        scheduler_send_frame(descriptor, response_text);
        const auto acknowledgement = py::cast<py::dict>(
            json.attr("loads")(scheduler_read_frame(descriptor)));
        if (!acknowledgement.contains("ack_output_name")
            || py::cast<std::string>(acknowledgement["ack_output_name"])
                != output_name) {
            throw std::runtime_error("host scheduler output acknowledgement mismatch");
        }
        output.attr("unlink")();
        output.attr("close")();
        output_name.clear();
        py::dict released;
        released["ok"] = true;
        released["released"] = true;
        scheduler_send_frame(
            descriptor,
            py::cast<std::string>(json.attr("dumps")(
                released,
                py::arg("sort_keys") = true,
                py::arg("separators") = py::make_tuple(",", ":"))));
    } catch (const std::exception& error) {
        try {
            py::gil_scoped_acquire acquire;
            const auto json = py::module_::import("json");
            if (!output_name.empty()) {
                const auto shared_memory = py::module_::import(
                    "multiprocessing.shared_memory");
                const auto orphan = shared_memory.attr("SharedMemory")(
                    py::arg("name") = output_name,
                    py::arg("track") = false);
                orphan.attr("close")();
                orphan.attr("unlink")();
            }
            py::dict response;
            response["ok"] = false;
            response["error_type"] = "RuntimeError";
            response["error"] = error.what();
            const auto response_text = py::cast<std::string>(json.attr("dumps")(
                response,
                py::arg("sort_keys") = true,
                py::arg("separators") = py::make_tuple(",", ":")));
            scheduler_send_frame(descriptor, response_text);
        } catch (const std::exception& response_error) {
            std::fprintf(
                stderr,
                "host scheduler could not report transaction failure: %s\n",
                response_error.what());
            std::fflush(stderr);
        }
    }
    ::close(descriptor);
}

}  // namespace
#endif

void run_host_scheduler_service_v1(
    const std::string& socket_path,
    std::int64_t worker_threads) {
#ifndef __linux__
    static_cast<void>(socket_path);
    static_cast<void>(worker_threads);
    throw std::runtime_error("host scheduler service requires Linux Unix-domain sockets");
#else
    if (socket_path.empty() || socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        throw std::invalid_argument("host scheduler socket path is invalid");
    }
    if (worker_threads != 24) {
        throw std::invalid_argument("host scheduler requires exactly 24 worker threads");
    }
    ::unlink(socket_path.c_str());
    const auto server = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server < 0) {
        throw std::runtime_error("host scheduler could not create its Unix-domain socket");
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (::bind(
            server,
            reinterpret_cast<const sockaddr*>(&address),
            sizeof(address)) != 0
        || ::listen(server, 64) != 0) {
        ::close(server);
        ::unlink(socket_path.c_str());
        throw std::runtime_error("host scheduler could not bind/listen on its control socket");
    }
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<int> connections;
    bool stopping = false;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(worker_threads));
    int accept_error = 0;
    {
        py::gil_scoped_release release;
        for (std::int64_t index = 0; index < worker_threads; ++index) {
            workers.emplace_back([&]() {
                while (true) {
                    int connection = -1;
                    {
                        std::unique_lock lock(mutex);
                        condition.wait(lock, [&]() {
                            return stopping || !connections.empty();
                        });
                        if (connections.empty()) {
                            if (stopping) {
                                return;
                            }
                            continue;
                        }
                        connection = connections.front();
                        connections.pop_front();
                    }
                    scheduler_handle_connection(connection);
                }
            });
        }
        while (true) {
            const auto connection = ::accept(server, nullptr, nullptr);
            if (connection < 0) {
                if (errno == EINTR) {
                    continue;
                }
                accept_error = errno;
                break;
            }
            char control = '\0';
            const auto peeked = ::recv(connection, &control, 1, MSG_PEEK);
            if (peeked == 1 && control == 'X') {
                ::recv(connection, &control, 1, 0);
                ::close(connection);
                break;
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
            {
                std::lock_guard lock(mutex);
                connections.push_back(connection);
            }
            condition.notify_one();
        }
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        condition.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
        if (accept_error != 0) {
            std::fprintf(
                stderr,
                "host scheduler accept failed with errno %d\n",
                accept_error);
            std::fflush(stderr);
        }
    }
    ::close(server);
    ::unlink(socket_path.c_str());
    if (accept_error != 0) {
        throw std::runtime_error("host scheduler accept failed");
    }
#endif
}

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
        py::arg("context_ids"));
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
        "full_native_alns_v2",
        &full_native_alns_v2,
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
        py::arg("deadline_remaining"),
        py::arg("protocol_control"),
        py::arg("protocol_options"),
        py::arg("stage04_integer"),
        py::arg("stage04_float"),
        py::arg("operator_integer"),
        py::arg("operator_float"));
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
        "run_host_scheduler_service_v1",
        &run_host_scheduler_service_v1,
        py::arg("socket_path"),
        py::arg("worker_threads"));
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
