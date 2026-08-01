#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <descrobject.h>

namespace py = pybind11;

using Point = std::pair<double, double>;

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
    std::unordered_map<std::string, std::size_t> first_by_key;
    std::int64_t duplicate_count = 0;
    std::int64_t negative_hit_count = 0;
    std::int64_t screened_count = 0;
    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < candidate_count; ++index) {
            const auto begin = static_cast<std::size_t>(offsets[index]);
            const auto end = static_cast<std::size_t>(offsets[index + 1]);
            const auto key = canonical_key(routes + begin, end - begin);
            const auto duplicate = first_by_key.find(key);
            if (duplicate != first_by_key.end()) {
                statuses[index] = 1;
                duplicate_of[index] = ids[duplicate->second];
                outputs[index] = outputs[duplicate->second];
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
            ++screened_count;
        }
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
    const auto digest = py::cast<std::string>(
        py::module_::import("hashlib")
            .attr("sha256")(py::bytes(evidence))
            .attr("hexdigest")());
    return py::make_tuple(
        std::move(returned_ids),
        std::move(status_array),
        std::move(duplicate_array),
        std::move(codes_array),
        std::move(metrics_array),
        std::move(counters),
        digest);
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
