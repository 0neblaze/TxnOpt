#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace txnopt::native {

enum class PreparedRoundPhase : std::int64_t {
    prepared = 0,
    reserved = 1,
    evaluating = 2,
    validated = 3,
    aborted = 4,
    interrupted = 5,
};

constexpr std::string_view phase_name(const PreparedRoundPhase phase) {
    switch (phase) {
        case PreparedRoundPhase::prepared:
            return "PREPARED";
        case PreparedRoundPhase::reserved:
            return "RESERVED";
        case PreparedRoundPhase::evaluating:
            return "EVALUATING";
        case PreparedRoundPhase::validated:
            return "VALIDATED";
        case PreparedRoundPhase::aborted:
            return "ABORTED";
        case PreparedRoundPhase::interrupted:
            return "INTERRUPTED";
    }
    throw std::logic_error("unknown prepared-round phase");
}

struct PreparedRoundReceipt final {
    PreparedRoundPhase phase = PreparedRoundPhase::prepared;
    std::int64_t requested_work = 0;
    std::int64_t budget_limit = 0;
    std::int64_t reserved_work = 0;
    std::int64_t remaining_work = 0;
    std::int64_t started_work = 0;
    std::int64_t completed_work = 0;
    std::int64_t interrupted_work = 0;
    std::int64_t prepared_cache_write_count = 0;
    std::uint64_t prepared_cache_key_checksum = 0;
    std::vector<PreparedRoundPhase> phase_trace;
};

// Deep internal module behind one native-round call. It owns the native
// reservation, phase trace, and prepared cache delta, but it cannot publish
// Python state or cache entries. The outer TxnRuntime remains the sole commit
// owner and treats this receipt as a prepared result.
class PreparedRoundTransaction final {
public:
    PreparedRoundTransaction(
        const std::int64_t requested_work,
        const std::int64_t budget_limit)
        : requested_work_(requested_work), budget_limit_(budget_limit) {
        if (requested_work < 0 || budget_limit < 0) {
            throw std::invalid_argument(
                "prepared-round work and budget must be non-negative");
        }
        phase_trace_.push_back(PreparedRoundPhase::prepared);
    }

    void reserve() {
        require_phase(PreparedRoundPhase::prepared);
        if (requested_work_ > budget_limit_) {
            transition(PreparedRoundPhase::aborted);
            throw std::invalid_argument(
                "native round budget cannot reserve the complete transaction");
        }
        reserved_work_ = requested_work_;
        transition(PreparedRoundPhase::reserved);
    }

    void begin_evaluation() {
        require_phase(PreparedRoundPhase::reserved);
        transition(PreparedRoundPhase::evaluating);
    }

    void resolve(
        const std::int64_t started_work,
        const std::int64_t completed_work,
        const std::int64_t interrupted_work,
        const std::span<const std::uint64_t> prepared_cache_keys) {
        require_phase(PreparedRoundPhase::evaluating);
        if (started_work < 0 || completed_work < 0 || interrupted_work < 0
            || started_work > reserved_work_
            || completed_work > started_work
            || interrupted_work > started_work
            || completed_work + interrupted_work != started_work) {
            transition(PreparedRoundPhase::aborted);
            throw std::runtime_error(
                "native round work settlement violates the reserved ledger");
        }
        started_work_ = started_work;
        completed_work_ = completed_work;
        interrupted_work_ = interrupted_work;
        if (interrupted_work != 0 || started_work != requested_work_) {
            if (!prepared_cache_keys.empty()) {
                transition(PreparedRoundPhase::aborted);
                throw std::runtime_error(
                    "interrupted native round cannot prepare cache writes");
            }
            transition(PreparedRoundPhase::interrupted);
            return;
        }
        if (completed_work != requested_work_
            || prepared_cache_keys.size()
                != static_cast<std::size_t>(completed_work)) {
            transition(PreparedRoundPhase::aborted);
            throw std::runtime_error(
                "validated native round lacks a complete prepared cache delta");
        }
        prepared_cache_write_count_ = completed_work;
        prepared_cache_key_checksum_ = ordered_key_checksum(prepared_cache_keys);
        transition(PreparedRoundPhase::validated);
    }

    void abort() noexcept {
        if (phase_ == PreparedRoundPhase::prepared
            || phase_ == PreparedRoundPhase::reserved
            || phase_ == PreparedRoundPhase::evaluating) {
            phase_ = PreparedRoundPhase::aborted;
            phase_trace_.push_back(PreparedRoundPhase::aborted);
        }
        prepared_cache_write_count_ = 0;
        prepared_cache_key_checksum_ = 0;
    }

    [[nodiscard]] PreparedRoundReceipt receipt() const {
        if (phase_ != PreparedRoundPhase::validated
            && phase_ != PreparedRoundPhase::interrupted
            && phase_ != PreparedRoundPhase::aborted) {
            throw std::logic_error(
                "prepared-round receipt requires a terminal prepared phase");
        }
        return PreparedRoundReceipt{
            .phase = phase_,
            .requested_work = requested_work_,
            .budget_limit = budget_limit_,
            .reserved_work = reserved_work_,
            .remaining_work = budget_limit_ - reserved_work_,
            .started_work = started_work_,
            .completed_work = completed_work_,
            .interrupted_work = interrupted_work_,
            .prepared_cache_write_count = prepared_cache_write_count_,
            .prepared_cache_key_checksum = prepared_cache_key_checksum_,
            .phase_trace = phase_trace_,
        };
    }

private:
    static constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
    static constexpr std::uint64_t fnv_prime = 1099511628211ULL;

    static std::uint64_t ordered_key_checksum(
        const std::span<const std::uint64_t> keys) noexcept {
        auto checksum = fnv_offset;
        for (const auto key : keys) {
            for (std::size_t byte = 0; byte < sizeof(key); ++byte) {
                checksum ^= (key >> (byte * 8U)) & 0xffU;
                checksum *= fnv_prime;
            }
            checksum ^= 0xffU;
            checksum *= fnv_prime;
        }
        return checksum;
    }

    void require_phase(const PreparedRoundPhase expected) const {
        if (phase_ != expected) {
            throw std::logic_error("prepared-round transition is out of order");
        }
    }

    void transition(const PreparedRoundPhase next) {
        phase_ = next;
        phase_trace_.push_back(next);
    }

    std::int64_t requested_work_ = 0;
    std::int64_t budget_limit_ = 0;
    std::int64_t reserved_work_ = 0;
    std::int64_t started_work_ = 0;
    std::int64_t completed_work_ = 0;
    std::int64_t interrupted_work_ = 0;
    std::int64_t prepared_cache_write_count_ = 0;
    std::uint64_t prepared_cache_key_checksum_ = 0;
    PreparedRoundPhase phase_ = PreparedRoundPhase::prepared;
    std::vector<PreparedRoundPhase> phase_trace_;
};

}  // namespace txnopt::native
