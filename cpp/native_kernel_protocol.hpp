#pragma once

#ifdef __linux__

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace evrptw::native_protocol {

constexpr std::uint64_t kernel_magic = 0x4556525054574b33ULL;
constexpr std::uint32_t kernel_protocol_version = 3;
constexpr std::size_t maximum_arrays = 24;
constexpr std::size_t maximum_segment_name = 128;
constexpr std::size_t maximum_payload_bytes = 256U * 1024U * 1024U;

enum class KernelOperation : std::uint32_t {
    exact_charging = 1,
    screen_route = 2,
    screen_routes = 3,
    search_request_receipt = 4,
    search_initial_state = 5,
    candidate_transaction_wire = 6,
    candidate_session_open = 7,
    candidate_session_close = 8,
    candidate_transaction_execute = 9,
    candidate_transaction_commit = 10,
    candidate_transaction_rollback = 11,
    candidate_transaction_status = 12,
};

enum class NumericType : std::uint32_t {
    int64 = 1,
    float64 = 2,
    uint8 = 3,
};

enum class ControlMessage : std::uint32_t {
    request = 1,
    response = 2,
    acknowledgement = 3,
    released = 4,
    failure = 5,
    shutdown = 6,
    test_request = 7,
};

enum class FailureCode : std::uint64_t {
    unspecified = 0,
    deadline = 1,
};

struct ArrayDescriptor final {
    NumericType type{};
    std::uint32_t dimensions = 0;
    std::uint64_t offset = 0;
    std::uint64_t count = 0;
    std::array<std::uint64_t, 2> shape{};
};

struct PayloadHeader final {
    std::uint64_t magic = kernel_magic;
    std::uint32_t version = kernel_protocol_version;
    KernelOperation operation{};
    std::uint64_t request_id = 0;
    std::uint32_t array_count = 0;
    std::uint32_t reserved = 0;
    std::array<ArrayDescriptor, maximum_arrays> arrays{};
};

struct ControlFrame final {
    std::uint64_t magic = kernel_magic;
    std::uint32_t version = kernel_protocol_version;
    ControlMessage message{};
    std::uint64_t request_id = 0;
    std::uint64_t segment_size = 0;
    std::array<char, maximum_segment_name> segment_name{};
    std::array<char, 64> sha256{};
    std::array<char, 192> error{};
};

static_assert(sizeof(ArrayDescriptor) == 40);
static_assert(sizeof(ControlFrame) == 416);

inline std::size_t numeric_size(NumericType type) {
    switch (type) {
    case NumericType::int64:
    case NumericType::float64:
        return 8;
    case NumericType::uint8:
        return 1;
    }
    throw std::runtime_error("native kernel protocol has an unknown numeric type");
}

inline std::size_t align_eight(std::size_t value) {
    if (value > std::numeric_limits<std::size_t>::max() - 7U) {
        throw std::overflow_error("native kernel payload alignment overflow");
    }
    return (value + 7U) & ~std::size_t{7U};
}

inline std::size_t checked_product(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view message) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::runtime_error(std::string(message));
    }
    const auto product = left * right;
    if (product > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(message));
    }
    return static_cast<std::size_t>(product);
}

class PayloadBuilder final {
public:
    PayloadBuilder(KernelOperation operation, std::uint64_t request_id) {
        header_.operation = operation;
        header_.request_id = request_id;
        bytes_.resize(sizeof(PayloadHeader), 0U);
    }

    void add(
        NumericType type,
        const void* source,
        std::uint64_t count,
        std::uint64_t first_extent,
        std::uint64_t second_extent = 0) {
        if (header_.array_count >= maximum_arrays
            || (count > 0 && source == nullptr)
            || (second_extent == 0 && count != first_extent)
            || (second_extent != 0
                && checked_product(
                       first_extent, second_extent,
                       "native kernel payload shape overflows") != count)) {
            throw std::invalid_argument("native kernel payload array is invalid");
        }
        bytes_.resize(align_eight(bytes_.size()), 0U);
        auto& descriptor = header_.arrays[header_.array_count++];
        descriptor.type = type;
        descriptor.dimensions = second_extent == 0 ? 1U : 2U;
        descriptor.offset = bytes_.size();
        descriptor.count = count;
        descriptor.shape = {first_extent, second_extent};
        const auto size = checked_product(
            count, numeric_size(type),
            "native kernel payload byte size overflows");
        if (bytes_.size() > maximum_payload_bytes
            || size > maximum_payload_bytes - bytes_.size()) {
            throw std::length_error("native kernel payload exceeds its size limit");
        }
        if (size > 0) {
            const auto* first = static_cast<const std::uint8_t*>(source);
            bytes_.insert(bytes_.end(), first, first + size);
        }
    }

    std::vector<std::uint8_t> finish() {
        std::memcpy(bytes_.data(), &header_, sizeof(header_));
        return std::move(bytes_);
    }

private:
    PayloadHeader header_{};
    std::vector<std::uint8_t> bytes_;
};

class PayloadView final {
public:
    PayloadView(const void* address, std::size_t size)
        : address_(static_cast<const std::uint8_t*>(address)), size_(size) {
        if (address_ == nullptr || size_ < sizeof(PayloadHeader)
            || size_ > maximum_payload_bytes) {
            throw std::runtime_error("native kernel payload is truncated");
        }
        std::memcpy(&header_, address_, sizeof(header_));
        if (header_.magic != kernel_magic
            || header_.version != kernel_protocol_version
            || header_.array_count > maximum_arrays
            || header_.reserved != 0
            || (header_.operation != KernelOperation::exact_charging
                && header_.operation != KernelOperation::screen_route
                && header_.operation != KernelOperation::screen_routes
                && header_.operation
                    != KernelOperation::search_request_receipt
                && header_.operation
                    != KernelOperation::search_initial_state
                && header_.operation
                    != KernelOperation::candidate_transaction_wire
                && header_.operation
                    != KernelOperation::candidate_session_open
                && header_.operation
                    != KernelOperation::candidate_session_close
                && header_.operation
                    != KernelOperation::candidate_transaction_execute
                && header_.operation
                    != KernelOperation::candidate_transaction_commit
                && header_.operation
                    != KernelOperation::candidate_transaction_rollback
                && header_.operation
                    != KernelOperation::candidate_transaction_status)) {
            throw std::runtime_error("native kernel payload header is invalid");
        }
        std::size_t previous_end = sizeof(PayloadHeader);
        for (std::size_t index = 0; index < header_.array_count; ++index) {
            const auto& descriptor = header_.arrays[index];
            const auto bytes = checked_product(
                descriptor.count, numeric_size(descriptor.type),
                "native kernel array byte size overflows");
            if ((descriptor.dimensions != 1 && descriptor.dimensions != 2)
                || descriptor.offset < sizeof(PayloadHeader)
                || descriptor.offset % 8 != 0
                || descriptor.offset < previous_end
                || descriptor.offset > size_
                || bytes > size_ - descriptor.offset
                || (descriptor.dimensions == 1
                    && (descriptor.shape[1] != 0
                        || descriptor.shape[0] != descriptor.count))
                || (descriptor.dimensions == 2
                    && (descriptor.shape[1] == 0
                        || checked_product(
                               descriptor.shape[0], descriptor.shape[1],
                               "native kernel array shape overflows")
                            != descriptor.count))) {
                throw std::runtime_error("native kernel array descriptor is invalid");
            }
            previous_end = static_cast<std::size_t>(descriptor.offset) + bytes;
        }
        if (previous_end != size_) {
            throw std::runtime_error("native kernel payload has trailing bytes");
        }
    }

    const PayloadHeader& header() const noexcept { return header_; }

    const ArrayDescriptor& descriptor(std::size_t index) const {
        if (index >= header_.array_count) {
            throw std::out_of_range("native kernel array index is invalid");
        }
        return header_.arrays[index];
    }

    template <typename T>
    const T* data(std::size_t index, NumericType expected) const {
        const auto& item = descriptor(index);
        if (item.type != expected || numeric_size(expected) != sizeof(T)) {
            throw std::runtime_error("native kernel array type is invalid");
        }
        return reinterpret_cast<const T*>(address_ + item.offset);
    }

private:
    const std::uint8_t* address_;
    std::size_t size_;
    PayloadHeader header_{};
};

class SharedMapping final {
public:
    SharedMapping() = default;
    SharedMapping(const SharedMapping&) = delete;
    SharedMapping& operator=(const SharedMapping&) = delete;
    SharedMapping(SharedMapping&& other) noexcept { swap(other); }
    SharedMapping& operator=(SharedMapping&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }
    ~SharedMapping() noexcept { reset(); }

    static SharedMapping create(std::string name, std::size_t size) {
        if (name.empty() || size == 0 || size > maximum_payload_bytes) {
            throw std::invalid_argument("native kernel shared segment is invalid");
        }
        SharedMapping mapping;
        mapping.name_ = std::move(name);
        mapping.descriptor_ = ::shm_open(
            mapping.name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        if (mapping.descriptor_ < 0) {
            throw std::runtime_error("native kernel could not create shared memory");
        }
        mapping.owner_ = true;
        if (::ftruncate(mapping.descriptor_, static_cast<off_t>(size)) != 0) {
            throw std::runtime_error("native kernel could not size shared memory");
        }
        mapping.address_ = ::mmap(
            nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
            mapping.descriptor_, 0);
        if (mapping.address_ == MAP_FAILED) {
            mapping.address_ = nullptr;
            throw std::runtime_error("native kernel could not map shared memory");
        }
        mapping.size_ = size;
        return mapping;
    }

    static SharedMapping open(
        std::string name, std::size_t size, bool writable = false) {
        if (name.empty() || size == 0 || size > maximum_payload_bytes) {
            throw std::invalid_argument("native kernel shared segment is invalid");
        }
        SharedMapping mapping;
        mapping.name_ = std::move(name);
        mapping.descriptor_ = ::shm_open(
            mapping.name_.c_str(), writable ? O_RDWR : O_RDONLY, 0600);
        if (mapping.descriptor_ < 0) {
            throw std::runtime_error("native kernel could not open shared memory");
        }
        struct stat status{};
        if (::fstat(mapping.descriptor_, &status) != 0
            || status.st_size != static_cast<off_t>(size)) {
            throw std::runtime_error("native kernel shared-memory size mismatch");
        }
        mapping.address_ = ::mmap(
            nullptr, size, writable ? PROT_READ | PROT_WRITE : PROT_READ,
            MAP_SHARED, mapping.descriptor_, 0);
        if (mapping.address_ == MAP_FAILED) {
            mapping.address_ = nullptr;
            throw std::runtime_error("native kernel could not map shared memory");
        }
        mapping.size_ = size;
        return mapping;
    }

    void release_ownership() noexcept { owner_ = false; }
    void* address() const noexcept { return address_; }
    std::size_t size() const noexcept { return size_; }
    const std::string& name() const noexcept { return name_; }

private:
    void reset() noexcept {
        if (address_ != nullptr) {
            ::munmap(address_, size_);
        }
        if (descriptor_ >= 0) {
            ::close(descriptor_);
        }
        if (owner_ && !name_.empty()) {
            ::shm_unlink(name_.c_str());
        }
        address_ = nullptr;
        descriptor_ = -1;
        size_ = 0;
        owner_ = false;
        name_.clear();
    }

    void swap(SharedMapping& other) noexcept {
        std::swap(name_, other.name_);
        std::swap(descriptor_, other.descriptor_);
        std::swap(address_, other.address_);
        std::swap(size_, other.size_);
        std::swap(owner_, other.owner_);
    }

    std::string name_;
    int descriptor_ = -1;
    void* address_ = nullptr;
    std::size_t size_ = 0;
    bool owner_ = false;
};

inline std::string bounded_string(const char* data, std::size_t capacity) {
    const auto end = std::find(data, data + capacity, '\0');
    return std::string(data, end);
}

inline void copy_bounded(std::string_view source, char* target, std::size_t capacity) {
    if (source.size() > capacity) {
        throw std::runtime_error("native kernel control field is too long");
    }
    std::fill(target, target + capacity, '\0');
    std::copy(source.begin(), source.end(), target);
}

}  // namespace evrptw::native_protocol

#endif
