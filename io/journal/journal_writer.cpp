#include "io/journal/journal_writer.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace clob {

namespace {

constexpr std::uint8_t TAG_NEW_LIMIT  = 0x01;
constexpr std::uint8_t TAG_NEW_MARKET = 0x02;
constexpr std::uint8_t TAG_NEW_IOC    = 0x03;
constexpr std::uint8_t TAG_CANCEL     = 0x04;
constexpr std::uint8_t TAG_REPLACE    = 0x05;

// Append a little-endian integer to `buf` (host is LE on x86/arm64; if you ever
// build on a BE target this needs byte-reversal — gate with std::endian).
template <typename T>
void append_le(std::vector<std::uint8_t>& buf, T value) {
    static_assert(std::endian::native == std::endian::little,
                  "Journal format assumes little-endian host");
    auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);
    buf.insert(buf.end(), bytes.begin(), bytes.end());
}

void serialize_payload(std::vector<std::uint8_t>& out, const NewLimit& e) {
    out.push_back(TAG_NEW_LIMIT);
    append_le(out, e.id.value());
    out.push_back(static_cast<std::uint8_t>(e.side));
    append_le(out, e.price.value());
    append_le(out, e.qty.value());
}

void serialize_payload(std::vector<std::uint8_t>& out, const NewMarket& e) {
    out.push_back(TAG_NEW_MARKET);
    append_le(out, e.id.value());
    out.push_back(static_cast<std::uint8_t>(e.side));
    append_le(out, e.qty.value());
}

void serialize_payload(std::vector<std::uint8_t>& out, const NewIoc& e) {
    out.push_back(TAG_NEW_IOC);
    append_le(out, e.id.value());
    out.push_back(static_cast<std::uint8_t>(e.side));
    append_le(out, e.price.value());
    append_le(out, e.qty.value());
}

void serialize_payload(std::vector<std::uint8_t>& out, const Cancel& e) {
    out.push_back(TAG_CANCEL);
    append_le(out, e.id.value());
}

void serialize_payload(std::vector<std::uint8_t>& out, const Replace& e) {
    out.push_back(TAG_REPLACE);
    append_le(out, e.old_id.value());
    append_le(out, e.new_id.value());
    append_le(out, e.price.value());
    append_le(out, e.qty.value());
}

void write_all(int fd, const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::system_category(), "JournalWriter write");
        }
        p += w;
        n -= static_cast<std::size_t>(w);
    }
}

}  // namespace

JournalWriter::JournalWriter(const std::filesystem::path& path) {
    fd_ = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd_ < 0) {
        throw std::system_error(errno, std::system_category(),
                                "JournalWriter open: " + path.string());
    }
}

JournalWriter::~JournalWriter() {
    if (fd_ >= 0) ::close(fd_);
}

JournalWriter::JournalWriter(JournalWriter&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
}

JournalWriter& JournalWriter::operator=(JournalWriter&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

void JournalWriter::write(const OrderEvent& ev) {
    std::vector<std::uint8_t> payload;
    payload.reserve(40);
    std::visit([&](const auto& e) { serialize_payload(payload, e); }, ev);

    const std::uint32_t len = static_cast<std::uint32_t>(payload.size());
    std::array<std::uint8_t, 4> length_bytes =
        std::bit_cast<std::array<std::uint8_t, 4>>(len);

    write_all(fd_, length_bytes.data(), length_bytes.size());
    write_all(fd_, payload.data(), payload.size());

    // Per-call fsync — durability ceiling is the OS cache flush. Tradeoff:
    // throughput drops vs. correctness on crash. v1 ships durable; W10 can
    // batch if benchmarking demands.
    if (::fsync(fd_) != 0) {
        throw std::system_error(errno, std::system_category(), "JournalWriter fsync");
    }
}

}  // namespace clob
