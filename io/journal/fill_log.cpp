#include "io/journal/fill_log.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <system_error>

namespace clob {

namespace {

constexpr std::streamsize FILL_BYTES = 32;

template <typename T>
void append_le(std::array<std::uint8_t, FILL_BYTES>& buf, std::size_t offset, T value) {
    static_assert(std::endian::native == std::endian::little);
    auto bytes = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(value);
    for (std::size_t i = 0; i < sizeof(T); ++i) buf[offset + i] = bytes[i];
}

template <typename T>
T read_le(const std::uint8_t* p) {
    static_assert(std::endian::native == std::endian::little);
    std::array<std::uint8_t, sizeof(T)> buf{};
    for (std::size_t i = 0; i < sizeof(T); ++i) buf[i] = p[i];
    return std::bit_cast<T>(buf);
}

void write_all(int fd, const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::system_category(), "FillLogWriter write");
        }
        p += w;
        n -= static_cast<std::size_t>(w);
    }
}

}  // namespace

FillLogWriter::FillLogWriter(const std::filesystem::path& path) {
    fd_ = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd_ < 0) {
        throw std::system_error(errno, std::system_category(),
                                "FillLogWriter open: " + path.string());
    }
}

FillLogWriter::~FillLogWriter() {
    if (fd_ >= 0) ::close(fd_);
}

void FillLogWriter::write(const Fill& fill) {
    std::array<std::uint8_t, FILL_BYTES> buf{};
    append_le(buf, 0,  fill.taker_id.value());
    append_le(buf, 8,  fill.maker_id.value());
    append_le(buf, 16, fill.price.value());
    append_le(buf, 24, fill.quantity.value());
    write_all(fd_, buf.data(), buf.size());
    if (::fsync(fd_) != 0) {
        throw std::system_error(errno, std::system_category(), "FillLogWriter fsync");
    }
}

FillLogReader::FillLogReader(const std::filesystem::path& path)
    : stream_(path, std::ios::binary) {
    if (!stream_) {
        throw std::runtime_error("FillLogReader: cannot open " + path.string());
    }
}

std::optional<Fill> FillLogReader::next() {
    std::array<std::uint8_t, FILL_BYTES> buf{};
    stream_.read(reinterpret_cast<char*>(buf.data()), FILL_BYTES);
    const auto got = stream_.gcount();
    if (got == 0 && stream_.eof()) {
        end_reason_ = EndReason::CleanEof;
        return std::nullopt;
    }
    if (got != FILL_BYTES) {
        end_reason_ = EndReason::TruncatedTrailingRecord;
        return std::nullopt;
    }
    return Fill{
        OrderId{read_le<std::uint64_t>(buf.data() + 0)},
        OrderId{read_le<std::uint64_t>(buf.data() + 8)},
        Price{read_le<std::int64_t>(buf.data() + 16)},
        Quantity{read_le<std::int64_t>(buf.data() + 24)},
    };
}

}  // namespace clob
