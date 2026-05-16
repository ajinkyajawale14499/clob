#pragma once

#include <filesystem>
#include <fstream>
#include <optional>

#include "core/matching/fill.hpp"

namespace clob {

// Fill log binary format (see docs/adr/0001-replay-determinism.md):
//
//   [ 8 bytes  taker_id  uint64 LE ]
//   [ 8 bytes  maker_id  uint64 LE ]
//   [ 8 bytes  price     int64  LE ]
//   [ 8 bytes  quantity  int64  LE ]
//
// = 32 bytes per fill, no header, no length prefix. The whole file must be a
// multiple of 32 bytes; a trailing partial record is treated as EOF by the
// reader (kill-9 mid-write recovery).
class FillLogWriter {
public:
    explicit FillLogWriter(const std::filesystem::path& path);
    ~FillLogWriter();

    FillLogWriter(const FillLogWriter&) = delete;
    FillLogWriter& operator=(const FillLogWriter&) = delete;

    void write(const Fill& fill);

private:
    int fd_ = -1;
};

class FillLogReader {
public:
    enum class EndReason { CleanEof, TruncatedTrailingRecord };

    explicit FillLogReader(const std::filesystem::path& path);

    FillLogReader(const FillLogReader&) = delete;
    FillLogReader& operator=(const FillLogReader&) = delete;

    std::optional<Fill> next();
    [[nodiscard]] EndReason end_reason() const noexcept { return end_reason_; }

private:
    std::ifstream stream_;
    EndReason end_reason_ = EndReason::CleanEof;
};

}  // namespace clob
