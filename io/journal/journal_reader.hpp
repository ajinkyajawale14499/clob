#pragma once

#include <filesystem>
#include <fstream>
#include <optional>

#include "io/journal/journal_event.hpp"

namespace clob {

// Streaming reader for journals written by JournalWriter.
//
// Recovery semantics: if the trailing record is incomplete (the process was
// killed mid-write), `next()` returns nullopt and `end_reason()` returns
// TruncatedTrailingRecord. Earlier complete records are still returned.
// Corrupt records (invalid tag, length exceeds cap) throw — those are not
// crash-recovery scenarios, they're file corruption.
class JournalReader {
public:
    enum class EndReason { CleanEof, TruncatedTrailingRecord };

    explicit JournalReader(const std::filesystem::path& path);

    JournalReader(const JournalReader&) = delete;
    JournalReader& operator=(const JournalReader&) = delete;

    std::optional<OrderEvent> next();
    [[nodiscard]] EndReason end_reason() const noexcept { return end_reason_; }

private:
    std::ifstream stream_;
    EndReason end_reason_ = EndReason::CleanEof;
};

}  // namespace clob
