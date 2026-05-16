#pragma once

#include <filesystem>

#include "io/journal/journal_event.hpp"

namespace clob {

// Append-only binary writer. Record layout (little-endian throughout):
//
//   [4-byte length][1-byte tag][payload]
//
// `length` = bytes of (tag + payload), NOT counting the 4 length bytes.
// Tag values:
//   0x01 NewLimit  : OrderId(8) Side(1) Price(8) Quantity(8) = 25 bytes
//   0x02 NewMarket : OrderId(8) Side(1) Quantity(8)          = 17 bytes
//   0x03 NewIoc    : OrderId(8) Side(1) Price(8) Quantity(8) = 25 bytes
//   0x04 Cancel    : OrderId(8)                              =  8 bytes
//   0x05 Replace   : OrderId(8) OrderId(8) Price(8) Quantity(8) = 32 bytes
//
// Opens the file with O_WRONLY | O_APPEND | O_CREAT and fsyncs after each write
// so that a process crash mid-call only loses the in-flight record.
class JournalWriter {
public:
    explicit JournalWriter(const std::filesystem::path& path);
    ~JournalWriter();

    JournalWriter(const JournalWriter&) = delete;
    JournalWriter& operator=(const JournalWriter&) = delete;
    JournalWriter(JournalWriter&&) noexcept;
    JournalWriter& operator=(JournalWriter&&) noexcept;

    void write(const OrderEvent& ev);

private:
    int fd_ = -1;
};

}  // namespace clob
