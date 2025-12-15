#pragma once
//
// ram_based_hashtable.h
// Simple RAM DP table for viral wild kangaroos
//

#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cstring>
#include "defs.h"

// We assume GPU_DP_SIZE is defined in defs.h (e.g. 48 bytes)
struct DPRecord {
    uint8_t data[GPU_DP_SIZE];
};

// Helper to pull pieces out of the DP record
inline void dp_extract_x12_and_type(const DPRecord& rec,
                                    uint64_t& x_lo,
                                    uint32_t& x_hi,
                                    uint8_t& type)
{
    // X: first 12 bytes (little-endian)
    std::memcpy(&x_lo, rec.data + 0,  8);
    std::memcpy(&x_hi, rec.data + 8,  4);

    // Type at byte 40 (matches CPU DP layout in your RCCpuKang code)
    type = rec.data[40];
}

class RAMBasedHashTable {
public:
    // Returns true if a *collision* is detected (same X, different type)
    bool checkOrAdd(const DPRecord& rec) {
        uint64_t x_lo;
        uint32_t x_hi;
        uint8_t  type;
        dp_extract_x12_and_type(rec, x_lo, x_hi, type);

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = table_.find(x_lo);
        if (it == table_.end()) {
            // First time we see this low 64 bits
            Entry e;
            e.x_lo   = x_lo;
            e.x_hi   = x_hi;
            e.type   = type;
            e.record = rec;
            table_.emplace(x_lo, e);
            return false;
        }

        Entry& existing = it->second;

        if (existing.x_hi == x_hi) {
            // Same 96-bit X. If type differs, we treat it as a collision.
            if (existing.type != type) {
                return true;
            }
            // Same type: ignore (duplicate DP from same side).
            return false;
        }

        // Rare low-64 collision but different high-32 bits:
        // overwrite but don’t call it a real solution.
        existing.x_hi   = x_hi;
        existing.type   = type;
        existing.record = rec;
        return false;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return table_.size();
    }

private:
    struct Entry {
        uint64_t x_lo;
        uint32_t x_hi;
        uint8_t  type;
        DPRecord record;
    };

    std::unordered_map<uint64_t, Entry> table_;
    mutable std::mutex mutex_;
};
