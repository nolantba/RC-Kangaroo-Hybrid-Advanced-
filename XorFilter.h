// ============================================================================
// XOR Filter - Fast probabilistic duplicate detection for DPs
// Based on "Xor Filters: Faster and Smaller Than Bloom Filters" (2019)
// Paper: https://arxiv.org/abs/1912.08258
// ============================================================================
// Perfect for DP deduplication:
// - 1.23 bits per DP (vs 10+ bits for Bloom filter)
// - No false positives on stored keys
// - Fast O(1) lookups
// - Minimal memory for puzzle 135 (billions of DPs)
// ============================================================================

#ifndef RCKANGAROO_XORFILTER_H
#define RCKANGAROO_XORFILTER_H

#include <stdint.h>
#include <vector>
#include <cstring>

// XOR filter configuration
#define XOR_FILTER_MAGIC 0x584F5246  // "XORF"
#define XOR_FILTER_VERSION 1

// ============================================================================
// XorFilter8 - 8-bit fingerprints (optimal for most cases)
// ============================================================================

class XorFilter8 {
private:
    uint64_t seed;
    uint64_t block_length;
    uint64_t fingerprints_count;
    uint8_t* fingerprints;  // Array of 8-bit fingerprints

    // Hashing
    uint64_t Hash(uint64_t key, uint32_t index) const;
    uint64_t Fingerprint(uint64_t key) const;
    uint64_t HashToRange(uint64_t hash, uint64_t range) const;

public:
    XorFilter8();
    ~XorFilter8();

    // Build filter from keys
    bool Build(const uint64_t* keys, size_t num_keys);
    bool Build(const std::vector<uint64_t>& keys);

    // Query filter
    bool Contains(uint64_t key) const;

    // Serialization
    bool Save(const char* filename) const;
    bool Load(const char* filename);
    bool SaveToBuffer(uint8_t* buffer, size_t* size_out) const;
    bool LoadFromBuffer(const uint8_t* buffer, size_t size);

    // Info
    size_t GetSizeBytes() const;
    double GetBitsPerKey() const;
    uint64_t GetKeyCount() const { return fingerprints_count * 3 / 4; }

    // Clear
    void Clear();
    bool IsBuilt() const { return fingerprints != nullptr; }
};

// ============================================================================
// DP-specific wrapper - handles 96-bit DP X coordinates
// ============================================================================

class DPXorFilter {
private:
    XorFilter8 filter;

    // Convert 96-bit DP to 64-bit hash for filter
    uint64_t DPToHash(const uint8_t* dp_x) const;

public:
    DPXorFilter();
    ~DPXorFilter();

    // Build from DPs (array of 12-byte DP X coordinates)
    bool BuildFromDPs(const uint8_t* dp_array, size_t num_dps);
    bool BuildFromDPs(const std::vector<uint8_t>& dp_data, size_t dp_size = 12);

    // Check if DP exists
    bool ContainsDP(const uint8_t* dp_x) const;

    // Serialization
    bool Save(const char* filename) const;
    bool Load(const char* filename);

    // Info
    size_t GetSizeBytes() const { return filter.GetSizeBytes(); }
    double GetBitsPerKey() const { return filter.GetBitsPerKey(); }
    uint64_t GetDPCount() const { return filter.GetKeyCount(); }

    void Clear() { filter.Clear(); }
    bool IsBuilt() const { return filter.IsBuilt(); }
};

// ============================================================================
// Usage Example:
// ============================================================================
//
// // Building filter from work file DPs
// DPXorFilter dp_filter;
// std::vector<uint8_t> dp_data;
// for (auto& dp : work_file.GetDPs()) {
//     dp_data.insert(dp_data.end(), dp.dp_x, dp.dp_x + 12);
// }
// dp_filter.BuildFromDPs(dp_data);
//
// // Fast duplicate detection (O(1))
// if (dp_filter.ContainsDP(new_dp)) {
//     printf("Duplicate DP found!\n");
// }
//
// // Memory usage for puzzle 135 (1 billion DPs):
// // Traditional hash table: 1B * 40 bytes = 40 GB
// // XOR filter: 1B * 1.23 bits = 154 MB (260x smaller!)
//
// ============================================================================

#endif // RCKANGAROO_XORFILTER_H
