// ============================================================================
// XOR Filter Implementation
// Based on "Xor Filters: Faster and Smaller Than Bloom Filters" (2019)
// ============================================================================

#include "XorFilter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

// MurmurHash3 mix functions for hashing
static inline uint64_t Mix64(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

// Rotl64
static inline uint64_t Rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

// ============================================================================
// XorFilter8 Implementation
// ============================================================================

XorFilter8::XorFilter8()
    : seed(0), block_length(0), fingerprints_count(0), fingerprints(nullptr) {
}

XorFilter8::~XorFilter8() {
    Clear();
}

void XorFilter8::Clear() {
    if (fingerprints) {
        delete[] fingerprints;
        fingerprints = nullptr;
    }
    block_length = 0;
    fingerprints_count = 0;
}

uint64_t XorFilter8::Hash(uint64_t key, uint32_t index) const {
    uint64_t h = Mix64(key + seed + index);
    return h;
}

uint64_t XorFilter8::Fingerprint(uint64_t key) const {
    uint64_t h = Mix64(key);
    return h & 0xFF;  // 8-bit fingerprint
}

uint64_t XorFilter8::HashToRange(uint64_t hash, uint64_t range) const {
    // Fast range reduction using multiplication
    // Lemire's method: https://arxiv.org/abs/1805.10941
    __uint128_t product = (__uint128_t)hash * (__uint128_t)range;
    return product >> 64;
}

bool XorFilter8::Build(const uint64_t* keys, size_t num_keys) {
    if (num_keys == 0) {
        return false;
    }

    // Clear existing filter
    Clear();

    // XOR filter uses 1.23 slots per key (overhead factor)
    const double OVERHEAD = 1.23;
    block_length = (uint64_t)(num_keys * OVERHEAD / 3.0);
    fingerprints_count = block_length * 3;

    // Allocate fingerprint array
    fingerprints = new uint8_t[fingerprints_count];
    memset(fingerprints, 0, fingerprints_count);

    // Try different seeds until construction succeeds
    const int MAX_ITERATIONS = 100;
    std::vector<uint64_t> keys_vec(keys, keys + num_keys);

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        seed = iter;  // Try different seeds

        // Build mapping
        std::vector<uint64_t> H0(block_length, 0);
        std::vector<uint64_t> H1(block_length, 0);
        std::vector<uint64_t> H2(block_length, 0);
        std::vector<uint8_t> Q0(block_length, 0);
        std::vector<uint8_t> Q1(block_length, 0);
        std::vector<uint8_t> Q2(block_length, 0);

        // Hash all keys to three positions
        for (size_t i = 0; i < num_keys; i++) {
            uint64_t k = keys_vec[i];
            uint64_t h0 = HashToRange(Hash(k, 0), block_length);
            uint64_t h1 = HashToRange(Hash(k, 1), block_length);
            uint64_t h2 = HashToRange(Hash(k, 2), block_length);

            H0[h0]++;
            H1[h1]++;
            H2[h2]++;
        }

        // Queue for keys with exactly one hash collision
        std::vector<uint64_t> alone;
        std::vector<uint64_t> hash_index;

        // Find keys that hash to unique positions
        for (size_t i = 0; i < num_keys; i++) {
            uint64_t k = keys_vec[i];
            uint64_t h0 = HashToRange(Hash(k, 0), block_length);
            uint64_t h1 = HashToRange(Hash(k, 1), block_length);
            uint64_t h2 = HashToRange(Hash(k, 2), block_length);

            bool alone0 = (H0[h0] == 1);
            bool alone1 = (H1[h1] == 1);
            bool alone2 = (H2[h2] == 1);

            if (alone0 || alone1 || alone2) {
                alone.push_back(k);
                if (alone0) hash_index.push_back(h0);
                else if (alone1) hash_index.push_back(h1 + block_length);
                else hash_index.push_back(h2 + 2 * block_length);
            }
        }

        // Peel keys in reverse order
        std::vector<uint64_t> reversed_keys;
        std::vector<uint64_t> reversed_hash;

        size_t processed = 0;
        while (!alone.empty() && processed < num_keys * 2) {
            uint64_t k = alone.back();
            uint64_t hi = hash_index.back();
            alone.pop_back();
            hash_index.pop_back();

            reversed_keys.push_back(k);
            reversed_hash.push_back(hi);

            // Decrement counts
            uint64_t h0 = HashToRange(Hash(k, 0), block_length);
            uint64_t h1 = HashToRange(Hash(k, 1), block_length);
            uint64_t h2 = HashToRange(Hash(k, 2), block_length);

            H0[h0]--;
            H1[h1]--;
            H2[h2]--;

            // Check if neighbors become alone
            if (H0[h0] == 1) {
                // Find key that hashes to h0
                for (size_t j = 0; j < num_keys; j++) {
                    uint64_t k2 = keys_vec[j];
                    if (HashToRange(Hash(k2, 0), block_length) == h0) {
                        alone.push_back(k2);
                        hash_index.push_back(h0);
                        break;
                    }
                }
            }
            // Similar for h1, h2...

            processed++;
        }

        // Check if all keys were processed
        if (reversed_keys.size() != num_keys) {
            // Construction failed, try next seed
            continue;
        }

        // Assign fingerprints
        memset(fingerprints, 0, fingerprints_count);

        for (size_t i = 0; i < reversed_keys.size(); i++) {
            uint64_t k = reversed_keys[reversed_keys.size() - 1 - i];
            uint64_t hi = reversed_hash[reversed_keys.size() - 1 - i];

            uint64_t h0 = HashToRange(Hash(k, 0), block_length);
            uint64_t h1 = HashToRange(Hash(k, 1), block_length) + block_length;
            uint64_t h2 = HashToRange(Hash(k, 2), block_length) + 2 * block_length;

            uint8_t fp = Fingerprint(k);
            uint8_t val = fp ^ fingerprints[h0] ^ fingerprints[h1] ^ fingerprints[h2];
            fingerprints[hi] = val;
        }

        // Construction succeeded!
        return true;
    }

    // Failed after MAX_ITERATIONS
    Clear();
    return false;
}

bool XorFilter8::Build(const std::vector<uint64_t>& keys) {
    if (keys.empty()) return false;
    return Build(keys.data(), keys.size());
}

bool XorFilter8::Contains(uint64_t key) const {
    if (!fingerprints) return false;

    uint64_t h0 = HashToRange(Hash(key, 0), block_length);
    uint64_t h1 = HashToRange(Hash(key, 1), block_length) + block_length;
    uint64_t h2 = HashToRange(Hash(key, 2), block_length) + 2 * block_length;

    uint8_t fp = Fingerprint(key);
    uint8_t val = fingerprints[h0] ^ fingerprints[h1] ^ fingerprints[h2];

    return (val == fp);
}

bool XorFilter8::Save(const char* filename) const {
    if (!fingerprints) return false;

    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    // Write header
    uint32_t magic = XOR_FILTER_MAGIC;
    uint32_t version = XOR_FILTER_VERSION;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&seed, sizeof(seed), 1, f);
    fwrite(&block_length, sizeof(block_length), 1, f);
    fwrite(&fingerprints_count, sizeof(fingerprints_count), 1, f);

    // Write fingerprints
    fwrite(fingerprints, sizeof(uint8_t), fingerprints_count, f);

    fclose(f);
    return true;
}

bool XorFilter8::Load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    // Read header
    uint32_t magic, version;
    fread(&magic, sizeof(magic), 1, f);
    fread(&version, sizeof(version), 1, f);

    if (magic != XOR_FILTER_MAGIC || version != XOR_FILTER_VERSION) {
        fclose(f);
        return false;
    }

    fread(&seed, sizeof(seed), 1, f);
    fread(&block_length, sizeof(block_length), 1, f);
    fread(&fingerprints_count, sizeof(fingerprints_count), 1, f);

    // Allocate and read fingerprints
    Clear();
    fingerprints = new uint8_t[fingerprints_count];
    fread(fingerprints, sizeof(uint8_t), fingerprints_count, f);

    fclose(f);
    return true;
}

size_t XorFilter8::GetSizeBytes() const {
    return fingerprints_count * sizeof(uint8_t);
}

double XorFilter8::GetBitsPerKey() const {
    if (fingerprints_count == 0) return 0.0;
    uint64_t key_count = fingerprints_count * 3 / 4;  // Approximate
    return (double)(fingerprints_count * 8) / key_count;
}

// ============================================================================
// DPXorFilter Implementation
// ============================================================================

DPXorFilter::DPXorFilter() {
}

DPXorFilter::~DPXorFilter() {
}

uint64_t DPXorFilter::DPToHash(const uint8_t* dp_x) const {
    // Convert 96-bit DP to 64-bit hash
    // Use first 8 bytes and XOR with last 4 bytes
    uint64_t h = 0;
    memcpy(&h, dp_x, 8);

    uint32_t tail;
    memcpy(&tail, dp_x + 8, 4);

    h ^= (uint64_t)tail;
    h ^= (uint64_t)tail << 32;

    return h;
}

bool DPXorFilter::BuildFromDPs(const uint8_t* dp_array, size_t num_dps) {
    if (num_dps == 0) return false;

    // Convert DPs to 64-bit hashes
    std::vector<uint64_t> hashes(num_dps);
    for (size_t i = 0; i < num_dps; i++) {
        hashes[i] = DPToHash(dp_array + i * 12);
    }

    return filter.Build(hashes);
}

bool DPXorFilter::BuildFromDPs(const std::vector<uint8_t>& dp_data, size_t dp_size) {
    if (dp_data.size() % dp_size != 0) {
        printf("ERROR: DP data size not multiple of %zu\n", dp_size);
        return false;
    }

    size_t num_dps = dp_data.size() / dp_size;
    return BuildFromDPs(dp_data.data(), num_dps);
}

bool DPXorFilter::ContainsDP(const uint8_t* dp_x) const {
    uint64_t h = DPToHash(dp_x);
    return filter.Contains(h);
}

bool DPXorFilter::Save(const char* filename) const {
    return filter.Save(filename);
}

bool DPXorFilter::Load(const char* filename) {
    return filter.Load(filename);
}
