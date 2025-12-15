// ============================================================================
// RCKangaroo Work File Format - Save/Resume/Merge System
// Compatible with distributed solving and long-running puzzles
// ============================================================================

#ifndef RCKANGAROO_WORKFILE_H
#define RCKANGAROO_WORKFILE_H

#include <stdint.h>
#include <string>
#include <vector>

// Work file format version
#define WORK_FILE_VERSION 1
#define WORK_FILE_MAGIC 0x524B574B  // "RKWK" (RCKangaroo Work)

// DP record types
#define DP_TYPE_TAME   0
#define DP_TYPE_WILD   1

#pragma pack(push, 1)

// Work file header (256 bytes total)
struct WorkFileHeader {
    uint32_t magic;              // Magic number: 0x524B574B
    uint32_t version;            // File format version

    // Puzzle parameters
    uint32_t range_bits;         // Puzzle range (e.g., 75, 90, 135)
    uint32_t dp_bits;            // Distinguished point bits
    uint8_t  pubkey_x[32];       // Public key X coordinate (compressed)
    uint8_t  pubkey_y[32];       // Public key Y coordinate (compressed)
    uint64_t range_start[4];     // Start offset (256-bit)
    uint64_t range_stop[4];      // Stop offset (256-bit)

    // Progress tracking
    uint64_t total_ops;          // Total operations performed
    uint64_t dp_count;           // Number of DPs found
    uint64_t dead_kangaroos;     // Dead kangaroo count
    uint64_t start_time;         // Unix timestamp when started
    uint64_t last_save_time;     // Unix timestamp of last save
    uint64_t elapsed_seconds;    // Total elapsed time

    // Flags and metadata
    uint32_t dp_mask_bits;       // DP mask bits for compatibility
    uint32_t flags;              // Reserved flags
    uint64_t rng_seed;           // RNG seed for kangaroo position initialization (auto-restart fix)
    uint8_t  reserved[32];       // Reserved for future use

    // Checksum
    uint32_t header_checksum;    // CRC32 of header (excluding this field)
};

// Distinguished Point record (variable length, but typically 35-45 bytes)
struct DPRecord {
    uint8_t  dp_x[12];           // Distinguished point X (96 bits, sufficient for DP check)
    uint8_t  distance[22];       // Distance from start (176 bits, enough for 2^135+)
    uint8_t  type;               // DP_TYPE_TAME or DP_TYPE_WILD
    uint8_t  reserved;           // Padding/future use
};

#pragma pack(pop)

// ============================================================================
// WorkFile Class - Handles save/load/merge operations
// ============================================================================

class RCWorkFile {
private:
    std::string filename;
    WorkFileHeader header;
    std::vector<DPRecord> dp_records;
    bool is_loaded;

    uint32_t CalculateChecksum(const void* data, size_t len);
    bool ValidateHeader();

public:
    RCWorkFile();
    RCWorkFile(const std::string& filename);
    ~RCWorkFile();

    // Save operations
    bool Create(uint32_t range_bits, uint32_t dp_bits,
                const uint8_t* pubkey_x, const uint8_t* pubkey_y,
                const uint64_t* range_start, const uint64_t* range_stop);
    bool Save();
    bool SaveAs(const std::string& new_filename);
    bool AddDP(const uint8_t* dp_x, const uint8_t* distance, uint8_t type);
    void UpdateProgress(uint64_t ops, uint64_t dps, uint64_t dead, uint64_t elapsed);

    // Load operations
    bool Load();
    bool Load(const std::string& filename);
    bool IsLoaded() const { return is_loaded; }

    // Resume info
    uint64_t GetTotalOps() const { return header.total_ops; }
    uint64_t GetDPCount() const { return header.dp_count; }
    uint64_t GetElapsedSeconds() const { return header.elapsed_seconds; }
    uint32_t GetRangeBits() const { return header.range_bits; }
    uint32_t GetDPBits() const { return header.dp_bits; }
    uint64_t GetRNGSeed() const { return header.rng_seed; }
    void SetRNGSeed(uint64_t seed) { header.rng_seed = seed; }

    // DP access
    const std::vector<DPRecord>& GetDPs() const { return dp_records; }
    bool HasDP(const uint8_t* dp_x) const;

    // Merge operations (combine multiple work files)
    static bool Merge(const std::vector<std::string>& input_files,
                     const std::string& output_file);

    // Validation
    bool VerifyIntegrity();
    bool IsCompatible(uint32_t range_bits, uint32_t dp_bits,
                     const uint8_t* pubkey_x, const uint8_t* pubkey_y);

    // Info/debug
    void PrintInfo() const;
    std::string GetInfoString() const;
};

// ============================================================================
// Auto-save manager for periodic checkpoints
// ============================================================================

class AutoSaveManager {
private:
    RCWorkFile* work_file;
    uint64_t save_interval_seconds;  // How often to save (default: 60s)
    uint64_t last_save_time;
    bool enabled;

public:
    AutoSaveManager(RCWorkFile* wf, uint64_t interval_sec = 60);

    void Enable() { enabled = true; }
    void Disable() { enabled = false; }
    bool IsEnabled() const { return enabled; }

    void SetInterval(uint64_t seconds) { save_interval_seconds = seconds; }
    uint64_t GetInterval() const { return save_interval_seconds; }

    // Call this periodically from main loop
    bool CheckAndSave(uint64_t current_ops, uint64_t current_dps,
                     uint64_t dead_kangaroos, uint64_t elapsed_sec);

    // Force immediate save
    bool ForceSave(uint64_t current_ops, uint64_t current_dps,
                  uint64_t dead_kangaroos, uint64_t elapsed_sec);
};

// ============================================================================
// Utility functions
// ============================================================================

// Create work filename from puzzle parameters
std::string GenerateWorkFilename(uint32_t range_bits, const uint8_t* pubkey_x);

// Check if work file exists and is valid
bool WorkFileExists(const std::string& filename);

// Get work file info without loading DPs (fast)
bool GetWorkFileInfo(const std::string& filename, WorkFileHeader* header_out);

#endif // RCKANGAROO_WORKFILE_H
