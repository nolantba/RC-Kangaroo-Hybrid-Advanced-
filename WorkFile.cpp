// ============================================================================
// RCKangaroo Work File Implementation - Save/Resume/Merge System
// Compatible with distributed solving and long-running puzzles
// ============================================================================

#include "WorkFile.h"
#include "XorFilter.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <algorithm>

// CRC32 lookup table for checksum calculation
static const uint32_t crc32_table[256] = {
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};

uint32_t RCWorkFile::CalculateChecksum(const void* data, size_t len)
{
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++)
    {
        crc = crc32_table[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
    }
    return ~crc;
}

bool RCWorkFile::ValidateHeader()
{
    if (header.magic != WORK_FILE_MAGIC)
    {
        printf("ERROR: Invalid magic number (0x%08X)\n", header.magic);
        return false;
    }
    if (header.version != WORK_FILE_VERSION)
    {
        printf("ERROR: Unsupported version %u (expected %u)\n",
               header.version, WORK_FILE_VERSION);
        return false;
    }

    // Calculate checksum (excluding the checksum field itself)
    uint32_t saved_checksum = header.header_checksum;
    header.header_checksum = 0;
    uint32_t calculated = CalculateChecksum(&header, sizeof(header));
    header.header_checksum = saved_checksum;

    if (calculated != saved_checksum)
    {
        printf("WARNING: Header checksum mismatch (calculated: 0x%08X, saved: 0x%08X)\n",
               calculated, saved_checksum);
        // Don't fail on checksum mismatch, just warn
    }

    return true;
}

// ============================================================================
// Constructor/Destructor
// ============================================================================

RCWorkFile::RCWorkFile()
    : is_loaded(false)
{
    memset(&header, 0, sizeof(header));
}

RCWorkFile::RCWorkFile(const std::string& fname)
    : filename(fname), is_loaded(false)
{
    memset(&header, 0, sizeof(header));
}

RCWorkFile::~RCWorkFile()
{
    // Nothing to clean up (using std::vector)
}

// ============================================================================
// Save operations
// ============================================================================

bool RCWorkFile::Create(uint32_t range_bits, uint32_t dp_bits,
                        const uint8_t* pubkey_x, const uint8_t* pubkey_y,
                        const uint64_t* range_start, const uint64_t* range_stop)
{
    memset(&header, 0, sizeof(header));

    header.magic = WORK_FILE_MAGIC;
    header.version = WORK_FILE_VERSION;
    header.range_bits = range_bits;
    header.dp_bits = dp_bits;
    header.dp_mask_bits = dp_bits;

    if (pubkey_x) memcpy(header.pubkey_x, pubkey_x, 32);
    if (pubkey_y) memcpy(header.pubkey_y, pubkey_y, 32);
    if (range_start) memcpy(header.range_start, range_start, 32);
    if (range_stop) memcpy(header.range_stop, range_stop, 32);

    header.start_time = time(NULL);
    header.last_save_time = header.start_time;
    header.rng_seed = header.start_time;  // Default RNG seed

    dp_records.clear();
    is_loaded = true;

    return true;
}

bool RCWorkFile::Save()
{
    if (filename.empty())
    {
        printf("ERROR: No filename specified for save\n");
        return false;
    }
    return SaveAs(filename);
}

bool RCWorkFile::SaveAs(const std::string& new_filename)
{
    // Update header
    header.dp_count = dp_records.size();
    header.last_save_time = time(NULL);

    // Calculate header checksum
    header.header_checksum = 0;
    header.header_checksum = CalculateChecksum(&header, sizeof(header));

    // Open file
    FILE* f = fopen(new_filename.c_str(), "wb");
    if (!f)
    {
        printf("ERROR: Failed to open file for writing: %s\n", new_filename.c_str());
        return false;
    }

    // Write header
    if (fwrite(&header, sizeof(header), 1, f) != 1)
    {
        printf("ERROR: Failed to write header\n");
        fclose(f);
        return false;
    }

    // Write DP records
    if (!dp_records.empty())
    {
        if (fwrite(dp_records.data(), sizeof(DPRecord), dp_records.size(), f) != dp_records.size())
        {
            printf("ERROR: Failed to write DP records\n");
            fclose(f);
            return false;
        }
    }

    fclose(f);

    printf("Work file saved: %s (%lu ops, %lu DPs)\n",
           new_filename.c_str(), (unsigned long)header.total_ops, (unsigned long)header.dp_count);

    filename = new_filename;
    return true;
}

bool RCWorkFile::AddDP(const uint8_t* dp_x, const uint8_t* distance, uint8_t type)
{
    if (!is_loaded)
    {
        printf("ERROR: Cannot add DP to unloaded work file\n");
        return false;
    }

    DPRecord dp;
    memcpy(dp.dp_x, dp_x, 12);
    memcpy(dp.distance, distance, 22);
    dp.type = type;
    dp.reserved = 0;

    dp_records.push_back(dp);
    return true;
}

void RCWorkFile::UpdateProgress(uint64_t ops, uint64_t dps, uint64_t dead, uint64_t elapsed)
{
    header.total_ops = ops;
    header.dp_count = dps;
    header.dead_kangaroos = dead;
    header.elapsed_seconds = elapsed;
}

// ============================================================================
// Load operations
// ============================================================================

bool RCWorkFile::Load()
{
    if (filename.empty())
    {
        printf("ERROR: No filename specified for load\n");
        return false;
    }
    return Load(filename);
}

bool RCWorkFile::Load(const std::string& fname)
{
    filename = fname;

    FILE* f = fopen(filename.c_str(), "rb");
    if (!f)
    {
        printf("ERROR: Failed to open file: %s\n", filename.c_str());
        return false;
    }

    // Read header
    if (fread(&header, sizeof(header), 1, f) != 1)
    {
        printf("ERROR: Failed to read header\n");
        fclose(f);
        return false;
    }

    // Validate header
    if (!ValidateHeader())
    {
        fclose(f);
        return false;
    }

    // Read DP records
    dp_records.clear();
    if (header.dp_count > 0)
    {
        dp_records.resize(header.dp_count);
        size_t read_count = fread(dp_records.data(), sizeof(DPRecord), header.dp_count, f);
        if (read_count != header.dp_count)
        {
            printf("ERROR: Failed to read DP records (%zu/%lu)\n",
                   read_count, (unsigned long)header.dp_count);
            fclose(f);
            return false;
        }
    }

    fclose(f);

    printf("Work file loaded: %s\n", filename.c_str());
    printf("  Range: %u bits, DP: %u bits\n", header.range_bits, header.dp_bits);
    printf("  Progress: %lu ops, %lu DPs\n", (unsigned long)header.total_ops, (unsigned long)header.dp_count);
    printf("  Elapsed: %lu seconds (%.1f hours)\n",
           (unsigned long)header.elapsed_seconds, header.elapsed_seconds / 3600.0);

    is_loaded = true;
    return true;
}

// ============================================================================
// DP access
// ============================================================================

bool RCWorkFile::HasDP(const uint8_t* dp_x) const
{
    for (const auto& dp : dp_records)
    {
        if (memcmp(dp.dp_x, dp_x, 12) == 0)
            return true;
    }
    return false;
}

// ============================================================================
// Merge operations
// ============================================================================

bool RCWorkFile::Merge(const std::vector<std::string>& input_files,
                       const std::string& output_file)
{
    if (input_files.empty())
    {
        printf("ERROR: No input files specified\n");
        return false;
    }

    // Load first file as base
    RCWorkFile merged;
    if (!merged.Load(input_files[0]))
    {
        printf("ERROR: Failed to load base file: %s\n", input_files[0].c_str());
        return false;
    }

    printf("Base file: %s (%lu ops, %lu DPs)\n",
           input_files[0].c_str(), (unsigned long)merged.header.total_ops, (unsigned long)merged.header.dp_count);

    // Build XOR filter for deduplication
    XorFilter8 dp_filter;
    std::vector<uint64_t> dp_hashes;
    for (const auto& dp : merged.dp_records)
    {
        uint64_t hash = 0;
        memcpy(&hash, dp.dp_x, 8);  // Use first 8 bytes as hash
        dp_hashes.push_back(hash);
    }

    if (!dp_hashes.empty())
    {
        dp_filter.Build(dp_hashes.data(), dp_hashes.size());
        printf("XOR filter built: %.1f MB for %lu DPs\n",
               dp_filter.GetSizeBytes() / 1024.0 / 1024.0, (unsigned long)merged.header.dp_count);
    }

    // Merge remaining files
    uint64_t total_ops = merged.header.total_ops;
    uint64_t duplicates_found = 0;

    for (size_t i = 1; i < input_files.size(); i++)
    {
        RCWorkFile wf;
        if (!wf.Load(input_files[i]))
        {
            printf("WARNING: Skipping file: %s\n", input_files[i].c_str());
            continue;
        }

        // Check compatibility
        if (wf.header.range_bits != merged.header.range_bits ||
            wf.header.dp_bits != merged.header.dp_bits)
        {
            printf("WARNING: Incompatible file (range=%u, dp=%u): %s\n",
                   wf.header.range_bits, wf.header.dp_bits, input_files[i].c_str());
            continue;
        }

        total_ops += wf.header.total_ops;

        // Add DPs (skip duplicates)
        size_t added = 0;
        for (const auto& dp : wf.dp_records)
        {
            uint64_t hash = 0;
            memcpy(&hash, dp.dp_x, 8);

            // Check XOR filter first (fast)
            if (!dp_hashes.empty() && dp_filter.Contains(hash))
            {
                // Might be duplicate, check full comparison
                if (merged.HasDP(dp.dp_x))
                {
                    duplicates_found++;
                    continue;
                }
            }

            merged.dp_records.push_back(dp);
            dp_hashes.push_back(hash);
            added++;
        }

        printf("  Merged: %s (+%zu DPs)\n", input_files[i].c_str(), added);
        if (duplicates_found > 0)
        {
            printf("  Added %zu DPs, skipped %lu duplicates\n", added, (unsigned long)duplicates_found);
        }
        printf("  Total: %lu ops, %zu DPs\n", (unsigned long)total_ops, merged.dp_records.size());
    }

    // Update merged header
    merged.header.total_ops = total_ops;
    merged.header.dp_count = merged.dp_records.size();
    merged.header.dead_kangaroos = 0;  // Can't merge this reliably

    // Save merged file
    if (!merged.SaveAs(output_file))
    {
        return false;
    }

    printf("\nMerge Summary:\n");
    printf("  Input files: %zu\n", input_files.size());
    printf("  Total operations: %lu\n", (unsigned long)total_ops);
    printf("  Total DPs: %lu\n", (unsigned long)merged.header.dp_count);
    printf("  Duplicates removed: %lu\n", (unsigned long)duplicates_found);

    return true;
}

// ============================================================================
// Validation
// ============================================================================

bool RCWorkFile::VerifyIntegrity()
{
    if (!is_loaded)
    {
        printf("ERROR: Cannot verify unloaded file\n");
        return false;
    }

    if (dp_records.size() != header.dp_count)
    {
        printf("ERROR: DP count mismatch (%lu in header, %zu in records)\n",
               (unsigned long)header.dp_count, dp_records.size());
        return false;
    }

    // Check for obvious corruption in DPs
    for (size_t i = 0; i < dp_records.size(); i++)
    {
        const auto& dp = dp_records[i];
        if (dp.type != DP_TYPE_TAME && dp.type != DP_TYPE_WILD)
        {
            printf("WARNING: Invalid DP type at index %zu: %u\n", i, dp.type);
        }
    }

    printf("Integrity check passed\n");
    return true;
}

bool RCWorkFile::IsCompatible(uint32_t range_bits, uint32_t dp_bits,
                              const uint8_t* pubkey_x, const uint8_t* pubkey_y)
{
    if (!is_loaded)
        return false;

    if (header.range_bits != range_bits || header.dp_bits != dp_bits)
        return false;

    if (pubkey_x && memcmp(header.pubkey_x, pubkey_x, 32) != 0)
        return false;

    if (pubkey_y && memcmp(header.pubkey_y, pubkey_y, 32) != 0)
        return false;

    return true;
}

// ============================================================================
// Info/debug
// ============================================================================

void RCWorkFile::PrintInfo() const
{
    printf("=== Work File Info ===\n");
    printf("File: %s\n", filename.c_str());
    printf("Range: %u bits, DP: %u bits\n", header.range_bits, header.dp_bits);
    printf("Operations: %lu (2^%.2f)\n", (unsigned long)header.total_ops,
           log2((double)header.total_ops));
    printf("DPs found: %lu\n", (unsigned long)header.dp_count);
    printf("Dead kangaroos: %lu\n", (unsigned long)header.dead_kangaroos);

    if (header.elapsed_seconds > 0)
    {
        uint64_t hours = header.elapsed_seconds / 3600;
        uint64_t minutes = (header.elapsed_seconds % 3600) / 60;
        uint64_t seconds = header.elapsed_seconds % 60;
        printf("Elapsed: %02lu:%02lu:%02lu\n", (unsigned long)hours, (unsigned long)minutes, (unsigned long)seconds);
    }
}

std::string RCWorkFile::GetInfoString() const
{
    char buf[256];
    snprintf(buf, sizeof(buf),
             "Range:%u DP:%u Ops:%lu DPs:%lu Elapsed:%lus",
             header.range_bits, header.dp_bits, (unsigned long)header.total_ops,
             (unsigned long)header.dp_count, (unsigned long)header.elapsed_seconds);
    return std::string(buf);
}

// ============================================================================
// AutoSaveManager Implementation
// ============================================================================

AutoSaveManager::AutoSaveManager(RCWorkFile* wf, uint64_t interval_sec)
    : work_file(wf), save_interval_seconds(interval_sec),
      last_save_time(0), enabled(false)
{
}

bool AutoSaveManager::CheckAndSave(uint64_t current_ops, uint64_t current_dps,
                                   uint64_t dead_kangaroos, uint64_t elapsed_sec)
{
    if (!enabled || !work_file)
        return false;

    uint64_t now = time(NULL);
    if (last_save_time == 0 || (now - last_save_time) >= save_interval_seconds)
    {
        return ForceSave(current_ops, current_dps, dead_kangaroos, elapsed_sec);
    }

    return false;
}

bool AutoSaveManager::ForceSave(uint64_t current_ops, uint64_t current_dps,
                                uint64_t dead_kangaroos, uint64_t elapsed_sec)
{
    if (!work_file)
        return false;

    work_file->UpdateProgress(current_ops, current_dps, dead_kangaroos, elapsed_sec);
    bool success = work_file->Save();

    if (success)
    {
        last_save_time = time(NULL);
    }

    return success;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string GenerateWorkFilename(uint32_t range_bits, const uint8_t* pubkey_x)
{
    char buf[128];
    if (pubkey_x)
    {
        snprintf(buf, sizeof(buf), "puzzle%u_%02x%02x.work",
                 range_bits, pubkey_x[0], pubkey_x[1]);
    }
    else
    {
        snprintf(buf, sizeof(buf), "puzzle%u.work", range_bits);
    }
    return std::string(buf);
}

bool WorkFileExists(const std::string& filename)
{
    FILE* f = fopen(filename.c_str(), "rb");
    if (f)
    {
        fclose(f);
        return true;
    }
    return false;
}

bool GetWorkFileInfo(const std::string& filename, WorkFileHeader* header_out)
{
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f)
        return false;

    WorkFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1)
    {
        fclose(f);
        return false;
    }

    fclose(f);

    if (hdr.magic != WORK_FILE_MAGIC)
        return false;

    if (header_out)
        *header_out = hdr;

    return true;
}
