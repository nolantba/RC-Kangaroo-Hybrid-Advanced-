#ifndef HERD_CONFIG_OPTIMIZED_H
#define HERD_CONFIG_OPTIMIZED_H

#include "HerdConfig.h"

// ============================================================================
// RTX 3060 Optimized Herd Configuration
// ============================================================================
// Tuned specifically for RTX 3060 architecture:
// - 3584 CUDA cores (28 SMs × 128 cores/SM)
// - 12GB GDDR6 memory
// - 360 GB/s memory bandwidth
// - 48KB shared memory per SM
// - 65536 registers per SM
// ============================================================================

struct HerdConfig_RTX3060 : public HerdConfig {

    HerdConfig_RTX3060() {
        // ====================================================================
        // Optimal settings for RTX 3060
        // ====================================================================

        // Use 14 herds for even distribution across 28 SMs (2 blocks per SM)
        herds_per_gpu = 14;

        // 896 kangaroos per herd = optimal warp utilization
        // (896 / 8 kangs_per_thread = 112 threads)
        // (112 threads fits in 256 thread blocks with room for other blocks)
        kangaroos_per_herd = 896;

        // Jump table: 256 entries fits perfectly in 12KB shared memory
        // (256 jumps × 12 u64s × 8 bytes = 24KB < 48KB limit)
        jump_table_size = 256;
        adaptive_jumps = true;
        jump_update_interval = 5000000;  // 5M ops (more frequent for RTX 3060)

        // DP storage tuned for 12GB VRAM
        herd_dp_buffer_size = 2048;    // Per-herd buffer (larger for RTX 3060)
        gpu_dp_buffer_size = 32768;     // Per-GPU shared (more VRAM available)
        dp_bits = 14;

        // Spatial distribution
        herd_separation_bits = 5;       // 2^5 = 32x separation
        deterministic_starts = true;

        // Performance monitoring
        enable_herd_stats = true;
        stats_update_interval = 500000;  // More frequent stats
        enable_rebalancing = true;
        rebalance_threshold = 0.85f;

        // Kernel launch: optimal for 28 SMs
        threads_per_block = 256;
        iterations_per_launch = 10000;
        use_async_streams = true;
    }

    // ========================================================================
    // RTX 3060 Memory Hierarchy
    // ========================================================================

    static constexpr int COMPUTE_CAPABILITY_MAJOR = 8;  // Ampere
    static constexpr int COMPUTE_CAPABILITY_MINOR = 6;
    static constexpr int NUM_SMS = 28;
    static constexpr int THREADS_PER_SM = 1536;
    static constexpr int SHARED_MEM_PER_SM = 49152;     // 48KB
    static constexpr int REGISTERS_PER_SM = 65536;
    static constexpr int MAX_THREADS_PER_BLOCK = 1024;
    static constexpr int WARP_SIZE = 32;

    // ========================================================================
    // Calculated properties for RTX 3060
    // ========================================================================

    int getOptimalBlocksPerSM() const {
        // With 256 threads/block, we can fit 6 blocks per SM (1536/256)
        // But use 4 for better register allocation
        return 4;
    }

    int getTotalActiveThreads() const {
        return NUM_SMS * getOptimalBlocksPerSM() * threads_per_block;
    }

    double getTheoreticalMaxGKs() const {
        // RTX 3060: 3584 cores @ 1.78 GHz boost = 6.38 TFLOPS
        // With optimal kernel utilization: ~8-9 GK/s theoretical max
        return 8.5;  // GK/s
    }
};

// ============================================================================
// Preset for different puzzle sizes on RTX 3060
// ============================================================================

inline HerdConfig_RTX3060 GetOptimalConfigRTX3060(int puzzle_bits) {
    HerdConfig_RTX3060 cfg;

    if (puzzle_bits < 100) {
        // Small puzzles: fewer herds, more kangaroos per herd
        cfg.herds_per_gpu = 7;   // 1 per 4 SMs
        cfg.kangaroos_per_herd = 1024;
        cfg.dp_bits = 12;
    }
    else if (puzzle_bits < 120) {
        // Medium puzzles: balanced (default)
        cfg.herds_per_gpu = 14;
        cfg.kangaroos_per_herd = 896;
        cfg.dp_bits = 14;
    }
    else if (puzzle_bits < 140) {
        // Large puzzles: more herds
        cfg.herds_per_gpu = 28;  // 1 per SM
        cfg.kangaroos_per_herd = 512;
        cfg.herd_separation_bits = 6;
        cfg.dp_bits = 16;
    }
    else {
        // Very large puzzles: maximum herds
        cfg.herds_per_gpu = 28;
        cfg.kangaroos_per_herd = 768;
        cfg.herd_separation_bits = 7;
        cfg.dp_bits = 18;
    }

    return cfg;
}

#endif // HERD_CONFIG_OPTIMIZED_H
