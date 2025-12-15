#ifndef HERD_CONFIG_H
#define HERD_CONFIG_H

#include <cstdint>

// ============================================================================
// SOTA++ Herd Configuration
// ============================================================================
// This file configures the multi-herd Kangaroo algorithm for GPU-only solving
// Each GPU runs multiple independent "herds" with different jump patterns
// ============================================================================

struct HerdConfig {
    // ========================================================================
    // Core Herd Parameters
    // ========================================================================
    
    // Number of herds per GPU (recommended: 8-16)
    int herds_per_gpu = 8;
    
    // Kangaroos per herd (must be multiple of 256 for warp alignment)
    int kangaroos_per_herd = 256;
    
    // ========================================================================
    // Jump Table Configuration
    // ========================================================================
    
    int jump_table_size = 256;
    bool adaptive_jumps = true;
    uint64_t jump_update_interval = 10000000;  // 10M ops
    
    // ========================================================================
    // Distinguished Point Storage Hierarchy
    // ========================================================================
    
    int herd_dp_buffer_size = 1024;   // Per-herd buffer
    int gpu_dp_buffer_size = 16384;    // Per-GPU shared
    int dp_bits = 14;
    
    // ========================================================================
    // Herd Spatial Distribution
    // ========================================================================
    
    int herd_separation_bits = 4;      // Herds start 2^4 = 16x apart
    bool deterministic_starts = true;
    
    // ========================================================================
    // Performance Monitoring
    // ========================================================================
    
    bool enable_herd_stats = true;
    uint64_t stats_update_interval = 1000000;
    bool enable_rebalancing = true;
    float rebalance_threshold = 0.80f;
    
    // ========================================================================
    // Kernel Launch Configuration
    // ========================================================================
    
    int threads_per_block = 256;
    int iterations_per_launch = 10000;
    bool use_async_streams = true;
    
    // ========================================================================
    // Calculated Properties
    // ========================================================================
    
    int getTotalKangaroosPerGpu() const {
        return herds_per_gpu * kangaroos_per_herd;
    }
    
    int getNumBlocks() const {
        int total = getTotalKangaroosPerGpu();
        return (total + threads_per_block - 1) / threads_per_block;
    }
    
    // ========================================================================
    // Preset Configurations
    // ========================================================================
    
    static HerdConfig forPuzzleSize(int bits) {
        HerdConfig cfg;
        
        if (bits < 100) {
            // Small puzzles: minimal herds
            cfg.herds_per_gpu = 4;
            cfg.kangaroos_per_herd = 512;
            cfg.adaptive_jumps = false;
        } else if (bits < 120) {
            // Medium puzzles: balanced
            cfg.herds_per_gpu = 8;
            cfg.kangaroos_per_herd = 256;
        } else {
            // Large puzzles: maximum herds
            cfg.herds_per_gpu = 16;
            cfg.kangaroos_per_herd = 256;
            cfg.herd_separation_bits = 6;
        }
        
        cfg.dp_bits = (bits >= 120) ? 18 : (bits >= 100) ? 16 : 14;
        return cfg;
    }
};

// Golden ratio for herd bias generation
constexpr uint64_t HERD_BIAS_MULTIPLIER = 0x9e3779b97f4a7c15ULL;

#endif // HERD_CONFIG_H
