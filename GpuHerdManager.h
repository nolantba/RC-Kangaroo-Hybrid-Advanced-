#ifndef GPU_HERD_MANAGER_H
#define GPU_HERD_MANAGER_H

#include "HerdConfig.h"
#include "defs.h"
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// Distinguished Point Structure
// ============================================================================
// NOTE: DP struct is defined in GpuKang.h to avoid circular dependencies
// This is just a forward declaration
struct DP;

// ============================================================================
// GPU Herd State (device-side)
// ============================================================================

struct GpuHerdState {
    uint64_t operations;          // Total operations by this herd
    uint64_t dps_found;          // DPs found by this herd
    uint64_t local_collisions;   // Collisions within herd
    float progress_rate;         // Ops per second
    
    void reset() {
        operations = 0;
        dps_found = 0;
        local_collisions = 0;
        progress_rate = 0.0f;
    }
};

// ============================================================================
// Herd DP Buffer (device-side)
// ============================================================================

struct HerdDPBuffer {
    DP* dps;              // Circular buffer of recent DPs
    int count;            // Number of DPs in buffer
    int capacity;         // Buffer capacity
    int write_idx;        // Current write position
};

// ============================================================================
// GPU Memory Layout for Herds
// ============================================================================

struct GpuHerdMemory {
    // Per-herd data
    uint64_t* d_jump_tables;     // [herds][jump_table_size]
    GpuHerdState* d_herd_states; // [herds]
    HerdDPBuffer* d_herd_buffers;// [herds]
    
    // Per-GPU shared data
    DP* d_gpu_dp_buffer;         // Shared DP buffer for GPU
    int* d_gpu_dp_count;         // Number of DPs in GPU buffer
    
    // Configuration
    HerdConfig config;
    int gpu_id;
};

// ============================================================================
// GPU Herd Manager (host-side)
// ============================================================================

class GpuHerdManager {
public:
    GpuHerdManager(int gpu_id, const HerdConfig& config);
    ~GpuHerdManager();
    
    // Initialize herds on GPU
    bool Initialize(int range_bits);
    
    // Generate herd-specific jump tables
    void GenerateHerdJumpTables(int bits);
    
    // Get herd statistics
    void GetHerdStats(std::vector<GpuHerdState>& stats);
    
    // Print herd statistics
    void PrintHerdStats();
    
    // Adaptive: Rebalance herds based on performance
    void RebalanceHerds();
    
    // Cleanup
    void Shutdown();
    
    // Get GPU memory structure (for kernel launch)
    GpuHerdMemory* GetMemory() { return &mem_; }
    
    // Get CUDA streams
    cudaStream_t* GetStreams() { return streams_; }
    
private:
    int gpu_id_;
    HerdConfig config_;
    GpuHerdMemory mem_;
    
    // Host-side tracking
    std::vector<uint64_t> herd_operations_;
    std::vector<GpuHerdState> herd_stats_;
    
    // CUDA streams for async operations
    cudaStream_t* streams_;
    
    // Helper functions
    void allocateGpuMemory();
    void freeGpuMemory();
    void initializeHerdStates();
    uint64_t generateHerdBias(int herd_id);
};

#endif // GPU_HERD_MANAGER_H
