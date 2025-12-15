// ============================================================================
// GPU Herd Manager Implementation
// Manages multiple independent kangaroo herds on GPU
// ============================================================================

#include "GpuHerdManager.h"
#include "GpuKang.h"  // For DP struct definition
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

// Simple 64-bit random number generator
static uint64_t GetRnd64() {
    static bool initialized = false;
    if (!initialized) {
        srand((unsigned int)time(NULL));
        initialized = true;
    }
    return ((uint64_t)rand() << 32) | (uint64_t)rand();
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

GpuHerdManager::GpuHerdManager(int gpu_id, const HerdConfig& config)
    : gpu_id_(gpu_id)
    , config_(config)
    , streams_(nullptr)
{
    mem_.gpu_id = gpu_id;
    mem_.config = config;

    // Initialize device pointers to null
    mem_.d_jump_tables = nullptr;
    mem_.d_herd_states = nullptr;
    mem_.d_herd_buffers = nullptr;
    mem_.d_gpu_dp_buffer = nullptr;
    mem_.d_gpu_dp_count = nullptr;

    // Allocate host-side tracking
    herd_operations_.resize(config.herds_per_gpu, 0);
    herd_stats_.resize(config.herds_per_gpu);

    for (auto& stat : herd_stats_) {
        stat.reset();
    }
}

GpuHerdManager::~GpuHerdManager()
{
    Shutdown();
}

// ============================================================================
// Initialization
// ============================================================================

bool GpuHerdManager::Initialize(int range_bits)
{
    cudaError_t err;

    // Set GPU device
    err = cudaSetDevice(gpu_id_);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to set GPU %d: %s\n", gpu_id_, cudaGetErrorString(err));
        return false;
    }

    printf("[GPU %d] Initializing SOTA++ herds (range=%d bits, herds=%d)\n",
           gpu_id_, range_bits, config_.herds_per_gpu);

    // Allocate GPU memory
    allocateGpuMemory();

    // Initialize herd states
    initializeHerdStates();

    // Generate herd-specific jump tables
    GenerateHerdJumpTables(range_bits);

    // Create CUDA streams for async operations
    if (config_.use_async_streams) {
        streams_ = new cudaStream_t[config_.herds_per_gpu];
        for (int i = 0; i < config_.herds_per_gpu; i++) {
            err = cudaStreamCreate(&streams_[i]);
            if (err != cudaSuccess) {
                printf("WARNING: Failed to create stream %d: %s\n", i, cudaGetErrorString(err));
            }
        }
    }

    printf("[GPU %d] Herd manager initialized: %d kangaroos total (%d per herd)\n",
           gpu_id_, config_.getTotalKangaroosPerGpu(), config_.kangaroos_per_herd);

    return true;
}

// ============================================================================
// Memory Management
// ============================================================================

void GpuHerdManager::allocateGpuMemory()
{
    cudaError_t err;

    // Allocate jump tables [herds][jump_table_size]
    size_t jump_table_bytes = config_.herds_per_gpu * config_.jump_table_size * sizeof(uint64_t);
    err = cudaMalloc(&mem_.d_jump_tables, jump_table_bytes);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate jump tables: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[GPU %d] Allocated jump tables: %.2f MB\n",
           gpu_id_, jump_table_bytes / (1024.0 * 1024.0));

    // Allocate herd states
    size_t state_bytes = config_.herds_per_gpu * sizeof(GpuHerdState);
    err = cudaMalloc(&mem_.d_herd_states, state_bytes);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate herd states: %s\n", cudaGetErrorString(err));
        return;
    }

    // Allocate herd DP buffers
    size_t herd_buffer_bytes = config_.herds_per_gpu * sizeof(HerdDPBuffer);
    err = cudaMalloc(&mem_.d_herd_buffers, herd_buffer_bytes);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate herd buffers: %s\n", cudaGetErrorString(err));
        return;
    }

    // Allocate individual DP arrays for each herd buffer
    for (int i = 0; i < config_.herds_per_gpu; i++) {
        HerdDPBuffer host_buffer;
        host_buffer.capacity = config_.herd_dp_buffer_size;
        host_buffer.count = 0;
        host_buffer.write_idx = 0;

        // Allocate DP storage
        size_t dp_bytes = config_.herd_dp_buffer_size * sizeof(DP);
        err = cudaMalloc(&host_buffer.dps, dp_bytes);
        if (err != cudaSuccess) {
            printf("ERROR: Failed to allocate herd %d DP buffer: %s\n",
                   i, cudaGetErrorString(err));
            return;
        }

        // Copy buffer metadata to device
        err = cudaMemcpy(mem_.d_herd_buffers + i, &host_buffer,
                        sizeof(HerdDPBuffer), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("ERROR: Failed to copy herd buffer metadata: %s\n",
                   cudaGetErrorString(err));
            return;
        }
    }

    // Allocate GPU-wide DP buffer
    size_t gpu_dp_bytes = config_.gpu_dp_buffer_size * sizeof(DP);
    err = cudaMalloc(&mem_.d_gpu_dp_buffer, gpu_dp_bytes);
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate GPU DP buffer: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[GPU %d] Allocated GPU DP buffer: %.2f MB\n",
           gpu_id_, gpu_dp_bytes / (1024.0 * 1024.0));

    // Allocate GPU DP counter
    err = cudaMalloc(&mem_.d_gpu_dp_count, sizeof(int));
    if (err != cudaSuccess) {
        printf("ERROR: Failed to allocate GPU DP counter: %s\n", cudaGetErrorString(err));
        return;
    }

    // Initialize counter to 0
    int zero = 0;
    cudaMemcpy(mem_.d_gpu_dp_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
}

void GpuHerdManager::freeGpuMemory()
{
    if (mem_.d_jump_tables) {
        cudaFree(mem_.d_jump_tables);
        mem_.d_jump_tables = nullptr;
    }

    if (mem_.d_herd_states) {
        cudaFree(mem_.d_herd_states);
        mem_.d_herd_states = nullptr;
    }

    if (mem_.d_herd_buffers) {
        // First free individual DP arrays
        HerdDPBuffer* host_buffers = new HerdDPBuffer[config_.herds_per_gpu];
        cudaMemcpy(host_buffers, mem_.d_herd_buffers,
                  config_.herds_per_gpu * sizeof(HerdDPBuffer),
                  cudaMemcpyDeviceToHost);

        for (int i = 0; i < config_.herds_per_gpu; i++) {
            if (host_buffers[i].dps) {
                cudaFree(host_buffers[i].dps);
            }
        }
        delete[] host_buffers;

        cudaFree(mem_.d_herd_buffers);
        mem_.d_herd_buffers = nullptr;
    }

    if (mem_.d_gpu_dp_buffer) {
        cudaFree(mem_.d_gpu_dp_buffer);
        mem_.d_gpu_dp_buffer = nullptr;
    }

    if (mem_.d_gpu_dp_count) {
        cudaFree(mem_.d_gpu_dp_count);
        mem_.d_gpu_dp_count = nullptr;
    }
}

void GpuHerdManager::initializeHerdStates()
{
    // Initialize all herd states to zero
    std::vector<GpuHerdState> initial_states(config_.herds_per_gpu);
    for (auto& state : initial_states) {
        state.reset();
    }

    cudaMemcpy(mem_.d_herd_states, initial_states.data(),
              config_.herds_per_gpu * sizeof(GpuHerdState),
              cudaMemcpyHostToDevice);
}

// ============================================================================
// Jump Table Generation
// ============================================================================

void GpuHerdManager::GenerateHerdJumpTables(int bits)
{
    printf("[GPU %d] Generating %d jump tables (adaptive=%s)...\n",
           gpu_id_, config_.herds_per_gpu, config_.adaptive_jumps ? "yes" : "no");

    // Allocate host-side jump tables
    size_t total_jumps = config_.herds_per_gpu * config_.jump_table_size;
    uint64_t* host_jumps = new uint64_t[total_jumps];

    // Base jump size for this range
    uint64_t min_jump = 1ULL << (bits / 2 + 1);  // Teske-optimized

    for (int herd = 0; herd < config_.herds_per_gpu; herd++) {
        // Generate herd-specific bias for spatial separation
        uint64_t herd_bias = generateHerdBias(herd);

        // Generate jump distances for this herd
        uint64_t* herd_table = host_jumps + (herd * config_.jump_table_size);

        for (int j = 0; j < config_.jump_table_size; j++) {
            // Base random jump
            uint64_t rnd = GetRnd64();
            uint64_t jump = min_jump + (rnd % min_jump);

            // Apply herd bias for spatial separation
            jump ^= (herd_bias * (j + 1));

            // Ensure even (for EC operations)
            jump &= ~1ULL;

            herd_table[j] = jump;
        }
    }

    // Copy to GPU
    size_t jump_bytes = total_jumps * sizeof(uint64_t);
    cudaMemcpy(mem_.d_jump_tables, host_jumps, jump_bytes, cudaMemcpyHostToDevice);

    delete[] host_jumps;

    printf("[GPU %d] Jump tables generated (%.2f MB)\n",
           gpu_id_, jump_bytes / (1024.0 * 1024.0));
}

uint64_t GpuHerdManager::generateHerdBias(int herd_id)
{
    // Generate deterministic but unique bias for each herd
    // Uses golden ratio multiplier for good distribution
    uint64_t bias = (uint64_t)herd_id * HERD_BIAS_MULTIPLIER;

    // Apply separation bits for spatial distribution
    bias <<= config_.herd_separation_bits;

    return bias;
}

// ============================================================================
// Statistics & Monitoring
// ============================================================================

void GpuHerdManager::GetHerdStats(std::vector<GpuHerdState>& stats)
{
    if (mem_.d_herd_states) {
        stats.resize(config_.herds_per_gpu);
        cudaMemcpy(stats.data(), mem_.d_herd_states,
                  config_.herds_per_gpu * sizeof(GpuHerdState),
                  cudaMemcpyDeviceToHost);
    }
}

void GpuHerdManager::PrintHerdStats()
{
    std::vector<GpuHerdState> stats;
    GetHerdStats(stats);

    printf("\n[GPU %d] Herd Statistics:\n", gpu_id_);
    printf("Herd | Ops (M)  | DPs Found | Rate (MK/s)\n");
    printf("-----+----------+-----------+------------\n");

    uint64_t total_ops = 0;
    uint64_t total_dps = 0;
    double total_rate = 0.0;

    for (int i = 0; i < config_.herds_per_gpu; i++) {
        double ops_m = stats[i].operations / 1000000.0;
        double rate = stats[i].progress_rate / 1000000.0;

        printf("  %2d | %8.1f | %9llu | %10.2f\n",
               i, ops_m, (unsigned long long)stats[i].dps_found, rate);

        total_ops += stats[i].operations;
        total_dps += stats[i].dps_found;
        total_rate += rate;
    }

    printf("-----+----------+-----------+------------\n");
    printf("Total| %8.1f | %9llu | %10.2f\n",
           total_ops / 1000000.0, (unsigned long long)total_dps, total_rate);
    printf("\n");

    // Check for imbalanced herds
    if (config_.enable_rebalancing && config_.herds_per_gpu > 1) {
        double avg_rate = total_rate / config_.herds_per_gpu;
        bool needs_rebalance = false;

        for (int i = 0; i < config_.herds_per_gpu; i++) {
            double rate = stats[i].progress_rate / 1000000.0;
            if (rate < avg_rate * config_.rebalance_threshold) {
                printf("WARNING: Herd %d underperforming (%.1f%% of average)\n",
                       i, (rate / avg_rate) * 100.0);
                needs_rebalance = true;
            }
        }

        if (needs_rebalance) {
            printf("Considering herd rebalancing...\n");
            // Future: implement actual rebalancing
        }
    }
}

void GpuHerdManager::RebalanceHerds()
{
    // TODO: Implement adaptive rebalancing
    // - Adjust jump tables for underperforming herds
    // - Redistribute kangaroos if needed
    // - Reset herd states
    printf("[GPU %d] Herd rebalancing not yet implemented\n", gpu_id_);
}

// ============================================================================
// Cleanup
// ============================================================================

void GpuHerdManager::Shutdown()
{
    // Destroy CUDA streams
    if (streams_) {
        for (int i = 0; i < config_.herds_per_gpu; i++) {
            cudaStreamDestroy(streams_[i]);
        }
        delete[] streams_;
        streams_ = nullptr;
    }

    // Free GPU memory
    freeGpuMemory();

    printf("[GPU %d] Herd manager shut down\n", gpu_id_);
}
