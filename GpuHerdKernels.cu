#include "GpuHerdManager.h"
#include "GpuKang.h"  // For DP struct definition
#include "RCGpuUtils.h"
#include "defs.h"  // For JMP_CNT, TAME, WILD1, WILD2
#include <cuda_runtime.h>
// Note: Removed stdio - not needed for device code

// ============================================================================
// OPTIMIZED Herd Kernel with Montgomery Batch Inversion
// ============================================================================
// Each thread processes KANGS_PER_THREAD kangaroos using batch inversion
// This reduces N expensive InvModP calls to just 1 per thread
// ============================================================================

#define KANGS_PER_THREAD 8  // Process 8 kangaroos per thread (optimized for register pressure)

// Count leading zeros across all 256 bits
__device__ __forceinline__ int clz256(const u64* x) {
    if (x[3] != 0) return __clzll(x[3]);
    if (x[2] != 0) return 64 + __clzll(x[2]);
    if (x[1] != 0) return 128 + __clzll(x[1]);
    if (x[0] != 0) return 192 + __clzll(x[0]);
    return 256;
}

__global__ void kangarooHerdKernel(
    u64* __restrict__ jump_table,    // Packed jump table [JMP_CNT][12]: X(4), Y(4), D(3), pad(1)
    GpuHerdState* __restrict__ herd_states,
    HerdDPBuffer* __restrict__ herd_buffers,
    DP* __restrict__ gpu_dp_buffer,
    int* __restrict__ gpu_dp_count,
    const HerdConfig config,
    u64* kangaroo_x,     // Kangaroo X coordinates [total_kangs][4]
    u64* kangaroo_y,     // Kangaroo Y coordinates [total_kangs][4]
    u64* kangaroo_dist,  // Kangaroo distances [total_kangs][3]
    int iterations,
    u64 dp_mask64
)
{
    // ========================================================================
    // Load jump table into shared memory (CRITICAL OPTIMIZATION!)
    // ========================================================================
    __shared__ u64 shared_jmp_table[JMP_CNT * 12];  // 12 u64s per jump

    // All threads cooperatively load jump table
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    for (int i = tid; i < JMP_CNT * 12; i += block_size) {
        shared_jmp_table[i] = jump_table[i];
    }
    __syncthreads();  // Wait for all threads to finish loading

    // ========================================================================
    // Process kangaroos
    // ========================================================================
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_kangaroos = config.herds_per_gpu * config.kangaroos_per_herd;

    // Each thread handles KANGS_PER_THREAD kangaroos starting from base_kang
    int base_kang = thread_id * KANGS_PER_THREAD;
    if (base_kang >= total_kangaroos) return;

    int kangs_this_thread = min(KANGS_PER_THREAD, total_kangaroos - base_kang);

    // Determine herd for this thread (all kangaroos in thread belong to same herd)
    int herd_id = base_kang / config.kangaroos_per_herd;
    int herd_bias = herd_id * 17;  // Herd-specific jump bias

    // Local storage for this thread's kangaroos
    __align__(16) u64 local_x[KANGS_PER_THREAD][4];
    __align__(16) u64 local_y[KANGS_PER_THREAD][4];
    u64 local_dist[KANGS_PER_THREAD][3];

    // Load kangaroo states
    for (int k = 0; k < kangs_this_thread; k++) {
        int kang_idx = base_kang + k;
        Copy_int4_x2(local_x[k], kangaroo_x + kang_idx * 4);
        Copy_int4_x2(local_y[k], kangaroo_y + kang_idx * 4);
        local_dist[k][0] = kangaroo_dist[kang_idx * 3 + 0];
        local_dist[k][1] = kangaroo_dist[kang_idx * 3 + 1];
        local_dist[k][2] = kangaroo_dist[kang_idx * 3 + 2];
    }

    // ========================================================================
    // Main iteration loop
    // ========================================================================
    for (int iter = 0; iter < iterations; iter++) {

        // ====================================================================
        // PHASE 1: Compute prefix product of denominators (Montgomery trick)
        // ====================================================================
        __align__(16) u64 dx_values[KANGS_PER_THREAD][4];
        __align__(16) u64 jmp_x[KANGS_PER_THREAD][4];
        __align__(16) u64 jmp_y[KANGS_PER_THREAD][4];
        u64 jmp_d[KANGS_PER_THREAD][3];

        // Compute dx = x0 - jmp_x for all kangaroos
        for (int k = 0; k < kangs_this_thread; k++) {
            // Select jump with herd bias
            u16 jmp_ind = ((u16)local_x[k][0] + herd_bias) & (JMP_CNT - 1);

            // Load jump point from SHARED MEMORY (fast!)
            u64* jmp_ptr = shared_jmp_table + 12 * jmp_ind;
            Copy_int4_x2(jmp_x[k], jmp_ptr + 0);
            Copy_int4_x2(jmp_y[k], jmp_ptr + 4);
            jmp_d[k][0] = jmp_ptr[8];
            jmp_d[k][1] = jmp_ptr[9];
            jmp_d[k][2] = jmp_ptr[10];

            // Compute dx = x0 - jmp_x
            SubModP(dx_values[k], local_x[k], jmp_x[k]);
        }

        // Build prefix products
        __align__(16) u64 prefix[KANGS_PER_THREAD][4];
        Copy_u64_x4(prefix[0], dx_values[0]);

        for (int k = 1; k < kangs_this_thread; k++) {
            MulModP(prefix[k], prefix[k-1], dx_values[k]);
        }

        // ====================================================================
        // PHASE 2: ONE expensive inverse operation
        // ====================================================================
        __align__(16) u64 inverse[5];
        Copy_u64_x4(inverse, prefix[kangs_this_thread - 1]);
        InvModP((u32*)inverse);  // ⚡ ONLY ONE INVERSE FOR ALL KANGAROOS

        // ====================================================================
        // PHASE 3: Recover individual inverses and perform EC operations
        // ====================================================================
        for (int k = kangs_this_thread - 1; k >= 0; k--) {
            __align__(16) u64 dxs[4];  // This is dx[k]^-1

            if (k > 0) {
                MulModP(dxs, inverse, prefix[k-1]);
                MulModP(inverse, inverse, dx_values[k]);
            } else {
                Copy_u64_x4(dxs, inverse);
            }

            // ================================================================
            // SOTA+ bidirectional jumping
            // ================================================================
            __align__(16) u64 tmp[4], tmp2[4];
            __align__(16) u64 jmp_y_neg[4];
            __align__(16) u64 x3_plus[4], x3_minus[4];
            __align__(16) u64 x0[4], y0[4];

            Copy_u64_x4(x0, local_x[k]);
            Copy_u64_x4(y0, local_y[k]);

            // --- P + Jump ---
            SubModP(tmp2, y0, jmp_y[k]);
            MulModP(tmp, tmp2, dxs);  // slope_plus
            SqrModP(tmp2, tmp);
            SubModP(x3_plus, tmp2, jmp_x[k]);
            SubModP(x3_plus, x3_plus, x0);

            // --- P - Jump ---
            Copy_u64_x4(jmp_y_neg, jmp_y[k]);
            NegModP(jmp_y_neg);
            SubModP(tmp2, y0, jmp_y_neg);
            MulModP(tmp, tmp2, dxs);  // slope_minus
            SqrModP(tmp2, tmp);
            SubModP(x3_minus, tmp2, jmp_x[k]);
            SubModP(x3_minus, x3_minus, x0);

            // Choose direction with most leading zeros
            int zeros_plus = clz256(x3_plus);
            int zeros_minus = clz256(x3_minus);
            bool use_plus = (zeros_plus >= zeros_minus);

            // Commit the chosen direction
            if (use_plus) {
                // Update X
                Copy_u64_x4(local_x[k], x3_plus);

                // Compute Y: y = slope * (x0 - x3) - y0
                SubModP(tmp2, y0, jmp_y[k]);
                MulModP(tmp, tmp2, dxs);  // slope
                SubModP(tmp2, x0, local_x[k]);
                MulModP(local_y[k], tmp, tmp2);
                SubModP(local_y[k], local_y[k], y0);

                // Update distance
                Add192to192(local_dist[k], jmp_d[k]);

            } else {
                // Update X
                Copy_u64_x4(local_x[k], x3_minus);

                // Compute Y
                SubModP(tmp2, y0, jmp_y_neg);
                MulModP(tmp, tmp2, dxs);  // slope
                SubModP(tmp2, x0, local_x[k]);
                MulModP(local_y[k], tmp, tmp2);
                SubModP(local_y[k], local_y[k], y0);

                // Update distance (subtract)
                Sub192from192(local_dist[k], jmp_d[k]);
            }

            // ================================================================
            // Check for distinguished point
            // ================================================================
            if ((local_x[k][3] & dp_mask64) == 0) {
                DP new_dp;

                // Copy X tail (96 bits = 12 bytes)
                ((u32*)new_dp.x)[0] = (u32)local_x[k][0];
                ((u32*)new_dp.x)[1] = (u32)(local_x[k][0] >> 32);
                ((u32*)new_dp.x)[2] = (u32)local_x[k][1];

                // Copy distance (176 bits = 22 bytes)
                ((u64*)new_dp.d)[0] = local_dist[k][0];
                ((u64*)new_dp.d)[1] = local_dist[k][1];
                ((u32*)new_dp.d)[4] = (u32)local_dist[k][2];
                ((u16*)new_dp.d)[10] = (u16)(local_dist[k][2] >> 32);

                // Set type based on herd (cycle through TAME/WILD1/WILD2)
                new_dp.type = (u8)(herd_id % 3);

                // Add to GPU DP buffer
                int dp_idx = atomicAdd(gpu_dp_count, 1);
                if (dp_idx < config.gpu_dp_buffer_size) {
                    gpu_dp_buffer[dp_idx] = new_dp;
                    atomicAdd((unsigned long long*)&herd_states[herd_id].dps_found, 1ULL);
                }
            }
        }

        // Update operation counter (each thread counts its own work)
        if (iter % 100 == 0) {
            atomicAdd((unsigned long long*)&herd_states[herd_id].operations,
                     (unsigned long long)(kangs_this_thread * 100));
        }
    }

    // ========================================================================
    // Store kangaroo states back to global memory
    // ========================================================================
    for (int k = 0; k < kangs_this_thread; k++) {
        int kang_idx = base_kang + k;
        Copy_int4_x2(kangaroo_x + kang_idx * 4, local_x[k]);
        Copy_int4_x2(kangaroo_y + kang_idx * 4, local_y[k]);
        kangaroo_dist[kang_idx * 3 + 0] = local_dist[k][0];
        kangaroo_dist[kang_idx * 3 + 1] = local_dist[k][1];
        kangaroo_dist[kang_idx * 3 + 2] = local_dist[k][2];
    }
}

// ============================================================================
// Kernel Launch Wrapper
// ============================================================================

extern "C" void launchHerdKernels(
    GpuHerdMemory* mem,
    u64* d_jump_table,
    u64* d_kangaroo_x,
    u64* d_kangaroo_y,
    u64* d_kangaroo_dist,
    int iterations,
    int dp_bits
)
{
    int total_kangaroos = mem->config.herds_per_gpu * mem->config.kangaroos_per_herd;
    int num_threads_needed = (total_kangaroos + KANGS_PER_THREAD - 1) / KANGS_PER_THREAD;

    // Launch configuration: 256 threads per block (optimal for RTX 3060)
    int threads_per_block = 256;
    int num_blocks = (num_threads_needed + threads_per_block - 1) / threads_per_block;

    // Calculate DP mask
    u64 dp_mask64 = ~((1ull << (64 - dp_bits)) - 1);

    // Launch kernel with batch inversion optimization
    kangarooHerdKernel<<<num_blocks, threads_per_block>>>(
        d_jump_table,
        mem->d_herd_states,
        mem->d_herd_buffers,
        mem->d_gpu_dp_buffer,
        mem->d_gpu_dp_count,
        mem->config,
        d_kangaroo_x,
        d_kangaroo_y,
        d_kangaroo_dist,
        iterations,
        dp_mask64
    );

    // Note: Caller should check for errors with cudaGetLastError()
}

// ============================================================================
// DP Collection Function
// ============================================================================

extern "C" int checkHerdCollisions(
    GpuHerdMemory* mem,
    DP* h_dp_buffer,
    int buffer_size
)
{
    // Get DP count from GPU
    int dp_count = 0;
    cudaMemcpy(&dp_count, mem->d_gpu_dp_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (dp_count > buffer_size) {
        printf("WARNING: DP buffer overflow! Found %d DPs, buffer size %d\n",
               dp_count, buffer_size);
        dp_count = buffer_size;
    }

    if (dp_count > 0) {
        // Copy DPs from GPU to host
        cudaMemcpy(h_dp_buffer, mem->d_gpu_dp_buffer,
                  dp_count * sizeof(DP), cudaMemcpyDeviceToHost);

        // Reset DP counter for next iteration
        cudaMemset(mem->d_gpu_dp_count, 0, sizeof(int));
    }

    return dp_count;
}
