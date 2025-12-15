// JLP-Inspired CPU Kangaroo Implementation for RC-Kangaroo-Hybrid
// Optimizations based on JeanLucPons' Kangaroo CPU engine
// Maintains compatibility with RC's DP format and infrastructure

#include <iostream>
#include <cstring>
#include "CpuKang.h"
#include "defs.h"
#include "utils.h"

extern Ec ec;
extern EcPoint g_G;
extern EcPoint Pnt_HalfRange;
extern EcPoint Pnt_NegHalfRange;
extern EcInt Int_TameOffset;
extern bool gGenMode;
extern volatile bool gSolved;

void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt);

// ============================================================================
// JLP-Style Optimized CPU Execution
// ============================================================================

void RCCpuKang::Execute_JLP()
{
    u64 last_stats_time = GetTickCount64();
    u64 ops_at_last_stats = 0;

    // JLP optimization: Use larger work chunks
    const int LARGE_BATCH = 10000;  // Much larger than RC's 100!
    const int DP_CHECK_INTERVAL = 100;  // Check for DPs every N kangaroos

    int current_kang = 0;  // Round-robin starting point

    while (!StopFlag && !gSolved)
    {
        // Process kangaroos in cyclic fashion (JLP style)
        for (int batch = 0; batch < DP_CHECK_INTERVAL && !StopFlag && !gSolved; batch++)
        {
            // Cyclic kangaroo selection (better than sequential)
            int kang_idx = current_kang;
            current_kang = (current_kang + 1) % KangCnt;

            TCpuKang* kang = &Kangaroos[kang_idx];

            // ================================================================
            // Inner loop: Process LARGE_BATCH steps WITHOUT overhead
            // ================================================================
            for (int step = 0; step < LARGE_BATCH; step++)
            {
                // Select jump based on x-coordinate (standard kangaroo method)
                u32 jmp_idx = (u32)(kang->point.x.data[0]) & JMP_MASK;
                EcJMP* jump = &EcJumps1[jmp_idx];

                // EC point addition
                kang->point = ec.AddPoints(kang->point, jump->p);
                kang->dist.Add(jump->dist);
            }

            TotalOps += LARGE_BATCH;

            // ================================================================
            // Check if distinguished point (only after full batch)
            // ================================================================
            if (IsDistinguishedPoint(kang->point.x))
            {
                // Format: 12 bytes X + 4 bytes padding + 22 bytes dist + 2 bytes type
                u8* dp_rec = (u8*)&DPBuffer[DPBufferIndex * (GPU_DP_SIZE / 4)];

                // Copy x-coordinate (first 12 bytes)
                memcpy(dp_rec, &kang->point.x.data[0], 12);

                // Padding (4 bytes)
                memset(dp_rec + 12, 0, 4);

                // Copy distance (22 bytes)
                memcpy(dp_rec + 16, &kang->dist.data[0], 22);

                // Type (wild = WILD1)
                dp_rec[38] = WILD1;
                dp_rec[39] = 0;

                DPBufferIndex++;

                // Reseed this kangaroo as new wild
                InitializeWildKangaroo(kang_idx);
            }

            // Flush DP buffer if full
            if (DPBufferIndex >= 256)
            {
                FlushDPBuffer();
            }
        }

        // ====================================================================
        // Update statistics (less frequently than RC's original)
        // ====================================================================
        u64 now = GetTickCount64();
        if (now - last_stats_time >= 1000)
        {
            u64 ops_delta = TotalOps - ops_at_last_stats;
            int speed_kkeys = (int)(ops_delta / 1000);

            SpeedStats[cur_stats_ind] = speed_kkeys;
            cur_stats_ind = (cur_stats_ind + 1) % CPU_STATS_WND_SIZE;

            last_stats_time = now;
            ops_at_last_stats = TotalOps;

            // Periodic DP flush
            FlushDPBuffer();
        }
    }

    // Final flush
    FlushDPBuffer();
}

// ============================================================================
// Alternative: Even more aggressive optimization (JLP extreme)
// ============================================================================

void RCCpuKang::Execute_JLP_Extreme()
{
    u64 last_stats_time = GetTickCount64();
    u64 ops_at_last_stats = 0;

    // Extreme JLP style: Massive batches, minimal overhead
    const int MEGA_BATCH = 50000;  // 50K steps before DP check!

    int kang_idx = 0;

    while (!StopFlag && !gSolved)
    {
        TCpuKang* kang = &Kangaroos[kang_idx];

        // ====================================================================
        // Ultra-tight inner loop (minimal overhead)
        // ====================================================================
        for (int step = 0; step < MEGA_BATCH; step++)
        {
            u32 jmp_idx = (u32)(kang->point.x.data[0]) & JMP_MASK;
            EcJMP* jump = &EcJumps1[jmp_idx];

            kang->point = ec.AddPoints(kang->point, jump->p);
            kang->dist.Add(jump->dist);
        }

        TotalOps += MEGA_BATCH;

        // Check DP
        if (IsDistinguishedPoint(kang->point.x))
        {
            u8* dp_rec = (u8*)&DPBuffer[DPBufferIndex * (GPU_DP_SIZE / 4)];
            memcpy(dp_rec, &kang->point.x.data[0], 12);
            memset(dp_rec + 12, 0, 4);
            memcpy(dp_rec + 16, &kang->dist.data[0], 22);
            dp_rec[38] = WILD1;
            dp_rec[39] = 0;

            DPBufferIndex++;
            InitializeWildKangaroo(kang_idx);
        }

        // Move to next kangaroo (cyclic)
        kang_idx = (kang_idx + 1) % KangCnt;

        // Flush if needed
        if (DPBufferIndex >= 256)
        {
            FlushDPBuffer();
        }

        // Stats update (every second)
        u64 now = GetTickCount64();
        if (now - last_stats_time >= 1000)
        {
            u64 ops_delta = TotalOps - ops_at_last_stats;
            int speed_kkeys = (int)(ops_delta / 1000);

            SpeedStats[cur_stats_ind] = speed_kkeys;
            cur_stats_ind = (cur_stats_ind + 1) % CPU_STATS_WND_SIZE;

            last_stats_time = now;
            ops_at_last_stats = TotalOps;

            FlushDPBuffer();
        }

        // Check stop conditions less frequently
        if ((TotalOps & 0xFFFF) == 0)  // Every 65K ops
        {
            if (StopFlag || gSolved) break;
        }
    }

    FlushDPBuffer();
}
