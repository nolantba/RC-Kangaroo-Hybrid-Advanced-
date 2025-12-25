// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC
// CPU Kangaroo Worker - Hybrid GPU+CPU implementation

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

RCCpuKang::RCCpuKang()
{
	Kangaroos = nullptr;
	DPBuffer = nullptr;
	StopFlag = false;
	Failed = false;
	TotalOps = 0;
	DPBufferIndex = 0;
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;
}

RCCpuKang::~RCCpuKang()
{
	if (Kangaroos)
		delete[] Kangaroos;
	if (DPBuffer)
		delete[] DPBuffer;
#if USE_LISSAJOUS_CPU
	if (LissajousGen)
		delete LissajousGen;
#endif
}

bool RCCpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	TotalOps = 0;
	DPBufferIndex = 0;

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = Pnt_HalfRange;
	NegPntHalfRange = Pnt_NegHalfRange;

	// Calculate points A and B for WILD1 and WILD2
	PntA = ec.AddPoints(PntToSolve, PntHalfRange);
	PntB = ec.AddPoints(PntToSolve, NegPntHalfRange);

	KangCnt = CPU_KANGS_PER_THREAD;
	Kangaroos = new TCpuKang[KangCnt];

	// Allocate DP buffer (enough for all kangaroos)
	DPBuffer = new u32[KangCnt * (GPU_DP_SIZE / 4)];

#if USE_LISSAJOUS_CPU
	// Initialize Lissajous jump generator for this range
	LissajousGen = new LissajousJumpGenerator();
	auto config = LissajousJumpGenerator::optimized_config(Range);
	config.pattern_type = LissajousJumpGenerator::Config::DAMPED_HARMONOGRAPH;
	config.damping_x = 0.00005; // Slow spiral for large ranges
	LissajousGen->initialize(config);
	printf("CPU Thread %d: Lissajous generator initialized (%zu jumps)\n",
		ThreadIndex, LissajousGen->get_table_size());
#endif

#if USE_GR_EQUIVALENCE
	printf("CPU Thread %d: Galbraith-Ruprai equivalence ENABLED\n", ThreadIndex);
#endif

	InitializeKangaroos();
	return true;
}

void RCCpuKang::InitializeKangaroos()
{
	// Initialize kangaroos with random starting positions using FAST jump-based method
	// Split equally: 1/3 TAME, 1/3 WILD1, 1/3 WILD2
	for (int i = 0; i < KangCnt; i++)
	{
		if (i % 3 == 0)
		{
			// TAME kangaroo - start from G + random jumps
			Kangaroos[i].type = TAME;
			Kangaroos[i].point = g_G;
			Kangaroos[i].dist.SetZero();

			// Do random jumps to scatter starting position (FAST)
			int num_jumps = 50 + (i % 100); // 50-150 jumps
			for (int j = 0; j < num_jumps; j++)
			{
				u32 jmp_idx = (u32)(Kangaroos[i].point.x.data[0]) % JMP_CNT;
				Kangaroos[i].point = ec.AddPoints(Kangaroos[i].point, EcJumps1[jmp_idx].p);
				Kangaroos[i].dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
				NormalizePoint_GR(&Kangaroos[i].point);
#endif
			}
		}
		else if (i % 3 == 1)
		{
			// WILD1 kangaroo - start from A (PubKey + HalfRange) + random jumps
			Kangaroos[i].type = WILD1;
			Kangaroos[i].point = PntA;
			Kangaroos[i].dist.SetZero();

			// Do random jumps to scatter starting position
			int num_jumps = 50 + (i % 100);
			for (int j = 0; j < num_jumps; j++)
			{
				u32 jmp_idx = (u32)(Kangaroos[i].point.x.data[0]) % JMP_CNT;
				Kangaroos[i].point = ec.AddPoints(Kangaroos[i].point, EcJumps1[jmp_idx].p);
				Kangaroos[i].dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
				NormalizePoint_GR(&Kangaroos[i].point);
#endif
			}
		}
		else
		{
			// WILD2 kangaroo - start from B (PubKey - HalfRange) + random jumps
			Kangaroos[i].type = WILD2;
			Kangaroos[i].point = PntB;
			Kangaroos[i].dist.SetZero();

			// Do random jumps to scatter starting position
			int num_jumps = 50 + (i % 100);
			for (int j = 0; j < num_jumps; j++)
			{
				u32 jmp_idx = (u32)(Kangaroos[i].point.x.data[0]) % JMP_CNT;
				Kangaroos[i].point = ec.AddPoints(Kangaroos[i].point, EcJumps1[jmp_idx].p);
				Kangaroos[i].dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
				NormalizePoint_GR(&Kangaroos[i].point);
#endif
			}
		}
	}
}

bool RCCpuKang::IsDistinguishedPoint(EcInt& x)
{
	// Check if lower DP bits are zero
	u64 mask = (1ull << DP) - 1;
	return (x.data[0] & mask) == 0;
}

void RCCpuKang::FlushDPBuffer()
{
	if (DPBufferIndex > 0)
	{
		// Send DPs to shared table
		u64 ops = TotalOps;
		AddPointsToList(DPBuffer, DPBufferIndex, ops);
		DPBufferIndex = 0;
	}
}

void RCCpuKang::InitializeWildKangaroo(int kang_idx)
{
	// Reseed a single kangaroo as WILD1 (for JLP-style execution)
	// Used after finding a DP to get a fresh kangaroo
	Kangaroos[kang_idx].type = WILD1;
	Kangaroos[kang_idx].point = PntA;  // Start from PubKey + HalfRange
	Kangaroos[kang_idx].dist.SetZero();

	// Do random jumps to scatter starting position
	int num_jumps = 50 + (kang_idx % 100);  // 50-150 jumps
	for (int j = 0; j < num_jumps; j++)
	{
		u32 jmp_idx = (u32)(Kangaroos[kang_idx].point.x.data[0]) % JMP_CNT;
		Kangaroos[kang_idx].point = ec.AddPoints(Kangaroos[kang_idx].point, EcJumps1[jmp_idx].p);
		Kangaroos[kang_idx].dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
		NormalizePoint_GR(&Kangaroos[kang_idx].point);
#endif
	}
}

void RCCpuKang::ProcessKangaroo(int kang_idx)
{
	TCpuKang* kang = &Kangaroos[kang_idx];

	// Perform multiple jumps before checking for DP
	const int BATCH_SIZE = 100;

	for (int step = 0; step < BATCH_SIZE && !StopFlag && !gSolved; step++)
	{
		// Select jump based on x-coordinate
		u32 jmp_idx = (u32)(kang->point.x.data[0]) & JMP_MASK;

#if USE_LISSAJOUS_CPU
		// Use Lissajous-generated jump parameters
		const auto& liss_params = LissajousGen->get_jump_params(TotalOps + step);
		jmp_idx = (jmp_idx + liss_params.mean) % JMP_CNT;
#endif

		// Use Jumps1 for most iterations
		EcJMP* jump = &EcJumps1[jmp_idx];

		// Add jump to point
		kang->point = ec.AddPoints(kang->point, jump->p);
		kang->dist.Add(jump->dist);

#if USE_GR_EQUIVALENCE
		// Normalize to canonical form (even y-coordinate)
		NormalizePoint_GR(&kang->point);
#endif

		TotalOps++;
	}

	// Check if this is a distinguished point
	if (IsDistinguishedPoint(kang->point.x))
	{
		// Format: 12 bytes X + 4 bytes padding + 22 bytes dist + 2 bytes type + 2 bytes padding
		u8* dp_rec = (u8*)&DPBuffer[DPBufferIndex * (GPU_DP_SIZE / 4)];

		// Copy x-coordinate (first 12 bytes)
		memcpy(dp_rec, &kang->point.x.data[0], 12);

		// Padding (4 bytes)
		memset(dp_rec + 12, 0, 4);

		// Copy distance (22 bytes)
		memcpy(dp_rec + 16, &kang->dist.data[0], 22);

		// Copy type
		dp_rec[38] = 0;
		dp_rec[39] = 0;
		dp_rec[40] = kang->type;
		dp_rec[41] = 0;

		// Padding
		memset(dp_rec + 42, 0, 6);

		DPBufferIndex++;

		// Restart this kangaroo with new random position using FAST jump-based method
		// Select starting point based on kangaroo type
		EcPoint start_point;
		if (kang->type == TAME)
			start_point = g_G;
		else if (kang->type == WILD1)
			start_point = PntA;
		else
			start_point = PntB;

		// Restart from starting point with random jumps to scatter
		kang->point = start_point;
		kang->dist.SetZero();
		int num_jumps = 50 + (kang_idx % 100);
		for (int j = 0; j < num_jumps; j++)
		{
			u32 jmp_idx = (u32)(kang->point.x.data[0]) % JMP_CNT;
			kang->point = ec.AddPoints(kang->point, EcJumps1[jmp_idx].p);
			kang->dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
			NormalizePoint_GR(&kang->point);
#endif
		}
	}

	// Flush buffer if full
	if (DPBufferIndex >= 256)
	{
		FlushDPBuffer();
	}
}

void RCCpuKang::Execute()
{
	u64 last_stats_time = GetTickCount64();
	u64 ops_at_last_stats = 0;

	while (!StopFlag && !gSolved)
	{
		// Process all kangaroos
		for (int i = 0; i < KangCnt && !StopFlag && !gSolved; i++)
		{
			ProcessKangaroo(i);
		}

		// Update statistics every second
		u64 now = GetTickCount64();
		if (now - last_stats_time >= 1000)
		{
			u64 ops_delta = TotalOps - ops_at_last_stats;
			int speed_kkeys = (int)(ops_delta / 1000); // KKeys/s (CPU does fewer ops than GPU)

			SpeedStats[cur_stats_ind] = speed_kkeys;
			cur_stats_ind = (cur_stats_ind + 1) % CPU_STATS_WND_SIZE;

			last_stats_time = now;
			ops_at_last_stats = TotalOps;

			// Flush any pending DPs
			FlushDPBuffer();
		}
	}

	// Final flush
	FlushDPBuffer();
}

// Optimized execution with larger batches while maintaining cache locality
void RCCpuKang::Execute_Optimized()
{
	u64 last_stats_time = GetTickCount64();
	u64 ops_at_last_stats = 0;

	while (!StopFlag && !gSolved)
	{
		// Process all kangaroos in sequence (preserves cache locality)
		for (int kang_idx = 0; kang_idx < KangCnt && !StopFlag && !gSolved; kang_idx++)
		{
			TCpuKang* kang = &Kangaroos[kang_idx];

			// Larger batch size (5000 steps) - balanced between overhead reduction and cache locality
			const int BATCH_SIZE = 5000;

			for (int step = 0; step < BATCH_SIZE && !StopFlag && !gSolved; step++)
			{
				// Select jump based on x-coordinate
				u32 jmp_idx = (u32)(kang->point.x.data[0]) & JMP_MASK;
				EcJMP* jump = &EcJumps1[jmp_idx];

				// EC point addition
				kang->point = ec.AddPoints(kang->point, jump->p);
				kang->dist.Add(jump->dist);

				TotalOps++;
			}

			// Check if this is a distinguished point (after batch)
			if (IsDistinguishedPoint(kang->point.x))
			{
				// Format: 12 bytes X + 4 bytes padding + 22 bytes dist + 2 bytes type + 2 bytes padding
				u8* dp_rec = (u8*)&DPBuffer[DPBufferIndex * (GPU_DP_SIZE / 4)];

				// Copy x-coordinate (first 12 bytes)
				memcpy(dp_rec, &kang->point.x.data[0], 12);

				// Padding (4 bytes)
				memset(dp_rec + 12, 0, 4);

				// Copy distance (22 bytes)
				memcpy(dp_rec + 16, &kang->dist.data[0], 22);

				// Copy type
				dp_rec[38] = 0;
				dp_rec[39] = 0;
				dp_rec[40] = kang->type;
				dp_rec[41] = 0;

				// Padding
				memset(dp_rec + 42, 0, 6);

				DPBufferIndex++;

				// Restart this kangaroo with new random position
				if (kang->type == TAME)
				{
					kang->point = g_G;
					kang->dist.SetZero();
					int num_jumps = 50 + (kang_idx % 100);
					for (int j = 0; j < num_jumps; j++)
					{
						u32 jmp_idx = (u32)(kang->point.x.data[0]) % JMP_CNT;
						kang->point = ec.AddPoints(kang->point, EcJumps1[jmp_idx].p);
						kang->dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
						NormalizePoint_GR(&kang->point);
#endif
					}
				}
				else if (kang->type == WILD1)
				{
					kang->point = PntA;
					kang->dist.SetZero();
					int num_jumps = 50 + (kang_idx % 100);
					for (int j = 0; j < num_jumps; j++)
					{
						u32 jmp_idx = (u32)(kang->point.x.data[0]) % JMP_CNT;
						kang->point = ec.AddPoints(kang->point, EcJumps1[jmp_idx].p);
						kang->dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
						NormalizePoint_GR(&kang->point);
#endif
					}
				}
				else
				{
					kang->point = PntB;
					kang->dist.SetZero();
					int num_jumps = 50 + (kang_idx % 100);
					for (int j = 0; j < num_jumps; j++)
					{
						u32 jmp_idx = (u32)(kang->point.x.data[0]) % JMP_CNT;
						kang->point = ec.AddPoints(kang->point, EcJumps1[jmp_idx].p);
						kang->dist.Add(EcJumps1[jmp_idx].dist);
#if USE_GR_EQUIVALENCE
						NormalizePoint_GR(&kang->point);
#endif
					}
				}
			}

			// Flush buffer if full
			if (DPBufferIndex >= 256)
			{
				FlushDPBuffer();
			}
		}

		// Update statistics every second
		u64 now = GetTickCount64();
		if (now - last_stats_time >= 1000)
		{
			u64 ops_delta = TotalOps - ops_at_last_stats;
			int speed_kkeys = (int)(ops_delta / 1000);

			SpeedStats[cur_stats_ind] = speed_kkeys;
			cur_stats_ind = (cur_stats_ind + 1) % CPU_STATS_WND_SIZE;

			last_stats_time = now;
			ops_at_last_stats = TotalOps;

			// Flush any pending DPs
			FlushDPBuffer();
		}
	}

	// Final flush
	FlushDPBuffer();
}

void RCCpuKang::Stop()
{
	StopFlag = true;
}

int RCCpuKang::GetStatsSpeed()
{
	int sum = 0;
	int cnt = 0;
	for (int i = 0; i < CPU_STATS_WND_SIZE; i++)
	{
		if (SpeedStats[i] > 0)
		{
			sum += SpeedStats[i];
			cnt++;
		}
	}
	if (cnt == 0)
		return 0;
	return sum / cnt; // Return average KKeys/s
}
