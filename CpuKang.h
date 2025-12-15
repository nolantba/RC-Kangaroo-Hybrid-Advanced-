// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC
// CPU Kangaroo Worker - Hybrid GPU+CPU implementation


#pragma once

#include "Ec.h"
#include "GpuKang.h"
#include "LissajousJumps.h"
#include "GalbraithRuprai.h"

#define CPU_STATS_WND_SIZE	16
#define CPU_KANGS_PER_THREAD	1024

// EXPERIMENTAL: Enable Lissajous jump patterns (default: OFF for testing)
#define USE_LISSAJOUS_CPU 0

// EXPERIMENTAL: Enable Galbraith-Ruprai equivalence (DISABLED - breaks GPU collision detection)
#define USE_GR_EQUIVALENCE 0

// CPU Kangaroo state
struct TCpuKang
{
	EcPoint point;
	EcInt dist;
	u8 type; // TAME, WILD1, or WILD2
};

class RCCpuKang
{
private:
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; // in bits
	int DP; // in bits
	Ec ec;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	EcPoint PntA;
	EcPoint PntB;

	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

#if USE_LISSAJOUS_CPU
	LissajousJumpGenerator* LissajousGen;
#endif

	TCpuKang* Kangaroos;
	int KangCnt;

	int cur_stats_ind;
	int SpeedStats[CPU_STATS_WND_SIZE];
	u64 TotalOps;

	u32* DPBuffer;
	int DPBufferIndex;

	void InitializeKangaroos();
	void InitializeWildKangaroo(int kang_idx);  // For reseeding after DP found
	void ProcessKangaroo(int kang_idx);
	bool IsDistinguishedPoint(EcInt& x);
	void FlushDPBuffer();

public:
	int ThreadIndex;
	bool Failed;

	RCCpuKang();
	~RCCpuKang();

	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();  // Original RC implementation
	void Execute_Optimized();  // Optimized with larger batches + cache locality
	void Execute_JLP();  // JLP-inspired optimized version
	void Execute_JLP_Extreme();  // Ultra-aggressive JLP version
	int GetStatsSpeed();
};
