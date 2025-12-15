// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "Ec.h"

#define STATS_WND_SIZE	16

// Forward declarations for herd support
class GpuHerdManager;

// DP struct definition (must come before class that uses it)
#pragma pack(push, 1)
struct DP {
    u8 x[12];   // X-coordinate tail (96 bits)
    u8 d[22];   // Distance (176 bits)
    u8 type;    // TAME/WILD1/WILD2
};
#pragma pack(pop)

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class RCGpuKang
{
private:
	bool StopFlag = false;  // Initialize to false
	EcPoint PntToSolve;
	int Range = 0;  // Initialize to 0
	int DP_bits = 0;  // Initialize to 0
	Ec ec;

	u32* DPs_out = nullptr;  // Initialize to nullptr
	TKparams Kparams{}; // Zero-initialize struct

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts = nullptr;
	EcJMP* EcJumps1 = nullptr;
	EcJMP* EcJumps2 = nullptr;
	EcJMP* EcJumps3 = nullptr;

	EcPoint PntA;
	EcPoint PntB;

	int cur_stats_ind = 0;
	int SpeedStats[STATS_WND_SIZE] = {};  // Zero-initialize array

	void GenerateRndDistances();
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif
	// SOTA++ Herd support
	bool use_herds_ = false;  // CRITICAL: Initialize to false!
	GpuHerdManager* herd_manager_ = nullptr;  // CRITICAL: Initialize to nullptr!

	// Herd mode GPU arrays (separate X/Y/Dist format)
	u64* d_herd_kangaroo_x_ = nullptr;
	u64* d_herd_kangaroo_y_ = nullptr;
	u64* d_herd_kangaroo_dist_ = nullptr;
	DP* h_herd_dp_buffer_ = nullptr;  // Host buffer for DP collection

public:
	int persistingL2CacheMaxSize = 0;
	int CudaIndex = 0;  // GPU index in cuda
	int mpCnt = 0;
	int KangCnt = 0;
	bool Failed = false;
	bool IsOldGpu = false;

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();

	u32 dbg[256];

	int GetStatsSpeed();

	// SOTA++ Herd methods
	void SetUseHerds(bool enable, int range_bits);
	bool IsUsingHerds() const { return use_herds_; }
	GpuHerdManager* GetHerdManager() { return herd_manager_; }
};
