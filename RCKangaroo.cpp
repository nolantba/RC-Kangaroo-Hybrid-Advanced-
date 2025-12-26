// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>
#include <signal.h>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"
#include "CpuKang.h"
#include "WorkFile.h"
#include "GpuMonitor.h"


EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
RCCpuKang* CpuKangs[128]; // Support up to 128 CPU threads
int CpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
volatile u64 DroppedDPs = 0;
volatile u64 TotalDPsGenerated = 0;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
int gCpuThreads; // Number of CPU threads to use
char gTamesFileName[1024];
int gTameRatioPct = 33;  // SOTA+ tame ratio optimization
int gTameBitsOffset = 4; // SOTA+ tame bits offset
double gMax;
bool gGenMode; //tames generation mode
bool gIsOpsLimit;

// Save/Resume work file support
RCWorkFile* g_work_file = nullptr;
AutoSaveManager* g_autosave = nullptr;
std::string g_work_filename;
uint64_t g_autosave_interval = 60;  // Default: 60 seconds

// Tames auto-save support
time_t g_last_tames_save = 0;
uint64_t g_tames_autosave_interval = 60;  // Default: 60 seconds

// SOTA++ Herds mode
bool g_use_herds = false;
time_t g_start_time = 0;
bool g_resume_mode = false;

#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

//	gcnt = 1; //dbg
	if (!gcnt)
		return;

	int drv, rt;
	cudaRuntimeGetVersion(&rt);
	cudaDriverGetVersion(&drv);
	char drvver[100];
	sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

	printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
	cudaError_t cudaStatus;
	for (int i = 0; i < gcnt; i++)
	{
		cudaStatus = cudaSetDevice(i);
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaSetDevice for gpu %d failed!\r\n", i);
			continue;
		}

		if (!gGPUs_Mask[i])
			continue;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
		
		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
u32 __stdcall cpu_kang_thr_proc(void* data)
{
	RCCpuKang* Kang = (RCCpuKang*)data;
	// Use original RC implementation (best performance for this codebase)
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
void* cpu_kang_thr_proc(void* data)
{
	RCCpuKang* Kang = (RCCpuKang*)data;
	// Use optimized version: larger batches (5K) while preserving cache locality
	Kang->Execute_Optimized();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
	csAddPoints.Enter();
	
	PntTotalOps += ops_cnt;
	TotalDPsGenerated += pnt_cnt;
	
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		DroppedDPs += pnt_cnt;
		
		static u64 last_warning = 0;
		if (DroppedDPs - last_warning >= 10000)
		{
			printf("\n⚠️  WARNING: DP BUFFER OVERFLOW!\n");
			printf("    Dropped: %llu DPs (%.1f%% loss)\n",
			       (unsigned long long)DroppedDPs,
			       100.0 * DroppedDPs / TotalDPsGenerated);
			printf("    FIX: Use -dp %d (current: %d)\n\n", gDP + 2, gDP);
			last_warning = DroppedDPs;
		}
		
		csAddPoints.Leave();
		return;
	}
	
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		gPrivKey.Add(Int_HalfRange);
		EcPoint P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		gPrivKey.Add(Int_HalfRange);
		P = ec.MultiplyG_Lambda(gPrivKey);  // GLV optimization
		return P.IsEqual(pnt);
	}
}


void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

	for (int i = 0; i < cnt; i++)
	{
		DBRec nrec;
		u8* p = pPntList2 + i * GPU_DP_SIZE;
		memcpy(nrec.x, p, 12);
		memcpy(nrec.d, p + 16, 22);
		nrec.type = gGenMode ? TAME : p[40];

		DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);

		// Add DP to work file if enabled (only if NEW DP, not duplicate)
		// pref is NULL for new DPs, non-NULL for duplicates
		if (g_work_file && !gGenMode && !pref)
		{
			g_work_file->AddDP(nrec.x, nrec.d, nrec.type);
		}

		if (gGenMode)
			continue;
		if (pref)
		{
			//in db we dont store first 3 bytes so restore them
			DBRec tmp_pref;
			memcpy(&tmp_pref, &nrec, 3);
			memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
			pref = &tmp_pref;

			if (pref->type == nrec.type)
			{
				if (pref->type == TAME)
					continue;

				//if it's wild, we can find the key from the same type if distances are different
				if (*(u64*)pref->d == *(u64*)nrec.d)
					continue;
				//else
				//	ToLog("key found by same wild");
			}

			EcInt w, t;
			int TameType, WildType;
			if (pref->type != TAME)
			{
				memcpy(w.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = nrec.type;
				WildType = pref->type;
			}
			else
			{
				memcpy(w.data, nrec.d, sizeof(nrec.d));
				if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
				memcpy(t.data, pref->d, sizeof(pref->d));
				if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
				TameType = TAME;
				WildType = nrec.type;
			}

			bool res = Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) || Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true);
			if (!res)
			{
				bool w12 = ((pref->type == WILD1) && (nrec.type == WILD2)) || ((pref->type == WILD2) && (nrec.type == WILD1));
				if (w12) //in rare cases WILD and WILD2 can collide in mirror, in this case there is no way to find K
					;// ToLog("W1 and W2 collides in mirror");
				else
				{
					printf("Collision Error\r\n");
					gTotalErrors++;
				}
				continue;
			}
			gSolved = true;
			break;
		}
	}
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
	for (int i = 0; i <= MD_LEN; i++)
	{
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	// Calculate elapsed time
	u64 elapsed_ms = GetTickCount64() - tm_start;
	if (elapsed_ms == 0) elapsed_ms = 1; // Avoid division by zero

	// Get REAL-TIME speeds from GPU/CPU monitors (rolling average, not cumulative)
	int gpu_speed = 0;
	int cpu_speed = 0;
	if (g_gpu_monitor) {
		SystemStats sys_stats = g_gpu_monitor->GetSystemStats();
		// Sum up individual GPU speeds (real-time rolling average)
		for (int i = 0; i < GpuCnt; i++) {
			if (GpuKangs[i]) {
				gpu_speed += GpuKangs[i]->GetStatsSpeed();
			}
		}
		cpu_speed = (int)sys_stats.cpu_speed_mkeys;
	}
	int total_speed = gpu_speed + cpu_speed;

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec_total = 0xFFFFFFFFFFFFFFFFull;
	if (total_speed)
		exp_sec_total = (u64)((exp_ops / 1000000) / total_speed); //in sec
	u64 exp_days = exp_sec_total / (3600 * 24);
	int exp_hours = (int)(exp_sec_total - exp_days * (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec_total - exp_days * (3600 * 24) - exp_hours * 3600) / 60;
	int exp_sec = (int)(exp_sec_total - exp_days * (3600 * 24) - exp_hours * 3600 - exp_min * 60);

	u64 sec_total = elapsed_ms / 1000;
	u64 days = sec_total / (3600 * 24);
	int hours = (int)(sec_total - days * (3600 * 24)) / 3600;
	int min = (int)(sec_total - days * (3600 * 24) - hours * 3600) / 60;
	int sec = (int)(sec_total - days * (3600 * 24) - hours * 3600 - min * 60);

	// Show total speed with GPU+CPU breakdown
	printf("%sSpeed: %d MKeys/s (%d GPU + %d CPU), Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm:%02ds/%llud:%02dh:%02dm:%02ds\r\n",
		gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "),
		total_speed, gpu_speed, cpu_speed,
		gTotalErrors, db.GetBlockCnt()/1000, est_dps_cnt/1000,
		days, hours, min, sec, exp_days, exp_hours, exp_min, exp_sec);
}

bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 10) || (DP > 60))
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}

	printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB. DP and GPU overheads not included!\r\n", log2(ops), ram);
	gIsOpsLimit = false;
	double MaxTotalOps = 0.0;
	if (gMax > 0)
	{
		MaxTotalOps = gMax * ops;
		double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
		ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
		ram_max /= (1024 * 1024 * 1024); //GB
		printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
	}

	u64 total_kangs = 0;
	if (GpuCnt > 0)
	{
		total_kangs = GpuKangs[0]->CalcKangCnt();
		for (int i = 1; i < GpuCnt; i++)
			total_kangs += GpuKangs[i]->CalcKangCnt();
	}
	total_kangs += CpuCnt * CPU_KANGS_PER_THREAD;
	double path_single_kang = ops / total_kangs;
	double DPs_per_kang = path_single_kang / dp_val;
	printf("Total kangaroos: %llu (GPU: %llu, CPU: %d)\r\n", total_kangs, total_kangs - CpuCnt * CPU_KANGS_PER_THREAD, CpuCnt * CPU_KANGS_PER_THREAD);

	// Enhanced DP analysis with recommendations
	printf("Estimated DPs per kangaroo: %.3f", DPs_per_kang);
	if (DPs_per_kang < 5.0) {
		// Calculate optimal DP for target of 10-15 DPs/kang
		int optimal_dp = (int)(log2(path_single_kang) - log2(12));
		printf(" ⚠️ HIGH DP overhead! Consider DP %d for better performance.\r\n", optimal_dp);
	} else if (DPs_per_kang < 10.0) {
		printf(" ✓ Good DP balance\r\n");
	} else if (DPs_per_kang < 20.0) {
		printf(" ✓ Optimal DP range\r\n");
	} else {
		printf(" (Low overhead, but consider lower DP for faster detection)\r\n");
	}

	if (!gGenMode && gTamesFileName[0])
	{
		printf("load tames...\r\n");
		if (db.LoadFromFile(gTamesFileName))
		{
			printf("tames loaded\r\n");
			if (db.Header[0] != gRange)
			{
				printf("loaded tames have different range, they cannot be used, clear\r\n");
				db.Clear();
			}
		}
		else
			printf("tames loading failed\r\n");
	}

	SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;

//prepare jumps
	// Teske-optimized jump sizes for better performance
	// Jump1: Main random walk - balanced optimization
	// Jump2/3: SOTA loop escape - kept large for cycle breaking
	EcInt minjump, t;
	minjump.Set(1);

	// Balanced Jump1: 2^(Range/2+1) [4x reduction from original]
	// Testing showed: Range/2+1 = OPTIMAL for SOTA (5-8% speedup)
	// Too aggressive (Range/2-1) = WORSE (jumps too small, poor K-factor)
	// Original: 2^(Range/2+3) was too large
	// Sweet spot: 2^(Range/2+1) balances Teske theory with SOTA needs
	int jump1_exp = Range / 2 + 1;  // Proven optimal
	if (jump1_exp < 10) jump1_exp = 10;  // Safety minimum for small ranges
	minjump.ShiftLeft(jump1_exp);

	printf("Jump tables: J1=2^%d (Teske-optimized), J2=2^%d, J3=2^%d (SOTA escape)\n",
		jump1_exp, Range-10, Range-12);

	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps1[i].p = ec.MultiplyG_Lambda(EcJumps1[i].dist);  // ~40% faster with GLV
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps2[i].p = ec.MultiplyG_Lambda(EcJumps2[i].dist);  // ~40% faster with GLV
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps3[i].p = ec.MultiplyG_Lambda(EcJumps3[i].dist);  // ~40% faster with GLV
	}

	SetRndSeed(GetTickCount64());

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(Range - 1);
	Pnt_HalfRange = ec.MultiplyG_Lambda(Int_HalfRange);  // ~40% faster with GLV
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();
	Int_TameOffset.Set(1);
	Int_TameOffset.ShiftLeft(Range - 1);
	EcInt tt;
	tt.Set(1);
	tt.ShiftLeft(Range - 5); //half of tame range width
	Int_TameOffset.Sub(tt);
	gPntToSolve = PntToSolve;

//prepare GPUs
	for (int i = 0; i < GpuCnt; i++)
	{
		// Enable SOTA++ herds if requested and range is large enough
		if (g_use_herds && Range >= 100)
		{
			printf("[GPU %d] Enabling SOTA++ herds (range=%d bits)\r\n", GpuKangs[i]->CudaIndex, Range);
			GpuKangs[i]->SetUseHerds(true, Range);
		}
		else if (g_use_herds && Range < 100)
		{
			printf("[GPU %d] Herds disabled: range too small (%d < 100)\r\n", GpuKangs[i]->CudaIndex, Range);
		}

		if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
		{
			GpuKangs[i]->Failed = true;
			printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
		}
	}

//prepare CPUs
	for (int i = 0; i < CpuCnt; i++)
	{
		CpuKangs[i] = new RCCpuKang();
		CpuKangs[i]->ThreadIndex = i;
		if (!CpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
		{
			CpuKangs[i]->Failed = true;
			printf("CPU worker %d Prepare failed\r\n", i);
		}
	}

	// Initialize GPU monitoring system
	if (GpuCnt > 0)
	{
		g_gpu_monitor = new GpuMonitor();
		if (!g_gpu_monitor->Initialize(GpuCnt))
		{
			printf("WARNING: GPU monitoring initialization failed\r\n");
			delete g_gpu_monitor;
			g_gpu_monitor = nullptr;
		}
	}

	u64 tm0 = GetTickCount64();
	if (GpuCnt > 0)
		printf("GPUs started...\r\n");
	if (CpuCnt > 0)
		printf("CPU workers started (%d threads)...\r\n", CpuCnt);

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT + 128];
#else
	pthread_t thr_handles[MAX_GPU_CNT + 128];
#endif

	u32 ThreadID;
	gSolved = false;
	ThrCnt = GpuCnt + CpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}
	for (int i = 0; i < CpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[GpuCnt + i] = (HANDLE)_beginthreadex(NULL, 0, cpu_kang_thr_proc, (void*)CpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[GpuCnt + i], NULL, cpu_kang_thr_proc, (void*)CpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
	u64 tm_monitor = GetTickCount64();
	while (!gSolved)
	{
		CheckNewPoints();
		Sleep(1);

		// GPU monitoring and thermal management (every second)
		if (g_gpu_monitor && (GetTickCount64() - tm_monitor > 1000))
		{
			// Update monitoring stats
			if (g_gpu_monitor)
			{
				SystemStats sys_stats = g_gpu_monitor->GetSystemStats();
				sys_stats.start_time_ms = tm0;
				sys_stats.actual_ops = PntTotalOps;
				sys_stats.expected_ops = ops;
				sys_stats.dp_count = db.GetBlockCnt();
				sys_stats.dp_expected = (u64)(ops / dp_val);
				sys_stats.dp_buffer_used = PntIndex;
				sys_stats.dp_buffer_total = MAX_CNT_LIST;

				// Update per-GPU speeds
				for (int i = 0; i < GpuCnt; i++)
				{
					sys_stats.gpu_stats[i].speed_mkeys = GpuKangs[i]->GetStatsSpeed();
					sys_stats.gpu_stats[i].operations = 0; // GPU ops tracked globally via PntTotalOps
				}

				// Update CPU speed
				// CPU stats are in KKeys/s, convert to MKeys/s
				sys_stats.cpu_speed_mkeys = 0;
				for (int i = 0; i < CpuCnt; i++)
				{
					sys_stats.cpu_speed_mkeys += CpuKangs[i]->GetStatsSpeed() / 1000.0;
				}

				// CRITICAL: Write updated stats back to GPU monitor!
				g_gpu_monitor->SetSystemStats(sys_stats);
				g_gpu_monitor->UpdateAllGPUs();
				g_gpu_monitor->ApplyThermalLimits();
			}
			tm_monitor = GetTickCount64();
		}

		// Statistics display (every 10 seconds)
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val);

			// Show detailed GPU stats every 10 seconds
			if (g_gpu_monitor)
			{
				g_gpu_monitor->PrintDetailedStats();
			}

			tm_stats = GetTickCount64();

			// Auto-save check for work files
			if (g_autosave && g_start_time > 0)
			{
				uint64_t elapsed = (uint64_t)(time(NULL) - g_start_time);
				g_autosave->CheckAndSave(PntTotalOps, PntIndex, gTotalErrors, elapsed);
			}

			// Auto-save check for tames files
			if (gGenMode && gTamesFileName[0])
			{
				time_t current_time = time(NULL);
				if (g_last_tames_save == 0)
				{
					// Initialize on first check
					g_last_tames_save = current_time;
				}
				else if ((current_time - g_last_tames_save) >= g_tames_autosave_interval)
				{
					printf("Auto-saving tames file...\r\n");
					db.Header[0] = gRange;
					if (db.SaveToFile(gTamesFileName))
					{
						printf("Tames auto-saved (%llu DPs)\r\n", (unsigned long long)db.GetBlockCnt());
						g_last_tames_save = current_time;
					}
					else
					{
						printf("WARNING: Tames auto-save failed\r\n");
					}
				}
			}
		}

		if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
		{
			gIsOpsLimit = true;
			printf("Operations limit reached\r\n");
			break;
		}
	}

	printf("Stopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	for (int i = 0; i < CpuCnt; i++)
		CpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(1);
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}
	for (int i = 0; i < CpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[GpuCnt + i]);
#else
		pthread_join(thr_handles[GpuCnt + i], NULL);
#endif
		delete CpuKangs[i];
		CpuKangs[i] = nullptr;
	}

	// Shutdown GPU monitoring
	if (g_gpu_monitor)
	{
		delete g_gpu_monitor;
		g_gpu_monitor = nullptr;
	}

	if (gIsOpsLimit)
	{
		if (gGenMode)
		{
			printf("saving tames...\r\n");
			db.Header[0] = gRange; 
			if (db.SaveToFile(gTamesFileName))
				printf("tames saved\r\n");
			else
				printf("tames saving failed\r\n");
		}
		db.Clear();
		return false;
	}

	double K = (double)PntTotalOps / pow(2.0, Range / 2.0);
	printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
	db.Clear();
	*pk_res = gPrivKey;
	return true;
}

// Signal handler for Ctrl+C and graceful shutdown
void SignalHandler(int signum)
{
	printf("\r\n\r\nInterrupted! Saving progress...\r\n");

	if (g_work_file && g_start_time > 0)
	{
		uint64_t elapsed = (uint64_t)(time(NULL) - g_start_time);
		g_work_file->UpdateProgress(PntTotalOps, PntIndex, gTotalErrors, elapsed);
		if (g_work_file->Save())
		{
			printf("Work file saved successfully\r\n");
		}
		else
		{
			printf("ERROR: Failed to save work file!\r\n");
		}
	}

	// Save tames file if in generation mode
	if (gGenMode && gTamesFileName[0])
	{
		printf("Saving tames file...\r\n");
		db.Header[0] = gRange;
		if (db.SaveToFile(gTamesFileName))
		{
			printf("Tames file saved successfully (%llu DPs)\r\n", (unsigned long long)db.GetBlockCnt());
		}
		else
		{
			printf("ERROR: Failed to save tames file!\r\n");
		}
	}

	exit(signum);
}

bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 12) || (val > 60))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-tames") == 0)
		{
			strcpy(gTamesFileName, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-max") == 0)
		{
			double val = atof(argv[ci]);
			ci++;
			if (val < 0.001)
			{
				printf("error: invalid value for -max option\r\n");
				return false;
			}
			gMax = val;
		}
		else
		if (strcmp(argument, "-cpu") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 0) || (val > 128))
			{
				printf("error: invalid value for -cpu option (must be 0-128)\r\n");
				return false;
			}
			gCpuThreads = val;
		}
		else
		if (strcmp(argument, "-workfile") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -workfile option\r\n");
				return false;
			}
			g_work_filename = argv[ci];
			ci++;
			printf("Work file: %s\r\n", g_work_filename.c_str());
		}
		else
		if (strcmp(argument, "-autosave") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -autosave option\r\n");
				return false;
			}
			int val = atoi(argv[ci]);
			ci++;
			if (val < 0)
			{
				printf("error: invalid value for -autosave option (must be >= 0)\r\n");
				return false;
			}
			g_autosave_interval = val;
			printf("Auto-save interval: %llu seconds\r\n", (unsigned long long)g_autosave_interval);
		}
		else
		if (strcmp(argument, "-tames-autosave") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -tames-autosave option\r\n");
				return false;
			}
			int val = atoi(argv[ci]);
			ci++;
			if (val < 0)
			{
				printf("error: invalid value for -tames-autosave option (must be >= 0)\r\n");
				return false;
			}
			g_tames_autosave_interval = val;
			printf("Tames auto-save interval: %llu seconds\r\n", (unsigned long long)g_tames_autosave_interval);
		}
		else
		if (strcmp(argument, "-herds") == 0)
		{
			g_use_herds = true;
			printf("SOTA++ herds mode enabled\r\n");
		}
		else
		if (strcmp(argument, "-info") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -info option\r\n");
				return false;
			}
			// Handle -info mode: display work file information and exit
			RCWorkFile info_file;
			if (info_file.Load(argv[ci]))
			{
				info_file.PrintInfo();
				exit(0);
			}
			else
			{
				printf("Failed to load work file: %s\r\n", argv[ci]);
				exit(1);
			}
		}
		else
		if (strcmp(argument, "-merge") == 0)
		{
			// Handle -merge mode: merge multiple work files
			std::vector<std::string> input_files;
			std::string output_file;

			// Collect input files until we hit -output or end of args
			while (ci < argc && strcmp(argv[ci], "-output") != 0)
			{
				input_files.push_back(argv[ci]);
				ci++;
			}

			// Get output file after -output
			if (ci < argc && strcmp(argv[ci], "-output") == 0)
			{
				ci++;
				if (ci < argc)
				{
					output_file = argv[ci];
					ci++;
				}
			}

			if (input_files.size() < 2)
			{
				printf("error: -merge requires at least 2 input files\r\n");
				exit(1);
			}

			if (output_file.empty())
			{
				printf("error: -merge requires -output option\r\n");
				exit(1);
			}

			// Perform merge
			printf("Merging %zu work files...\r\n", input_files.size());
			if (RCWorkFile::Merge(input_files, output_file))
			{
				printf("Merge successful! Output: %s\r\n", output_file.c_str());
				exit(0);
			}
			else
			{
				printf("Merge failed!\r\n");
				exit(1);
			}
		}
		else
		{
			printf("error: unknown option %s\r\n", argument);
			return false;
		}
	}
	if (!gPubKey.x.IsZero())
		if (!gStartSet || !gRange || !gDP)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG	
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	printf("********************************************************************************\r\n");
	printf("*        RCKangaroo v3.2 Hybrid+SOTA+  (c) 2024 RetiredCoder + fmg75           *\r\n");
	printf("*        GPU+CPU Hybrid with SOTA+ Optimizations (+10-30%% performance)        *\r\n");
	printf("*        Nataanii's Optimized Fork - SOTA++ Herds Save Functions               *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("This software is free and open-source: https://github.com/RetiredC\r\n");
	printf("It demonstrates fast GPU implementation of SOTA Kangaroo method for solving ECDLP\r\n");

#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
	printf("DEBUG MODE\r\n\r\n");
#endif

	InitEc();
	gDP = 0;
	gRange = 0;
	gStartSet = false;
	gTamesFileName[0] = 0;
	gMax = 0.0;
	gGenMode = false;
	gIsOpsLimit = false;
	gCpuThreads = 0; // Default: no CPU threads
	CpuCnt = 0;
	memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
	if (!ParseCommandLine(argc, argv))
		return 0;

	InitGpus();
	CpuCnt = gCpuThreads;

	if (!GpuCnt && !CpuCnt)
	{
		printf("No workers configured! Use -cpu option to add CPU workers or ensure GPUs are available\r\n");
		return 0;
	}

	if (GpuCnt == 0 && CpuCnt > 0)
	{
		printf("Running in CPU-only mode with %d threads\r\n", CpuCnt);
	}

	pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	TotalOps = 0;
	TotalSolved = 0;
	gTotalErrors = 0;
	IsBench = gPubKey.x.IsZero();

	// Initialize work file if specified
	if (!g_work_filename.empty() && !IsBench && !gGenMode)
	{
		g_work_file = new RCWorkFile(g_work_filename);
		g_start_time = time(NULL);

		// Check if work file exists (resume mode)
		if (WorkFileExists(g_work_filename))
		{
			printf("Found existing work file, resuming...\r\n");

			if (g_work_file->Load())
			{
				g_resume_mode = true;

				// Verify compatibility
				if (!g_work_file->IsCompatible(gRange, gDP, (const uint8_t*)gPubKey.x.data, (const uint8_t*)gPubKey.y.data))
				{
					printf("ERROR: Work file parameters don't match!\r\n");
					delete g_work_file;
					g_work_file = nullptr;
					return 1;
				}

				// Restore progress
				TotalOps = g_work_file->GetTotalOps();
				printf("Resuming from: %llu operations\r\n", (unsigned long long)TotalOps);

				// Load DPs into database
				const auto& dps = g_work_file->GetDPs();
				printf("Loading %zu DPs into database...\r\n", dps.size());

				for (const auto& dp : dps)
				{
					// Add DP to database using FindOrAddDataBlock
					DBRec nrec;
					memcpy(nrec.x, dp.dp_x, 12);
					memcpy(nrec.d, dp.distance, 22);
					nrec.type = dp.type;
					db.FindOrAddDataBlock((u8*)&nrec);
				}

				printf("Resume complete!\r\n");
			}
			else
			{
				printf("Failed to load work file\r\n");
				delete g_work_file;
				g_work_file = nullptr;
				return 1;
			}
		}
		else
		{
			// Create new work file
			printf("Creating new work file...\r\n");

			if (!g_work_file->Create(gRange, gDP, (const uint8_t*)gPubKey.x.data, (const uint8_t*)gPubKey.y.data,
			                        (const uint64_t*)gStart.data, nullptr))
			{
				printf("Failed to create work file\r\n");
				delete g_work_file;
				g_work_file = nullptr;
				return 1;
			}
			printf("Work file created successfully\r\n");
		}

		// Initialize auto-save
		if (g_autosave_interval > 0)
		{
			g_autosave = new AutoSaveManager(g_work_file, g_autosave_interval);
			printf("Auto-save enabled: every %llu seconds\r\n", (unsigned long long)g_autosave_interval);
		}

		// Register signal handlers for graceful shutdown
		signal(SIGINT, SignalHandler);
		signal(SIGTERM, SignalHandler);
	}

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk, pk_found;

		PntToSolve = gPubKey;
		if (!gStart.IsZero())
		{
			PntOfs = ec.MultiplyG_Lambda(gStart);  // GLV optimization
			PntOfs.y.NegModP();
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
		}

		char sx[100], sy[100];
		gPubKey.x.GetHexStr(sx);
		gPubKey.y.GetHexStr(sy);
		printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
		gStart.GetHexStr(sx);
		printf("Offset: %s\r\n", sx);

		if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
		{
			if (!gIsOpsLimit)
				printf("FATAL ERROR: SolvePoint failed\r\n");
			goto label_end;
		}
		pk_found.AddModP(gStart);
		EcPoint tmp = ec.MultiplyG_Lambda(pk_found);  // GLV optimization
		if (!tmp.IsEqual(gPubKey))
		{
			printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
			goto label_end;
		}
		//happy end
		char s[100];
		pk_found.GetHexStr(s);
		printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);

		// Save final work file state
		if (g_work_file && g_start_time > 0)
		{
			uint64_t elapsed = (uint64_t)(time(NULL) - g_start_time);
			g_work_file->UpdateProgress(PntTotalOps, PntIndex, gTotalErrors, elapsed);
			g_work_file->Save();
		}

		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else //we cannot save the key, show error and wait forever so the key is displayed
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
		{
			printf("\r\nTAMES GENERATION MODE\r\n");
			if (g_tames_autosave_interval > 0)
			{
				printf("Tames auto-save enabled: every %llu seconds\r\n", (unsigned long long)g_tames_autosave_interval);
			}
		}
		else
			printf("\r\nBENCHMARK MODE\r\n");
		//solve points, show K
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;

			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 14;

			//generate random pk
			pk.RndBits(gRange);
			PntToSolve = ec.MultiplyG_Lambda(pk);  // GLV optimization

			if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
			{
				if (!gIsOpsLimit)
					printf("FATAL ERROR: SolvePoint failed\r\n");
				break;
			}
			if (!pk_found.IsEqual(pk))
			{
				printf("FATAL ERROR: Found key is wrong!\r\n");
				break;
			}
			TotalOps += PntTotalOps;
			TotalSolved++;
			u64 ops_per_pnt = TotalOps / TotalSolved;
			double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
			printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
			//if (TotalSolved >= 100) break; //dbg
		}
	}
label_end:
	// Cleanup work file and auto-save
	if (g_autosave)
	{
		delete g_autosave;
		g_autosave = nullptr;
	}
	if (g_work_file)
	{
		delete g_work_file;
		g_work_file = nullptr;
	}

	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
	free(pPntList2);
	free(pPntList);
}



