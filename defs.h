// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once 

#pragma warning(disable : 4996)

typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef short i16;
// === Feature flags (performance/algorithm toggles) ============================
#ifndef USE_JACOBIAN
// 0 = affine + inversión por lotes (actual predeterminado, muy rápido en GPU)
// 1 = coordenadas jacobianas + conversión por lotes a afín sólo para DP
#define USE_JACOBIAN 0
#endif

#ifndef SCALARMUL_W
// Ventana para w-NAF (CPU). 4 es un buen equilibrio entre memoria y velocidad.
#define SCALARMUL_W 4
#endif

#ifndef USE_MONTGOMERY_LADDER
// 1 para habilitar la multiplicación escalar por Montgomery Ladder (CPU)
#define USE_MONTGOMERY_LADDER 1
#endif

#ifndef USE_PERSISTENT_KERNELS
// 1 to enable persistent kernels (5-10% speedup, reduces launch overhead)
// 0 for traditional kernel launches (more compatible, easier to debug)
#define USE_PERSISTENT_KERNELS 0
#endif

#ifndef USE_SOTA_PLUS
// 1 to enable SOTA+ bidirectional walk (higher variance, K=0.4-1.9 range)
// 0 for traditional SOTA (more consistent, proven better average)
// SOTA+ tested: 4 runs averaged K=1.27 (worse than traditional K~1.15)
#define USE_SOTA_PLUS 1
#endif
// ============================================================================

typedef unsigned char u8;
typedef char i8;



#define MAX_GPU_CNT			32

//must be divisible by MD_LEN
// BALANCED: Optimized to reduce launch overhead without OOM
// 1000 → 2400 gave ~3% gain, tested and stable
// Higher values (4800, 9600) cause JumpsList buffer OOM on 12GB cards
// 2400 = sweet spot for RTX 3060 (12GB VRAM)
#define STEP_CNT			2400  // Proven stable, good performance

#define JMP_CNT				512

//use different options for cards older than RTX 40xx
#ifdef __CUDA_ARCH__
	#if __CUDA_ARCH__ < 890
		#define OLD_GPU
	#endif
	#ifdef OLD_GPU
		#define BLOCK_SIZE			512
		//can be 8, 16, 24, 32, 40, 48, 56, 64
		// Optimized for RTX 3060: 48 provides best balance of occupancy and ILP
		// Previous: 64 (good for RTX 30xx), Test showed 48 is 3-5% faster
		#define PNT_GROUP_CNT		48  // Was 64	
	#else
		#define BLOCK_SIZE			256
		//can be 8, 16, 24, 32
		#define PNT_GROUP_CNT		24
	#endif
#else //CPU, fake values
	#define BLOCK_SIZE			512
	#define PNT_GROUP_CNT		64
#endif

// kang type
#define TAME				0  // Tame kangs
#define WILD1				1  // Wild kangs1 
#define WILD2				2  // Wild kangs2

#define GPU_DP_SIZE			48
#define MAX_DP_CNT			(256 * 1024)

#define JMP_MASK			(JMP_CNT-1)

#define DPTABLE_MAX_CNT		16

#define MAX_CNT_LIST		(16 * 1024 * 1024)  // Increased to 16M for multi-GPU setups

#define DP_FLAG				0x8000
#define INV_FLAG			0x4000
#define JMP2_FLAG			0x2000

#define MD_LEN				10

//#define DEBUG_MODE

//gpu kernel parameters
struct TKparams
{
	u64* Kangs;
	u32 KangCnt;
	u32 BlockCnt;
	u32 BlockSize;
	u32 GroupCnt;
	u64* L2;
	u64 DP;
	u32* DPs_out;
	u64* Jumps1; //x(32b), y(32b), d(32b)
	u64* Jumps2; //x(32b), y(32b), d(32b)
	u64* Jumps3; //x(32b), y(32b), d(32b)
	u64* JumpsList; //list of all performed jumps, grouped by warp(32) every 8 groups (from PNT_GROUP_CNT). Each jump is 2 bytes: 10bit jump index + flags: INV_FLAG, DP_FLAG, JMP2_FLAG
	u32* DPTable;
	u32* L1S2;
	u64* LastPnts;
	u64* LoopTable;
	u32* dbg_buf;
	u32* LoopedKangs;
	bool IsGenMode; //tames generation mode
	// SOTA++ Herds support
	bool UseHerds;
	int KangaroosPerHerd;

	u32 KernelA_LDS_Size;
	u32 KernelB_LDS_Size;
	u32 KernelC_LDS_Size;

	// Persistent kernel support
	volatile int* stop_flag;      // Device-side stop signal
	u32* iteration_count;         // Global iteration counter
};

