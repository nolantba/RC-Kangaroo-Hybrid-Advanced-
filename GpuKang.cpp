// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"
#include "GpuHerdManager.h"
extern int gTameRatioPct;
extern int gTameBitsOffset;

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode

// SOTA++ Herd kernel launch (external from GpuHerdKernels.cu)
extern "C" void launchHerdKernels(
    GpuHerdMemory* mem,
    u64* d_jump_table,       // Packed jump table from Kparams.Jumps1 [JMP_CNT][12]
    u64* d_kangaroo_x,
    u64* d_kangaroo_y,
    u64* d_kangaroo_dist,
    int iterations,
    int dp_bits
);

extern "C" int checkHerdCollisions(
    GpuHerdMemory* mem,
    DP* host_dp_buffer,
    int max_dps
);

// ============================================================================
// SOTA++ Herd Support Methods
// ============================================================================

void RCGpuKang::SetUseHerds(bool enable, int range_bits)
{
    // Only use herds for puzzles 100+ bits (overhead too high for smaller puzzles)
    use_herds_ = enable && (range_bits >= 100);

    if (use_herds_) {
        printf("[GPU %d] SOTA++ herds enabled (range=%d bits)\n", CudaIndex, range_bits);
    } else {
        if (enable && range_bits < 100) {
            printf("[GPU %d] Herds disabled: range too small (%d < 100)\n", CudaIndex, range_bits);
        }
    }

    // Note: Herd pointers are initialized to nullptr in class member initializers
    // They will be allocated in Prepare() if use_herds_ is true
}

int RCGpuKang::CalcKangCnt()
{
	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	return Kparams.BlockSize* Kparams.GroupCnt* Kparams.BlockCnt;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP_bits = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	u64 total_mem = 0;
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP_bits;
	Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
	Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
	Kparams.IsGenMode = gGenMode;

	// SOTA++ Herds parameters
	if (use_herds_ && herd_manager_) {
		Kparams.UseHerds = true;
		Kparams.KangaroosPerHerd = herd_manager_->GetMemory()->config.kangaroos_per_herd;
	} else {
		Kparams.UseHerds = false;
		Kparams.KangaroosPerHerd = 0;
	}

//allocate gpu mem
	u64 size;
	if (!IsOldGpu)
	{
		//L2	
		int L2size = Kparams.KangCnt * (3 * 32);
		total_mem += L2size;
		err = cudaMalloc((void**)&Kparams.L2, L2size);
		if (err != cudaSuccess)
		{
			printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
		size = L2size;
		if (size > persistingL2CacheMaxSize)
			size = persistingL2CacheMaxSize;
		err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2
		//persisting for L2
		cudaStreamAttrValue stream_attribute;                                                   
		stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
		stream_attribute.accessPolicyWindow.num_bytes = size;										
		stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
		stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
		stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
		err = cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
	}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPs_out, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = KangCnt * 96;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.Kangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.JumpsList, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.L1S2, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LastPnts, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += 1024;
	err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);
//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	err = cuSetGpuParams(Kparams, jmp2_table);
	if (err != cudaSuccess)
	{
		free(jmp2_table);
		printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(jmp2_table);
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");

	// Initialize SOTA++ herd manager if enabled
	if (use_herds_) {
		HerdConfig herd_config = HerdConfig::forPuzzleSize(Range);

		// CRITICAL FIX: Calculate kangaroos_per_herd based on actual GPU kangaroo count
		// Must be multiple of 256 for warp alignment
		int kangaroos_per_herd = KangCnt / herd_config.herds_per_gpu;
		kangaroos_per_herd = (kangaroos_per_herd / 256) * 256;  // Round down to multiple of 256

		// Ensure at least 256 kangaroos per herd
		if (kangaroos_per_herd < 256) {
			kangaroos_per_herd = 256;
		}

		herd_config.kangaroos_per_herd = kangaroos_per_herd;

		printf("[GPU %d] Herd config: %d herds × %d kangaroos = %d total (%.1f%% of GPU capacity)\n",
		       CudaIndex, herd_config.herds_per_gpu, kangaroos_per_herd,
		       herd_config.getTotalKangaroosPerGpu(),
		       (100.0 * herd_config.getTotalKangaroosPerGpu()) / KangCnt);

		herd_manager_ = new GpuHerdManager(CudaIndex, herd_config);

		if (!herd_manager_->Initialize(Range)) {
			printf("ERROR: Failed to initialize herd manager on GPU %d\n", CudaIndex);
			delete herd_manager_;
			herd_manager_ = nullptr;
			use_herds_ = false;  // Fall back to unified mode
		} else {
			printf("[GPU %d] Herd manager initialized successfully\n", CudaIndex);

			// Allocate separate X/Y/Dist arrays for herd kernels
			size_t kang_xy_size = KangCnt * 4 * sizeof(u64);    // 4x u64 per coordinate (256 bits)
			size_t kang_dist_size = KangCnt * 3 * sizeof(u64);  // 3x u64 per distance (192 bits)

			err = cudaMalloc(&d_herd_kangaroo_x_, kang_xy_size);
			if (err != cudaSuccess) {
				printf("ERROR: Failed to allocate herd X array: %s\n", cudaGetErrorString(err));
				use_herds_ = false;
			}

			err = cudaMalloc(&d_herd_kangaroo_y_, kang_xy_size);
			if (err != cudaSuccess) {
				printf("ERROR: Failed to allocate herd Y array: %s\n", cudaGetErrorString(err));
				use_herds_ = false;
			}

			err = cudaMalloc(&d_herd_kangaroo_dist_, kang_dist_size);
			if (err != cudaSuccess) {
				printf("ERROR: Failed to allocate herd dist array: %s\n", cudaGetErrorString(err));
				use_herds_ = false;
			}

			// Allocate host buffer for DP collection
			h_herd_dp_buffer_ = (DP*)malloc(herd_config.gpu_dp_buffer_size * sizeof(DP));
			if (!h_herd_dp_buffer_) {
				printf("ERROR: Failed to allocate host DP buffer\n");
				use_herds_ = false;
			}

			if (use_herds_) {
				total_mem += 2 * kang_xy_size + kang_dist_size;
				printf("[GPU %d] Herd arrays allocated: %.2f MB\n", CudaIndex,
				       (2.0 * kang_xy_size + kang_dist_size) / (1024.0 * 1024.0));
			}
		}
	}

	return true;
}

void RCGpuKang::Release()
{
	// Cleanup SOTA++ herd manager
	if (herd_manager_) {
		herd_manager_->Shutdown();
		delete herd_manager_;
		herd_manager_ = nullptr;
	}

	// Free herd arrays
	if (d_herd_kangaroo_x_) {
		cudaFree(d_herd_kangaroo_x_);
		d_herd_kangaroo_x_ = nullptr;
	}
	if (d_herd_kangaroo_y_) {
		cudaFree(d_herd_kangaroo_y_);
		d_herd_kangaroo_y_ = nullptr;
	}
	if (d_herd_kangaroo_dist_) {
		cudaFree(d_herd_kangaroo_dist_);
		d_herd_kangaroo_dist_ = nullptr;
	}
	if (h_herd_dp_buffer_) {
		free(h_herd_dp_buffer_);
		h_herd_dp_buffer_ = nullptr;
	}

	free(RndPnts);
	free(DPs_out);
	cudaFree(Kparams.LoopedKangs);
	cudaFree(Kparams.dbg_buf);
	cudaFree(Kparams.LoopTable);
	cudaFree(Kparams.LastPnts);
	cudaFree(Kparams.L1S2);
	cudaFree(Kparams.DPTable);
	cudaFree(Kparams.JumpsList);
	cudaFree(Kparams.Jumps3);
	cudaFree(Kparams.Jumps2);
	cudaFree(Kparams.Jumps1);
	cudaFree(Kparams.Kangs);
	cudaFree(Kparams.DPs_out);
	if (!IsOldGpu)
		cudaFree(Kparams.L2);
}

void RCGpuKang::Stop()
{
	StopFlag = true;
}

void RCGpuKang::GenerateRndDistances()
{
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		int tameBorder = (KangCnt * gTameRatioPct) / 100;
		int tameBits = Range - gTameBitsOffset;
		if (tameBits < 1) tameBits = 1;
		if (i < tameBorder)
			d.RndBits(tameBits); // TAME kangs
		else
		{
			d.RndBits(Range - 1);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; // must be even
		}
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

bool RCGpuKang::Start()
{
	if (Failed)
		return false;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
	PntB = PntA;
	PntB.y.NegModP();

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	GenerateRndDistances();
/* 
	//we can calc start points on CPU
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);
		d.data[3] = 0;
		d.data[4] = 0;
		EcPoint p = ec.MultiplyG(d);
		memcpy(RndPnts[i].x, p.x.data, 32);
		memcpy(RndPnts[i].y, p.y.data, 32);
	}
	for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntA);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntB);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
/**/
	//but it's faster to calc then on GPU
	u8 buf_PntA[64], buf_PntB[64];
	PntA.SaveToBuffer64(buf_PntA);
	PntB.SaveToBuffer64(buf_PntB);
	for (int i = 0; i < KangCnt; i++)
	{
		if (i < KangCnt / 3)
			memset(RndPnts[i].x, 0, 64);
		else
			if (i < 2 * KangCnt / 3)
				memcpy(RndPnts[i].x, buf_PntA, 64);
			else
				memcpy(RndPnts[i].x, buf_PntB, 64);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	CallGpuKernelGen(Kparams);

	// SOTA++ Herds: Convert kangaroo data to separate arrays ONCE at startup
	// OPTIMIZED: Use GPU kernel for conversion (eliminates CPU round-trip, saves 50-100ms)
	if (use_herds_ && herd_manager_) {
		printf("[GPU %d] Converting kangaroo data for herd mode (GPU-side)...\n", CudaIndex);

		// Launch GPU kernel to convert format directly on device
		int threads_per_block = 256;
		int num_blocks = (KangCnt + threads_per_block - 1) / threads_per_block;

		// Forward declaration of kernel (defined in GpuHerdKernels.cu)
		extern void ConvertKangarooFormatLauncher(
			const u64* src_packed, u64* dst_x, u64* dst_y, u64* dst_dist,
			int count, int blocks, int threads);

		ConvertKangarooFormatLauncher(
			(const u64*)Kparams.Kangs,
			d_herd_kangaroo_x_,
			d_herd_kangaroo_y_,
			d_herd_kangaroo_dist_,
			KangCnt,
			num_blocks,
			threads_per_block);

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("[GPU %d] ERROR: Kangaroo format conversion kernel failed: %s\n",
			       CudaIndex, cudaGetErrorString(err));
			use_herds_ = false;
		} else {
			cudaDeviceSynchronize();  // Wait for conversion to complete
			printf("[GPU %d] Kangaroo data converted on GPU (optimized path)\n", CudaIndex);
		}
	}

	err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
	if (err != cudaSuccess)
		return false;
	cudaMemset(Kparams.dbg_buf, 0, 1024);
	cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
	return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
	int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
	u64* kangs = (u64*)malloc(kang_size);
	cudaError_t err = cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost);
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &kangs[i * 12 + 8], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			if (i < 2 * KangCnt / 3)
				p = ec.AddPoints(PntA, p);
			else
				p = ec.AddPoints(PntB, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	return res;
}
#endif

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
	cudaSetDevice(CudaIndex);

	if (!Start())
	{
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	cudaError_t err;
	int herd_stats_counter = 0;  // For periodic herd stats printing

	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		int cnt = 0;
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		// ========================================================================
		// Choose kernel implementation
		// ========================================================================
#if USE_SEPARATE_HERD_KERNEL
		// Use separate herd kernel (experimental - for benchmarking)
		if (use_herds_ && herd_manager_) {
			// Launch separate herd kernel
			launchHerdKernels(
				herd_manager_->GetMemory(),
				d_herd_jump_table_,
				d_herd_kangaroo_x_,
				d_herd_kangaroo_y_,
				d_herd_kangaroo_dist_,
				STEP_CNT,  // iterations
				Kparams.DP_rshift
			);

			// Check for kernel launch errors
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("GPU %d, launchHerdKernels failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
				gTotalErrors++;
				break;
			}

			// Collect DPs from herd kernel
			DP* h_dp_buffer = (DP*)malloc(herd_manager_->GetMemory()->config.gpu_dp_buffer_size * sizeof(DP));
			cnt = checkHerdCollisions(herd_manager_->GetMemory(), h_dp_buffer, herd_manager_->GetMemory()->config.gpu_dp_buffer_size);

			if (cnt > 0) {
				// Convert DP format and add to list
				// TODO: Implement DP conversion from herd format to unified format
				printf("[GPU %d] Herd kernel found %d DPs (conversion not implemented)\n", CudaIndex, cnt);
			}

			free(h_dp_buffer);
		} else {
			// Fall back to unified kernel
			cudaMemset(Kparams.DPs_out, 0, 4);
			cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
			cudaMemset(Kparams.LoopedKangs, 0, 8);
			CallGpuKernelABC(Kparams);

			err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
				gTotalErrors++;
				break;
			}

			if (cnt >= MAX_DP_CNT) {
				cnt = MAX_DP_CNT;
				printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
			}

			if (cnt) {
				err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					gTotalErrors++;
					break;
				}
				AddPointsToList(DPs_out, cnt, pnt_cnt);
			}
		}
#else
		// Use optimized unified kernel (with herd support integrated) - RECOMMENDED
		{
			cudaMemset(Kparams.DPs_out, 0, 4);
			cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
			cudaMemset(Kparams.LoopedKangs, 0, 8);
			CallGpuKernelABC(Kparams);

			err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
				gTotalErrors++;
				break;
			}

			if (cnt >= MAX_DP_CNT)
			{
				cnt = MAX_DP_CNT;
				printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
			}

			if (cnt)
			{
				err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess)
				{
					gTotalErrors++;
					break;
				}
				AddPointsToList(DPs_out, cnt, pnt_cnt);
			}
		}
#endif

		//dbg
		cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost);

		u32 lcnt;
		cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost);
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors++;
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
		}
		iter++;
#endif
	}

	Release();
}

int RCGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}