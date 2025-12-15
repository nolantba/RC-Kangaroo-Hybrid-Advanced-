# Persistent Kernel Implementation Design (Week 2)

## Goal

Eliminate kernel launch overhead by combining KernelA, KernelB, KernelC into a single persistent kernel that loops internally. Target: **+6-8% performance improvement**.

## Current Architecture Analysis

### Traditional 3-Kernel Launch Pattern

**From host code analysis:**
```cpp
void CallGpuKernelABC(...) {
    // Launch 3 separate kernels per iteration
    KernelA<<<blocks, threads>>>(...);  // Random walk (hot path)
    KernelB<<<blocks, threads>>>(...);  // Distance tracking
    KernelC<<<blocks, threads>>>(...);  // DP collision detection
}

// Host loop
while (!StopFlag) {
    CallGpuKernelABC();  // 3 kernel launches
    // Check for collisions
    // Update stats
}
```

**Overhead per iteration:**
- KernelA launch: ~8-15 μs
- KernelB launch: ~5-10 μs
- KernelC launch: ~5-10 μs
- **Total: ~18-35 μs per iteration**

**At 100 iterations/sec:**
- Lost time: 1.8-3.5 ms/sec = **0.18-0.35% overhead**

**At 200 iterations/sec (aggressive):**
- Lost time: 3.6-7.0 ms/sec = **0.36-0.70% overhead**

**Hidden costs:**
- Warp eviction/reload between kernels
- L1/L2 cache invalidation
- Driver scheduling latency
- **Estimated total overhead: 3-5%**

### Kernel Responsibilities

**KernelA (Main workload ~95% of time):**
- Random walk computation (affine or Jacobian)
- DP detection
- Cycle detection (L1S2)
- Store jump history for KernelB

**KernelB (Distance tracking):**
- Accumulate distances from jump history
- Update kangaroo distance counters
- Detect overflow/wraparound

**KernelC (Collision detection):**
- Check DP matches between tame/wild
- Output collision candidates
- Loop exit point detection

## Persistent Kernel Design

### Architecture Overview

**Single persistent kernel that loops until stop signal:**

```cpp
__global__ void PersistentKernelABC(TKparams Kparams, volatile int* stop_flag) {
    // Initialize shared memory, constants
    __shared__ ...;

    // Initialize thread-local state
    u32 iteration = 0;

    // MAIN LOOP: Runs until host signals stop
    while (!(*stop_flag)) {
        // ====== PHASE A: Random Walk ======
        for (int step = 0; step < STEP_CNT; step++) {
            // Kangaroo walk logic (same as KernelA)
            // DP detection
            // Cycle detection
            // Store jump history
        }
        __syncthreads();  // Ensure all threads complete Phase A

        // ====== PHASE B: Distance Tracking ======
        // Distance accumulation (same as KernelB)
        __syncthreads();

        // ====== PHASE C: Collision Detection ======
        // DP collision check (same as KernelC)
        __syncthreads();

        // Increment iteration counter (for host polling)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            atomicAdd(&Kparams.iteration_count, 1);
        }

        // Optional: Check stop flag periodically
        if (iteration % 10 == 0) {
            if (*stop_flag) break;
        }
        iteration++;
    }
}
```

### Host-Side Changes

**Current (traditional):**
```cpp
while (!StopFlag) {
    CallGpuKernelABC();  // 3 launches
    cudaMemcpy(DP_results, ...);
    ProcessResults();
}
```

**Persistent kernel:**
```cpp
// Allocate stop flag in device memory
int* d_stop_flag;
cudaMalloc(&d_stop_flag, sizeof(int));
cudaMemset(d_stop_flag, 0, sizeof(int));

// Launch persistent kernel ONCE
PersistentKernelABC<<<blocks, threads>>>(..., d_stop_flag);

// Host loop: Poll for results periodically
u32 last_iteration = 0;
while (!StopFlag) {
    sleep(100ms);  // or condition variable

    // Check iteration count
    u32 current_iter;
    cudaMemcpy(&current_iter, &Kparams.iteration_count, sizeof(u32), D2H);

    if (current_iter > last_iteration + BATCH_SIZE) {
        // Copy results
        cudaMemcpy(DP_results, ...);
        ProcessResults();
        last_iteration = current_iter;
    }
}

// Signal kernel to stop
int stop = 1;
cudaMemcpy(d_stop_flag, &stop, sizeof(int), H2D);
cudaDeviceSynchronize();
```

## Implementation Details

### Memory Barriers and Synchronization

**Critical requirement:** Ensure result visibility to host

```cpp
// Before host reads results
if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence_system();  // Flush writes to system memory
}
__syncthreads();
```

**Stop flag checking:**
```cpp
volatile int* stop_flag;  // Must be volatile!
if (*stop_flag) break;    // No caching
```

### Iteration Batching

**Problem:** Copying results every iteration is expensive

**Solution:** Batch N iterations before host reads

```cpp
#define PERSISTENT_BATCH_SIZE 100  // Copy results every 100 iterations

// In kernel
if (iteration % PERSISTENT_BATCH_SIZE == 0) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        atomicAdd(&Kparams.ready_flag, 1);
    }
    __syncthreads();
}

// Host waits for ready_flag increment
```

### Backward Compatibility

**Compile-time switch:**
```cpp
#if USE_PERSISTENT_KERNELS
    // Persistent kernel path
    PersistentKernelABC<<<...>>>(...);
    // Polling loop
#else
    // Traditional path
    while (!StopFlag) {
        KernelA<<<...>>>(...);
        KernelB<<<...>>>(...);
        KernelC<<<...>>>(...);
    }
#endif
```

### Error Handling

**Problem:** Persistent kernel may hang on error

**Solutions:**
1. **Timeout watchdog:** Host-side timer kills kernel after N seconds
2. **Error flags:** Kernel sets error flag, host checks periodically
3. **Debug mode:** Traditional kernels for debugging

```cpp
// In kernel
if (error_detected) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Kparams.error_code = ERROR_XYZ;
    }
    __syncthreads();
    break;  // Exit kernel
}
```

## Performance Optimization

### Minimize Stop Flag Checks

**Problem:** Checking stop flag every iteration adds overhead

**Solution:** Check every N iterations
```cpp
#define STOP_CHECK_INTERVAL 10

if ((iteration & (STOP_CHECK_INTERVAL - 1)) == 0) {
    if (*stop_flag) break;
}
```

**Cost:** ~1 ns per check (negligible)
**Responsiveness:** Kernel exits within N iterations (~100ms @ 1000 iter/sec)

### Reduce __syncthreads() Overhead

**Current design:** 3 __syncthreads() per iteration (after A, B, C)

**Optimization:** Merge phases where possible
```cpp
// If KernelB and KernelC are independent:
// Phase A
...
__syncthreads();

// Phase B and C in parallel (if dependencies allow)
if (threadIdx.x < HALF_THREADS) {
    // KernelB logic
} else {
    // KernelC logic
}
__syncthreads();
```

**Analysis needed:** Check if B and C have dependencies

### Register Pressure Mitigation

**Persistent kernel has same register usage as traditional**, but:
- Longer lifetime → Better locality
- No eviction → L1 cache benefits

**No register overhead expected.**

## Expected Performance Impact

### Theoretical Speedup

**Kernel launch elimination:**
- 3 launches/iter × 10μs = 30μs saved
- At 100 iter/sec: 3ms/sec = **0.3% gain**

**Cache/occupancy benefits:**
- Warps stay resident: +2-3%
- L1/L2 locality: +1-2%

**Conservative estimate:** +3-5%
**Optimistic estimate:** +6-10%
**Target:** +6-8%

### Real-World Validation

**Test methodology:**
```bash
# Baseline (traditional)
make clean && make SM=86 USE_JACOBIAN=1 USE_PERSISTENT_KERNELS=0 -j
./rckangaroo -gpu 0 -range 80 ...  # Measure speed

# Persistent kernels
make clean && make SM=86 USE_JACOBIAN=1 USE_PERSISTENT_KERNELS=1 -j
./rckangaroo -gpu 0 -range 80 ...  # Compare speed

Expected:
- Traditional: 6.2 GKeys/s (with SOTA+)
- Persistent:  6.6-6.7 GKeys/s (+6-8%)
```

## Implementation Phases

### Phase 1: Infrastructure (1-2 days)

1. **Add device memory for control:**
   ```cpp
   int* stop_flag;
   u32* iteration_count;
   u32* error_code;
   ```

2. **Create PersistentKernelABC shell:**
   - Copy KernelA logic
   - Add while loop and stop flag check
   - Add __syncthreads() between phases

3. **Update host code:**
   - Polling loop instead of launch loop
   - Iteration-based result copying

### Phase 2: Integration (1 day)

1. **Integrate KernelB logic:**
   - Distance accumulation
   - Test correctness

2. **Integrate KernelC logic:**
   - Collision detection
   - Test correctness

3. **Full system test:**
   - Puzzle 75 correctness
   - Puzzle 80 performance

### Phase 3: Optimization (1-2 days)

1. **Tune batch size:**
   - Measure at 10, 50, 100, 200 iterations/batch
   - Find optimal for latency vs. throughput

2. **Minimize synchronization:**
   - Reduce __syncthreads() where safe
   - Optimize stop flag checks

3. **Validate on puzzle 90:**
   - Multi-hour stability test
   - Confirm +6-8% speedup

## Risks and Mitigations

### Risk 1: Kernel Hang
**Mitigation:**
- Timeout watchdog on host
- Error flags
- Debug mode with traditional kernels

### Risk 2: Result Corruption
**Mitigation:**
- __threadfence_system() before host reads
- Validate results on small puzzles first

### Risk 3: Responsiveness
**Mitigation:**
- Stop flag checked every 10 iterations
- Max exit latency: ~100ms (acceptable)

### Risk 4: No Performance Gain
**Mitigation:**
- Fallback to traditional kernels
- Re-measure overhead with profiler
- Investigate cache behavior

## Code Structure

### Files to Modify

1. **RCGpuCore.cu:**
   - Add `PersistentKernelABC` function
   - Wrap existing kernels in `#if !USE_PERSISTENT_KERNELS`

2. **GpuKang.cpp:**
   - Add persistent kernel launch logic
   - Polling loop for results

3. **defs.h:**
   - Already has `USE_PERSISTENT_KERNELS` flag

4. **Makefile:**
   - Already has `USE_PERSISTENT_KERNELS` build flag

### New Data Structures

```cpp
// In TKparams
struct TKparams {
    // ... existing fields ...

    // Persistent kernel control
    volatile int* stop_flag;
    u32* iteration_count;
    u32* error_code;
    u32* ready_flag;
};
```

## Success Criteria

**Phase 1:**
- ✓ Compiles with `USE_PERSISTENT_KERNELS=1`
- ✓ Runs without hanging
- ✓ Correct results on puzzle 75

**Phase 2:**
- ✓ Correct K-factor on puzzle 80
- ✓ Performance: +3-5% vs. traditional

**Phase 3:**
- ✓ Optimized: +6-8% vs. traditional
- ✓ Stable on multi-hour puzzle 90 run
- ✓ Combined with SOTA+: 6.6-7.0 GKeys/s

**Final target:**
- SOTA+ (K=1.02) + Persistent (+7%) = 6.6-6.8 GKeys/s effective

## Next Steps

After persistent kernels (Week 2), proceed to **Week 3: Kernel Micro-optimizations** for final 10-15% gain:
- ILP tuning
- Memory coalescing
- Register pressure fixes (from REGISTER_ANALYSIS.md)
- Prefetching

**Combined target (Week 4):** 8.5-9.0 GKeys/s to compete for Puzzle 135!
