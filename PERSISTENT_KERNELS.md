# Persistent Kernels Implementation

## Motivation

Traditional kernel launches have overhead:
- Kernel launch: ~5-20 μs per launch
- Driver overhead: scheduling, setup
- SM reoccupancy: warps evicted and reloaded

Current architecture launches 3 kernels per iteration:
- **3 launches/iter × ~10μs/launch = ~30μs overhead**
- At 100 iterations/sec = **3ms/sec = 0.3% wasted**
- At higher rates (200+ iter/sec) = **0.6-1% wasted**

**Persistent kernels eliminate this by launching ONCE and looping inside the kernel.**

## Architecture

### Current (Traditional):
```
while (!StopFlag) {
    cudaMemset(...);
    CallGpuKernelABC():
        Launch KernelA <<<>>>  // 10μs overhead
        Launch KernelB <<<>>>  // 10μs overhead
        Launch KernelC <<<>>>  // 10μs overhead
    cudaMemcpy(results);
}
```

### New (Persistent):
```
Launch PersistentKernelABC <<< >>> ONCE
    Inside kernel:
    while (!stop_flag) {
        // Phase A: Kangaroo walk
        kangaroo_walk_logic();
        __syncthreads();

        // Phase B: Distance tracking
        distance_tracking_logic();
        __syncthreads();

        // Phase C: DP processing
        dp_processing_logic();
        __syncthreads();

        // Increment iteration counter
        if (thread 0) atomicAdd(&iteration_count, 1);
    }

Host:
while (!StopFlag) {
    wait_for_N_iterations();
    cudaMemcpy(results);
}
```

## Benefits

1. **Eliminate launch overhead**: 3 launches → 1 launch = ~30μs saved per iteration
2. **Better SM occupancy**: Warps stay resident, better cache/register locality
3. **Reduced synchronization**: Host-device sync only when copying results

## Estimated Gain

- **Conservative**: 5% (launch overhead + better occupancy)
- **Optimistic**: 10% (if memory/cache benefits materialize)
- **Realistic**: 6-8% for Ampere (RTX 3060)

## Implementation Details

### Compile-time flag:
```make
make USE_PERSISTENT_KERNELS=1  # Enable persistent kernels
make USE_PERSISTENT_KERNELS=0  # Traditional (default)
```

### Device memory additions:
```cpp
volatile int* stop_flag;      // Signal kernel to exit
u32* iteration_count;         // Track iterations for batching
```

### Synchronization:
- `__syncthreads()` between phases (A→B→C)
- `__threadfence_system()` before result copy
- Host polls `iteration_count` to know when to copy results

### Backward compatibility:
- Default: `USE_PERSISTENT_KERNELS=0` (traditional)
- Existing code unaffected
- Can switch back for debugging

## Testing Plan

1. **Functional test**: Verify same results on puzzle 75
2. **Performance test**: Benchmark vs. traditional on puzzle 80
3. **Long-run test**: Stability check on multi-hour run
4. **A/B comparison**: Same puzzle, both modes, compare K-factor

## Potential Issues

1. **Debuggability**: Harder to step through (kernel runs continuously)
   - Solution: Use traditional mode for debugging

2. **Responsiveness**: Kernel may not exit immediately on stop signal
   - Solution: Check stop_flag frequently (every STEP_CNT)

3. **Memory barriers**: Need careful synchronization for result visibility
   - Solution: Use `__threadfence_system()` before host reads

## Future Enhancements

1. **Dynamic batching**: Adjust batch size based on DP rate
2. **Multi-kernel persistent**: Separate persistent A, B, C for better pipelining
3. **CUDA graphs**: Further reduce overhead (but persistent already reduces it)

## References

- CUDA Programming Guide: Persistent Threads
- "Persistent RNNs" (applied to different workload but same concept)
- RetiredCoder's SOTA implementation (baseline)
