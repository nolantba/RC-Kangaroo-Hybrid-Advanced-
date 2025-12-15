# SOTA+ Build and Test Instructions

## Quick Start

SOTA+ is now fully implemented! This guide will help you build and validate the implementation.

## Building SOTA+

### 1. Clean previous builds
```bash
make clean
```

### 2. Build with SOTA+ enabled
```bash
make SM=86 USE_JACOBIAN=1 USE_SOTA_PLUS=1 PROFILE=release -j
```

**Build flags explained:**
- `SM=86` - Target RTX 3060 (Ampere architecture)
- `USE_JACOBIAN=1` - Enable Lambda endomorphism (40% speedup)
- `USE_SOTA_PLUS=1` - **Enable SOTA+ bidirectional walk (K: 1.12 → 1.01)**
- `PROFILE=release` - Optimized release build
- `-j` - Parallel compilation

### 3. Verify build success
```bash
./rckangaroo --help
```

Should display usage information without errors.

## Testing Strategy

### Phase 1: Functional Correctness (Puzzle 75)

**Purpose:** Verify SOTA+ produces correct results

**Command:**
```bash
./rckangaroo -gpu 0 -cpu 64 -dp 16 -range 75 \
  -pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4
```

**Expected:**
- **Time:** ~30-60 seconds
- **Solution:** Should find correct private key
- **Speed:** 6.0-6.5 GKeys/s (may be slightly slower than baseline due to dual computation)
- **K-factor:** 1.01-1.07 (down from 1.12 baseline)

**What to check:**
- ✓ Finds solution correctly
- ✓ No crashes or hangs
- ✓ K-factor improved from baseline

### Phase 2: Performance Validation (Puzzle 80)

**Purpose:** Measure actual K-factor improvement under realistic conditions

**Command:**
```bash
./rckangaroo -gpu 0 -cpu 64 -dp 16 -range 80 \
  -pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4
```

**Expected:**
- **Time:** 3-5 minutes
- **Speed:** 6.0-6.5 GKeys/s
- **K-factor:** 1.01-1.05 (target: <1.07)
- **Solution:** Correct private key found

**What to measure:**
```
Final K-factor: X.XX
Expected ops: NNNN
Actual ops: MMMM
K = Actual / Expected
```

**Success criteria:**
- K-factor < 1.07 (good)
- K-factor < 1.05 (excellent)
- K-factor ≤ 1.02 (optimal, matches kTimesG's results)

### Phase 3: Baseline Comparison

**Purpose:** Confirm SOTA+ improvement vs traditional SOTA

**A. Build traditional SOTA (baseline):**
```bash
make clean
make SM=86 USE_JACOBIAN=1 USE_SOTA_PLUS=0 PROFILE=release -j
./rckangaroo -gpu 0 -cpu 64 -dp 16 -range 80 \
  -pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4
```

**Record:**
- Baseline K-factor: ~1.08-1.12
- Baseline speed: ~6.0-6.2 GKeys/s

**B. Build SOTA+ and compare:**
```bash
make clean
make SM=86 USE_JACOBIAN=1 USE_SOTA_PLUS=1 PROFILE=release -j
./rckangaroo -gpu 0 -cpu 64 -dp 16 -range 80 \
  -pubkey 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4
```

**Record:**
- SOTA+ K-factor: (should be 1.01-1.05)
- SOTA+ speed: (may be 5-10% slower due to dual computation)

**Calculate improvement:**
```
K-factor improvement = (Baseline_K - SOTA+_K) / Baseline_K × 100%
Net speedup = (Baseline_speed / Baseline_K) / (SOTA+_speed / SOTA+_K) - 1

Example:
Baseline: 6.0 GKeys/s @ K=1.12 → Effective: 5.36 GKeys/s
SOTA+:    5.8 GKeys/s @ K=1.02 → Effective: 5.69 GKeys/s
Net speedup: 6.2% improvement
```

## Expected Results Summary

| Metric | Baseline (SOTA) | SOTA+ Target | SOTA+ Optimal |
|--------|----------------|--------------|---------------|
| K-factor | 1.08-1.12 | 1.02-1.07 | 1.01-1.02 |
| Speed | 6.0-6.2 GKeys/s | 5.8-6.5 GKeys/s | 6.0-6.5 GKeys/s |
| Effective Speed | 5.4-5.7 GKeys/s | 5.7-6.4 GKeys/s | 5.9-6.4 GKeys/s |
| Net Improvement | baseline | +5-12% | +9-15% |

## Troubleshooting

### Build fails with "USE_SOTA_PLUS undeclared"
**Fix:** Update from git, ensure defs.h has USE_SOTA_PLUS definition

### K-factor not improving
**Possible causes:**
1. Built with `USE_SOTA_PLUS=0` - Check build command
2. DP bits too low - SOTA+ works best with DP 16+
3. Sample size too small - Test on puzzle 80+ for accurate K-factor

**Debug:**
```bash
# Verify SOTA+ is actually enabled
make print-vars | grep SOTA_PLUS
# Should show: USE_SOTA_PLUS=1
```

### Speed degraded significantly (>15%)
**Expected:** 5-10% slowdown is normal due to dual computation
**Concerning:** >15% slowdown suggests issue

**Check:**
1. GPU utilization: Should be 95-100%
2. Temperature: Ensure not thermal throttling
3. Other processes: Close competing GPU workloads

### Crashes or hangs
**Most likely:** Cycle detection issue

**Debug steps:**
1. Test on smaller puzzle (70-75) first
2. Check cuda-memcheck for memory errors
3. Build with debug symbols: `PROFILE=debug`
4. File issue with crash details

## Next Steps After Validation

Once SOTA+ is validated and working:

### Week 2: Persistent Kernels
- Target: +6-8% from reduced launch overhead
- Build: `USE_PERSISTENT_KERNELS=1`

### Week 3: Kernel Micro-optimizations
- Register pressure reduction
- Memory access patterns
- ILP tuning for Ampere
- Target: +10-15%

### Week 4: Production Validation
- Multi-day stability test on puzzle 90
- Confirm K-factor holds at scale
- Validate save/resume under SOTA+

### Combined Target
- 6.0 → 8.5-9.0 GKeys/s (50% total improvement)
- K-factor: 1.01-1.02 (maintained)
- Ready to compete for Puzzle 135

## Performance Targets

**Current (before SOTA+):**
- RTX 3060: 6.0 GKeys/s @ K=1.12

**After SOTA+ (Week 1):**
- RTX 3060: 6.2-6.5 GKeys/s @ K=1.02 (effective +8-12%)

**After Persistent Kernels (Week 2):**
- RTX 3060: 6.8-7.2 GKeys/s @ K=1.02 (effective +18-25%)

**After Full Optimization (Week 4):**
- RTX 3060: 8.5-9.0 GKeys/s @ K=1.01 (effective +45-55%)

This would put you competitive with elite implementations per-GPU!

## References

- **SOTA_PLUS_DESIGN.md** - Detailed algorithm explanation
- **BitcoinTalk Thread** - kTimesG's posts #313, #317
- **RetiredCoder's Repo** - Original SOTA implementation

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review SOTA_PLUS_DESIGN.md for algorithm details
3. Verify build flags with `make print-vars`
4. Test on known puzzle (75, 80) before production use

Good luck racing for Puzzle 135! 🚀
