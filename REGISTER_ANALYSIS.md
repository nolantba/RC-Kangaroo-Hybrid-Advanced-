# SOTA+ Register Pressure Analysis

## Executive Summary

SOTA+ adds ~6 additional u64[4] arrays (192 bytes) per thread compared to traditional SOTA. On Ampere (sm_86), this may reduce occupancy if not optimized. This document analyzes register usage and proposes optimizations to maintain high occupancy.

## Register Usage Breakdown

### Traditional SOTA (Baseline)

**Per-thread stack arrays:**
```cpp
__align__(16) u64 x0[4];      // 32 bytes (4 regs)
__align__(16) u64 y0[4];      // 32 bytes (4 regs)
__align__(16) u64 dxs[4];     // 32 bytes (4 regs)
__align__(16) u64 x[4];       // 32 bytes (4 regs) - outer scope
__align__(16) u64 y[4];       // 32 bytes (4 regs) - outer scope
__align__(16) u64 tmp[4];     // 32 bytes (4 regs) - outer scope
__align__(16) u64 tmp2[4];    // 32 bytes (4 regs) - outer scope
__align__(16) u64 jmp_x[4];   // 32 bytes (4 regs) - shared/reused
__align__(16) u64 jmp_y[4];   // 32 bytes (4 regs) - shared/reused
```

**Scalar variables:**
```cpp
u64 dp_mask64;                // 1 reg
u16 jmp_ind;                  // 1 reg
u32 inv_flag;                 // 1 reg
u32 L1S2;                     // 1 reg
int step_ind;                 // 1 reg
```

**Estimated baseline:** ~35-40 registers per thread (plus compiler overhead)

### SOTA+ Additional Registers

**New arrays in SOTA+ path:**
```cpp
__align__(16) u64 jmp_y_neg[4];    // 32 bytes (4 regs)
__align__(16) u64 slope_plus[4];   // 32 bytes (4 regs)
__align__(16) u64 slope_minus[4];  // 32 bytes (4 regs)
__align__(16) u64 x3_plus[4];      // 32 bytes (4 regs)
__align__(16) u64 x3_minus[4];     // 32 bytes (4 regs)
__align__(16) u64 y3_plus[4];      // 32 bytes (4 regs) - declared but unused!
__align__(16) u64 y3_minus[4];     // 32 bytes (4 regs) - declared but unused!
```

**New scalars:**
```cpp
int zeros_plus;               // 1 reg
int zeros_minus;              // 1 reg
bool use_plus;                // 1 reg
```

**Additional overhead:** +28-32 registers

**SOTA+ estimated total:** ~63-72 registers per thread

## Ampere (sm_86) Occupancy Analysis

### RTX 3060 Specifications
- **Register file per SM:** 65,536 registers
- **Max threads per SM:** 1,536
- **Max registers per thread:** 255
- **Block size:** 256 threads (from code)

### Occupancy Calculator

**Traditional SOTA (40 regs/thread):**
- Registers needed per block: 256 threads × 40 regs = 10,240 regs
- Blocks per SM: 65,536 / 10,240 = 6.4 → **6 blocks**
- Active threads per SM: 6 × 256 = **1,536 threads (100% occupancy)**

**SOTA+ (70 regs/thread):**
- Registers needed per block: 256 threads × 70 regs = 17,920 regs
- Blocks per SM: 65,536 / 17,920 = 3.66 → **3 blocks**
- Active threads per SM: 3 × 256 = **768 threads (50% occupancy)**

**⚠️ CRITICAL: SOTA+ may reduce occupancy to 50% if register usage is not optimized!**

## Optimization Strategies

### Strategy 1: Eliminate Unused Variables (IMMEDIATE)

**Problem:** `y3_plus` and `y3_minus` arrays are declared but never used!

**Current code (lines 210, 236-241, 243-248):**
```cpp
__align__(16) u64 y3_plus[4], y3_minus[4];  // ← Declared

if (use_plus) {
    Copy_u64_x4(x, x3_plus);
    // Compute Y3 for plus direction
    SubModP(y, x0, x);          // ← Writes to 'y' (outer scope)
    MulModP(y, y, slope_plus);
    SubModP(y, y, y0);
} else {
    Copy_u64_x4(x, x3_minus);
    jmp_ind |= INV_FLAG;
    // Compute Y3 for minus direction
    SubModP(y, x0, x);          // ← Writes to 'y' (outer scope)
    MulModP(y, y, slope_minus);
    SubModP(y, y, y0);
}
```

**Fix:** Remove `y3_plus` and `y3_minus` declarations entirely.

**Savings:** 8 registers (2 × u64[4])

### Strategy 2: Reuse Existing Temporaries

**Problem:** `slope_plus` and `slope_minus` are only used for Y3 computation, which could reuse `tmp` and `tmp2`.

**Current:**
```cpp
__align__(16) u64 slope_plus[4], slope_minus[4];
```

**Optimized:**
```cpp
// Reuse outer-scope tmp/tmp2 for slopes
// slope_plus → tmp
// slope_minus → tmp2
```

**Implementation:**
```cpp
// Compute slope for P + Jump (reuse tmp as slope_plus)
SubModP(tmp2, y0, jmp_y);
MulModP(tmp, tmp2, dxs);        // tmp = slope_plus
SqrModP(tmp2, tmp);
SubModP(x3_plus, tmp2, jmp_x);
SubModP(x3_plus, x3_plus, x0);

// Compute slope for P - Jump (reuse tmp2 as slope_minus)
Copy_u64_x4(jmp_y_neg, jmp_y);
NegModP(jmp_y_neg);
SubModP(tmp, y0, jmp_y_neg);    // Reuse tmp (slope_plus no longer needed)
MulModP(tmp2, tmp, dxs);        // tmp2 = slope_minus
SqrModP(tmp, tmp2);
SubModP(x3_minus, tmp, jmp_x);
SubModP(x3_minus, x3_minus, x0);

// Choose best
int zeros_plus = __clzll(x3_plus[3]);
int zeros_minus = __clzll(x3_minus[3]);
bool use_plus = zeros_plus >= zeros_minus;

// Compute Y3 using selected direction
if (use_plus) {
    Copy_u64_x4(x, x3_plus);
    // Recompute slope_plus in tmp (already there from first computation)
    SubModP(tmp, y0, jmp_y);
    MulModP(tmp, tmp, dxs);
    SubModP(y, x0, x);
    MulModP(y, y, tmp);
    SubModP(y, y, y0);
} else {
    Copy_u64_x4(x, x3_minus);
    jmp_ind |= INV_FLAG;
    // Recompute slope_minus in tmp2 (already there)
    SubModP(tmp, y0, jmp_y_neg);
    MulModP(tmp2, tmp, dxs);
    SubModP(y, x0, x);
    MulModP(y, y, tmp2);
    SubModP(y, y, y0);
}
```

**Problem with above:** Slopes get overwritten during X3 computation. Need to preserve them.

**Better approach:** Only store slopes if we need them for Y3. Recompute when needed.

**Savings:** 8 registers (2 × u64[4])

### Strategy 3: Compute-on-Demand (Trade compute for registers)

**Observation:** We only need ONE slope for Y3 computation (the chosen direction).

**Optimized flow:**
1. Compute BOTH X3 values
2. Choose best direction
3. **Recompute only the chosen slope** for Y3

**Pseudocode:**
```cpp
// Compute X3 for both directions (no slope storage)
__align__(16) u64 x3_plus[4], x3_minus[4];
__align__(16) u64 jmp_y_neg[4];

// Plus direction: X3 = (dy/dx)^2 - x0 - jmp_x
SubModP(tmp2, y0, jmp_y);
MulModP(tmp, tmp2, dxs);    // tmp = slope (temporary)
SqrModP(tmp2, tmp);
SubModP(x3_plus, tmp2, jmp_x);
SubModP(x3_plus, x3_plus, x0);

// Minus direction
Copy_u64_x4(jmp_y_neg, jmp_y);
NegModP(jmp_y_neg);
SubModP(tmp2, y0, jmp_y_neg);
MulModP(tmp, tmp2, dxs);    // tmp = slope (overwritten, OK)
SqrModP(tmp2, tmp);
SubModP(x3_minus, tmp2, jmp_x);
SubModP(x3_minus, x3_minus, x0);

// Choose
int zeros_plus = __clzll(x3_plus[3]);
int zeros_minus = __clzll(x3_minus[3]);
bool use_plus = zeros_plus >= zeros_minus;

// Compute Y3: Recompute slope for chosen direction
if (use_plus) {
    Copy_u64_x4(x, x3_plus);
    // Recompute slope for plus
    SubModP(tmp2, y0, jmp_y);
    MulModP(tmp, tmp2, dxs);
    SubModP(y, x0, x);
    MulModP(y, y, tmp);
    SubModP(y, y, y0);
} else {
    Copy_u64_x4(x, x3_minus);
    jmp_ind |= INV_FLAG;
    // Recompute slope for minus
    SubModP(tmp2, y0, jmp_y_neg);
    MulModP(tmp, tmp2, dxs);
    SubModP(y, x0, x);
    MulModP(y, y, tmp);
    SubModP(y, y, y0);
}
```

**Cost:** 2 extra SubModP + 1 extra MulModP per iteration (chosen direction)
**Savings:** 8 registers (slope_plus, slope_minus)

### Strategy 4: Merge jmp_y_neg into conditional (Advanced)

**Current:**
```cpp
__align__(16) u64 jmp_y_neg[4];
Copy_u64_x4(jmp_y_neg, jmp_y);
NegModP(jmp_y_neg);
```

**Optimized:** Compute jmp_y_neg only when needed for Y3

**Savings:** 4 registers

**Tradeoff:** More complex control flow, may hurt pipeline

## Optimized SOTA+ Register Usage

**After all optimizations:**

**Eliminated:**
- y3_plus[4], y3_minus[4]: -8 regs
- slope_plus[4], slope_minus[4]: -8 regs (recomputed on-demand)
- jmp_y_neg[4]: -4 regs (conditional computation)

**Remaining additions:**
- x3_plus[4], x3_minus[4]: +8 regs (unavoidable, need both for comparison)
- zeros_plus, zeros_minus, use_plus: +3 regs

**Net addition:** ~11 registers (vs. 28-32 original)

**SOTA+ optimized total:** ~46-51 registers per thread

**New occupancy:**
- Registers per block: 256 × 51 = 13,056 regs
- Blocks per SM: 65,536 / 13,056 = 5.02 → **5 blocks**
- Active threads: 5 × 256 = **1,280 threads (83% occupancy)**

**Result:** Acceptable occupancy, much better than 50%!

## Implementation Priority

### Phase 1: Quick Wins (5 min)
1. Remove `y3_plus` and `y3_minus` declarations (lines 210)
2. Verify no references to these variables

### Phase 2: Slope Recomputation (30 min)
1. Remove `slope_plus` and `slope_minus` declarations
2. Implement recompute-on-demand for Y3
3. Test correctness on puzzle 75

### Phase 3: Advanced (1-2 hours)
1. Conditional jmp_y_neg computation
2. Measure actual register usage with ptxas -v
3. Validate occupancy with CUDA profiler

## Testing Plan

**Correctness:**
```bash
# Before optimization
make clean && make SM=86 USE_JACOBIAN=1 USE_SOTA_PLUS=1 PROFILE=release -j
./rckangaroo -gpu 0 -range 75 ... # Record K-factor

# After optimization
make clean && make SM=86 USE_JACOBIAN=1 USE_SOTA_PLUS=1 PROFILE=release -j
./rckangaroo -gpu 0 -range 75 ... # Verify same K-factor, check speed
```

**Performance:**
- Measure speed (MKeys/s) before/after
- Expected: 0-5% improvement from better occupancy
- K-factor should remain 1.01-1.05

**Occupancy verification:**
```bash
nvcc --ptxas-options=-v RCGpuCore.cu # Check "registers per thread"
# Target: 46-55 registers/thread
# Critical threshold: < 65 regs (to maintain 3+ blocks/SM)
```

## Expected Impact

**Before optimization:**
- SOTA+ speed: 5.7-6.2 GKeys/s @ 50-83% occupancy
- Risk of slowdown vs. baseline

**After optimization:**
- SOTA+ speed: 6.2-6.5 GKeys/s @ 83% occupancy
- K-factor: 1.01-1.05 maintained
- Net improvement: +3-8% vs. baseline SOTA

## Summary

SOTA+ introduces register pressure that could reduce occupancy to 50%. By:
1. Removing unused y3 arrays
2. Recomputing slopes on-demand
3. Optimizing temporary usage

We can maintain **83% occupancy** and achieve target performance of 6.2-6.5 GKeys/s with K-factor 1.01-1.05.

**Next step:** Implement Phase 1-2 optimizations before testing SOTA+ on real hardware.
