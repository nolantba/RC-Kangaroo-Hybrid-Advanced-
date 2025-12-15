# SOTA+ Implementation Guide

## Goal
Improve K-factor from ~1.12 to ~1.01 (10% algorithmic improvement) by implementing bidirectional walk selection.

## Background

**Current SOTA:**
- Single direction walk: `P → P + Jump`
- DP check on resulting point
- K-factor ~1.08-1.12

**SOTA+ (kTimesG's approach):**
- Compute BOTH: `P + Jump` AND `P - Jump`
- Choose direction with higher DP probability
- Effectively doubles DP density
- K-factor ~1.01

## Key Insight from BitcoinTalk

From kTimesG (Nov 2025):
> "I compute both X3(P+J) and X3(P-J) and their slopes, before Y3. The route is chosen based on what X3 is better. This keeps symmetry without having to do weird checks for Y parity and conditional branching."

> "The best/default route is going in the direction that has the higher chances of being a DP. If the DP criteria is on X, then you know what to pick."

> "DP density doubles, so the overhead is half. The 'worse' point can never be a DP if the 'best' point is not a DP itself."

## Implementation Plan

### Phase 1: Dual X3 Computation (Week 1)

**Location:** `RCGpuCore.cu`, KernelA, affine path (lines 156-260)

**Current code structure:**
```cpp
for (int group = 0; group < PNT_GROUP_CNT; group++) {
    LOAD_VAL_256(x0, L2x, group);
    LOAD_VAL_256(y0, L2y, group);

    // Select jump
    jmp_ind = x0[0] % JMP_CNT;
    Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
    Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);

    // Check Y parity, invert if needed
    if (y0[0] & 1) {
        NegModP(jmp_y);  // This chooses P-Jump
    }

    // Compute P' = P + Jump
    // ... point addition math ...
}
```

**SOTA+ modification:**
```cpp
for (int group = 0; group < PNT_GROUP_CNT; group++) {
    LOAD_VAL_256(x0, L2x, group);
    LOAD_VAL_256(y0, L2y, group);

    // Select jump
    jmp_ind = x0[0] % JMP_CNT;
    Copy_int4_x2(jmp_x, jmp_table + 8 * jmp_ind);
    Copy_int4_x2(jmp_y, jmp_table + 8 * jmp_ind + 4);

    // SOTA+: Compute BOTH slopes for P+J and P-J
    __align__(16) u64 jmp_y_neg[4];
    Copy_u64_x4(jmp_y_neg, jmp_y);
    NegModP(jmp_y_neg);  // Negative jump

    // Slope for P + Jump
    __align__(16) u64 slope_plus[4];
    SubModP(tmp, y0, jmp_y);
    SubModP(tmp2, x0, jmp_x);
    // slope_plus = (y0 - jmp_y) / (x0 - jmp_x) [will be computed with batch inv]

    // Slope for P - Jump
    __align__(16) u64 slope_minus[4];
    SubModP(tmp, y0, jmp_y_neg);
    SubModP(tmp2, x0, jmp_x);
    // slope_minus = (y0 - jmp_y_neg) / (x0 - jmp_x)

    // Compute X3 for BOTH directions (before full inversion)
    // X3 = slope^2 - x0 - jmp_x
    __align__(16) u64 x3_plus[4], x3_minus[4];

    // We need slope^2, but slope requires inversion
    // Optimization: Compare (y0 - jmp_y)^2 vs (y0 - jmp_y_neg)^2
    // to determine which direction is better WITHOUT full inversion

    // Actually, simpler approach:
    // Just compute BOTH X3 values after batch inversion
    // Compare leading zeros
    // Choose best

    // For now: Use Y parity as tie-breaker, but remember both options
    bool use_plus = (y0[0] & 1) == 0;  // Placeholder logic

    // Select the jump direction
    u64* selected_jmp_y = use_plus ? jmp_y : jmp_y_neg;

    // Continue with point addition using selected direction
    // ... existing code ...
}
```

**Problem with above:** Still doesn't compute BOTH X3 values

**Better approach:**
```cpp
// After batch inversion, compute BOTH X3 values

for (int group = PNT_GROUP_CNT - 1; group >= 0; group--) {
    // ... load point, jump, compute slope with batch inv ...

    // Compute X3 for P + Jump
    __align__(16) u64 x3_plus[4];
    Copy_u64_x4(jmp_y_use, jmp_y);  // Positive
    SubModP(tmp2, y0, jmp_y_use);
    MulModP(slope, tmp2, dxs);  // slope with batch inv
    SqrModP(tmp2, slope);
    SubModP(x3_plus, tmp2, jmp_x);
    SubModP(x3_plus, x3_plus, x0);

    // Compute X3 for P - Jump
    __align__(16) u64 x3_minus[4];
    NegModP(jmp_y_use);  // Now negative
    SubModP(tmp2, y0, jmp_y_use);
    MulModP(slope, tmp2, dxs);  // slope for negative jump
    SqrModP(tmp2, slope);
    SubModP(x3_minus, tmp2, jmp_x);
    SubModP(x3_minus, x3_minus, x0);

    // Choose best: Count leading zeros
    int zeros_plus = __clzll(x3_plus[3]);
    int zeros_minus = __clzll(x3_minus[3]);

    bool use_plus = zeros_plus >= zeros_minus;

    // Use the chosen X3
    if (use_plus) {
        Copy_u64_x4(x, x3_plus);
        Copy_u64_x4(jmp_y_use, jmp_y);  // Restore positive
        jmp_ind &= ~INV_FLAG;  // Clear inversion flag
    } else {
        Copy_u64_x4(x, x3_minus);
        // jmp_y_use already negative
        jmp_ind |= INV_FLAG;  // Set inversion flag
    }

    // Compute Y3 using chosen direction
    SubModP(tmp2, y0, jmp_y_use);
    MulModP(slope, tmp2, dxs);
    SubModP(y, x0, x);
    MulModP(y, y, slope);
    SubModP(y, y, y0);

    // Store result
    SAVE_VAL_256(L2x, x, group);
    SAVE_VAL_256(L2y, y, group);

    // Rest of the code (DP check, loop detection, etc.)
}
```

### Phase 2: Cycle Detection Updates

**Problem:** SOTA+ creates symmetric walks, but cycle detection assumes single direction.

**L1S2 (2-cycle) detection needs update:**

Current:
```cpp
u32 jmp_next = x[0] % JMP_CNT;
jmp_next |= ((u32)y[0] & 1) ? 0 : INV_FLAG;
L1S2 |= (jmp_ind == jmp_next) ? (1u << group) : 0;
```

SOTA+ modification:
```cpp
// After SOTA+ selection, jmp_ind already has correct INV_FLAG
u32 jmp_next = x[0] % JMP_CNT;
// Compute BOTH possible next jumps
u32 jmp_next_plus = jmp_next;
u32 jmp_next_minus = jmp_next | INV_FLAG;

// Check if EITHER would create a 2-cycle
bool cycle_plus = (jmp_ind == jmp_next_plus);
bool cycle_minus = (jmp_ind == jmp_next_minus);

if (cycle_plus || cycle_minus) {
    // 2-cycle detected, use Jump2 escape
    L1S2 |= (1u << group);
}
```

### Phase 3: Optimization

**Reduce overhead of dual computation:**

1. **Reuse computations:**
   - Both directions use same `(x0 - jmp_x)`
   - Only `(y0 - jmp_y)` differs

2. **Early exit:**
   - If one direction has DP-level leading zeros, choose immediately
   - No need to fully compute the other

3. **Batch better:**
   - Current batch inversion handles all groups
   - SOTA+ could batch ALL possible directions
   - More complex but more efficient

## Expected Performance Impact

### K-Factor Improvement
- **Current:** K = 1.08-1.12
- **Phase 1:** K = 1.05-1.07 (partial SOTA+)
- **Phase 2:** K = 1.01-1.02 (full SOTA+)

**Net benefit:** ~10% fewer operations

### Speed Impact
- **Phase 1:** -5% to -10% (extra computation, not optimized)
- **Phase 2:** -5% to -10% (cycle detection overhead)
- **Phase 3:** +5% to +10% (optimizations recover overhead)

**Net speed:** ~same or slightly faster

### Combined Benefit
- K-factor improvement: 10% fewer operations needed
- Speed stays roughly same
- **Overall:** ~10% faster puzzle solving

## Testing Plan

1. **Correctness test (Puzzle 75):**
   - Ensure same answer with SOTA+
   - Verify K-factor improves

2. **Performance test (Puzzle 80):**
   - Measure actual K-factor
   - Measure speed (MKeys/s)
   - Compare to baseline

3. **Long run (Puzzle 90):**
   - Multi-day stability
   - Confirm K-factor holds
   - Validate improvement

## Implementation Files

**Files to modify:**
1. `RCGpuCore.cu` - KernelA affine path
2. `defs.h` - Add `USE_SOTA_PLUS` flag
3. `Makefile` - Add compile option

**Compile flags:**
```make
USE_SOTA_PLUS ?= 0

# In CCFLAGS and NVCCFLAGS:
-DUSE_SOTA_PLUS=$(USE_SOTA_PLUS)
```

**Usage:**
```bash
# Traditional SOTA
make SM=86 USE_JACOBIAN=1 USE_PERSISTENT_KERNELS=0

# SOTA+
make SM=86 USE_JACOBIAN=1 USE_SOTA_PLUS=1
```

## References

- BitcoinTalk thread: Post #313, #317 (kTimesG)
- kTimesG's reported performance: 11.6 GKeys/s @ K=1.01 on RTX 4090
- Ykra's reported performance: 11.5 GKeys/s on RTX 5090

## Success Criteria

**Phase 1 success:**
- Code compiles and runs
- K-factor improves to <1.07
- No crashes on puzzle 75

**Phase 2 success:**
- K-factor reaches 1.01-1.02
- Stable on puzzle 80
- Correct results

**Phase 3 success:**
- Combined with persistent kernels: 6.0 → 7.0+ GKeys/s
- K-factor remains 1.01-1.02
- Stable on puzzle 90

**Final target:**
- 8-9 GKeys/s (with all optimizations)
- K = 1.01
- Ready to race for Puzzle 135
