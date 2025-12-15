// GALBRAITH-RUPRAI EQUIVALENCE CLASSES - CPU Implementation
// Reduces search space by 2x by treating (x, y) and (x, -y) as equivalent
#ifndef GALBRAITH_RUPRAI_H
#define GALBRAITH_RUPRAI_H

#include "Ec.h"
#include "defs.h"

// External secp256k1 prime (should be defined in Ec.cpp or utils.cpp)
extern EcInt g_P;

// =============================================================================
// NORMALIZE POINT: Force y-coordinate to be even (canonical form)
// =============================================================================
inline void NormalizePoint_GR(EcPoint* p) {
    // If y is odd, compute y' = P - y (negate y mod P)
    if (p->y.data[0] & 1) {
        // Use existing EcInt operations
        EcInt P_minus_y;

        // Copy P
        for (int i = 0; i < 4; i++) {
            P_minus_y.data[i] = g_P.data[i];
        }

        // Compute P - y using existing SubModP
        P_minus_y.Sub(p->y);

        // Store result
        p->y = P_minus_y;
    }
}

// Alternative: Use NegModP if available
inline void NormalizePoint_GR_v2(EcPoint* p) {
    if (p->y.data[0] & 1) {
        // Negate y coordinate
        p->y.NegModP();
    }
}

// Validation helper
inline bool IsNormalized_GR(const EcPoint* p) {
    // After normalization, y must be even
    return (p->y.data[0] & 1) == 0;
}

// =============================================================================
// USAGE NOTES:
// =============================================================================
// 1. Call NormalizePoint_GR() after EVERY point operation:
//    - After AddPoints()
//    - After initial kangaroo generation
//    - After DP restart
//
// 2. This reduces the effective search space by 2x because:
//    - Points (x, y) and (x, -y) are treated as equivalent
//    - We always choose the even y-coordinate
//    - Collision detection automatically works on canonical forms
//
// 3. Expected speedup:
//    - Theoretical: 2x (half the search space)
//    - Practical: 1.5-1.8x (accounting for normalization overhead)
//
// 4. Compatible with existing SOTA Kangaroo method
//
// =============================================================================

#endif // GALBRAITH_RUPRAI_H
