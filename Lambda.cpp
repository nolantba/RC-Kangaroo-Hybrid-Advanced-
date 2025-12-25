// ============================================================================
// Lambda Endomorphism Implementation (GLV Method) for secp256k1
// ============================================================================

#include "Lambda.h"
#include <cstring>

// ============================================================================
// Precomputed tables for fast MultiplyG with Lambda
// ============================================================================

static EcPoint* g_lambda_table = nullptr;  // Precomputed table for G
static bool g_lambda_initialized = false;

// ============================================================================
// Scalar Decomposition (Babai's Nearest Plane Algorithm)
// ============================================================================

// Decompose scalar k into k1, k2 such that k ≡ k1 + k2*λ (mod n)
// where |k1|, |k2| ≈ √n ≈ 2^128 (half-size scalars)
//
// Proper Babai nearest plane algorithm using precomputed constants:
// 1. Compute c1 = ⌊k * b2 / n⌋  (using efficient approximation)
// 2. Compute c2 = ⌊k * (-b1) / n⌋
// 3. k1 = k - c1*a1 - c2*a2
// 4. k2 = -c1*b1 - c2*b2
//
// For secp256k1, we use the fact that b2 and -b1 are ~128 bits
// and n ≈ 2^256, so we can compute c1 and c2 by multiplying
// the high 128 bits of k by precomputed constants.
ScalarDecomposition DecomposeScalar(const EcInt& k) {
    ScalarDecomposition result;

    // Helper lambda for 128x128 -> 256-bit multiplication
    // Multiplies two 128-bit values and returns full 256-bit result
    auto mul_128x128 = [](u64 a_lo, u64 a_hi, u64 b_lo, u64 b_hi) -> EcInt {
        EcInt res;
        res.SetZero();

        // Split into 64-bit limbs and compute partial products
        // (a_hi * 2^64 + a_lo) * (b_hi * 2^64 + b_lo)

        // Partial products using 64x64->128 bit multiplication
        u64 p0_lo, p0_hi, p1_lo, p1_hi, p2_lo, p2_hi, p3_lo, p3_hi;

        // p0 = a_lo * b_lo
        #ifdef _MSC_VER
            p0_lo = _umul128(a_lo, b_lo, &p0_hi);
        #else
            __uint128_t p0_full = (__uint128_t)a_lo * b_lo;
            p0_lo = (u64)p0_full;
            p0_hi = (u64)(p0_full >> 64);
        #endif

        // p1 = a_lo * b_hi
        #ifdef _MSC_VER
            p1_lo = _umul128(a_lo, b_hi, &p1_hi);
        #else
            __uint128_t p1_full = (__uint128_t)a_lo * b_hi;
            p1_lo = (u64)p1_full;
            p1_hi = (u64)(p1_full >> 64);
        #endif

        // p2 = a_hi * b_lo
        #ifdef _MSC_VER
            p2_lo = _umul128(a_hi, b_lo, &p2_hi);
        #else
            __uint128_t p2_full = (__uint128_t)a_hi * b_lo;
            p2_lo = (u64)p2_full;
            p2_hi = (u64)(p2_full >> 64);
        #endif

        // p3 = a_hi * b_hi
        #ifdef _MSC_VER
            p3_lo = _umul128(a_hi, b_hi, &p3_hi);
        #else
            __uint128_t p3_full = (__uint128_t)a_hi * b_hi;
            p3_lo = (u64)p3_full;
            p3_hi = (u64)(p3_full >> 64);
        #endif

        // Combine partial products
        res.data[0] = p0_lo;
        res.data[1] = p0_hi;
        res.data[2] = p3_lo;
        res.data[3] = p3_hi;

        // Add middle terms (p1 + p2) at offset 64 bits
        u64 carry = 0;
        u64 sum = res.data[1] + p1_lo + carry;
        carry = (sum < res.data[1]) ? 1 : 0;
        res.data[1] = sum;

        sum = res.data[2] + p1_hi + carry;
        carry = (sum < res.data[2]) ? 1 : 0;
        res.data[2] = sum;
        res.data[3] += carry;

        carry = 0;
        sum = res.data[1] + p2_lo + carry;
        carry = (sum < res.data[1]) ? 1 : 0;
        res.data[1] = sum;

        sum = res.data[2] + p2_hi + carry;
        carry = (sum < res.data[2]) ? 1 : 0;
        res.data[2] = sum;
        res.data[3] += carry;

        return res;
    };

    // Extract high 128 bits of k (this is the key for the Babai rounding)
    u64 k_hi_lo = k.data[2];
    u64 k_hi_hi = k.data[3];

    // Precomputed constants for c1 and c2 calculation
    // These are derived from b2 and -b1 for secp256k1
    // g1 corresponds to b2: 0x3086d221a7d46bcde86c90e49284eb15
    u64 g1_lo = 0xe86c90e49284eb15ULL;
    u64 g1_hi = 0x3086d221a7d46bcdULL;

    // g2 corresponds to -b1: 0xe4437ed6010e88286f547fa90abfe4c3
    u64 g2_lo = 0x6f547fa90abfe4c3ULL;
    u64 g2_hi = 0xe4437ed6010e8828ULL;

    // Compute c1 = (k_hi * g1) >> 128 (approximation of (k * b2) / n)
    EcInt c1_full = mul_128x128(k_hi_lo, k_hi_hi, g1_lo, g1_hi);
    EcInt c1;
    c1.data[0] = c1_full.data[2];
    c1.data[1] = c1_full.data[3];
    c1.data[2] = 0;
    c1.data[3] = 0;

    // Compute c2 = (k_hi * g2) >> 128 (approximation of (k * (-b1)) / n)
    EcInt c2_full = mul_128x128(k_hi_lo, k_hi_hi, g2_lo, g2_hi);
    EcInt c2;
    c2.data[0] = c2_full.data[2];
    c2.data[1] = c2_full.data[3];
    c2.data[2] = 0;
    c2.data[3] = 0;

    // Compute k1 = k - c1*a1 - c2*a2
    result.k1 = k;

    EcInt tmp, a1, a2, b1, b2;
    a1 = LATTICE_A1;
    a2 = LATTICE_A2;
    b1 = LATTICE_B1;
    b2 = LATTICE_B2;

    // Multiply c1 * a1 (both are ~128 bits, need full multiplication)
    EcInt c1_a1 = mul_128x128(c1.data[0], c1.data[1], a1.data[0], a1.data[1]);
    result.k1.Sub(c1_a1);

    // Multiply c2 * a2
    EcInt c2_a2 = mul_128x128(c2.data[0], c2.data[1], a2.data[0], a2.data[1]);
    result.k1.Sub(c2_a2);

    // Compute k2 = -c1*b1 - c2*b2
    // Note: b1 is stored as positive value but represents negative in the lattice
    // So -c1*b1 means we multiply c1 by the absolute value
    EcInt abs_b1;
    abs_b1.data[0] = 0x6f547fa90abfe4c3ULL;  // |b1| low bits
    abs_b1.data[1] = 0xe4437ed6010e8828ULL;  // |b1| high bits
    abs_b1.data[2] = 0;
    abs_b1.data[3] = 0;

    result.k2 = mul_128x128(c1.data[0], c1.data[1], abs_b1.data[0], abs_b1.data[1]);

    // Subtract c2*b2
    EcInt c2_b2 = mul_128x128(c2.data[0], c2.data[1], b2.data[0], b2.data[1]);
    result.k2.Sub(c2_b2);

    // Handle signs
    result.k1_neg = false;
    result.k2_neg = false;

    // If k1 is negative (high bit set), negate it and set flag
    if (result.k1.data[3] & 0x8000000000000000ULL) {
        result.k1.Neg256();
        result.k1_neg = true;
    }

    // If k2 is negative (high bit set), negate it and set flag
    if (result.k2.data[3] & 0x8000000000000000ULL) {
        result.k2.Neg256();
        result.k2_neg = true;
    }

    return result;
}

// ============================================================================
// Lambda-based Multiplication Functions
// ============================================================================

// Precomputed φ(G) = (β*Gx, Gy) for secp256k1
static EcPoint g_phi_G;

// Multiply point P by scalar k where P is φ(G)
// This is a specialized version that uses φ(G) as the base point
static EcPoint MultiplyPhiG(EcInt& k) {
    // Use double-and-add with φ(G) as base
    EcPoint res;
    EcPoint t = g_phi_G;
    bool first = true;
    int n = 3;
    while ((n >= 0) && !k.data[n])
        n--;
    if (n < 0)
        return res; // Zero
    u32 index;
    _BitScanReverse64(&index, k.data[n]);
    for (int i = 0; i <= 64 * n + index; i++)
    {
        u8 v = (k.data[i / 64] >> (i % 64)) & 1;
        if (v)
        {
            if (first)
            {
                first = false;
                res = t;
            }
            else
                res = Ec::AddPoints(res, t);
        }
        t = Ec::DoublePoint(t);
    }
    return res;
}

// Multiply generator G by scalar k using Lambda endomorphism
// k*G = k1*G + k2*λ*G = k1*G + k2*φ(G)
// Speedup: ~40% by using two ~128-bit multiplications instead of one 256-bit
EcPoint MultiplyG_Lambda(const EcInt& k) {
    // Decompose scalar into k = k1 + k2*λ (mod n)
    ScalarDecomposition decomp = DecomposeScalar(k);

    // Compute k1*G using standard multiplication (with ~128-bit scalar)
    EcPoint P1 = Ec::MultiplyG(decomp.k1);
    if (decomp.k1_neg) {
        P1.y.NegModP();  // Negate point if k1 was negative
    }

    // Compute k2*φ(G) (with ~128-bit scalar)
    EcPoint P2 = MultiplyPhiG(decomp.k2);
    if (decomp.k2_neg) {
        P2.y.NegModP();  // Negate point if k2 was negative
    }

    // Handle edge cases
    if (decomp.k1.IsZero()) {
        return P2;
    }
    if (decomp.k2.IsZero()) {
        return P1;
    }

    // Add the two results: k*G = k1*G + k2*φ(G)
    EcPoint result = Ec::AddPoints(P1, P2);

    return result;
}

// Multiply arbitrary point P by scalar k using Lambda endomorphism
EcPoint Multiply_Lambda(const EcPoint& P, const EcInt& k) {
    // Decompose scalar
    ScalarDecomposition decomp = DecomposeScalar(k);

    // For arbitrary point multiplication, we need to:
    // 1. Compute k1*P using standard scalar multiplication
    // 2. Apply endomorphism to P: φ(P) = (β*Px, Py)
    // 3. Compute k2*φ(P)
    // 4. Add results: k1*P + k2*φ(P)

    // This requires implementing general point multiplication
    // For now, this is a placeholder - would need full implementation
    // Most use cases only need MultiplyG_Lambda for generator multiplication

    EcPoint phi_P = ApplyEndomorphism(P);

    // TODO: Implement full scalar multiplication for arbitrary points
    // For RCKangaroo, we primarily use MultiplyG, so this can be deferred

    EcPoint result = P;  // Placeholder
    return result;
}

// ============================================================================
// Initialization
// ============================================================================

void InitLambda() {
    if (g_lambda_initialized) {
        return;
    }

    // Precompute φ(G) = (β*Gx, Gy)
    // G = (0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
    //      0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)
    // φ(G) = (β*Gx, Gy) where β = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee

    g_phi_G.x.SetHexStr("79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    EcInt beta = BETA_CONST;  // Make mutable copy
    g_phi_G.x.MulModP(beta);  // β*Gx mod p
    g_phi_G.y.SetHexStr("483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");

    g_lambda_initialized = true;
}

void DeInitLambda() {
    if (g_lambda_table) {
        delete[] g_lambda_table;
        g_lambda_table = nullptr;
    }
    g_lambda_initialized = false;
}
