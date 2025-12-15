// ============================================================================
// Lambda Endomorphism (GLV Method) for secp256k1
// Provides ~40% speedup for scalar multiplications
// ============================================================================
// Based on: "Faster Point Multiplication on Elliptic Curves"
// by Gallant, Lambert, Vanstone (GLV method)
//
// secp256k1 has efficiently computable endomorphism:
// φ(x, y) = (β*x, y) where φ(P) = λ*P
//
// This allows decomposing k into k1 + k2*λ where k1, k2 are ~128 bits
// Computing k*P as k1*P + k2*φ(P) uses two half-size scalar mults
// Net speedup: ~40%
// ============================================================================

#pragma once

#include "defs.h"
#include "Ec.h"

// ============================================================================
// secp256k1 Lambda Endomorphism Constants
// ============================================================================

// Lambda (eigenvalue of endomorphism, mod n)
// λ ≡ 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72 (mod n)
const EcInt LAMBDA_CONST = []() {
    EcInt lambda;
    lambda.SetHexStr("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
    return lambda;
}();

// Beta (cube root of 1 mod p, used in endomorphism)
// β ≡ 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee (mod p)
const EcInt BETA_CONST = []() {
    EcInt beta;
    beta.SetHexStr("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
    return beta;
}();

// Decomposition lattice basis vectors (for Babai's nearest plane algorithm)
// These allow decomposing k = k1 + k2*λ where |k1|, |k2| ≈ √n ≈ 2^128

// a1 = 0x3086d221a7d46bcde86c90e49284eb15
const EcInt LATTICE_A1 = []() {
    EcInt a1;
    a1.SetHexStr("3086d221a7d46bcde86c90e49284eb15");
    return a1;
}();

// b1 = -0xe4437ed6010e88286f547fa90abfe4c3
const EcInt LATTICE_B1 = []() {
    EcInt b1;
    b1.SetHexStr("e4437ed6010e88286f547fa90abfe4c3");
    b1.Neg();  // Make negative
    return b1;
}();

// a2 = 0x114ca50f7a8e2f3f657c1108d9d44cfd8
const EcInt LATTICE_A2 = []() {
    EcInt a2;
    a2.SetHexStr("114ca50f7a8e2f3f657c1108d9d44cfd8");
    return a2;
}();

// b2 = 0x3086d221a7d46bcde86c90e49284eb15
const EcInt LATTICE_B2 = []() {
    EcInt b2;
    b2.SetHexStr("3086d221a7d46bcde86c90e49284eb15");
    return b2;
}();

// secp256k1 curve order n
const EcInt SECP256K1_N = []() {
    EcInt n;
    n.SetHexStr("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
    return n;
}();

// ============================================================================
// Lambda Endomorphism Functions
// ============================================================================

// Apply endomorphism: φ(x, y) = (β*x, y)
// This satisfies φ(P) = λ*P for secp256k1
inline EcPoint ApplyEndomorphism(const EcPoint& P) {
    EcPoint result;
    result.x = P.x;
    EcInt beta = BETA_CONST;  // Make mutable copy
    result.x.MulModP(beta);  // β*x mod p
    result.y = P.y;           // y unchanged
    return result;
}

// Decompose scalar k into k1, k2 such that k ≡ k1 + k2*λ (mod n)
// where |k1|, |k2| ≈ √n ≈ 2^128 (half-size scalars)
//
// Uses Babai's nearest plane algorithm with precomputed lattice basis
struct ScalarDecomposition {
    EcInt k1;
    EcInt k2;
    bool k1_neg;  // True if k1 should be negated
    bool k2_neg;  // True if k2 should be negated
};

ScalarDecomposition DecomposeScalar(const EcInt& k);

// Multiply generator G by scalar k using Lambda endomorphism
// k*G = k1*G + k2*λ*G = k1*G + k2*φ(G)
// Roughly 40% faster than standard scalar multiplication
EcPoint MultiplyG_Lambda(const EcInt& k);

// Multiply arbitrary point P by scalar k using Lambda endomorphism
EcPoint Multiply_Lambda(const EcPoint& P, const EcInt& k);

// Initialize Lambda endomorphism (called during InitEc)
void InitLambda();

// Cleanup Lambda endomorphism (called during DeInitEc)
void DeInitLambda();
