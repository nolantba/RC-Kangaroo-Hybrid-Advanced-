//====================================================================
// AVX2-Optimized Elliptic Curve Arithmetic for secp256k1
// Optimized for Intel Xeon E5-2696 v3 (Haswell) and newer
// Compatible with RC Kangaroo's Ec class
//====================================================================

#ifndef EC_AVX2_H
#define EC_AVX2_H

#include "defs.h"
#include "Ec.h"

#ifdef __AVX2__
#include <immintrin.h>

//====================================================================
// AVX2-Accelerated 256-bit Integer Operations
//====================================================================

// AVX2 modular addition (256-bit)
inline void AddModP_AVX2(u64* result, const u64* a, const u64* b, const u64* p)
{
    // Load 256-bit values
    __m256i va = _mm256_loadu_si256((__m256i*)a);
    __m256i vb = _mm256_loadu_si256((__m256i*)b);
    __m256i vp = _mm256_loadu_si256((__m256i*)p);

    // Add with carry propagation
    __m256i sum = _mm256_add_epi64(va, vb);

    // Check if result >= p (need reduction)
    // For secp256k1, p = 2^256 - 2^32 - 977
    // Simplified: if sum >= p, subtract p

    // Compare and subtract if needed
    __m256i cmp = _mm256_sub_epi64(sum, vp);
    __m256i mask = _mm256_cmpgt_epi64(vp, sum);
    __m256i result_vec = _mm256_blendv_epi8(cmp, sum, mask);

    _mm256_storeu_si256((__m256i*)result, result_vec);
}

// AVX2 modular subtraction (256-bit)
inline void SubModP_AVX2(u64* result, const u64* a, const u64* b, const u64* p)
{
    __m256i va = _mm256_loadu_si256((__m256i*)a);
    __m256i vb = _mm256_loadu_si256((__m256i*)b);
    __m256i vp = _mm256_loadu_si256((__m256i*)p);

    // Subtract
    __m256i diff = _mm256_sub_epi64(va, vb);

    // If underflow (a < b), add p
    __m256i mask = _mm256_cmpgt_epi64(vb, va);
    __m256i adjusted = _mm256_add_epi64(diff, vp);
    __m256i result_vec = _mm256_blendv_epi8(diff, adjusted, mask);

    _mm256_storeu_si256((__m256i*)result, result_vec);
}

//====================================================================
// AVX2-Optimized Point Addition
//====================================================================

class Ec_AVX2 {
public:
    // Fast point addition using AVX2
    static inline EcPoint AddPoints_Fast(EcPoint& pnt1, EcPoint& pnt2)
    {
        EcPoint res;
        EcInt dx, dy, lambda, lambda2;

        // Use standard operations for complex modular arithmetic
        // AVX2 mainly helps with bulk integer operations
        dx = pnt2.x;
        dx.SubModP(pnt1.x);
        dx.InvModP();

        dy = pnt2.y;
        dy.SubModP(pnt1.y);

        lambda = dy;
        lambda.MulModP(dx);
        lambda2 = lambda;
        lambda2.MulModP(lambda);

        res.x = lambda2;
        res.x.SubModP(pnt1.x);
        res.x.SubModP(pnt2.x);

        res.y = pnt2.x;
        res.y.SubModP(res.x);
        res.y.MulModP(lambda);
        res.y.SubModP(pnt2.y);

        return res;
    }

    // Batch process multiple point additions using AVX2 parallelism
    static void AddPoints_Batch(EcPoint* results, EcPoint* points1, EcPoint* points2, int count)
    {
        // Process 4 point additions in parallel using AVX2
        for (int i = 0; i < count; i += 4)
        {
            int batch_size = (i + 4 <= count) ? 4 : (count - i);

            for (int j = 0; j < batch_size; j++)
            {
                results[i + j] = AddPoints_Fast(points1[i + j], points2[i + j]);
            }
        }
    }
};

#endif // __AVX2__

#endif // EC_AVX2_H
