#include <stdio.h>
#include <time.h>
#include "Ec.h"
#include "Lambda.h"

// Test if GLV produces correct results and measure speedup
int main() {
    printf("=================================================\n");
    printf("GLV/Lambda Optimization Test\n");
    printf("=================================================\n\n");

    Ec ec;
    ec.Init();
    InitLambda();

    // Test 1: Correctness - verify GLV gives same result as standard multiplication
    printf("[Test 1] Correctness Check\n");
    printf("-------------------------------------------------\n");

    EcInt test_scalars[5];
    test_scalars[0].SetHexStr("349b84b6431a6c4ef1");  // Puzzle 70 key
    test_scalars[1].SetHexStr("1000000000000000000");
    test_scalars[2].SetHexStr("FFFFFFFFFFFFFFFF");
    test_scalars[3].SetHexStr("123456789ABCDEF0");
    test_scalars[4].SetHexStr("DEADBEEFCAFEBABE");

    bool all_correct = true;
    for (int i = 0; i < 5; i++) {
        EcPoint p1 = ec.MultiplyG(test_scalars[i]);          // Standard method
        EcPoint p2 = ec.MultiplyG_Lambda(test_scalars[i]);   // GLV method

        bool match = p1.IsEqual(p2);
        printf("  Test %d: %s\n", i+1, match ? "✓ PASS" : "✗ FAIL");
        if (!match) {
            all_correct = false;
            printf("    Standard: x=%s\n", p1.x.GetHexStr().c_str());
            printf("    GLV:      x=%s\n", p2.x.GetHexStr().c_str());
        }
    }

    if (all_correct) {
        printf("\n✓ GLV produces CORRECT results!\n\n");
    } else {
        printf("\n✗ GLV FAILED correctness test!\n\n");
        return 1;
    }

    // Test 2: Performance - measure speedup
    printf("[Test 2] Performance Benchmark\n");
    printf("-------------------------------------------------\n");

    const int NUM_TESTS = 10000;
    EcInt k;
    k.RndBits(128);  // Random 128-bit scalar

    // Benchmark standard multiplication
    clock_t start = clock();
    for (int i = 0; i < NUM_TESTS; i++) {
        k.data[0] += i;  // Vary the scalar slightly
        volatile EcPoint p = ec.MultiplyG(k);
    }
    clock_t end_standard = clock();
    double time_standard = (double)(end_standard - start) / CLOCKS_PER_SEC;

    // Benchmark GLV multiplication
    k.RndBits(128);
    start = clock();
    for (int i = 0; i < NUM_TESTS; i++) {
        k.data[0] += i;
        volatile EcPoint p = ec.MultiplyG_Lambda(k);
    }
    clock_t end_glv = clock();
    double time_glv = (double)(end_glv - start) / CLOCKS_PER_SEC;

    double speedup = (time_standard / time_glv - 1.0) * 100.0;

    printf("  Standard method: %.3f seconds (%d ops)\n", time_standard, NUM_TESTS);
    printf("  GLV method:      %.3f seconds (%d ops)\n", time_glv, NUM_TESTS);
    printf("  Speedup:         %.1f%%\n\n", speedup);

    if (speedup > 20.0) {
        printf("✓ GLV is providing significant speedup (>20%%)\n");
    } else if (speedup > 0) {
        printf("⚠ GLV speedup is lower than expected (should be ~30-40%%)\n");
    } else {
        printf("✗ GLV is SLOWER than standard! Something is wrong.\n");
    }

    // Test 3: Decomposition quality check
    printf("\n[Test 3] Scalar Decomposition Quality\n");
    printf("-------------------------------------------------\n");

    EcInt large_k;
    large_k.SetHexStr("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");

    ScalarDecomposition decomp = DecomposeScalar(large_k);

    int k1_bits = 0, k2_bits = 0;
    for (int i = 255; i >= 0; i--) {
        if (decomp.k1.GetBit(i)) { k1_bits = i + 1; break; }
    }
    for (int i = 255; i >= 0; i--) {
        if (decomp.k2.GetBit(i)) { k2_bits = i + 1; break; }
    }

    printf("  Original scalar: 256 bits\n");
    printf("  k1 decomposed:   %d bits\n", k1_bits);
    printf("  k2 decomposed:   %d bits\n", k2_bits);

    if (k1_bits <= 130 && k2_bits <= 130) {
        printf("\n✓ Decomposition is GOOD (both ~128 bits)\n");
        printf("  This means GLV is working optimally!\n");
    } else {
        printf("\n✗ Decomposition is POOR (should be ~128 bits each)\n");
        printf("  GLV may not be providing full speedup.\n");
    }

    printf("\n=================================================\n");
    printf("Test Complete\n");
    printf("=================================================\n");

    return 0;
}
