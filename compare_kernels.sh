#!/bin/bash
# ============================================================================
# RC-Kangaroo Herd Kernel Comparison Script
# Builds and benchmarks both integrated and separate kernel implementations
# ============================================================================

set -e

echo "============================================================================"
echo "RC-Kangaroo SOTA++ Herd Kernel Comparison"
echo "============================================================================"
echo ""

# Configuration
PUZZLE=135
PUBKEY="0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b"
RANGE="800000000000000:fffffffffffffffff"
TEST_DURATION=60  # seconds
DP_BITS=14
SM_ARCH=86  # RTX 3060

# Create results directory
RESULTS_DIR="kernel_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

# ============================================================================
# Test 1: Integrated Herd Kernel (Current - Recommended)
# ============================================================================

echo "============================================================================"
echo "Test 1: Building with INTEGRATED herd support"
echo "============================================================================"
echo ""

cd ..
make clean
echo "Building: make SM=$SM_ARCH USE_SEPARATE_HERD_KERNEL=0 PROFILE=release -j"
make SM=$SM_ARCH USE_SEPARATE_HERD_KERNEL=0 PROFILE=release -j

if [ ! -f rckangaroo ]; then
    echo "ERROR: Build failed for integrated kernel"
    exit 1
fi

# Backup binary
cp rckangaroo "$RESULTS_DIR/rckangaroo_integrated"
cd "$RESULTS_DIR"

echo ""
echo "Running integrated kernel test for ${TEST_DURATION} seconds..."
echo ""

timeout ${TEST_DURATION}s ./rckangaroo_integrated \
    -herds \
    -range $PUZZLE \
    -pubkey $PUBKEY \
    -rangekey1 $RANGE \
    -s integrated_test \
    -d $DP_BITS \
    > integrated_output.txt 2>&1 || true

# Extract performance metrics
echo "Extracting metrics..."
INTEGRATED_SPEED=$(grep -oP 'MAIN: Speed: \K[0-9]+' integrated_output.txt | tail -1 || echo "0")
INTEGRATED_DPS=$(grep -oP 'DPs: \K[0-9]+K' integrated_output.txt | tail -1 || echo "0K")
INTEGRATED_OPS=$(grep -oP 'Operations: \K[0-9]+' integrated_output.txt | tail -1 || echo "0")

echo "Integrated Kernel Results:"
echo "  Speed: ${INTEGRATED_SPEED} MKeys/s"
echo "  DPs: ${INTEGRATED_DPS}"
echo "  Operations: ${INTEGRATED_OPS}"
echo ""

# GPU stats during integrated run
echo "GPU utilization (during test):"
nvidia-smi --query-gpu=index,name,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader > integrated_gpu_stats.txt
cat integrated_gpu_stats.txt
echo ""

# ============================================================================
# Test 2: Separate Herd Kernel (Experimental)
# ============================================================================

echo "============================================================================"
echo "Test 2: Building with SEPARATE herd kernel"
echo "============================================================================"
echo ""

cd ..
make clean
echo "Building: make SM=$SM_ARCH USE_SEPARATE_HERD_KERNEL=1 PROFILE=release -j"
make SM=$SM_ARCH USE_SEPARATE_HERD_KERNEL=1 PROFILE=release -j

if [ ! -f rckangaroo ]; then
    echo "ERROR: Build failed for separate kernel"
    exit 1
fi

# Backup binary
cp rckangaroo "$RESULTS_DIR/rckangaroo_separate"
cd "$RESULTS_DIR"

echo ""
echo "Running separate kernel test for ${TEST_DURATION} seconds..."
echo ""

timeout ${TEST_DURATION}s ./rckangaroo_separate \
    -herds \
    -range $PUZZLE \
    -pubkey $PUBKEY \
    -rangekey1 $RANGE \
    -s separate_test \
    -d $DP_BITS \
    > separate_output.txt 2>&1 || true

# Extract performance metrics
echo "Extracting metrics..."
SEPARATE_SPEED=$(grep -oP 'MAIN: Speed: \K[0-9]+' separate_output.txt | tail -1 || echo "0")
SEPARATE_DPS=$(grep -oP 'DPs: \K[0-9]+K' separate_output.txt | tail -1 || echo "0K")
SEPARATE_OPS=$(grep -oP 'Operations: \K[0-9]+' separate_output.txt | tail -1 || echo "0")

echo "Separate Kernel Results:"
echo "  Speed: ${SEPARATE_SPEED} MKeys/s"
echo "  DPs: ${SEPARATE_DPS}"
echo "  Operations: ${SEPARATE_OPS}"
echo ""

# GPU stats during separate run
echo "GPU utilization (during test):"
nvidia-smi --query-gpu=index,name,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader > separate_gpu_stats.txt
cat separate_gpu_stats.txt
echo ""

# ============================================================================
# Test 3: No Herds Baseline
# ============================================================================

echo "============================================================================"
echo "Test 3: Baseline (No Herds)"
echo "============================================================================"
echo ""

cd ..
make clean
echo "Building: make SM=$SM_ARCH USE_SEPARATE_HERD_KERNEL=0 PROFILE=release -j"
make SM=$SM_ARCH USE_SEPARATE_HERD_KERNEL=0 PROFILE=release -j

cp rckangaroo "$RESULTS_DIR/rckangaroo_baseline"
cd "$RESULTS_DIR"

echo ""
echo "Running baseline test for ${TEST_DURATION} seconds..."
echo ""

rm -f baseline_test*.txt

timeout ${TEST_DURATION}s ./rckangaroo_baseline \
    -range $PUZZLE \
    -pubkey $PUBKEY \
    -rangekey1 $RANGE \
    -s baseline_test \
    -d $DP_BITS \
    > baseline_output.txt 2>&1 || true

# Extract performance metrics
echo "Extracting metrics..."
BASELINE_SPEED=$(grep -oP 'MAIN: Speed: \K[0-9]+' baseline_output.txt | tail -1 || echo "0")
BASELINE_DPS=$(grep -oP 'DPs: \K[0-9]+K' baseline_output.txt | tail -1 || echo "0K")
BASELINE_OPS=$(grep -oP 'Operations: \K[0-9]+' baseline_output.txt | tail -1 || echo "0")

echo "Baseline Results:"
echo "  Speed: ${BASELINE_SPEED} MKeys/s"
echo "  DPs: ${BASELINE_DPS}"
echo "  Operations: ${BASELINE_OPS}"
echo ""

# ============================================================================
# Performance Analysis and Comparison
# ============================================================================

echo "============================================================================"
echo "PERFORMANCE COMPARISON"
echo "============================================================================"
echo ""

# Create comparison table
{
    echo "| Implementation | Speed (MKeys/s) | DPs Found | Ratio vs Baseline |"
    echo "|----------------|-----------------|-----------|-------------------|"

    # Baseline
    if [ "$BASELINE_SPEED" != "0" ]; then
        echo "| Baseline (No Herds) | $BASELINE_SPEED | $BASELINE_DPS | 1.00x |"
    fi

    # Integrated
    if [ "$INTEGRATED_SPEED" != "0" ] && [ "$BASELINE_SPEED" != "0" ]; then
        INTEGRATED_RATIO=$(awk "BEGIN {printf \"%.2f\", $INTEGRATED_SPEED/$BASELINE_SPEED}")
        echo "| Integrated Herds | $INTEGRATED_SPEED | $INTEGRATED_DPS | ${INTEGRATED_RATIO}x |"
    fi

    # Separate
    if [ "$SEPARATE_SPEED" != "0" ] && [ "$BASELINE_SPEED" != "0" ]; then
        SEPARATE_RATIO=$(awk "BEGIN {printf \"%.2f\", $SEPARATE_SPEED/$BASELINE_SPEED}")
        echo "| Separate Kernel | $SEPARATE_SPEED | $SEPARATE_DPS | ${SEPARATE_RATIO}x |"
    fi
} | tee performance_comparison.txt

echo ""
echo ""

# Detailed analysis
echo "DETAILED ANALYSIS"
echo "-------------------------------------------------------------------------------"
echo ""

if [ "$INTEGRATED_SPEED" != "0" ] && [ "$SEPARATE_SPEED" != "0" ]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $INTEGRATED_SPEED/$SEPARATE_SPEED}")
    DIFF_PCT=$(awk "BEGIN {printf \"%.1f\", ($INTEGRATED_SPEED-$SEPARATE_SPEED)/$SEPARATE_SPEED*100}")

    echo "Integrated vs Separate:"
    echo "  Speed difference: ${SPEEDUP}x (${DIFF_PCT}% faster)"

    if (( $(echo "$SPEEDUP > 2.0" | bc -l) )); then
        echo "  ✅ RECOMMENDATION: Use INTEGRATED kernel (${SPEEDUP}x faster)"
    elif (( $(echo "$SPEEDUP > 1.2" | bc -l) )); then
        echo "  ✅ RECOMMENDATION: Use INTEGRATED kernel (significantly faster)"
    elif (( $(echo "$SPEEDUP < 0.8" | bc -l) )); then
        echo "  ⚠️  UNEXPECTED: Separate kernel is faster - investigate!"
    else
        echo "  ⚡ Both implementations have similar performance"
    fi
fi

echo ""

# GPU efficiency analysis
echo "GPU Efficiency:"
if [ -f integrated_gpu_stats.txt ]; then
    INTEGRATED_GPU_UTIL=$(cut -d',' -f3 integrated_gpu_stats.txt | head -1)
    INTEGRATED_POWER=$(cut -d',' -f4 integrated_gpu_stats.txt | head -1)
    echo "  Integrated: ${INTEGRATED_GPU_UTIL}% GPU, ${INTEGRATED_POWER}W power"
fi

if [ -f separate_gpu_stats.txt ]; then
    SEPARATE_GPU_UTIL=$(cut -d',' -f3 separate_gpu_stats.txt | head -1)
    SEPARATE_POWER=$(cut -d',' -f4 separate_gpu_stats.txt | head -1)
    echo "  Separate:   ${SEPARATE_GPU_UTIL}% GPU, ${SEPARATE_POWER}W power"
fi

echo ""

# ============================================================================
# Summary and Recommendations
# ============================================================================

echo "============================================================================"
echo "SUMMARY"
echo "============================================================================"
echo ""

echo "Test Configuration:"
echo "  Puzzle: $PUZZLE bits"
echo "  Test duration: ${TEST_DURATION}s per test"
echo "  DP bits: $DP_BITS"
echo "  SM architecture: $SM_ARCH"
echo ""

echo "All test outputs saved to: $(pwd)/"
echo "  - integrated_output.txt"
echo "  - separate_output.txt"
echo "  - baseline_output.txt"
echo "  - performance_comparison.txt"
echo ""

if [ "$INTEGRATED_SPEED" != "0" ] && [ "$BASELINE_SPEED" != "0" ]; then
    INTEGRATED_VS_BASELINE=$(awk "BEGIN {printf \"%.1f\", ($INTEGRATED_SPEED-$BASELINE_SPEED)/$BASELINE_SPEED*100}")

    if (( $(echo "$INTEGRATED_VS_BASELINE > 5" | bc -l) )); then
        echo "✅ WINNER: Integrated herds (+${INTEGRATED_VS_BASELINE}% vs baseline)"
    elif (( $(echo "$INTEGRATED_VS_BASELINE < -5" | bc -l) )); then
        echo "⚠️  Integrated herds are SLOWER (${INTEGRATED_VS_BASELINE}% vs baseline)"
    else
        echo "✅ Integrated herds have ZERO overhead (${INTEGRATED_VS_BASELINE}% vs baseline)"
    fi
fi

echo ""
echo "Comparison complete!"

# Return to original directory
cd ..
