#!/bin/bash
# ============================================================================
# RC-Kangaroo Herd Performance Benchmark
# Compares integrated vs separate kernel implementations
# ============================================================================

set -e

echo "================================================"
echo "RC-Kangaroo SOTA++ Herds Benchmark"
echo "================================================"
echo ""

# Test configuration
PUZZLE=135
PUBKEY="0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b"
RANGE="800000000000000:fffffffffffffffff"
TEST_DURATION=60  # seconds per test
DP_BITS=14

# Create test directory
TEST_DIR="benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

echo "Test Configuration:"
echo "  Puzzle: $PUZZLE"
echo "  Duration: ${TEST_DURATION}s per test"
echo "  DP Bits: $DP_BITS"
echo ""

# ============================================================================
# Test 1: Integrated Herd Mode (current implementation)
# ============================================================================

echo "================================================"
echo "Test 1: Integrated Herd Mode"
echo "================================================"
echo "Running for ${TEST_DURATION} seconds..."
echo ""

timeout ${TEST_DURATION}s ../rckangaroo \
    -herds \
    -range $PUZZLE \
    -pubkey $PUBKEY \
    -rangekey1 $RANGE \
    -s integrated_test \
    -d $DP_BITS \
    > integrated_output.txt 2>&1 || true

# Extract metrics
INTEGRATED_SPEED=$(grep -oP 'MAIN: Speed: \K[0-9]+' integrated_output.txt | tail -1)
INTEGRATED_DPS=$(grep -oP 'DPs: \K[0-9]+K' integrated_output.txt | tail -1)
INTEGRATED_KANGS=$(grep -oP 'Kangaroos: \K[0-9]+K' integrated_output.txt | tail -1)

echo "Integrated Results:"
echo "  Speed: ${INTEGRATED_SPEED} MKeys/s"
echo "  DPs Found: ${INTEGRATED_DPS}"
echo "  Kangaroos: ${INTEGRATED_KANGS}"
echo ""

# ============================================================================
# Test 2: Baseline (No Herds)
# ============================================================================

echo "================================================"
echo "Test 2: Baseline (No Herds)"
echo "================================================"
echo "Running for ${TEST_DURATION} seconds..."
echo ""

rm -f baseline_test*.txt

timeout ${TEST_DURATION}s ../rckangaroo \
    -range $PUZZLE \
    -pubkey $PUBKEY \
    -rangekey1 $RANGE \
    -s baseline_test \
    -d $DP_BITS \
    > baseline_output.txt 2>&1 || true

# Extract metrics
BASELINE_SPEED=$(grep -oP 'MAIN: Speed: \K[0-9]+' baseline_output.txt | tail -1)
BASELINE_DPS=$(grep -oP 'DPs: \K[0-9]+K' baseline_output.txt | tail -1)
BASELINE_KANGS=$(grep -oP 'Kangaroos: \K[0-9]+K' baseline_output.txt | tail -1)

echo "Baseline Results:"
echo "  Speed: ${BASELINE_SPEED} MKeys/s"
echo "  DPs Found: ${BASELINE_DPS}"
echo "  Kangaroos: ${BASELINE_KANGS}"
echo ""

# ============================================================================
# Analysis
# ============================================================================

echo "================================================"
echo "Performance Analysis"
echo "================================================"
echo ""

# Calculate speed comparison
if [ -n "$INTEGRATED_SPEED" ] && [ -n "$BASELINE_SPEED" ]; then
    SPEED_RATIO=$(awk "BEGIN {printf \"%.2f\", $INTEGRATED_SPEED/$BASELINE_SPEED}")
    SPEED_DIFF=$(awk "BEGIN {printf \"%.1f\", ($INTEGRATED_SPEED-$BASELINE_SPEED)/$BASELINE_SPEED*100}")

    echo "Speed Comparison:"
    echo "  Baseline:   ${BASELINE_SPEED} MKeys/s"
    echo "  Integrated: ${INTEGRATED_SPEED} MKeys/s"
    echo "  Ratio:      ${SPEED_RATIO}x"
    echo "  Difference: ${SPEED_DIFF}%"
    echo ""
fi

# Calculate DP generation rate
if [ -n "$INTEGRATED_DPS" ] && [ -n "$BASELINE_DPS" ]; then
    echo "DP Generation:"
    echo "  Baseline:   ${BASELINE_DPS} in ${TEST_DURATION}s"
    echo "  Integrated: ${INTEGRATED_DPS} in ${TEST_DURATION}s"
    echo ""
fi

# GPU utilization check
echo "GPU Utilization Check:"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,power.draw --format=csv,noheader
echo ""

# Summary
echo "================================================"
echo "Summary"
echo "================================================"
echo ""

if [ -n "$SPEED_DIFF" ]; then
    if (( $(echo "$SPEED_DIFF > 5" | bc -l) )); then
        echo "✅ Integrated herds show ${SPEED_DIFF}% improvement"
    elif (( $(echo "$SPEED_DIFF < -5" | bc -l) )); then
        echo "⚠️  Integrated herds show ${SPEED_DIFF}% regression"
    else
        echo "✅ Performance equivalent (${SPEED_DIFF}% difference)"
    fi
else
    echo "⚠️  Unable to calculate performance comparison"
fi

echo ""
echo "Test outputs saved to: $TEST_DIR/"
echo "  - integrated_output.txt"
echo "  - baseline_output.txt"
echo ""

# Return to original directory
cd ..

echo "Benchmark complete!"
