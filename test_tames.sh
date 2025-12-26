#!/bin/bash
# Test pre-generated tames against random public keys
# Usage: ./test_tames.sh <tames_file> <range> <dp_bits> <num_tests>

TAMES_FILE="${1:-tames93.dat}"
RANGE="${2:-93}"
DP_BITS="${3:-17}"
NUM_TESTS="${4:-5}"

if [ ! -f "$TAMES_FILE" ]; then
    echo "ERROR: Tames file not found: $TAMES_FILE"
    echo "Usage: $0 <tames_file> <range> <dp_bits> <num_tests>"
    exit 1
fi

echo "=================================================="
echo "Testing Tames File: $TAMES_FILE"
echo "Range: $RANGE bits"
echo "DP: $DP_BITS bits"
echo "Number of tests: $NUM_TESTS"
echo "=================================================="
echo ""

# Create results directory
RESULTS_DIR="tames_test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR/"
echo ""

# Run tests
for i in $(seq 1 $NUM_TESTS); do
    echo "=========================================="
    echo "Test $i of $NUM_TESTS"
    echo "=========================================="

    # Generate random private key using Python
    PRIVKEY=$(python3 << EOF
import random
min_key = 2**(${RANGE}-1)
max_key = 2**${RANGE} - 1
privkey = random.randint(min_key, max_key)
print(f"{privkey:064x}")
EOF
)

    echo "Generated random private key: $PRIVKEY"
    echo "Private key: $PRIVKEY" > "$RESULTS_DIR/test_${i}.txt"

    # Compute public key using RC-Kangaroo
    # We need to use the EC multiply function - for now we'll use a workaround
    # by letting RC-Kangaroo generate a random key in benchmark mode and capture it

    echo "Computing public key..."
    # TODO: This requires RC-Kangaroo to have a key-to-pubkey utility
    # For now, we'll run in benchmark mode which generates its own keys

    echo "Running RC-Kangaroo with tames file..."
    echo "Command: ./rckangaroo -tames $TAMES_FILE -range $RANGE -dp $DP_BITS"
    echo ""

    # Run once in benchmark mode with tames
    # This will solve a random key that RC-Kangaroo generates internally
    timeout 300 ./rckangaroo \
        -tames "$TAMES_FILE" \
        -range $RANGE \
        -dp $DP_BITS \
        2>&1 | tee "$RESULTS_DIR/test_${i}_output.txt"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Test $i: SUCCESS (solved within 5 minutes)"
        echo "Status: SUCCESS" >> "$RESULTS_DIR/test_${i}.txt"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "⏱ Test $i: TIMEOUT (did not solve in 5 minutes)"
        echo "Status: TIMEOUT" >> "$RESULTS_DIR/test_${i}.txt"
    else
        echo "✗ Test $i: FAILED (exit code $EXIT_CODE)"
        echo "Status: FAILED" >> "$RESULTS_DIR/test_${i}.txt"
    fi

    echo ""
    echo "Waiting 5 seconds before next test..."
    sleep 5
done

echo ""
echo "=================================================="
echo "All tests complete!"
echo "=================================================="
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Summary:"
grep -h "Status:" "$RESULTS_DIR"/test_*.txt | sort | uniq -c
