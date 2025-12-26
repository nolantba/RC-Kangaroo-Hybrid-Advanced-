#!/bin/bash
# Generate random 93-bit public keys for testing tames

RANGE=93
NUM_KEYS=10

echo "Generating ${NUM_KEYS} random ${RANGE}-bit test keys..."
echo ""

# Output file for the list
OUTPUT="test_keys_${RANGE}bit.txt"
> $OUTPUT  # Clear file

for i in $(seq 1 $NUM_KEYS); do
    # Generate random private key in range [2^92, 2^93)
    # Using Python for proper range calculation
    PRIVKEY=$(python3 << EOF
import random
min_key = 2**92
max_key = 2**93 - 1
privkey = random.randint(min_key, max_key)
print(f"{privkey:064x}")
EOF
)

    # Use rc-secp256k1 or btcrecover tools to get pubkey
    # For now, just save the private key and note for manual conversion
    echo "Test Key ${i}:" | tee -a $OUTPUT
    echo "  Private: ${PRIVKEY}" | tee -a $OUTPUT
    echo "  (Convert to pubkey using: ./rckangaroo or external tool)" | tee -a $OUTPUT
    echo "" | tee -a $OUTPUT
done

echo "Random keys saved to: ${OUTPUT}"
echo ""
echo "Next steps:"
echo "1. Convert private keys to public keys"
echo "2. Test solving with: ./rckangaroo -tames tames93.dat -pubkey <pubkey> -range 93 -dp 17"
