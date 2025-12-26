#!/bin/bash

echo "=== GLV Verification ==="
echo ""

if grep -q "mul_128x128" Lambda.cpp; then
    echo "✅ GLV FIX IS PRESENT (proper Babai algorithm)"
else
    echo "❌ GLV FIX MISSING - using old broken code"
fi

echo ""

calls=$(grep -c "MultiplyG_Lambda" RCKangaroo.cpp)
echo "GLV called $calls times in RCKangaroo.cpp"

echo ""
echo "=== Key GLV Usage Locations ==="
grep -n "MultiplyG_Lambda" RCKangaroo.cpp | head -5

echo ""
echo "=== Jump Table GLV Usage (most important for performance) ==="
grep -B 2 -A 2 "EcJumps.*MultiplyG_Lambda" RCKangaroo.cpp | head -10

echo ""
echo "Verification complete!"
