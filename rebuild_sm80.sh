#!/bin/bash
# ============================================================================
# SM 8.0 Clean Rebuild Script with Diagnostics
# For A100, RTX 3090, RTX 3080 Ti (compute capability 8.0)
# ============================================================================

set -e

echo "============================================================================"
echo "RC-Kangaroo SM 8.0 Clean Rebuild"
echo "============================================================================"
echo ""

# Check GPU architecture
echo "Step 1: Detecting GPU architecture..."
GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
echo "Detected compute capability: ${GPU_SM}"

if [ "$GPU_SM" != "80" ]; then
    echo "WARNING: Detected SM $GPU_SM, but this script is for SM 80"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Clean everything thoroughly
echo ""
echo "Step 2: Cleaning previous build..."
make clean
rm -f *.o gpu_dlink.o rckangaroo
rm -f *.a *.so
find . -name "*.o" -delete

# Verify CUDA installation
echo ""
echo "Step 3: Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Is CUDA installed?"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
echo "CUDA version: $CUDA_VERSION"

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Driver version: $DRIVER_VERSION"

# Build with SM 80 specific flags
echo ""
echo "Step 4: Building for SM 80..."
echo "Command: make SM=80 USE_JACOBIAN=1 PROFILE=release -j$(nproc)"
make SM=80 USE_JACOBIAN=1 PROFILE=release -j$(nproc)

if [ ! -f rckangaroo ]; then
    echo ""
    echo "ERROR: Build failed - rckangaroo binary not created"
    exit 1
fi

# Verify binary
echo ""
echo "Step 5: Verifying binary..."
file rckangaroo
ldd rckangaroo | grep cuda

# Check for SM 80 in binary
if strings rckangaroo | grep -q "sm_80"; then
    echo "✅ Binary contains SM 80 code"
else
    echo "⚠️  WARNING: SM 80 code not found in binary"
fi

# Test with single GPU first
echo ""
echo "Step 6: Testing with single GPU..."
echo "Running: ./rckangaroo -t 1 -benchmark"
timeout 10s ./rckangaroo -t 1 || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "✅ Test completed (timed out as expected)"
    else
        echo "❌ Test failed with exit code: $EXIT_CODE"
        echo ""
        echo "Debugging information:"
        echo "  - Check cuda-memcheck output below"
        echo "  - Look for 'illegal memory access' or 'out of bounds'"
        echo ""
        echo "Running cuda-memcheck (this may take a minute)..."
        cuda-memcheck --leak-check full ./rckangaroo -t 1 2>&1 | head -50
        exit 1
    fi
}

echo ""
echo "============================================================================"
echo "Build successful!"
echo "============================================================================"
echo ""
echo "To run with all 10 GPUs:"
echo "  ./rckangaroo -t 10"
echo ""
echo "To run in benchmark mode:"
echo "  ./rckangaroo -t 10 -benchmark"
echo ""
echo "If you still get illegal memory access errors, try:"
echo "  1. cuda-memcheck ./rckangaroo -t 1"
echo "  2. Build with debug: make SM=80 PROFILE=debug -j"
echo "  3. Check dmesg for GPU errors: sudo dmesg | tail -50"
echo ""
