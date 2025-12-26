#!/bin/bash
# Compile the GLV test program

echo "Compiling GLV test..."

g++ -std=c++17 -O2 -march=native -o test_glv \
    test_glv.cpp \
    Ec.cpp \
    EcInt.cpp \
    Lambda.cpp \
    -I. \
    -lgmp

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo ""
    echo "Run the test with: ./test_glv"
else
    echo "✗ Compilation failed"
    exit 1
fi
