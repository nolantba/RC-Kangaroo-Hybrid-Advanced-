#!/bin/bash
# RC-Kangaroo Tames Testing Commands
# Test pre-generated tames against solved puzzles

# Puzzle 60 - First generate tames if not exists
if [ ! -f "tames60.dat" ]; then
  echo "Generating 30x tames for 60-bit range..."
  ./rckangaroo -tames tames60.dat -range 60 -dp 13 -max 30
fi

echo "Testing Puzzle 60 with tames..."
./rckangaroo -tames tames60.dat -pubkey 0348E843DC5B1BD246E6309B4924B81543D02B16C8083DF973A89CE2C7EB89A10D -range 60 -dp 13
echo ""

# Puzzle 61 - First generate tames if not exists
if [ ! -f "tames61.dat" ]; then
  echo "Generating 30x tames for 61-bit range..."
  ./rckangaroo -tames tames61.dat -range 61 -dp 13 -max 30
fi

echo "Testing Puzzle 61 with tames..."
./rckangaroo -tames tames61.dat -pubkey 0249A43860D115143C35C09454863D6F82A95E47C1162FB9B2EBE0186EB26F453F -range 61 -dp 13
echo ""

# Puzzle 62 - First generate tames if not exists
if [ ! -f "tames62.dat" ]; then
  echo "Generating 30x tames for 62-bit range..."
  ./rckangaroo -tames tames62.dat -range 62 -dp 13 -max 30
fi

echo "Testing Puzzle 62 with tames..."
./rckangaroo -tames tames62.dat -pubkey 03231A67E424CAF7D01A00D5CD49B0464942255B8E48766F96602BDFA4EA14FEA8 -range 62 -dp 13
echo ""

# Puzzle 63 - First generate tames if not exists
if [ ! -f "tames63.dat" ]; then
  echo "Generating 30x tames for 63-bit range..."
  ./rckangaroo -tames tames63.dat -range 63 -dp 13 -max 30
fi

echo "Testing Puzzle 63 with tames..."
./rckangaroo -tames tames63.dat -pubkey 0365EC2994B8CC0A20D40DD69EDFE55CA32A54BCBBAA6B0DDCFF36049301A54579 -range 63 -dp 13
echo ""

# Puzzle 64 - First generate tames if not exists
if [ ! -f "tames64.dat" ]; then
  echo "Generating 30x tames for 64-bit range..."
  ./rckangaroo -tames tames64.dat -range 64 -dp 13 -max 30
fi

echo "Testing Puzzle 64 with tames..."
./rckangaroo -tames tames64.dat -pubkey 03100611C54DFEF604163B8358F7B7FAC13CE478E02CB224AE16D45526B25D9D4D -range 64 -dp 13
echo ""

# Puzzle 65 - First generate tames if not exists
if [ ! -f "tames65.dat" ]; then
  echo "Generating 30x tames for 65-bit range..."
  ./rckangaroo -tames tames65.dat -range 65 -dp 13 -max 30
fi

echo "Testing Puzzle 65 with tames..."
./rckangaroo -tames tames65.dat -pubkey 0230210C23B1A047BC9BDBB13448E67DEDDC108946DE6DE639BCC75D47C0216B1B -range 65 -dp 13
echo ""

# Puzzle 66 - First generate tames if not exists
if [ ! -f "tames66.dat" ]; then
  echo "Generating 30x tames for 66-bit range..."
  ./rckangaroo -tames tames66.dat -range 66 -dp 14 -max 30
fi

echo "Testing Puzzle 66 with tames..."
./rckangaroo -tames tames66.dat -pubkey 024EE2BE2D4E9F92D2F5A4A03058617DC45BEFE22938FEED5B7A6B7282DD74CBDD -range 66 -dp 14
echo ""

# Puzzle 67 - First generate tames if not exists
if [ ! -f "tames67.dat" ]; then
  echo "Generating 30x tames for 67-bit range..."
  ./rckangaroo -tames tames67.dat -range 67 -dp 14 -max 30
fi

echo "Testing Puzzle 67 with tames..."
./rckangaroo -tames tames67.dat -pubkey 0212209F5EC514A1580A2937BD833979D933199FC230E204C6CDC58872B7D46F75 -range 67 -dp 14
echo ""

# Puzzle 68 - First generate tames if not exists
if [ ! -f "tames68.dat" ]; then
  echo "Generating 30x tames for 68-bit range..."
  ./rckangaroo -tames tames68.dat -range 68 -dp 14 -max 30
fi

echo "Testing Puzzle 68 with tames..."
./rckangaroo -tames tames68.dat -pubkey 031FE02F1D740637A7127CDFE8A77A8A0CFC6435F85E7EC3282CB6243C0A93BA1B -range 68 -dp 14
echo ""

# Puzzle 69 - First generate tames if not exists
if [ ! -f "tames69.dat" ]; then
  echo "Generating 30x tames for 69-bit range..."
  ./rckangaroo -tames tames69.dat -range 69 -dp 15 -max 30
fi

echo "Testing Puzzle 69 with tames..."
./rckangaroo -tames tames69.dat -pubkey 024BABADCCC6CFD5F0E5E7FD2A50AA7D677CE0AA16FDCE26A0D0882EED03E7BA53 -range 69 -dp 15
echo ""

# Puzzle 70 - First generate tames if not exists
if [ ! -f "tames70.dat" ]; then
  echo "Generating 30x tames for 70-bit range..."
  ./rckangaroo -tames tames70.dat -range 70 -dp 15 -max 30
fi

echo "Testing Puzzle 70 with tames..."
./rckangaroo -tames tames70.dat -pubkey 0290E6900A58D33393BC1097B5AED31F2E4E7CBD3E5466AF958665BC0121248483 -range 70 -dp 15
echo ""

