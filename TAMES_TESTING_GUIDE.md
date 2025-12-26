# Testing Pre-Generated Tames

## Strategy: Use Benchmark Mode with Tames

RC-Kangaroo has a built-in **benchmark mode** that generates random public keys and solves them. You can use your pre-generated tames in benchmark mode!

---

## Method 1: Benchmark Mode (Simple)

### **Run with Pre-Generated Tames:**

```bash
# Test your tames file by solving random keys
./rckangaroo -tames tames93.dat -range 93 -dp 17
```

**What happens:**
1. Loads your pre-generated tames from `tames93.dat`
2. Generates a random 93-bit private key
3. Computes the public key
4. Uses tames + wilds to solve for the private key
5. Verifies the solution is correct
6. **Repeats forever** (solving new random keys each time)

**Output:**
```
BENCHMARK MODE
load tames...
tames loaded
Tames: 123456789 DPs loaded from tames93.dat

Range: 93 bits
DP: 17 bits
SOTA method, estimated ops: 2^47.5

Speed: 6734 MKeys/s (6386 GPU + 348 CPU)
...
PRIVATE KEY: 00000001a2b3c4d5e6f7...
Point solved, K: 1.234 (with DP and GPU overheads)

[Starts solving another random key...]
```

### **Run Multiple Tests:**

```bash
# Solve 10 random keys using your tames
for i in {1..10}; do
    echo "Test $i of 10"
    timeout 600 ./rckangaroo -tames tames93.dat -range 93 -dp 17
    echo "Test $i complete"
done
```

**This verifies:**
- ✅ Tames file loads correctly
- ✅ Tames are valid for the range
- ✅ Solving works with pre-generated tames
- ✅ Collision detection is working

---

## Method 2: Solve Multiple Known Public Keys

If you have a **list of specific public keys** to solve (e.g., multiple Bitcoin puzzle addresses):

### **Create a Key List:**

```bash
# Create file: puzzle_keys_93bit.txt
cat > puzzle_keys_93bit.txt << EOF
# Format: pubkey,description
02a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4,Test Key 1
03b472a266d0bd89c13706a4132ccfb16f7c3b9fcb..,Test Key 2
02c1234567890abcdef...,Test Key 3
EOF
```

### **Solve Each Key with Same Tames:**

```bash
#!/bin/bash
# solve_multiple_keys.sh

TAMES_FILE="tames93.dat"
RANGE=93
DP=17

while IFS=',' read -r pubkey description; do
    # Skip comments
    [[ "$pubkey" =~ ^#.*$ ]] && continue
    [[ -z "$pubkey" ]] && continue

    echo "======================================"
    echo "Solving: $description"
    echo "PubKey: $pubkey"
    echo "======================================"

    ./rckangaroo \
        -tames "$TAMES_FILE" \
        -range $RANGE \
        -dp $DP \
        -pubkey "$pubkey" \
        -workfile "solve_${description// /_}.work" \
        -autosave 60

    if grep -q "PRIVATE KEY:" RESULTS.TXT; then
        echo "✓ SOLVED: $description"
        tail -1 RESULTS.TXT
    else
        echo "✗ Failed to solve: $description"
    fi

    echo ""
done < puzzle_keys_93bit.txt
```

**This allows you to:**
- Use same tames for multiple puzzles
- Solve them sequentially
- Each puzzle saves its own work file
- Automatic resume if interrupted

---

## Method 3: Parallel Solving (Multiple Machines)

If you have **multiple machines/GPUs**, run them in parallel with the same tames:

### **Machine 1:**
```bash
# Copy tames file to Machine 1
scp tames93.dat user@machine1:/path/to/rckangaroo/

# SSH to Machine 1
ssh user@machine1
cd /path/to/rckangaroo
./rckangaroo -tames tames93.dat -pubkey <pubkey_A> -range 93 -dp 17 -gpu 0
```

### **Machine 2:**
```bash
# Copy same tames file to Machine 2
scp tames93.dat user@machine2:/path/to/rckangaroo/

# SSH to Machine 2
ssh user@machine2
cd /path/to/rckangaroo
./rckangaroo -tames tames93.dat -pubkey <pubkey_B> -range 93 -dp 17 -gpu 0
```

**Both machines:**
- Use the SAME tames file (saves generation time)
- Solve DIFFERENT puzzles (parallel progress)
- Independent wild kangaroos (no collision)

---

## Method 4: Statistical Testing

### **Test Solve Time Distribution:**

```bash
#!/bin/bash
# test_solve_times.sh - Measure how long it takes to solve with your tames

TAMES_FILE="tames93.dat"
RANGE=93
DP=17
NUM_TESTS=20

echo "Range,DP,Tames_DPs,Solve_Time_Seconds,K_Factor" > solve_times.csv

for i in $(seq 1 $NUM_TESTS); do
    echo "Running test $i of $NUM_TESTS..."

    START=$(date +%s)
    ./rckangaroo -tames "$TAMES_FILE" -range $RANGE -dp $DP > test_output.txt 2>&1
    END=$(date +%s)

    ELAPSED=$((END - START))
    K_FACTOR=$(grep "Point solved, K:" test_output.txt | awk '{print $4}')
    TAMES_DPS=$(grep "tames loaded" test_output.txt | awk '{print $1}')

    echo "$RANGE,$DP,$TAMES_DPS,$ELAPSED,$K_FACTOR" >> solve_times.csv
    echo "  Solved in ${ELAPSED}s, K=${K_FACTOR}"
done

echo ""
echo "Statistical analysis:"
python3 << EOF
import csv
times = []
with open('solve_times.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        times.append(float(row['Solve_Time_Seconds']))

import statistics
print(f"Mean solve time: {statistics.mean(times):.1f}s")
print(f"Median solve time: {statistics.median(times):.1f}s")
print(f"Std deviation: {statistics.stdev(times):.1f}s")
print(f"Min: {min(times):.1f}s, Max: {max(times):.1f}s")
EOF
```

**This tells you:**
- Average time to solve with your tames
- Variability in solve times
- If your tames are "good quality"
- Expected solve time for real puzzles

---

## Method 5: Competitive Testing (Who Solves First?)

### **Race Multiple Configurations:**

```bash
#!/bin/bash
# race_test.sh - Test which configuration solves fastest

# Generate one random key for all to solve
PRIVKEY=$(python3 -c "import random; print(f'{random.randint(2**92, 2**93-1):064x}')")
# Convert to pubkey (requires external tool or RC-Kangaroo modification)
PUBKEY="<computed_pubkey>"

echo "Race to solve: $PUBKEY"
echo ""

# Configuration 1: With tames
echo "Config 1: With pre-generated tames"
timeout 3600 ./rckangaroo -tames tames93.dat -pubkey "$PUBKEY" -range 93 -dp 17 &
PID1=$!

# Configuration 2: Without tames (fresh generation)
echo "Config 2: Without tames (generate on-the-fly)"
timeout 3600 ./rckangaroo -pubkey "$PUBKEY" -range 93 -dp 17 &
PID2=$!

# Wait for first to finish
wait -n $PID1 $PID2
WINNER=$?

if ps -p $PID1 > /dev/null; then
    echo "Config 2 (no tames) won!"
    kill $PID1
else
    echo "Config 1 (with tames) won!"
    kill $PID2 2>/dev/null
fi
```

**This compares:**
- Pre-generated tames vs. fresh generation
- Which is faster for your specific hardware
- Validates the tames advantage

---

## Quick Test Commands

### **Verify Tames File is Valid:**
```bash
# Quick test - solve one random key (5 min timeout)
timeout 300 ./rckangaroo -tames tames93.dat -range 93 -dp 17
```

### **Measure Tames Quality:**
```bash
# Check how many DPs are loaded
./rckangaroo -tames tames93.dat -range 93 -dp 17 2>&1 | grep "tames loaded"
```

### **Continuous Testing (Run Overnight):**
```bash
# Solve random keys all night, log results
while true; do
    date >> tames_test_log.txt
    timeout 600 ./rckangaroo -tames tames93.dat -range 93 -dp 17 >> tames_test_log.txt 2>&1
    grep "Point solved" tames_test_log.txt | tail -1
done
```

---

## When to Use Each Method

| Method | Use Case | Time | Complexity |
|--------|----------|------|------------|
| **Method 1** | Quick validation | 5-10 min | Easy |
| **Method 2** | Multiple known puzzles | Hours | Medium |
| **Method 3** | Distributed solving | Days | Medium |
| **Method 4** | Statistical analysis | Hours | Advanced |
| **Method 5** | Performance comparison | Hours | Advanced |

---

## Expected Results

### **For 93-Bit Range:**

**Theoretical:**
- Expected operations: ~2^47.5 ≈ 186 trillion
- With good tames: K factor ~1.2-1.5
- With 30x tames: K factor ~0.5-0.8

**Practical (at 20 GKeys/s):**
- 1x tames: ~2-3 hours per solve
- 10x tames: ~20-40 minutes per solve
- 30x tames: ~10-15 minutes per solve

**If solving takes much longer:**
- Check DP bits (too high = rare collisions)
- Verify tames file loaded correctly
- Check hardware performance (GPU utilization)

---

## Troubleshooting

### **"tames loaded" shows 0 DPs**
```bash
# Tames file is empty or wrong range
# Regenerate tames for correct range
./rckangaroo -tames tames93.dat -range 93 -dp 17 -max 1000
```

### **Solve time is same with/without tames**
```bash
# Not enough tames generated
# Need at least 1x theoretical to see benefit
# Check: ./rckangaroo -tames ... | grep "tames loaded"
```

### **Getting "range mismatch" error**
```bash
# Tames were generated for different range
# Must regenerate for correct range
# Tames for 93 bits won't work for 94 bits!
```

---

## Summary

✅ **YES** - You can run tames against random public keys
✅ **Benchmark mode** - Easiest way to test (generates random keys automatically)
✅ **Multiple puzzles** - Solve sequentially with same tames
✅ **Parallel machines** - Share tames, solve different puzzles
✅ **Statistical testing** - Validate tames quality and solve times

**Recommended testing workflow:**
1. Generate tames: `./rckangaroo -tames tames93.dat -range 93 -dp 17 -max 10000`
2. Quick test: `timeout 300 ./rckangaroo -tames tames93.dat -range 93 -dp 17`
3. Verify DPs: Check "tames loaded" count in output
4. Statistical test: Run 10-20 solves, measure average time
5. When ready: Use for real puzzles when revealed

**Your tames are ready for battle!** 🎯
