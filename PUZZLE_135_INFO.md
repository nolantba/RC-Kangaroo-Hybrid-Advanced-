# Bitcoin Puzzle 135 - Complete Information

## Puzzle Details

**Puzzle Number:** 135
**Range Size:** 134-bit (key range from 2^134 to 2^135-1)
**Address:** `16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v`
**Public Key:** `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`
**Prize:** 13.5 BTC (~$1.3 million USD)
**Status:** UNSOLVED ✅

**IMPORTANT:** Use `-range 134` NOT `-range 134` (the search space is 2^134 in size)

**Key Range (HEX):**
- Min: `4000000000000000000000000000000000` (2^134)
- Max: `7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF` (2^135-1)

**Key Range (Decimal):**
- Min: 2^134 = 21,778,071,482,940,061,661,655,974,875,633,165,533,184
- Max: 2^135-1 = 43,556,142,965,880,123,323,311,949,751,266,331,066,367
- **Range Size:** 2^134 ≈ 21.8 sextillion keys

---

## Why This Puzzle Has a Public Key

The puzzle creator made a **1000 satoshi outgoing transaction** from this address to intentionally reveal the public key. This makes it solvable using:
- ✅ Pollard's Kangaroo algorithm (RC-Kangaroo)
- ✅ Baby-step Giant-step algorithm
- ✅ Other ECDLP algorithms

**Without the public key**, these algorithms wouldn't work!

---

## Computational Difficulty

### Operations Required

**Theoretical operations:** ~2^67 × 1.15 = ~1.69 × 10^20 operations (for 134-bit range)

**Time estimates (Single GPU @ 20 GKeys/s):**

| Configuration | Estimated Time |
|---------------|----------------|
| No tames | ~2.1 billion years |
| 10x tames | ~210 million years |
| 30x tames | ~70 million years |
| 100x tames | ~21 million years |

### With Distributed Computing

| GPU Count | Time (30x tames) |
|-----------|------------------|
| 1 GPU | ~70 million years |
| 100 GPUs | ~700,000 years |
| 10,000 GPUs | ~7,000 years |
| 1,000,000 GPUs | ~70 years |

**Bottom line:** This requires **massive distributed effort** or breakthrough algorithms.

---

## DP Value Configuration

### Recommended DP Values

Based on your system (needed DP 20 for 93-bit with overflow):

**For 135-bit:**
- **Minimum safe:** `-dp 24`
- **Recommended:** `-dp 24` (start here)
- **If overflow:** `-dp 25` or `-dp 26`

### Why DP 24?

- 93-bit needed DP 20 on your system
- 135-bit is 42 bits larger
- DP 24 = 1 in 16,777,216 points saved
- Should prevent buffer overflow

**Monitor during first 5-10 minutes!** If you see overflow warnings, increase DP value immediately.

---

## Commands

### Generate 30x Tames for Puzzle 135

```bash
./rckangaroo -tames tames135.dat -range 134 -dp 24 -max 30 -tames-autosave 300
```

**Flags explained:**
- `-tames tames135.dat` = Save tames to this file
- `-range 134` = Search 134-bit keyspace (2^134 to 2^135-1) - CORRECT!
- `-dp 24` = Use 24 bits for distinguished points (prevents overflow)
- `-max 30` = Generate 30x theoretical tames
- `-tames-autosave 300` = Auto-save every 5 minutes (300 seconds)

### Solve With Tames (When Ready)

```bash
./rckangaroo \
  -tames tames135.dat \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -range 134 \
  -dp 24
```

### Solve Without Tames (Direct)

```bash
./rckangaroo \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -range 134 \
  -dp 24
```

---

## What to Monitor

### During First 5-10 Minutes (CRITICAL!)

**Watch for buffer overflow warnings:**

✅ **GOOD - No warnings:**
```
Auto-saving tames file...
Tames auto-saved (1234567 DPs)
```

❌ **BAD - Overflow warnings:**
```
⚠️  WARNING: DP BUFFER OVERFLOW!
    Dropped: 12345 DPs (0.1% loss)
    FIX: Use -dp 25 (current: 24)
```

**If you see overflow:**
1. Press Ctrl+C immediately (auto-saves)
2. Restart with higher DP value (e.g., `-dp 25` or `-dp 26`)
3. Monitor again

### During Generation

**Normal output:**
```
[GPU 0] 20.5 GKeys/s | DPs: 1234567 | Time: 1h 23m
Auto-saving tames file...
Tames auto-saved (1234567 DPs)
```

**What you want to see:**
- ✅ Stable GKeys/s rate
- ✅ DPs accumulating
- ✅ Auto-save every 5 minutes
- ✅ NO overflow warnings
- ✅ NO crashes

---

## Expected File Sizes

### Tames File Growth

| Time | DPs Expected | File Size (approx) |
|------|--------------|-------------------|
| 1 hour | ~1-10 million | ~100 MB - 1 GB |
| 1 day | ~25-250 million | ~2.5 GB - 25 GB |
| 1 week | ~175 million - 1.75 billion | ~17 GB - 175 GB |
| 1 month | ~750 million - 7.5 billion | ~75 GB - 750 GB |

**Make sure you have enough disk space!** (Recommend 1+ TB free)

---

## Realistic Strategy

### Option 1: Partial Contribution (Recommended)

Generate tames for as long as you can (days/weeks/months), then:
- Share tames file with community
- Join distributed solving effort
- Contribute to larger team

### Option 2: Long-term Solo Mining

Run continuously for months/years:
- Very low probability of success
- Requires extreme patience
- Massive electricity costs

### Option 3: Distributed Team Effort

Best chance of success:
- Multiple people generate different tames
- Share tames files
- Coordinate solving effort
- Split reward if found

---

## Cost Analysis

### Electricity Costs (Single GPU)

**Assumptions:**
- GPU power: 300W
- Electricity: $0.12/kWh
- Run time: 1 year

**Cost:** ~$315/year in electricity

**For 30x tames generation time (estimated months):**
- Cost: $79+ just to generate tames

### Is It Worth It?

**Puzzle 135 Prize:** ~$1.3 million
**Your Solo Success Probability:** Extremely low
**Electricity Cost:** Hundreds of dollars
**Time Investment:** Months+ of generation

**Verdict:** Only worth it if:
- You have cheap/free electricity
- You're joining a team effort
- You want to contribute to the community
- You understand it's a very long shot

---

## Progress Tracking

### Estimate Completion

The solver doesn't show % complete for tames generation, but you can estimate:

**For 30x tames on 135-bit:**
- Theoretical DP count: ~2^67.5 / 2^24 = ~2^43.5 DPs
- That's ~12.4 trillion DPs
- At 20 GKeys/s: Months to years

**Monitor your DP count:**
```bash
# Check current DPs (shown in auto-save messages)
# Compare to target: ~12.4 trillion DPs
```

---

## When to Stop / Pivot

**Consider stopping if:**
- ❌ Overflow warnings continue after increasing DP
- ❌ Costs exceed your budget
- ❌ GPU crashes frequently
- ❌ Disk space running out

**Consider pivoting to easier puzzle if:**
- You want faster results (try 66-70 bit puzzles)
- You want to test workflow first
- You want higher success probability

---

## Emergency Commands

### Check Tames File

```bash
ls -lh tames135.dat
```

### Resume After Crash

Just re-run the same command:
```bash
./rckangaroo -tames tames135.dat -range 134 -dp 24 -max 30 -tames-autosave 300
```

It should resume from last auto-save.

### Backup Tames File

```bash
# Backup periodically (while solver is stopped)
cp tames135.dat tames135_backup_$(date +%Y%m%d).dat
```

---

## Final Recommendations

1. **Start with DP 24** - Monitor for overflow
2. **Let it run for 1 week** - See how much progress you make
3. **Calculate remaining time** - Decide if it's worth continuing
4. **Consider sharing tames** - Team effort has better odds
5. **Set realistic expectations** - This is a VERY hard puzzle

---

## Quick Reference

**Start command:**
```bash
./rckangaroo -tames tames135.dat -range 134 -dp 24 -max 30 -tames-autosave 300
```

**Public Key:** `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`

**Monitor:** First 5-10 minutes for overflow warnings

**Auto-save:** Every 5 minutes

**Ctrl+C:** Saves before exit

**Resume:** Re-run same command

---

Good luck! 🍀 This is a marathon, not a sprint!
