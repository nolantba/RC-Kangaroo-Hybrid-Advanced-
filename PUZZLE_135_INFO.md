# Bitcoin Puzzle 135 - Complete Information

## Puzzle Details

**Puzzle Number:** 135
**Range Size:** 134-bit (key range from 2^134 to 2^135-1)
**Address:** `16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v`
**Public Key:** `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`
**Prize:** 13.5 BTC (~$1.3 million USD)
**Status:** UNSOLVED ✅

**IMPORTANT:** Use `-range 134` NOT `-range 135` (the search space is 2^134 in size)

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

**Theoretical operations:** ~2^67 × 1.15 = ~1.697 × 10^20 operations (for 134-bit range)

### Time Estimates (Average - Probabilistic Algorithm)

**IMPORTANT:** These are AVERAGE times. Actual time can be 50% faster or 2× slower due to luck!

**Direct Solving (Recommended for Solo):**

| GPU Speed | Average Solve Time |
|-----------|-------------------|
| 6.74 GK/s (3× RTX 3060) | ~798 years |
| 20 GK/s (High-end single GPU) | ~269 years |
| 67.4 GK/s (10× 3060 Ti) | ~80 years |
| 200 GK/s (Distributed) | ~27 years |

**With 30x Pre-Generated Tames (ONLY for teams/multiple keys):**

| Configuration | Time | When Useful |
|---------------|------|-------------|
| Generate 30x tames | ~26 years @ 6.74 GK/s | Team shares tames file |
| Solve with tames | ~532 years @ 6.74 GK/s | Each team member runs wilds |
| **Total (Solo)** | **~558 years** | ❌ SLOWER than direct! |

**WARNING:** For solo solving, generating 30x tames makes you **slower** because you do 30× the tames work but only solve 1 key!

### Distributed Computing (Direct Solving)

| GPU Count | Combined Speed | Average Time |
|-----------|---------------|--------------|
| 10 GPUs | ~67 GK/s | ~80 years |
| 100 GPUs | ~670 GK/s | ~8 years |
| 1,000 GPUs | ~6.7 TK/s | ~292 days |

**Bottom line:** Solo = not feasible. Requires **large distributed team** or extreme luck.

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

### ✅ RECOMMENDED: Direct Solving (Solo/Small Teams)

**For solo solving or small teams, DO NOT generate tames - just solve directly:**

```bash
./rckangaroo \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -range 134 \
  -dp 24
```

**Why this is optimal:**
- Runs both tames (33%) and wilds (67%) automatically
- K-factor ≈ 1.15 (optimal efficiency)
- No wasted time generating tames you'll only use once
- Average: ~798 years @ 6.74 GK/s (still extremely long but fastest option)

---

### 🔄 ALTERNATIVE: Generate Tames (Teams/Multiple Keys ONLY)

**⚠️ Only generate 30x tames if:**
- You have 30+ team members to share the tames file with
- You plan to solve 30+ different keys in the same 134-bit range
- You're contributing to a community tames pool

**Generate tames:**
```bash
./rckangaroo -tames tames135.dat -range 134 -dp 24 -max 30 -tames-autosave 300
```

**Each team member solves with shared tames:**
```bash
./rckangaroo \
  -tames tames135.dat \
  -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 \
  -range 134 \
  -dp 24
```

**Team Benefit:**
- 1 person generates tames (~26 years @ 6.74 GK/s)
- 30 people solve with shared tames in parallel (~532 years each)
- Combined speedup: Finding solution in ~18 years instead of 798 years
- **Requires coordination and trust!**

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

**If you were direct solving (recommended):**
```bash
# Just re-run - it continues automatically
./rckangaroo -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -range 134 -dp 24
```

**If you were generating tames (teams only):**
```bash
# Resumes from last auto-save
./rckangaroo -tames tames135.dat -range 134 -dp 24 -max 30 -tames-autosave 300
```

### Backup Tames File

```bash
# Backup periodically (while solver is stopped)
cp tames135.dat tames135_backup_$(date +%Y%m%d).dat
```

---

## Final Recommendations

### For Solo Miners

1. **Use direct solving** - Don't waste time generating tames
2. **Start with DP 24** - Monitor for overflow in first 5-10 minutes
3. **Set realistic expectations** - ~798 years average @ 6.74 GK/s
4. **Consider luck factor** - Could find in days (0.1% chance) or never
5. **Have a backup plan** - Maybe tackle Puzzle 93 instead?

### For Team Efforts

1. **Coordinate tames generation** - 1 person generates, all share
2. **Distribute work** - Each person runs wilds with shared tames
3. **Combine resources** - 100+ GPUs needed for reasonable timeframe
4. **Use cloud computing** - Rent GPUs when electricity is cheap
5. **Share the reward** - Agree on split before starting

### Reality Check

**At 6.74 GK/s solo:**
- Average: 798 years
- Best case (extreme luck): Days to weeks
- Worst case (bad luck): Never

**Better alternatives:**
- **Puzzle 93**: ~2.4 years average (with 30x tames: ~29 days)
- **Puzzle 70**: ~2.7 hours average
- **Puzzle 66**: ~21 minutes average

**Consider joining a distributed team for Puzzle 135 instead of solo mining!**

---

## Quick Reference

**✅ RECOMMENDED - Direct Solving (Solo):**
```bash
./rckangaroo -pubkey 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16 -range 134 -dp 24
```

**🔄 ALTERNATIVE - Tames Generation (Teams Only):**
```bash
./rckangaroo -tames tames135.dat -range 134 -dp 24 -max 30 -tames-autosave 300
```

**Public Key:** `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`

**Average Time:**
- Solo @ 6.74 GK/s: ~798 years (direct solving)
- Team @ 200 GK/s: ~27 years (100+ GPUs distributed)

**Monitor:** First 5-10 minutes for overflow warnings

**Reality Check:**
- This puzzle is effectively impossible for solo miners
- Requires massive distributed effort (hundreds of GPUs)
- Or extreme luck (could find in days, or never)
- Consider joining a team or tackling easier puzzles

---

Good luck! 🍀 This is a **century-long marathon**, not a sprint!
