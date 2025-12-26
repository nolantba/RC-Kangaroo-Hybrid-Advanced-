# Quick Start Testing Guide

You now have **real public keys** from solved Bitcoin puzzles to test your RC-Kangaroo implementation!

## What Was Created

✅ **pubkeys.txt** - Complete list of public keys for puzzles 60-70
✅ **test_commands.sh** - Ready-to-run test commands (without tames)
✅ **test_with_tames.sh** - Test commands using pre-generated tames
✅ **SOLVED_PUZZLES_TESTDATA.md** - Full documentation and reference
✅ **extract_pubkeys.py** - Script to regenerate public keys anytime

---

## Quick Test (5 minutes)

### Test 1: Verify Solver Works (Puzzle 60 - easiest)

```bash
# Should solve in ~10-30 seconds
./rckangaroo \
  -pubkey 0348E843DC5B1BD246E6309B4924B81543D02B16C8083DF973A89CE2C7EB89A10D \
  -range 60 \
  -dp 12
```

**Expected output:**
- Key found: `0FC07A1825367BBE`
- Solve time: ~10-30 seconds (depends on luck)

**If it works:** ✅ Your solver is working correctly!

---

### Test 2: Medium Difficulty (Puzzle 65)

```bash
# Should solve in ~2-10 minutes
./rckangaroo \
  -pubkey 0230210C23B1A047BC9BDBB13448E67DEDDC108946DE6DE639BCC75D47C0216B1B \
  -range 65 \
  -dp 13
```

**Expected key:** `1A838B13505B26867`

---

### Test 3: Generate Tames for Testing (Puzzle 66)

```bash
# Generate 10x tames (takes ~1-2 hours)
./rckangaroo -tames tames66_test.dat -range 66 -dp 14 -max 10
```

**Wait for completion**, then test with tames:

```bash
# Solve Puzzle 66 using tames (should be 3x faster)
./rckangaroo \
  -tames tames66_test.dat \
  -pubkey 024EE2BE2D4E9F92D2F5A4A03058617DC45BEFE22938FEED5B7A6B7282DD74CBDD \
  -range 66 \
  -dp 14
```

**Expected speedup:** ~3x faster than without tames

---

## Testing Your 30x Tames Strategy

### For Puzzle 93 (Your Real Target)

**Option A: Generate 30x Tames on Lower Puzzle First (Recommended)**

```bash
# Test on 66-bit first (proves the concept)
./rckangaroo -tames tames66.dat -range 66 -dp 14 -max 30

# Then solve with it
./rckangaroo \
  -tames tames66.dat \
  -pubkey 024EE2BE2D4E9F92D2F5A4A03058617DC45BEFE22938FEED5B7A6B7282DD74CBDD \
  -range 66 \
  -dp 14
```

**Expected tames generation time (66-bit, 30x):**
- Single GPU @ 20 GKeys/s: ~2-3 days
- Multi-GPU: Proportionally faster

**Expected solve time (66-bit with 30x tames):**
- Without tames: ~10 minutes
- With 30x tames: ~3 minutes
- **Speedup: 3.3x** ✅

---

**Option B: Go Straight for 93-Bit Tames**

```bash
# Generate 30x tames for 93-bit range (takes ~22 days)
./rckangaroo -tames tames93.dat -range 93 -dp 17 -max 30 -tames-autosave 300
```

**Important flags:**
- `-max 30` = Generate 30x theoretical tames
- `-tames-autosave 300` = Auto-save every 5 minutes
- Ctrl+C saves automatically (don't lose progress!)

**When Puzzle 93 public key is revealed:**

```bash
./rckangaroo \
  -tames tames93.dat \
  -pubkey <PUBKEY_WHEN_REVEALED> \
  -range 93 \
  -dp 17
```

**Expected solve time:**
- Without tames: ~659 days
- With 30x tames: ~22 days
- **Speedup: 30x** ✅

---

## Automated Testing

### Run All Tests (Without Tames)

```bash
./test_commands.sh
```

Tests puzzles 60-70 in sequence. **Watch for:**
- ✅ Each puzzle finds the correct private key
- ✅ Keys match the table in pubkeys.txt
- ✅ Solve times are reasonable

---

### Run All Tests (With Tames)

```bash
./test_with_tames.sh
```

**This script will:**
1. Generate 30x tames for each puzzle (if not exists)
2. Solve each puzzle using the tames
3. Compare solve times

**Note:** Generating tames for all puzzles 60-70 takes ~1 week total

---

## What You Should See (Success Indicators)

### ✅ Correct Behavior

```
[+] Key found!
Private key: 0FC07A1825367BBE
Verify: Public key matches target
Total time: 23.4 seconds
```

### ✅ Auto-Save Working

```
Auto-saving tames file...
Tames auto-saved (123456789 DPs)
```

### ✅ Ctrl+C Save Working

```
^C
Signal caught, gracefully shutting down...
Saving tames file...
Tames file saved successfully (987654321 DPs)
```

---

## Troubleshooting

### Issue: Solver doesn't find the key

**Check:**
1. Range matches puzzle number (`-range 66` for puzzle 66)
2. Public key is correct (copy from pubkeys.txt)
3. DP bits are appropriate (12-17 for puzzles 60-70)

**Try:**
```bash
# Add verbose flag
./rckangaroo -pubkey <PUBKEY> -range 60 -dp 12 -v
```

---

### Issue: "No public key" error

**Solution:** Make sure you're using the compressed public key (33 bytes, starts with 02 or 03)

**Correct format:**
```
0348E843DC5B1BD246E6309B4924B81543D02B16C8083DF973A89CE2C7EB89A10D
```

**Wrong format:**
```
1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su  ❌ This is the address, not pubkey!
```

---

### Issue: Tames don't speed up solving

**Check:**
1. Tames file generated for same range (`-range 66` for both)
2. Tames file has data (check file size, should be GBs)
3. Tames generation completed (wasn't interrupted)

**Verify tames file:**
```bash
ls -lh tames66.dat
# Should be 1-10+ GB for 30x tames
```

---

### Issue: Auto-save not working

**Solution:** Recompile with latest changes

```bash
make clean
make
```

**Verify flag works:**
```bash
./rckangaroo -h | grep -i autosave
# Should show: -tames-autosave <seconds>
```

---

## Reference Table

| Puzzle | Public Key | Private Key (Expected) | Est. Solve Time (No Tames) |
|--------|------------|------------------------|---------------------------|
| 60 | 0348E843...89A10D | 0FC07A1825367BBE | ~10-30 sec |
| 61 | 0249A438...6F453F | 13C96A3742F64906 | ~20-60 sec |
| 62 | 03231A67...14FEA8 | 363D541EB611ABEE | ~1-3 min |
| 63 | 0365EC29...A54579 | 7CCE5EFDACCF6808 | ~2-5 min |
| 64 | 03100611...5D9D4D | F7051F27B09112D4 | ~4-10 min |
| 65 | 0230210C...C0216B1B | 1A838B13505B26867 | ~8-20 min |
| 66 | 024EE2BE...DD74CBDD | 2832ED74F2B5E35EE | ~15-40 min |
| 67 | 0212209F...B7D46F75 | 730FC235C1942C1AE | ~30-80 min |
| 68 | 031FE02F...0A93BA1B | BEBB3940CD0FC1491 | ~1-3 hours |
| 69 | 024BABAD...03E7BA53 | 101D83275FB2BC7E0C | ~2-6 hours |
| 70 | 0290E690...21248483 | 349B84B6431A6C4EF1 | ~4-12 hours |

**Times assume:** Single GPU @ 20 GKeys/s, average luck

---

## Next Steps - Your 93-Bit Strategy

### Phase 1: Verify Everything Works ✅
1. Test Puzzle 60 (quick verification)
2. Test Puzzle 65 (medium difficulty)
3. Verify auto-save works
4. Verify Ctrl+C save works

### Phase 2: Test Tames on Puzzle 66 ✅
1. Generate 30x tames for 66-bit (~2-3 days)
2. Solve Puzzle 66 with tames
3. Verify 3-3.5x speedup
4. Confirm workflow is smooth

### Phase 3: Generate 93-Bit Tames 🎯
1. Start generating 30x tames for 93-bit (~22 days)
2. Monitor auto-saves (every 5 minutes recommended)
3. Let it run until completion
4. Keep tames file safe!

### Phase 4: Wait for Public Key Reveal ⏳
1. Monitor Puzzle 93 address: `1L2JsXHPMYuAa9ugvHGLwkdstCPUDemNCf`
2. Watch for spending transactions
3. Extract public key when revealed

### Phase 5: SOLVE! 🏆
1. Run solver with your 30x tames
2. Expected solve time: ~22 days (vs 659 days without tames)
3. Win the race against other solvers!

---

## Sources

- [Bitcoin Puzzle Private Keys Directory](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)
- [SecretScan Bitcoin Puzzle List](https://secretscan.org/Bitcoin_puzzle)
- [BTC Puzzle Info](https://btcpuzzle.info/puzzle)

---

**Ready to test? Start here:**

```bash
# Quick 30-second test
./rckangaroo -pubkey 0348E843DC5B1BD246E6309B4924B81543D02B16C8083DF973A89CE2C7EB89A10D -range 60 -dp 12
```

🚀 **Good luck with your 93-bit hunt!**
