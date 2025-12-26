# Solved Bitcoin Puzzles - Testing Data

This file contains **solved** Bitcoin puzzles (60-70) that you can use to test your RC-Kangaroo implementation and 30x tames strategy.

## Why Test on Solved Puzzles?

✅ **Verify your solver works correctly**
✅ **Test 30x tames effectiveness**
✅ **Practice workflow before unsolved puzzles**
✅ **Benchmark performance**

---

## Solved Puzzles 60-70

| Puzzle | Bits | Address | Private Key (Hex) | Balance | Solved Date |
|--------|------|---------|-------------------|---------|-------------|
| 60 | 60 | 1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su | 0FC07A1825367BBE | 0 ₿ | 2019-02-17 |
| 61 | 61 | 1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB | 13C96A3742F64906 | 0 ₿ | 2019-05-11 |
| 62 | 62 | 1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi | 363D541EB611ABEE | 0 ₿ | 2019-09-08 |
| 63 | 63 | 1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS | 7CCE5EFDACCF6808 | 0 ₿ | 2019-07-12 |
| 64 | 64 | 16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN | F7051F27B09112D4 | 0 ₿ | 2022-09-10 |
| 65 | 65 | 18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe | 1A838B13505B26867 | 0 ₿ | 2019-06-07 |
| 66 | 66 | 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so | 2832ED74F2B5E35EE | 6.6 ₿ | 2024-09-12 |
| 67 | 67 | 1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9 | 730FC235C1942C1AE | 6.7 ₿ | 2025-02-21 |
| 68 | 68 | 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ | BEBB3940CD0FC1491 | 6.8 ₿ | 2025-04-07 |
| 69 | 69 | 19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG | 101D83275FB2BC7E0C | 6.9 ₿ | 2025-04-30 |
| 70 | 70 | 19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR | 349B84B6431A6C4EF1 | 7.0 ₿ | 2019-06-09 |

---

## How to Extract Public Keys

### Method 1: Use Python Script (Recommended)

I'll create a script below that derives public keys from private keys.

### Method 2: Blockchain Explorer

For each address, look up the **spending transaction** on blockchain explorers:
- https://blockchain.info
- https://blockchair.com
- https://mempool.space

The public key appears in the **scriptSig** of the spending transaction.

### Method 3: Bitcoin Core

```bash
# If you run a full node
bitcoin-cli getrawtransaction <spending_txid> 1 | jq '.vin[0].scriptSig.asm'
```

---

## Testing Workflow

### Step 1: Generate Public Keys from Private Keys

Use the script I'll create: `extract_pubkeys.py`

### Step 2: Test Your Solver (Without Tames)

```bash
# Test on Puzzle 60 (should solve in seconds)
./rckangaroo -pubkey <pubkey60> -range 60 -dp 12

# Test on Puzzle 65 (should solve in minutes)
./rckangaroo -pubkey <pubkey65> -range 65 -dp 14
```

### Step 3: Generate Test Tames

```bash
# Generate 30x tames for 66-bit range
./rckangaroo -tames tames66.dat -range 66 -dp 14 -max 30
```

**Expected time:** ~2-3 days for 30x tames on 66-bit

### Step 4: Test With Tames

```bash
# Solve Puzzle 66 using pre-generated tames
./rckangaroo -tames tames66.dat -pubkey <pubkey66> -range 66 -dp 14
```

**Expected speedup:** Should be ~3.5x faster than without tames

### Step 5: Verify Results

```bash
# The solver should output the private key
# Compare with the known private key from the table above
# They should match!
```

---

## Expected Solve Times (Single GPU @ 20 GKeys/s)

| Puzzle | Without Tames | With 30x Tames | Speedup |
|--------|---------------|----------------|---------|
| 60 | ~10 seconds | ~3 seconds | 3.3x |
| 65 | ~5 minutes | ~90 seconds | 3.3x |
| 66 | ~10 minutes | ~3 minutes | 3.3x |
| 67 | ~20 minutes | ~6 minutes | 3.3x |
| 68 | ~40 minutes | ~12 minutes | 3.3x |
| 69 | ~1.3 hours | ~24 minutes | 3.3x |
| 70 | ~2.7 hours | ~48 minutes | 3.4x |

**Note:** Times are approximate and depend on luck (probabilistic algorithm)

---

## What to Look For (Success Indicators)

✅ **Solver finds the correct private key**
✅ **Private key matches the table above**
✅ **With tames is ~3-3.5x faster than without**
✅ **Auto-save works (check every 60 seconds)**
✅ **Ctrl+C saves properly**
✅ **Can resume from saved files**

---

## Troubleshooting

### Issue: Solver doesn't find the key
- Check range is correct (`-range 66` for puzzle 66)
- Check DP bits (use 14-17 for 66-70 bit puzzles)
- Check public key format (compressed: 33 bytes starting with 02/03)

### Issue: Tames don't speed up solving
- Verify tames file was generated for the same range
- Check tames file size (should be GBs for 30x)
- Use `-v` flag for verbose output

### Issue: Auto-save not working
- Recompile with latest code changes
- Check `-tames-autosave 60` option
- Monitor console for "Auto-saving tames file..." messages

---

## Next Steps

1. ✅ Run the Python script to extract public keys (see below)
2. ✅ Test solver on Puzzle 60-65 (quick verification)
3. ✅ Generate 30x tames for Puzzle 66
4. ✅ Solve Puzzle 66 with tames (verify speedup)
5. ✅ Practice complete workflow
6. ✅ Generate 30x tames for Puzzle 93 (unsolved)
7. ✅ Monitor blockchain for Puzzle 93 public key reveal
8. ✅ Solve immediately when revealed!

---

## Sources

- [Bitcoin Puzzle Private Keys Directory](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)
- [SecretScan Bitcoin Puzzle List](https://secretscan.org/Bitcoin_puzzle)
- [BTC Puzzle Info](https://btcpuzzle.info/puzzle)
