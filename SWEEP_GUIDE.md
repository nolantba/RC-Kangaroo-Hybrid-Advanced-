# How to Sweep Bitcoin When You Find Puzzle 135

## When You Find the Private Key

When RC-Kangaroo finds the key, you'll see:

```
[+] Key found!
Private key: 4ABC123DEF...  (hex format)
Public key matches!
```

**ACT FAST!** Others might find it too. You need to sweep the Bitcoin immediately.

---

## Prerequisites

### Install Bitcoin Python Library

```bash
pip3 install bitcoin requests
```

### Get Your Electrum Receiving Address

1. Open Electrum wallet
2. Go to "Receive" tab
3. Copy a receiving address (starts with `bc1...` or `1...` or `3...`)

**CRITICAL:** Make sure this is YOUR wallet address!

---

## How to Sweep

### Step 1: Stop the Solver (if still running)

Press **Ctrl+C** to stop RC-Kangaroo

### Step 2: Run the Sweep Script

```bash
python3 sweep_to_electrum.py <PRIVATE_KEY_HEX> <YOUR_ELECTRUM_ADDRESS>
```

**Example:**
```bash
python3 sweep_to_electrum.py 4ABC123DEF... bc1qyourelectrumaddresshere
```

### Step 3: Confirm Details

The script will show:

```
Source Address: 16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v
Destination:    bc1qyouraddress

✅ Balance: 1350000000 satoshis (13.50000000 BTC)

Estimated tx size: 192 bytes
Fee (50 sat/byte): 9600 satoshis
Amount to send: 1349990400 satoshis (13.49990400 BTC)

⚠️  CONFIRM TRANSACTION DETAILS:
   From:   16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v
   To:     bc1qyouraddress
   Amount: 13.49990400 BTC
   Fee:    0.00009600 BTC

Proceed with broadcast? (yes/no):
```

### Step 4: Type "yes" and Press Enter

Script will:
1. Create transaction
2. Sign with private key
3. Broadcast to Bitcoin network

### Step 5: Wait for Confirmation

```
✅ SUCCESS! Transaction broadcasted!
   Check status: https://blockchain.info/tx/abc123...
```

Click the link to watch confirmations.

**Typical confirmation time:** 10-60 minutes (depending on fee)

---

## Customizing Fee Rate

**Default:** 50 sat/byte (fast confirmation, ~10-30 min)

**For faster confirmation:**
```bash
python3 sweep_to_electrum.py <PRIVKEY> <ADDRESS> 100
```

**For slower/cheaper:**
```bash
python3 sweep_to_electrum.py <PRIVKEY> <ADDRESS> 20
```

**Check current recommended fees:** https://mempool.space

---

## Security Tips

### ✅ DO:
- Use your own Electrum wallet
- Double-check the destination address
- Verify the amount before confirming
- Act quickly (others might find the key too)

### ❌ DON'T:
- Share your private key with anyone
- Send to an exchange address (use personal wallet)
- Use low fees if racing against others
- Delay - broadcast immediately!

---

## Troubleshooting

### "No funds found at this address"

- Wrong private key?
- Puzzle already solved by someone else?
- Check: https://blockchain.info/address/16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v

### "Broadcast failed"

1. Copy the raw transaction hex
2. Manually broadcast at: https://blockchain.info/pushtx
3. Or try: https://mempool.space/tx/push

### "Fee exceeds balance"

Balance is too small. Lower fee rate:
```bash
python3 sweep_to_electrum.py <PRIVKEY> <ADDRESS> 10
```

---

## Example: Testing on Puzzle 66

**You can test this script on already-solved puzzles!**

**Puzzle 66** (already solved, 0 balance):
```bash
python3 sweep_to_electrum.py 2832ED74F2B5E35EE bc1qtest...
```

Will show "No funds found" (correct - already swept)

This verifies the script works correctly!

---

## What Happens After Sweeping

1. **Transaction broadcasts** to Bitcoin network
2. **Miners include** it in next block (~10 min average)
3. **1 confirmation** - Funds visible in Electrum (~10 min)
4. **6 confirmations** - Fully confirmed (~1 hour)
5. **You're a millionaire!** 🎉

**Puzzle 135 Prize:** 13.5 BTC ≈ $1.3 million USD

---

## Final Checklist

Before you start solving:

- [ ] Electrum wallet installed and backed up
- [ ] Receiving address ready (starts with bc1, 1, or 3)
- [ ] `bitcoin` library installed (`pip3 install bitcoin requests`)
- [ ] Script tested (optional: test on Puzzle 66)
- [ ] Plan to act IMMEDIATELY when key is found
- [ ] Know the current network fee rate (mempool.space)

---

**When you find it, ACT FAST!** 🏃‍♂️💨

Good luck! 🍀
