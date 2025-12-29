#!/usr/bin/env python3
"""
Sweep Bitcoin from found private key to Electrum wallet

Usage:
    python3 sweep_to_electrum.py <private_key_hex> <destination_address>

Example:
    python3 sweep_to_electrum.py 2832ED74F2B5E35EE bc1q...

IMPORTANT:
- Install required package: pip3 install bitcoin
- Use your own Electrum receiving address
- This broadcasts to mainnet - double check everything!
"""

import sys
import requests
import time
from bitcoin import *

def get_balance(address):
    """Get balance and UTXOs for an address using blockchain.info API"""
    try:
        url = f"https://blockchain.info/unspent?active={address}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            utxos = data.get('unspent_outputs', [])
            total = sum(utxo['value'] for utxo in utxos)
            return total, utxos
        elif response.status_code == 500:
            # No UTXOs found
            return 0, []
        else:
            print(f"Error fetching balance: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def create_and_broadcast_tx(private_key_hex, destination_address, fee_rate=50):
    """
    Create and broadcast transaction to sweep all funds

    Args:
        private_key_hex: Private key in hex format
        destination_address: Your Electrum receiving address
        fee_rate: Fee in satoshis per byte (default: 50 sat/byte for fast confirmation)
    """

    # Validate private key format
    if len(private_key_hex) != 64:
        # Pad with zeros if needed
        private_key_hex = private_key_hex.zfill(64)

    print(f"\n{'='*80}")
    print("Bitcoin Sweep Tool")
    print(f"{'='*80}\n")

    # Derive public key and address
    try:
        private_key_wif = encode_privkey(private_key_hex, 'wif_compressed')
        public_key = privkey_to_pubkey(private_key_hex)
        source_address = pubkey_to_address(public_key)

        print(f"Source Address: {source_address}")
        print(f"Destination:    {destination_address}\n")

    except Exception as e:
        print(f"Error deriving keys: {e}")
        return False

    # Get balance and UTXOs
    print("Fetching balance...")
    balance, utxos = get_balance(source_address)

    if balance is None:
        print("Failed to fetch balance. Check your internet connection.")
        return False

    if balance == 0:
        print("⚠️  No funds found at this address!")
        return False

    balance_btc = balance / 100000000
    print(f"✅ Balance: {balance} satoshis ({balance_btc:.8f} BTC)\n")

    if not utxos:
        print("No UTXOs to spend")
        return False

    # Estimate transaction size
    # Input: ~148 bytes per input (compressed pubkey)
    # Output: ~34 bytes
    # Overhead: ~10 bytes
    tx_size_estimate = len(utxos) * 148 + 34 + 10
    fee = tx_size_estimate * fee_rate

    print(f"Estimated tx size: {tx_size_estimate} bytes")
    print(f"Fee ({fee_rate} sat/byte): {fee} satoshis")

    # Calculate output amount
    output_amount = balance - fee

    if output_amount <= 0:
        print("Error: Fee exceeds balance!")
        return False

    output_btc = output_amount / 100000000
    print(f"Amount to send: {output_amount} satoshis ({output_btc:.8f} BTC)\n")

    # Confirm
    print("⚠️  CONFIRM TRANSACTION DETAILS:")
    print(f"   From:   {source_address}")
    print(f"   To:     {destination_address}")
    print(f"   Amount: {output_btc:.8f} BTC")
    print(f"   Fee:    {fee / 100000000:.8f} BTC")

    confirm = input("\nProceed with broadcast? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return False

    # Build transaction inputs
    inputs = []
    for utxo in utxos:
        inputs.append({
            'output': f"{utxo['tx_hash_big_endian']}:{utxo['tx_output_n']}",
            'value': utxo['value']
        })

    # Build transaction outputs
    outputs = [{
        'address': destination_address,
        'value': output_amount
    }]

    # Create transaction
    try:
        print("\nCreating transaction...")
        tx = mktx(inputs, outputs)

        print("Signing transaction...")
        # Sign all inputs
        for i in range(len(inputs)):
            tx = sign(tx, i, private_key_hex)

        print(f"Raw transaction: {tx}\n")

        # Broadcast
        print("Broadcasting to network...")
        url = "https://blockchain.info/pushtx"
        response = requests.post(url, data={'tx': tx}, timeout=30)

        if response.status_code == 200:
            print("\n✅ SUCCESS! Transaction broadcasted!")
            print(f"   Check status: https://blockchain.info/tx/{txhash(tx)}")
            return True
        else:
            print(f"\n❌ Broadcast failed: {response.text}")
            print(f"   You can manually broadcast: {tx}")
            return False

    except Exception as e:
        print(f"\n❌ Error creating/broadcasting transaction: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 sweep_to_electrum.py <private_key_hex> <destination_address>")
        print("\nExample:")
        print("  python3 sweep_to_electrum.py 2832ED74F2B5E35EE bc1qyourfancyelectrumaddress")
        print("\nNOTE: Make sure destination is YOUR Electrum receiving address!")
        sys.exit(1)

    private_key = sys.argv[1].strip()
    destination = sys.argv[2].strip()

    # Optional: fee rate (sat/byte)
    fee_rate = 50  # Default: 50 sat/byte for fast confirmation
    if len(sys.argv) > 3:
        try:
            fee_rate = int(sys.argv[3])
        except:
            print("Invalid fee rate, using default 50 sat/byte")

    success = create_and_broadcast_tx(private_key, destination, fee_rate)

    if success:
        print("\n🎉 Funds swept successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Sweep failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
