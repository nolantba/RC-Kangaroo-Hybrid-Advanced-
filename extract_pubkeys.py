#!/usr/bin/env python3
"""
Extract Public Keys from Bitcoin Puzzle Private Keys

This script derives compressed public keys from the known private keys
of solved Bitcoin puzzles (60-70) for testing RC-Kangaroo.

Dependencies: None (uses pure Python with basic secp256k1 implementation)
"""

# secp256k1 curve parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

def modinv(a, m):
    """Modular inverse using extended Euclidean algorithm"""
    if a < 0:
        a = (a % m + m) % m
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    return x % m

def extended_gcd(a, b):
    """Extended Euclidean Algorithm"""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def point_add(p1, p2):
    """Add two points on secp256k1"""
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        if y1 == y2:
            # Point doubling
            s = (3 * x1 * x1 * modinv(2 * y1, P)) % P
        else:
            # Points are inverses
            return None
    else:
        # Point addition
        s = ((y2 - y1) * modinv(x2 - x1, P)) % P

    x3 = (s * s - x1 - x2) % P
    y3 = (s * (x1 - x3) - y1) % P

    return (x3, y3)

def point_multiply(k, point):
    """Multiply a point by scalar k using double-and-add"""
    if k == 0:
        return None
    if k == 1:
        return point

    result = None
    addend = point

    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1

    return result

def privkey_to_pubkey(privkey_hex):
    """Convert private key to compressed public key"""
    # Convert hex to integer
    privkey = int(privkey_hex, 16)

    # Multiply generator point by private key
    pubkey_point = point_multiply(privkey, (Gx, Gy))

    if pubkey_point is None:
        raise Exception("Invalid private key")

    x, y = pubkey_point

    # Compressed format: 02 if y is even, 03 if y is odd
    prefix = "02" if y % 2 == 0 else "03"
    compressed_pubkey = prefix + format(x, '064x')

    return compressed_pubkey.upper()

# Solved puzzles data (private keys)
puzzles = {
    60: {"address": "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su", "privkey": "0FC07A1825367BBE"},
    61: {"address": "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB", "privkey": "13C96A3742F64906"},
    62: {"address": "1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi", "privkey": "363D541EB611ABEE"},
    63: {"address": "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS", "privkey": "7CCE5EFDACCF6808"},
    64: {"address": "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN", "privkey": "F7051F27B09112D4"},
    65: {"address": "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe", "privkey": "1A838B13505B26867"},
    66: {"address": "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so", "privkey": "2832ED74F2B5E35EE"},
    67: {"address": "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9", "privkey": "730FC235C1942C1AE"},
    68: {"address": "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", "privkey": "BEBB3940CD0FC1491"},
    69: {"address": "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", "privkey": "101D83275FB2BC7E0C"},
    70: {"address": "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR", "privkey": "349B84B6431A6C4EF1"},
}

def main():
    print("=" * 80)
    print("Bitcoin Puzzle Public Key Extractor")
    print("=" * 80)
    print()
    print("Deriving compressed public keys from known private keys...")
    print()

    # Store results for later use
    results = []

    for puzzle_num in sorted(puzzles.keys()):
        data = puzzles[puzzle_num]
        privkey = data["privkey"]
        address = data["address"]

        print(f"Puzzle #{puzzle_num} ({puzzle_num}-bit)")
        print(f"  Address:     {address}")
        print(f"  Private Key: {privkey}")

        try:
            pubkey = privkey_to_pubkey(privkey)
            print(f"  Public Key:  {pubkey}")
            print()

            results.append({
                "puzzle": puzzle_num,
                "address": address,
                "privkey": privkey,
                "pubkey": pubkey
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Write to file for easy reference
    print("=" * 80)
    print("Writing to pubkeys.txt...")
    print("=" * 80)
    print()

    with open("pubkeys.txt", "w") as f:
        f.write("Bitcoin Puzzle Public Keys (Solved Puzzles 60-70)\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"Puzzle #{r['puzzle']}\n")
            f.write(f"Address:     {r['address']}\n")
            f.write(f"Private Key: {r['privkey']}\n")
            f.write(f"Public Key:  {r['pubkey']}\n")
            f.write("\n")

    print("✅ Public keys written to: pubkeys.txt")
    print()

    # Generate test commands
    print("=" * 80)
    print("Test Commands for RC-Kangaroo")
    print("=" * 80)
    print()

    with open("test_commands.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# RC-Kangaroo Testing Commands\n")
        f.write("# Generated from solved Bitcoin puzzles 60-70\n\n")

        for r in results:
            puzzle = r['puzzle']
            pubkey = r['pubkey']

            # Determine DP bits based on puzzle size
            if puzzle <= 62:
                dp_bits = 12
            elif puzzle <= 65:
                dp_bits = 13
            elif puzzle <= 68:
                dp_bits = 14
            else:
                dp_bits = 15

            print(f"# Test Puzzle {puzzle} (without tames)")
            cmd = f'./rckangaroo -pubkey {pubkey} -range {puzzle} -dp {dp_bits}'
            print(f"{cmd}")
            print()

            f.write(f"# Puzzle {puzzle}\n")
            f.write(f'echo "Testing Puzzle {puzzle}..."\n')
            f.write(f'{cmd}\n')
            f.write(f'echo ""\n\n')

    print("✅ Test commands written to: test_commands.sh")
    print()

    # Generate tames test commands
    with open("test_with_tames.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# RC-Kangaroo Tames Testing Commands\n")
        f.write("# Test pre-generated tames against solved puzzles\n\n")

        for r in results:
            puzzle = r['puzzle']
            pubkey = r['pubkey']

            if puzzle <= 65:
                dp_bits = 13
            elif puzzle <= 68:
                dp_bits = 14
            else:
                dp_bits = 15

            f.write(f"# Puzzle {puzzle} - First generate tames if not exists\n")
            f.write(f'if [ ! -f "tames{puzzle}.dat" ]; then\n')
            f.write(f'  echo "Generating 30x tames for {puzzle}-bit range..."\n')
            f.write(f'  ./rckangaroo -tames tames{puzzle}.dat -range {puzzle} -dp {dp_bits} -max 30\n')
            f.write(f'fi\n\n')
            f.write(f'echo "Testing Puzzle {puzzle} with tames..."\n')
            f.write(f'./rckangaroo -tames tames{puzzle}.dat -pubkey {pubkey} -range {puzzle} -dp {dp_bits}\n')
            f.write(f'echo ""\n\n')

    print("✅ Tames test commands written to: test_with_tames.sh")
    print()
    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print()
    print("1. Test without tames:")
    print("   chmod +x test_commands.sh")
    print("   ./test_commands.sh")
    print()
    print("2. Test with tames:")
    print("   chmod +x test_with_tames.sh")
    print("   ./test_with_tames.sh")
    print()
    print("3. Manual test example:")
    print(f"   ./rckangaroo -pubkey {results[0]['pubkey']} -range 60 -dp 12")
    print()

if __name__ == "__main__":
    main()
