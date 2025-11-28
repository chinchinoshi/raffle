#!/usr/bin/env python3
"""
Provably Fair Raffle - Verification Script

500 prizes across 2 raffles with per-raffle caps.
Uses xoshiro128+ PRNG for deterministic, reproducible results.

Config: Raffle 1 (240 prizes, 12 days), Raffle 2 (260 prizes, 13 days)
Caps per raffle: 1-9 tickets→1, 10-19→2, 20+→3
Token IDs: 4 to 52 (incrementing by 2 each day)

Usage: python run_raffle_verify.py --seed <BLOCK_HASH> --input eligibility.csv --output results/
"""

import argparse, csv, sys
from pathlib import Path

def xoshiro128(seed):
    def splitmix64(x):
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return x ^ (x >> 31)
    state = [0, 0, 0, 0]
    s = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(4):
        s = splitmix64(s)
        state[i] = s & 0xFFFFFFFF
    if all(x == 0 for x in state): state[0] = 1
    def next_rand():
        nonlocal state
        result = (state[0] + state[3]) & 0xFFFFFFFF
        t = (state[1] << 9) & 0xFFFFFFFF
        state[2] ^= state[0]; state[3] ^= state[1]
        state[1] ^= state[2]; state[0] ^= state[3]
        state[2] ^= t
        state[3] = ((state[3] << 11) | (state[3] >> 21)) & 0xFFFFFFFF
        return result / 0x100000000
    return next_rand

def get_cap(tickets):
    if tickets < 1: return 0
    elif tickets < 10: return 1
    elif tickets < 20: return 2
    else: return 3

def run_raffle(participants, num_prizes, rng):
    pool = {p['addr']: p['tickets'] for p in participants if p['tickets'] > 0}
    caps = {p['addr']: get_cap(p['tickets']) for p in participants if p['tickets'] > 0}
    wins = {addr: 0 for addr in pool}
    winners = []
    for _ in range(num_prizes):
        if not pool: break
        total = sum(pool.values())
        if total <= 0: break
        r, cumul = rng() * total, 0.0
        winner = None
        for addr, tickets in pool.items():
            cumul += tickets
            if r < cumul: winner = addr; break
        if not winner: winner = list(pool.keys())[-1]
        winners.append(winner)
        wins[winner] += 1
        if wins[winner] >= caps[winner]: del pool[winner]
    return winners

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True)
    parser.add_argument('--input', default='raffle_eligibility_list.csv')
    parser.add_argument('--output', default='raffle_results')
    args = parser.parse_args()
    
    try:
        seed = int(args.seed, 16) if args.seed.startswith('0x') else int(args.seed)
        participants = []
        with open(args.input, 'r') as f:
            for row in csv.DictReader(f):
                t = float(row['ticket_sizes'])
                if t > 0: participants.append({'addr': row['address'], 'tickets': t})
        
        rng = xoshiro128(seed)
        out = Path(args.output)
        
        # Raffle 1: 240 prizes, 12 days, tokens 4-26
        r1 = run_raffle(participants, 240, rng)
        (out / 'raffle_1').mkdir(parents=True, exist_ok=True)
        for d in range(12):
            day_winners = r1[d*20:(d+1)*20]
            wins = {}
            for a in day_winners: wins[a] = wins.get(a, 0) + 1
            with open(out / 'raffle_1' / f'day_{d+1}.csv', 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['address', 'quantity', 'token_id'])
                for a, q in wins.items(): w.writerow([a, q, 4 + d*2])
        
        # Raffle 2: 260 prizes, 13 days, tokens 28-52
        r2 = run_raffle(participants, 260, rng)
        (out / 'raffle_2').mkdir(parents=True, exist_ok=True)
        for d in range(13):
            day_winners = r2[d*20:(d+1)*20]
            wins = {}
            for a in day_winners: wins[a] = wins.get(a, 0) + 1
            with open(out / 'raffle_2' / f'day_{d+1}.csv', 'w', newline='') as f:
                w = csv.writer(f); w.writerow(['address', 'quantity', 'token_id'])
                for a, q in wins.items(): w.writerow([a, q, 28 + d*2])
        
        print("✅ Complete")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

