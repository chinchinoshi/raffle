#!/usr/bin/env python3
"""
Weighted Raffle Script - 500 Prizes with Per-Raffle Caps

Runs 2 raffles with deterministic randomness (xoshiro128+) for provable fairness.
Winners are selected proportionally to their ticket count, with bracket-based prize caps PER RAFFLE.

Configuration:
  - Raffle 1: 240 prizes, 12 days (20/day), token_ids 4-26
  - Raffle 2: 260 prizes, 13 days (20/day), token_ids 28-52
  - Total: 500 prizes, 25 days

Prize caps PER RAFFLE (reset between raffles):
  - 1-9 tickets: max 1 prize per raffle
  - 10-19 tickets: max 2 prizes per raffle
  - 20+ tickets: max 3 prizes per raffle

Additional constraint:
  - Max 1 win per day per address (no duplicate token_ids per winner)

Usage:
    python run_raffle.py --seed <SEED_VALUE>
"""

import argparse
import csv
from pathlib import Path
from typing import Callable


# =============================================================================
# Deterministic PRNG: xoshiro128+
# =============================================================================

def create_xoshiro128(seed: int) -> Callable[[], float]:
    """
    Create a xoshiro128+ pseudo-random number generator.
    
    Returns a function that generates random floats in [0, 1).
    Given the same seed, it always produces the same sequence.
    """
    def splitmix64(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return x ^ (x >> 31)
    
    state = [0, 0, 0, 0]
    s = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(4):
        s = splitmix64(s)
        state[i] = s & 0xFFFFFFFF
    
    if all(s == 0 for s in state):
        state[0] = 1
    
    def next_random() -> float:
        nonlocal state
        result = (state[0] + state[3]) & 0xFFFFFFFF
        t = (state[1] << 9) & 0xFFFFFFFF
        state[2] ^= state[0]
        state[3] ^= state[1]
        state[1] ^= state[2]
        state[0] ^= state[3]
        state[2] ^= t
        state[3] = ((state[3] << 11) | (state[3] >> 21)) & 0xFFFFFFFF
        return result / 0x100000000
    
    return next_random


# =============================================================================
# Prize Bracket Configuration
# =============================================================================

def get_max_prizes_for_bracket(ticket_count: float) -> int:
    """
    Determine max prizes PER RAFFLE based on ticket count.
    
    - 1-9 tickets: max 1 prize per raffle
    - 10-19 tickets: max 2 prizes per raffle
    - 20+ tickets: max 3 prizes per raffle
    """
    if ticket_count < 1:
        return 0
    elif ticket_count < 10:
        return 1
    elif ticket_count < 20:
        return 2
    else:
        return 3


# =============================================================================
# Weighted Selection Algorithm (Day-by-Day with 1 win per day limit)
# =============================================================================

def select_daily_winners(
    pool: dict[str, float],
    num_prizes: int,
    rng: Callable[[], float]
) -> list[str]:
    """
    Select winners for a single day. Each address can only win once per day.
    
    Args:
        pool: dict of {address: tickets} for eligible participants this day
        num_prizes: prizes to distribute this day
        rng: random number generator
    
    Returns:
        List of winner addresses (unique, one per winner)
    """
    # Make a copy so we can remove winners
    day_pool = dict(pool)
    winners = []
    
    for _ in range(num_prizes):
        if not day_pool:
            break
        
        total_tickets = sum(day_pool.values())
        if total_tickets <= 0:
            break
        
        r = rng() * total_tickets
        cumulative = 0.0
        winner = None
        
        for addr, tickets in day_pool.items():
            cumulative += tickets
            if r < cumulative:
                winner = addr
                break
        
        if winner is None:
            winner = list(day_pool.keys())[-1]
        
        winners.append(winner)
        # Remove winner from today's pool (max 1 win per day)
        del day_pool[winner]
    
    return winners


def run_raffle_by_day(
    participants: list[dict],
    days: int,
    prizes_per_day: int,
    rng: Callable[[], float]
) -> list[list[str]]:
    """
    Run a raffle day by day with per-raffle caps and max 1 win per day.
    
    Returns:
        List of daily winner lists
    """
    # Build initial pool and caps
    base_pool = {}
    max_prizes = {}
    raffle_wins = {}
    
    for p in participants:
        addr = p['address']
        tickets = p['tickets']
        if tickets > 0:
            base_pool[addr] = tickets
            max_prizes[addr] = p['max_prizes']
            raffle_wins[addr] = 0
    
    daily_results = []
    
    for day in range(days):
        # Build today's pool: exclude those who hit their raffle cap
        today_pool = {
            addr: tickets 
            for addr, tickets in base_pool.items() 
            if raffle_wins[addr] < max_prizes[addr]
        }
        
        # Select winners for today (max 1 per address per day)
        day_winners = select_daily_winners(today_pool, prizes_per_day, rng)
        
        # Update raffle wins
        for addr in day_winners:
            raffle_wins[addr] += 1
        
        daily_results.append(day_winners)
    
    return daily_results


# =============================================================================
# Main Raffle Runner
# =============================================================================

def load_participants(csv_path: Path) -> list[dict]:
    """
    Load and filter participants from CSV.
    
    Accepts CSV with columns: address, ticket_sizes
    """
    participants = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            tickets = float(row['ticket_sizes'])
            
            if tickets <= 0:
                continue
            
            participant = {
                'address': row['address'],
                'tickets': tickets,
                'max_prizes': get_max_prizes_for_bracket(tickets),
            }
            
            participants.append(participant)
    
    return participants


def save_daily_results(
    daily_winners: list[list[str]],
    start_token_id: int,
    raffle_dir: Path
):
    """
    Save winners to daily CSV files with token_id column.
    Each winner appears only once per day (quantity always 1).
    """
    raffle_dir.mkdir(parents=True, exist_ok=True)
    
    for day, winners in enumerate(daily_winners):
        day_num = day + 1
        token_id = start_token_id + (day * 2)
        
        # Write daily CSV (quantity is always 1 now)
        day_path = raffle_dir / f'day_{day_num}.csv'
        with open(day_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['address', 'quantity', 'token_id'])
            
            for addr in winners:
                writer.writerow([addr, 1, token_id])
        
        print(f"    Day {day_num}: {len(winners)} winners, token_id={token_id}")


def save_summary(
    raffle_1_daily: list[list[str]],
    raffle_2_daily: list[list[str]],
    participants: list[dict],
    output_dir: Path
):
    """Save summary CSV with all winners."""
    participant_info = {p['address']: p['tickets'] for p in participants}
    
    # Count wins per address
    all_wins = {}
    raffle_1_wins = {}
    raffle_2_wins = {}
    
    for day_winners in raffle_1_daily:
        for addr in day_winners:
            raffle_1_wins[addr] = raffle_1_wins.get(addr, 0) + 1
            all_wins[addr] = all_wins.get(addr, 0) + 1
    
    for day_winners in raffle_2_daily:
        for addr in day_winners:
            raffle_2_wins[addr] = raffle_2_wins.get(addr, 0) + 1
            all_wins[addr] = all_wins.get(addr, 0) + 1
    
    # Sort by total prizes
    sorted_winners = sorted(all_wins.items(), key=lambda x: x[1], reverse=True)
    
    summary_path = output_dir / 'raffle_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['address', 'total_prizes', 'raffle_1_wins', 'raffle_2_wins', 'ticket_sizes'])
        
        for addr, total in sorted_winners:
            writer.writerow([
                addr,
                total,
                raffle_1_wins.get(addr, 0),
                raffle_2_wins.get(addr, 0),
                participant_info.get(addr, 0)
            ])
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"Total unique winners: {len(sorted_winners)}")
    print(f"Total prizes distributed: {sum(all_wins.values())}")


def parse_seed(seed_str: str) -> int:
    """Parse seed from string (supports hex like 0x... or decimal)."""
    seed_str = seed_str.strip()
    if seed_str.startswith('0x') or seed_str.startswith('0X'):
        return int(seed_str, 16)
    return int(seed_str)


def main():
    parser = argparse.ArgumentParser(
        description='Run weighted raffle with per-raffle caps (500 prizes, 2 raffles, 25 days)'
    )
    parser.add_argument(
        '--seed',
        required=True,
        help='Seed for PRNG (integer or hex string like 0x7f3a9b...)'
    )
    parser.add_argument(
        '--input',
        default='data/raffle_eligibility_list.csv',
        help='Path to input CSV with columns: address, ticket_sizes'
    )
    parser.add_argument(
        '--output',
        default='data/raffle_results',
        help='Output directory (default: data/raffle_results)'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent
    
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = workspace_root / input_path
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = workspace_root / output_dir
    
    seed = parse_seed(args.seed)
    
    # Configuration
    RAFFLE_1_DAYS = 12
    RAFFLE_2_DAYS = 13
    PRIZES_PER_DAY = 20
    START_TOKEN_ID = 4  # Token IDs: 4, 6, 8, ... 52
    
    raffle_1_prizes = RAFFLE_1_DAYS * PRIZES_PER_DAY
    raffle_2_prizes = RAFFLE_2_DAYS * PRIZES_PER_DAY
    
    print("=" * 70)
    print("WEIGHTED RAFFLE - 500 PRIZES (PER-RAFFLE CAPS + 1 WIN/DAY)")
    print("=" * 70)
    print(f"Seed: {args.seed} (decimal: {seed})")
    print()
    print("Configuration:")
    print(f"  - Raffle 1: {raffle_1_prizes} prizes, {RAFFLE_1_DAYS} days ({PRIZES_PER_DAY}/day)")
    print(f"  - Raffle 2: {raffle_2_prizes} prizes, {RAFFLE_2_DAYS} days ({PRIZES_PER_DAY}/day)")
    print(f"  - Total: {raffle_1_prizes + raffle_2_prizes} prizes, {RAFFLE_1_DAYS + RAFFLE_2_DAYS} days")
    print()
    print("Token IDs:")
    print(f"  - Raffle 1: {START_TOKEN_ID} to {START_TOKEN_ID + (RAFFLE_1_DAYS - 1) * 2} (days 1-12)")
    print(f"  - Raffle 2: {START_TOKEN_ID + RAFFLE_1_DAYS * 2} to {START_TOKEN_ID + (RAFFLE_1_DAYS + RAFFLE_2_DAYS - 1) * 2} (days 13-25)")
    print()
    print("Prize caps PER RAFFLE (reset between raffles):")
    print("  - 1-9 tickets: max 1 prize per raffle")
    print("  - 10-19 tickets: max 2 prizes per raffle")
    print("  - 20+ tickets: max 3 prizes per raffle")
    print()
    print("Additional: Max 1 win per day per address (unique token_ids)")
    print()
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()
    
    # Load participants
    participants = load_participants(input_path)
    print(f"Loaded {len(participants)} eligible participants (excluding 0 tickets)")
    
    total_tickets = sum(p['tickets'] for p in participants)
    print(f"Total tickets in pool: {total_tickets:,.0f}")
    
    # Create PRNG
    rng = create_xoshiro128(seed)
    
    # Run Raffle 1 (day by day)
    print(f"\n{'='*40}")
    print("RAFFLE 1")
    print(f"{'='*40}")
    print(f"  Prizes: {raffle_1_prizes} ({RAFFLE_1_DAYS} days x {PRIZES_PER_DAY}/day)")
    print(f"  Caps reset - all participants eligible")
    
    raffle_1_daily = run_raffle_by_day(participants, RAFFLE_1_DAYS, PRIZES_PER_DAY, rng)
    
    total_r1 = sum(len(day) for day in raffle_1_daily)
    unique_r1 = len(set(addr for day in raffle_1_daily for addr in day))
    print(f"  -> {total_r1} prizes distributed to {unique_r1} unique winners")
    
    # Save Raffle 1 daily results
    print("\n  Saving daily results:")
    raffle_1_dir = output_dir / 'raffle_1'
    save_daily_results(raffle_1_daily, START_TOKEN_ID, raffle_1_dir)
    
    # Run Raffle 2 (day by day)
    print(f"\n{'='*40}")
    print("RAFFLE 2")
    print(f"{'='*40}")
    print(f"  Prizes: {raffle_2_prizes} ({RAFFLE_2_DAYS} days x {PRIZES_PER_DAY}/day)")
    print(f"  Caps reset - all participants eligible")
    
    raffle_2_daily = run_raffle_by_day(participants, RAFFLE_2_DAYS, PRIZES_PER_DAY, rng)
    
    total_r2 = sum(len(day) for day in raffle_2_daily)
    unique_r2 = len(set(addr for day in raffle_2_daily for addr in day))
    print(f"  -> {total_r2} prizes distributed to {unique_r2} unique winners")
    
    # Save Raffle 2 daily results
    print("\n  Saving daily results:")
    raffle_2_start_token = START_TOKEN_ID + (RAFFLE_1_DAYS * 2)
    raffle_2_dir = output_dir / 'raffle_2'
    save_daily_results(raffle_2_daily, raffle_2_start_token, raffle_2_dir)
    
    # Save summary
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    save_summary(raffle_1_daily, raffle_2_daily, participants, output_dir)


if __name__ == '__main__':
    main()
