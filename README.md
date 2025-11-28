# Loomlock Advent Calendar Raffle

A deterministic raffle using xoshiro128+ PRNG seeded by a future block hash.

## Commitment

**Eligibility List Hash:** `1db51cfcb982eb1cdf2b519e4cceb0d250c129fbf4c92b376e3a92d95fe7e089`  
**Seed Block:** Ethereum Mainnet #23897900

## How It Works

1. The eligibility list was hashed and committed **before** the seed block was mined
2. The block hash of #23897900 is used as the random seed
3. The same seed + eligibility list always produces the same results

### Raffle Configuration

| | Prizes | Days | Token IDs |
|---|--------|------|-----------|
| Raffle 1 | 240 | 12 | 4 → 26 |
| Raffle 2 | 260 | 13 | 28 → 52 |
| **Total** | **500** | **25** | |

**Prize Caps (per raffle):**
- 1-9 tickets: max 1 prize
- 10-19 tickets: max 2 prizes  
- 20+ tickets: max 3 prizes

## Verify Results

### 1. Verify the eligibility list wasn't changed
NOTE: Eligibility CSV will be shared after the raffle is complete.

```bash
shasum -a 256 raffle_eligibility_list.csv
# Must output: 1db51cfcb982eb1cdf2b519e4cceb0d250c129fbf4c92b376e3a92d95fe7e089
```

### 2. Get the block hash
Check [Etherscan block #23897900](https://etherscan.io/block/23897900) for the block hash.

### 3. Run the raffle yourself
```bash
python raffle.py --seed 0x<BLOCK_HASH> --input raffle_eligibility_list.csv --output my_results/
```

### 4. Compare results
Your output should be **identical** to the published results.

## Files

| File | Description |
|------|-------------|
| `raffle.py` | Raffle script |
| `raffle_eligibility_list.csv` | Input: address, ticket_sizes |
| `raffle_results/` | Output: daily winner CSVs |

