# benchmarks/sweep_eviction.py
"""Capacity sweep: what eviction costs on top of page quantisation.

`sweep.py` runs at a capacity far above every working set, so it isolates the
cost of page granularity alone. This sweep instead sets capacity as a fraction
of each workload's distinct-token count, so the cache is forced to evict, and
reports hit rate as capacity tightens.

Capacity is always expressed in *tokens*; the block-hash cache is given
capacity // BLOCK_SIZE blocks so both ports hold the same amount of content.
"""

from caches.radix import RadixCache
from caches.block_hash import BlockHashCache
from traces.synthetic import WORKLOADS, to_block_hashes, BLOCK_SIZE

FRACTIONS = (1.0, 0.5, 0.25, 0.1)
PAGE_SIZES = (1, 16)


def working_set(trace):
    """Distinct tokens in the trace — capacity needed to never evict."""
    return len({t for seq in trace for t in seq})


def run_radix(trace, capacity_tokens, page_size):
    c = RadixCache(capacity_tokens, page_size=page_size)
    matched = total = 0
    for tokens in trace:
        matched += c.match(tokens)
        total += len(tokens)
        c.release(c.insert(tokens))
    return matched, total


def run_block(trace, capacity_tokens):
    c = BlockHashCache(max(1, capacity_tokens // BLOCK_SIZE))
    matched = total = 0
    for tokens in trace:
        hs = to_block_hashes(tokens)
        matched += c.match(hs) * BLOCK_SIZE
        total += len(tokens)
        c.insert(hs)
        c.release(hs)
    return matched, total


def main():
    for name, gen in WORKLOADS.items():
        trace = list(gen())
        ws = working_set(trace)
        print(f"\n{name}  (working set {ws} tokens)")
        print(f"  {'capacity':>18}  {'block-hash':>10}  "
              + "  ".join(f"radix p={p}" for p in PAGE_SIZES))

        for frac in FRACTIONS:
            cap = max(BLOCK_SIZE, int(ws * frac))
            cells = []

            try:
                bm, tot = run_block(trace, cap)
                cells.append(f"{bm / tot:9.2%}")
            except RuntimeError as e:
                cells.append(f"{'RuntimeError':>9}")
                tot = sum(len(t) for t in trace)

            for ps in PAGE_SIZES:
                try:
                    rm, _ = run_radix(trace, cap, ps)
                    cells.append(f"{rm / tot:8.2%}")
                except RuntimeError:
                    cells.append(f"{'RuntimeError':>8}")

            label = f"{frac:.0%} = {cap:>6d} tok"
            print(f"  {label:>18}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
