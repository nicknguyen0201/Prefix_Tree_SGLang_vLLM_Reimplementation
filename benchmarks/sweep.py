# benchmarks/sweep.py
from caches.radix import RadixCache
from caches.block_hash import BlockHashCache
from traces.synthetic import WORKLOADS, to_block_hashes


def run_radix(trace, capacity_tokens, page_size):
    c = RadixCache(capacity_tokens, page_size=page_size)
    matched = total = 0
    for tokens in trace:
        matched += c.match(tokens)
        total += len(tokens)
        c.release(c.insert(tokens))
    return matched, total


def run_block(trace, capacity_blocks):
    c = BlockHashCache(capacity_blocks)
    matched = total = 0
    for tokens in trace:
        hs = to_block_hashes(tokens)
        matched += c.match(hs) * 16          # blocks -> tokens
        total += len(tokens)
        c.insert(hs); c.release(hs)
    return matched, total


if __name__ == "__main__":
    CAP_TOKENS = 100_000                      # unlimited: isolate granularity
    for name, gen in WORKLOADS.items():
        trace = list(gen())
        print(f"\n{name}")

        bm, tot = run_block(trace, CAP_TOKENS // 16)
        print(f"  block-hash (page 16)   {bm/tot:6.2%}")

        for ps in (1, 2, 4, 8, 16, 32):
            rm, _ = run_radix(trace, CAP_TOKENS, ps)
            print(f"  radix page_size={ps:<3d}    {rm/tot:6.2%}")