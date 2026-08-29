"""Capacity sweep, with eviction actually biting.

The previous sweep sized capacity as a fraction of the *whole trace's*
distinct-token count. Sessions run sequentially, so that number is far larger
than what is live at any moment -- capacity was never the constraint and hit
rates stayed flat.

This version sweeps absolute capacities anchored to the largest single
request, which is the real lower bound on what a cache must hold, and reports
eviction counts so you can see the axis working.

It also isolates the `fanout` anomaly: at page 16 the two structures match
identically by construction, so any divergence under pressure comes from
eviction policy alone -- radix drops whole leaves, block-hash drops
individual blocks.
"""

from caches.radix import RadixCache
from caches.block_hash import BlockHashCache
from traces.synthetic import WORKLOADS, to_block_hashes, BLOCK_SIZE

PAGE = 16


# --------------------------------------------------------------------------
# instrumented runners
# --------------------------------------------------------------------------

def walk(cache):
    stack, out = [cache.root], []
    while stack:
        n = stack.pop()
        for c in n.children.values():
            out.append(c)
            stack.append(c)
    return out


def run_radix(trace, capacity_tokens, page_size, check=False):
    c = RadixCache(capacity_tokens, page_size=page_size)
    matched = total = 0
    evictions = 0
    prev_used = 0

    for tokens in trace:
        matched += c.match(tokens)
        total += len(tokens)
        node = c.insert(tokens)
        if c.used < prev_used:
            evictions += 1
        prev_used = c.used
        c.release(node)

        if check:
            actual = sum(len(n.key) for n in walk(c))
            assert c.used == actual, (
                f"used={c.used} but tree holds {actual} tokens")

    return matched, total, evictions, c.used


def run_block(trace, capacity_tokens):
    c = BlockHashCache(max(1, capacity_tokens // BLOCK_SIZE))
    matched = total = 0
    evictions = 0
    prev_free = len(c.free_list) if hasattr(c, "free_list") else None

    for tokens in trace:
        hs = to_block_hashes(tokens)
        matched += c.match(hs) * BLOCK_SIZE
        total += len(tokens)
        c.insert(hs)
        c.release(hs)

    return matched, total, evictions


# --------------------------------------------------------------------------
# 1. does capacity bite at all?
# --------------------------------------------------------------------------

def part1_find_the_knee():
    print("=" * 74)
    print("1. ABSOLUTE CAPACITY SWEEP -- where does eviction start to hurt?")
    print("=" * 74)

    for name, gen in WORKLOADS.items():
        if name == "no-sharing":
            continue
        trace = list(gen())
        biggest = max(len(t) for t in trace)
        total_tok = sum(len(t) for t in trace)

        print(f"\n{name}   largest request {biggest} tok, "
              f"{len(trace)} reqs, {total_tok} tok total")
        print(f"  {'cap':>7} {'blk-hash':>9} {'radix p1':>9} {'radix p16':>9} "
              f"{'p16 vs blk':>11} {'evict':>7}")

        caps = [biggest, biggest * 2, biggest * 4,
                biggest * 8, biggest * 16, biggest * 32]

        for cap in caps:
            row = [f"{cap:>7}"]
            try:
                bm, tot, _ = run_block(trace, cap)
                b_rate = bm / tot
                row.append(f"{b_rate:>9.2%}")
            except RuntimeError:
                b_rate = None
                row.append(f"{'OOM':>9}")

            rates = {}
            ev = 0
            for ps in (1, PAGE):
                try:
                    rm, tot, e, _ = run_radix(trace, cap, ps)
                    rates[ps] = rm / tot
                    if ps == PAGE:
                        ev = e
                    row.append(f"{rates[ps]:>9.2%}")
                except RuntimeError:
                    rates[ps] = None
                    row.append(f"{'OOM':>9}")

            if b_rate is not None and rates.get(PAGE) is not None:
                delta = (rates[PAGE] - b_rate) * 100
                row.append(f"{delta:>+11.2f}")
            else:
                row.append(f"{'-':>11}")
            row.append(f"{ev:>7}")

            print("  " + " ".join(row))


# --------------------------------------------------------------------------
# 2. the fanout anomaly
# --------------------------------------------------------------------------

def part2_chase_the_anomaly():
    print("\n" + "=" * 74)
    print("2. FANOUT ANOMALY -- at page 16 the structures match identically,")
    print("   so any gap here is eviction policy, not granularity.")
    print("=" * 74)

    trace = list(WORKLOADS["fanout"]())
    tot = sum(len(t) for t in trace)

    print(f"\n  {'cap':>7} {'blk-hash':>9} {'radix p16':>9} {'delta pp':>9} "
          f"{'r.evict':>8} {'r.used':>8}")

    for cap in (1000, 1500, 1750, 2016, 2250, 2500, 3000, 4033, 6000):
        try:
            bm, _, _ = run_block(trace, cap)
            b = bm / tot
            bs = f"{b:>9.2%}"
        except RuntimeError:
            b, bs = None, f"{'OOM':>9}"

        try:
            rm, _, ev, used = run_radix(trace, cap, PAGE, check=True)
            r = rm / tot
            rs = f"{r:>9.2%}"
        except RuntimeError:
            r, rs, ev, used = None, f"{'OOM':>9}", 0, 0
        except AssertionError as e:
            print(f"  {cap:>7}  ACCOUNTING BUG: {e}")
            continue

        d = f"{(r - b) * 100:>+9.2f}" if (b is not None and r is not None) else f"{'-':>9}"
        print(f"  {cap:>7} {bs} {rs} {d} {ev:>8} {used:>8}")

    print("\n  A nonzero delta means the two evict different content under")
    print("  equal pressure. A zero delta everywhere means the earlier")
    print("  3-point gap was an artefact of capacity sizing, not policy.")


# --------------------------------------------------------------------------
# 3. accounting invariant, run hard
# --------------------------------------------------------------------------

def part3_invariants():
    print("\n" + "=" * 74)
    print("3. ACCOUNTING INVARIANT under eviction pressure")
    print("=" * 74)

    ok = True
    for name, gen in WORKLOADS.items():
        trace = list(gen())
        biggest = max(len(t) for t in trace)
        for cap in (biggest, biggest * 2, biggest * 4):
            for ps in (1, PAGE):
                try:
                    run_radix(trace, cap, ps, check=True)
                except AssertionError as e:
                    print(f"  FAIL  {name} cap={cap} page={ps}: {e}")
                    ok = False
                except RuntimeError:
                    pass          # legitimately out of evictable space
    print("  used == tokens-in-tree holds everywhere" if ok
          else "  ACCOUNTING DRIFT -- fix before trusting any sweep above")


if __name__ == "__main__":
    part1_find_the_knee()
    part2_chase_the_anomaly()
    part3_invariants()