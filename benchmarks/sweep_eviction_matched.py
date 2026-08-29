"""Do the two structures evict differently?

At page_size = BLOCK_SIZE they match identically, so any hit-rate difference
under memory pressure comes from eviction policy alone:

    radix       evicts a whole leaf node, cascading to the parent when it
                becomes childless
    block-hash  evicts one block at a time, LRU over unreferenced blocks

**Why every capacity here is a multiple of BLOCK_SIZE.** The block cache is
sized `capacity // BLOCK_SIZE` blocks, which truncates. At capacity 2250 it
holds 140 blocks = 2240 tokens while radix holds 2250 -- not the same size,
so any comparison there is unfair. Only block-aligned capacities give both
caches identical budgets.

That matters because the earlier sweep found divergence at exactly one
capacity, 2016 = 126 x 16 -- the only block-aligned value it happened to
test. Either the effect is real and the other zeros were masked by unequal
capacity, or 2016 was a single unlucky eviction ordering. This settles it.

Run:  python3 -m benchmarks.sweep_eviction_matched
"""

from caches.radix import RadixCache
from caches.block_hash import BlockHashCache
from traces.synthetic import WORKLOADS, to_block_hashes, BLOCK_SIZE

PAGE = BLOCK_SIZE


# --------------------------------------------------------------------------
# instrumented runners
# --------------------------------------------------------------------------

def walk(cache):
    """Every node except the root."""
    stack, out = [cache.root], []
    while stack:
        n = stack.pop()
        for c in n.children.values():
            out.append(c)
            stack.append(c)
    return out


def run_radix(trace, capacity_tokens, page_size=PAGE, check=False):
    """Returns (hit_rate, evict_events, evicted_tokens)."""
    c = RadixCache(capacity_tokens, page_size=page_size)
    matched = total = 0
    events = evicted = 0
    prev_used = 0

    for tokens in trace:
        matched += c.match(tokens)
        total += len(tokens)
        node = c.insert(tokens)

        if c.used < prev_used:
            events += 1
            evicted += prev_used - c.used
        prev_used = c.used
        c.release(node)

        if check:
            actual = sum(len(n.key) for n in walk(c))
            assert c.used == actual, f"used={c.used}, tree has {actual}"

    return matched / total, events, evicted


def run_block(trace, capacity_tokens):
    """Same, reported in tokens so the two are directly comparable."""
    assert capacity_tokens % BLOCK_SIZE == 0, "capacity must be block-aligned"
    c = BlockHashCache(capacity_tokens // BLOCK_SIZE)

    matched = total = 0
    events = evicted_blocks = 0

    for tokens in trace:
        hs = to_block_hashes(tokens)
        hit = c.match(hs)
        matched += hit * BLOCK_SIZE
        total += len(tokens)

        before = len(c.table)
        c.insert(hs)
        # every miss should add a table entry; a shortfall means entries
        # were evicted to make room
        dropped = max(0, (before + (len(hs) - hit)) - len(c.table))
        if dropped:
            events += 1
            evicted_blocks += dropped

        c.release(hs)

    return matched / total, events, evicted_blocks * BLOCK_SIZE


def align(n):
    """Round down to a multiple of BLOCK_SIZE, at least one block."""
    return max(BLOCK_SIZE, (n // BLOCK_SIZE) * BLOCK_SIZE)


# --------------------------------------------------------------------------
# 0. is eviction firing at all?
# --------------------------------------------------------------------------

def part0_sanity():
    print("=" * 80)
    print("0. IS EVICTION FIRING?  (if rdx ev is 0 everywhere, stop here)")
    print("=" * 80)
    print("\n  A zero eviction count means either _evict never runs, or it")
    print("  runs without decrementing `used`. Either way nothing below is")
    print("  meaningful.\n")
    print(f"  {'workload':>14} {'cap':>7} {'blk ev':>7} {'rdx ev':>7} "
          f"{'blk hit':>9} {'rdx hit':>9}")

    ok = False
    for name, gen in WORKLOADS.items():
        if name == "no-sharing":
            continue
        trace = list(gen())
        cap = align(max(len(t) for t in trace))     # very tight
        try:
            bh, bev, _ = run_block(trace, cap)
            rh, rev, _ = run_radix(trace, cap)
        except RuntimeError as e:
            print(f"  {name:>14} {cap:>7}  {e}")
            continue
        if rev > 0:
            ok = True
        print(f"  {name:>14} {cap:>7} {bev:>7} {rev:>7} "
              f"{bh:>9.2%} {rh:>9.2%}")

    print("\n  " + ("radix eviction is firing -- continue"
                    if ok else
                    "*** radix never evicted. Check _evict and `used`. ***"))
    return ok


# --------------------------------------------------------------------------
# 1. matched-capacity sweep, all workloads
# --------------------------------------------------------------------------

def part1_matched_capacity():
    print("\n" + "=" * 80)
    print("1. MATCHED CAPACITY -- both caches hold exactly the same tokens")
    print("=" * 80)

    for name, gen in WORKLOADS.items():
        if name == "no-sharing":
            continue
        trace = list(gen())
        biggest = max(len(t) for t in trace)
        distinct = len({t for seq in trace for t in seq})

        lo, hi = align(biggest), align(distinct * 2)
        step = align(max(BLOCK_SIZE, (hi - lo) // 20))
        caps = list(range(lo, hi + 1, step))

        print(f"\n{name}   largest req {biggest}, distinct {distinct} tok")
        print(f"  {'cap':>7} {'blk hit':>9} {'rdx hit':>9} {'delta pp':>9} "
              f"{'blk ev':>7} {'rdx ev':>7} {'blk tok':>9} {'rdx tok':>9}")

        diverged = []
        for cap in caps:
            try:
                bh, bev, btok = run_block(trace, cap)
            except RuntimeError:
                print(f"  {cap:>7} {'OOM':>9}")
                continue
            try:
                rh, rev, rtok = run_radix(trace, cap)
            except RuntimeError:
                print(f"  {cap:>7} {bh:>9.2%} {'OOM':>9}")
                continue

            d = (rh - bh) * 100
            if abs(d) > 0.005:
                diverged.append(d)
            print(f"  {cap:>7} {bh:>9.2%} {rh:>9.2%} {d:>+9.2f} "
                  f"{bev:>7} {rev:>7} {btok:>9} {rtok:>9}")

        if diverged:
            signs = {d > 0 for d in diverged}
            verdict = ("consistent sign -- looks like a real policy difference"
                       if len(signs) == 1 else
                       "mixed signs -- looks like eviction-order noise")
            print(f"  -> {len(diverged)}/{len(caps)} diverged. {verdict}")
        else:
            print(f"  -> identical at all {len(caps)} capacities")


# --------------------------------------------------------------------------
# 2. zoom on the fanout anomaly
# --------------------------------------------------------------------------

def part2_zoom():
    print("\n" + "=" * 80)
    print("2. FANOUT, every multiple of 16 through the anomalous region")
    print("=" * 80)

    trace = list(WORKLOADS["fanout"]())
    print(f"\n  {'cap':>7} {'blk hit':>9} {'rdx hit':>9} {'delta pp':>9} "
          f"{'blk ev':>7} {'rdx ev':>7}")

    caps = list(range(1920, 2401, BLOCK_SIZE))
    diverged = []
    for cap in caps:
        try:
            bh, bev, _ = run_block(trace, cap)
            rh, rev, _ = run_radix(trace, cap)
        except RuntimeError:
            print(f"  {cap:>7}  OOM")
            continue
        d = (rh - bh) * 100
        flag = "  <--" if abs(d) > 0.005 else ""
        if flag:
            diverged.append(d)
        print(f"  {cap:>7} {bh:>9.2%} {rh:>9.2%} {d:>+9.2f} "
              f"{bev:>7} {rev:>7}{flag}")

    print(f"\n  {len(diverged)}/{len(caps)} block-aligned capacities diverged.")
    if len(diverged) > 2:
        signs = {d > 0 for d in diverged}
        print("  Consistent sign -> real eviction-policy difference."
              if len(signs) == 1 else
              "  Mixed signs -> ordering noise, not a policy difference.")
    else:
        print("  Too few to be a policy difference -- the earlier 2016 result")
        print("  was a single unlucky eviction ordering.")

def part2b_zoom_all():
    """Find each workload's saturation window, if it has one."""
    print("\n" + "=" * 80)
    print("2b. WHERE DOES EACH CACHE REACH ITS UNPRESSURED HIT RATE?")
    print("=" * 80)

    for name, gen in WORKLOADS.items():
        if name == "no-sharing":
            continue
        trace = list(gen())
        distinct = len({t for seq in trace for t in seq})

        # unpressured baseline
        ceiling, _, _ = run_block(trace, align(distinct * 4))

        print(f"\n{name}   ceiling {ceiling:.2%}")
        print(f"  {'cap':>7} {'blk hit':>9} {'rdx hit':>9} {'delta pp':>9}")

        blk_sat = rdx_sat = None
        lo = align(int(distinct * 0.08))
        hi = align(int(distinct * 0.9))
        for cap in range(lo, hi + 1, BLOCK_SIZE * 4):
            try:
                bh, _, _ = run_block(trace, cap)
                rh, _, _ = run_radix(trace, cap)
            except RuntimeError:
                continue
            if blk_sat is None and abs(bh - ceiling) < 1e-9:
                blk_sat = cap
            if rdx_sat is None and abs(rh - ceiling) < 1e-9:
                rdx_sat = cap
            d = (rh - bh) * 100
            if abs(d) > 0.005 or cap % (BLOCK_SIZE * 16) == 0:
                print(f"  {cap:>7} {bh:>9.2%} {rh:>9.2%} {d:>+9.2f}")

        if blk_sat and rdx_sat:
            extra = (rdx_sat - blk_sat) / blk_sat * 100
            print(f"  -> block-hash saturates at {blk_sat}, "
                  f"radix at {rdx_sat}  ({extra:+.1f}% more capacity)")
        else:
            print(f"  -> block-hash sat {blk_sat}, radix sat {rdx_sat}")
# --------------------------------------------------------------------------
# 3. how much does one eviction throw away?
# --------------------------------------------------------------------------

def part3_eviction_volume():
    print("\n" + "=" * 80)
    print("3. TOKENS DISCARDED PER EVICTION EVENT")
    print("=" * 80)
    print("\n  radix drops a whole leaf; block-hash drops one block (16 tok).")
    print("  A ratio well above 1 means radix throws away more per event,")
    print("  which is the mechanism that could make its hit rate differ.\n")
    print(f"  {'workload':>14} {'cap':>7} {'blk tok/ev':>11} "
          f"{'rdx tok/ev':>11} {'ratio':>7}")

    for name, gen in WORKLOADS.items():
        if name == "no-sharing":
            continue
        trace = list(gen())
        distinct = len({t for seq in trace for t in seq})
        cap = align(distinct // 3)
        try:
            _, bev, btok = run_block(trace, cap)
            _, rev, rtok = run_radix(trace, cap)
        except RuntimeError:
            print(f"  {name:>14} {cap:>7}  OOM")
            continue

        b_per = btok / bev if bev else 0.0
        r_per = rtok / rev if rev else 0.0
        ratio = r_per / b_per if b_per else 0.0
        print(f"  {name:>14} {cap:>7} {b_per:>11.1f} {r_per:>11.1f} "
              f"{ratio:>7.2f}")


# --------------------------------------------------------------------------
# 4. accounting invariant under pressure
# --------------------------------------------------------------------------

def part4_invariant():
    print("\n" + "=" * 80)
    print("4. ACCOUNTING INVARIANT under eviction pressure")
    print("=" * 80)

    ok = True
    for name, gen in WORKLOADS.items():
        trace = list(gen())
        biggest = max(len(t) for t in trace)
        for mult in (1, 2, 4):
            for ps in (1, PAGE):
                try:
                    run_radix(trace, align(biggest * mult), ps, check=True)
                except AssertionError as e:
                    print(f"  FAIL {name} cap={align(biggest*mult)} "
                          f"page={ps}: {e}")
                    ok = False
                except RuntimeError:
                    pass          # legitimately nothing evictable
    print("\n  used == tokens-in-tree holds everywhere" if ok else
          "\n  ACCOUNTING DRIFT -- fix before trusting anything above")


if __name__ == "__main__":
    if part0_sanity():
        part1_matched_capacity()
        part2_zoom()
        part2b_zoom_all()
        part3_eviction_volume()
    part4_invariant()