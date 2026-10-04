# Prefix_Tree_SGLang_vLLM_Reimplementation

Simplified Python reimplementations of the two prefix-caching (KV-cache reuse) designs used by LLM serving engines, plus a benchmark harness that compares them on the same synthetic workloads:

- **SGLang-style radix tree** ([caches/radix.py](caches/radix.py)): a token-indexed radix tree. Requests that diverge partway through a node split it, so prefixes can match at any `page_size` granularity. Eviction drops the least-recently-used unlocked *leaf* node.
- **vLLM-style block hash table** ([caches/block_hash.py](caches/block_hash.py)): a flat map from chained block hashes to block IDs, at a fixed 16-token block size. Eviction frees one unreferenced block at a time, in LRU order.

Both caches are simulations: they track which tokens are cached and ignore the actual KV tensors. Each follows the same `match` → `insert` → `release` lifecycle, and reference counts (`lock_ref` / `ref_count`) keep in-flight requests from being evicted.

The question the project answers is: **how much hit rate does vLLM's 16-token block granularity cost, compared with SGLang's token-level radix tree, and do the two eviction policies behave differently under memory pressure?**

## Project layout

| Path | Contents |
|---|---|
| [caches/](caches/) | `radix.py` (SGLang port) and `block_hash.py` (vLLM port) |
| [traces/synthetic.py](traces/synthetic.py) | Workload generators: `agent`, `linear`, `fanout`, `shared-prefix`, `no-sharing`. They emit token sequences, and `to_block_hashes` turns them into chained CRC32 block hashes (mirroring vLLM's `hash_block_tokens`), so both caches replay the exact same trace. Segment lengths are deliberately not multiples of 16, so divergence points fall inside blocks. |
| [benchmarks/sweep.py](benchmarks/sweep.py) | Hit rate at effectively unlimited capacity, sweeping radix `page_size` over 1–32. This isolates the cost of granularity. |
| [benchmarks/sweep_eviction.py](benchmarks/sweep_eviction.py), [sweep_eviction2.py](benchmarks/sweep_eviction2.py) | Earlier capacity sweeps, which led to the matched-capacity experiment below. |
| [benchmarks/sweep_eviction_matched.py](benchmarks/sweep_eviction_matched.py) | Eviction comparison at block-aligned capacities, so both caches get identical budgets. At `page_size=16` the two match the same tokens, so any gap comes only from the eviction policy. |
| [tests/](tests/) | Unit tests for both caches (results below). |

Run any module from the repo root, for example `python -m benchmarks.sweep`.

## Key findings

**1. Granularity matters, and the cost grows with page size.** Hit rates at unlimited capacity (`python -m benchmarks.sweep`):

| Workload | block-hash (16) | radix p=1 | radix p=16 | radix p=32 |
|---|---|---|---|---|
| agent | 83.97% | 87.93% | 83.97% | 79.24% |
| linear | 77.93% | 85.06% | 77.93% | 70.80% |
| fanout | 78.33% | 83.27% | 78.33% | 72.75% |
| shared-prefix | 93.53% | 94.28% | 93.53% | 90.51% |
| no-sharing | 0.00% | 0.00% | 0.00% | 0.00% |

- Token-level matching (radix, `page_size=1`) beats 16-token blocks by **4–7 percentage points** on agent-style workloads. Most of the loss comes from rounding each divergence point down to a block boundary.
- At `page_size=16`, radix and block-hash match **exactly the same tokens**, which confirms that the two ports are equivalent at equal granularity.
- When there is one long shared prefix (`shared-prefix`), the gap shrinks to under 1 point, because the at-most-15-token loss per request is small next to a 500-token prefix.
- `no-sharing` stays at 0% for every configuration, which shows that neither cache matches content it shouldn't.

**2. Comparing eviction policies: radix leaf eviction vs. block-hash block eviction** (`python -m benchmarks.sweep_eviction_matched`).

**Setup.** From finding 1, radix at `page_size=16` matches exactly the same tokens as the block-hash cache. Running both at `page_size=16` with equal memory therefore leaves one difference: what each cache throws out when memory is full.

- **Radix** evicts a whole leaf node. A node can be any length, such as one agent's last 37-token observation.
- **Block-hash** evicts one 16-token block at a time.

Every capacity tested is a multiple of 16. The block cache gets `capacity // 16` blocks, so at a capacity of 2250 it would hold only 2240 tokens while radix holds 2250. Using multiples of 16 gives both caches the same budget.

**Where the two caches differ.** At most capacities both caches score exactly the same. They differ only in a narrow window, about 1900–2200 tokens on `fanout`. That window is just below the point where the working set fits, so the cache evicts constantly and the choice of what to drop matters. Within the window:

- At 1920 tokens, radix is **+1.19** points ahead.
- From 1984 to 2208 tokens, radix is **behind** every time: −4.38 points at the worst, shrinking to −0.20.

So on `fanout`, radix is mostly a little worse in that window. The effect is small and depends on capacity.

**Capacity needed to reach the full (unpressured) hit rate.** This asks how small each cache can be before it starts losing hits:

| Workload | Block-hash reaches full rate at | Radix at | Meaning |
|---|---|---|---|
| agent | 560 tok | 496 tok | radix needs ~11% **less** memory |
| fanout | 1984 tok | 2240 tok | radix needs ~13% **more** memory |
| linear, shared-prefix | same | same | no difference |

The direction flips with the workload, which is the strongest evidence that neither policy is better in general.

**Tokens discarded per eviction.** One worry was that radix might drop large chunks at once. It doesn't: both caches drop about the same number of tokens per eviction (ratio 0.73–1.03), so that is not what causes the differences above.

**Accounting check.** The radix cache keeps a running `used` counter. After every request, the benchmark compares it with the actual number of tokens in the tree. They match at every workload and capacity tested, so the hit rates above are not caused by a bookkeeping bug.

**Takeaway.** When both caches can match the same tokens, the choice of eviction policy makes only a small, workload-dependent difference. The real gap between SGLang-style and vLLM-style caching is the 16-token block granularity from finding 1, not eviction.

## Test Results

Run with:

```bash
python -m pytest tests/sgl/radix_tests.py tests/vllm/block_hash_test.py -v
```

(The test files don't follow pytest's default `test_*.py` naming, so they are passed explicitly.)

**50 passed in 0.03s** — Python 3.10.19, pytest 9.1.1, macOS (darwin)

### SGLang radix cache — `tests/sgl/radix_tests.py` (47 tests)

| Test class | Tests | Result |
|---|---|---|
| `TestCommonPrefixLen` | 6 | ✅ all passed |
| `TestMatch` | 9 | ✅ all passed |
| `TestSplit` | 5 | ✅ all passed |
| `TestInsert` | 6 | ✅ all passed |
| `TestLockRef` | 5 | ✅ all passed |
| `TestEviction` | 5 | ✅ all passed |
| `TestPageSize` | 4 | ✅ all passed |
| `TestInvariants` | 5 | ✅ all passed |
| `TestEquivalence` | 2 | ✅ all passed |

### vLLM block-hash cache — `tests/vllm/block_hash_test.py` (3 tests)

| Test | Result |
|---|---|
| `test_exact_prefix_match` | ✅ passed |
| `test_refcount_blocks_eviction` | ✅ passed |
| `test_eviction_is_lru` | ✅ passed |
