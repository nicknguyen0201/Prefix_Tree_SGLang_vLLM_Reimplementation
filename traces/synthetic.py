"""Synthetic workload generators.

Traces yield **token sequences**, not block hashes. Both cache ports consume
the same trace: the radix cache takes tokens directly, and the block-hash
cache derives chained block hashes via ``to_block_hashes``. Feeding both from
one source is what makes the comparison fair.

Two properties matter for the comparison to be non-degenerate:

**Segment lengths are not multiples of the block size.** If every logical
unit were exactly 16 tokens, divergence between requests would always land on
a block boundary and the two structures would tie by construction. Real
prompts do not divide evenly, so segments here are deliberately ragged.

**Block hashes are chained.** ``hash(parent_hash, block_tokens)``, mirroring
vLLM's ``hash_block_tokens``. A block's identity encodes its whole prefix, so
identical content after *different* prefixes does not falsely match. CRC32 is
used rather than Python's ``hash()``, which is randomised per process and
would make traces irreproducible across runs.
"""

import zlib
from typing import Dict, Iterator, List, Optional

Token = int
BlockHash = int

BLOCK_SIZE = 16


# --------------------------------------------------------------------------
# token -> block hash, for the vLLM-style cache
# --------------------------------------------------------------------------

def to_block_hashes(tokens: List[Token],
                    block_size: int = BLOCK_SIZE) -> List[BlockHash]:
    """Chained hashes over **full** blocks only.

    A trailing partial block is not hashed: its contents can still change as
    generation continues, so it is not cacheable. Both real systems do this.
    """
    h: Optional[int] = None
    out: List[BlockHash] = []
    for i in range(0, len(tokens) - block_size + 1, block_size):
        h = zlib.crc32(f"{h}|{tuple(tokens[i:i + block_size])}".encode())
        out.append(h)
    return out


# --------------------------------------------------------------------------
# token allocation
# --------------------------------------------------------------------------

class _TokenSpace:
    """Hands out disjoint token ranges, so distinct content never collides.
cd 
    A segment is a run of consecutive ids. The same key always returns the
    same segment; different keys never overlap.
    """

    def __init__(self, start: int = 1000):
        self._next = start
        self._cache: Dict[str, List[Token]] = {}

    def segment(self, key: str, length: int) -> List[Token]:
        if key not in self._cache:
            self._cache[key] = list(range(self._next, self._next + length))
            self._next += length
        seg = self._cache[key]
        assert len(seg) == length, f"segment {key!r} requested at two lengths"
        return seg


def _ragged(base: int, spread: int, salt: int) -> int:
    """A length that is rarely a multiple of the block size."""
    return base + (salt * 7919) % spread


# --------------------------------------------------------------------------
# workloads
# --------------------------------------------------------------------------

def agent_trace(
    n_sessions: int = 20,
    n_tools: int = 3,
    depth: int = 6,
    sys_tokens: int = 61,
    tool_tokens: int = 37,
    obs_base: int = 13,
    obs_spread: int = 11,
) -> Iterator[List[Token]]:
    """An agent workload: every step extends the previous step's prompt.

        [system prompt]                     shared by all sessions
            |-- [tool 0 preamble]           shared by sessions 0, 3, 6, ...
            |-- [tool 1 preamble]           shared by sessions 1, 4, 7, ...
            `-- [tool 2 preamble]           shared by sessions 2, 5, 8, ...
                    `-- [observations]      unique per session and step

    Request *d* of a session contains everything request *d-1* contained plus
    one new observation. That accumulation is the defining shape of agent
    traffic and the reason its prefix reuse is so high -- Preble measured 97%
    shared tokens on a comparable workload.

    Default lengths (61, 37, 13-23) are deliberately awkward against the
    16-token block size, so divergence points land inside blocks rather than
    on their edges.

    Args:
        n_sessions:  independent agent runs
        n_tools:     branching factor at the tool level
        depth:       steps per session, i.e. requests per session
        sys_tokens:  length of the universally shared system prompt
        tool_tokens: length of each tool-specific preamble
        obs_base:    minimum observation length
        obs_spread:  observation lengths vary over [base, base + spread)

    Yields:
        One token sequence per request, in arrival order.
    """
    space = _TokenSpace()

    for s in range(n_sessions):
        tool = s % n_tools

        seq: List[Token] = []
        seq += space.segment("sys", sys_tokens)
        if tool_tokens:
            seq += space.segment(f"tool{tool}", tool_tokens)

        for d in range(depth):
            n = _ragged(obs_base, obs_spread, s * 31 + d)
            seq = seq + space.segment(f"obs{s}.{d}", n)
            yield list(seq)          # copy: callers must not alias state


def linear_trace(n_sessions: int = 20, depth: int = 6,
                 **kwargs) -> Iterator[List[Token]]:
    """No branching: one shared prefix, then per-session divergence.

    The control for the branching hypothesis. If the two structures differ
    only under branching, they should agree here.
    """
    yield from agent_trace(n_sessions=n_sessions, n_tools=1, depth=depth,
                           tool_tokens=0, **kwargs)


def fanout_trace(n_sessions: int = 60, n_tools: int = 20,
                 depth: int = 3, **kwargs) -> Iterator[List[Token]]:
    """Heavy branching, shallow chains -- the case most favourable to radix.

    Many short subtrees mean many divergence points per unit of cached
    content, which is where sub-block matching should pay off if it ever does.
    """
    yield from agent_trace(n_sessions=n_sessions, n_tools=n_tools,
                           depth=depth, **kwargs)


def no_sharing_trace(n_requests: int = 120,
                     length: int = 97) -> Iterator[List[Token]]:
    """Every request unique. Hit rate must be ~0 at any capacity.

    A floor check: if this reports hits, a cache is matching something it
    should not.
    """
    space = _TokenSpace()
    for r in range(n_requests):
        yield list(space.segment(f"req{r}", length))


def shared_prefix_trace(n_requests: int = 120, prefix_tokens: int = 500,
                        suffix_base: int = 20,
                        suffix_spread: int = 13) -> Iterator[List[Token]]:
    """One long shared prefix, short unique suffixes.

    The ceiling case: near-total sharing, exactly one divergence point per
    request. If quantisation costs anything measurable, it should show up
    here in its purest form -- the loss is bounded at ``block_size - 1``
    tokens per request, against a 500-token prefix.
    """
    space = _TokenSpace()
    prefix = space.segment("shared", prefix_tokens)
    for r in range(n_requests):
        n = _ragged(suffix_base, suffix_spread, r)
        yield prefix + space.segment(f"suffix{r}", n)


# --------------------------------------------------------------------------

WORKLOADS = {
    "agent": agent_trace,
    "linear": linear_trace,
    "fanout": fanout_trace,
    "shared-prefix": shared_prefix_trace,
    "no-sharing": no_sharing_trace,
}


if __name__ == "__main__":
    for name, gen in WORKLOADS.items():
        trace = list(gen())
        toks = sum(len(t) for t in trace)
        blks = sum(len(to_block_hashes(t)) for t in trace)
        lens = [len(t) for t in trace]
        aligned = sum(1 for t in trace if len(t) % BLOCK_SIZE == 0)
        print(f"{name:>14}  {len(trace):4d} reqs  {toks:7d} tok  "
              f"{blks:6d} blk  len {min(lens)}-{max(lens)}  "
              f"{aligned} block-aligned")