"""Test suite for both prefix-cache ports.

Layout:

  TestCommonPrefixLen   pure sequence comparison + page quantisation
  TestMatch             tree descent, all three termination cases
  TestSplit             node splitting -- where the bugs live
  TestInsert            growth, sharing, and `used` accounting
  TestLockRef           refcounting across insert / release / split
  TestEviction          LRU over leaves, protected nodes survive
  TestPageSize          the axis the whole project sweeps
  TestInvariants        structural checks that must hold after any sequence
  TestEquivalence       radix vs. block-hash must agree where they should

The last two matter most. Invariants catch accounting drift, which is the
bug class behind SGLang's evictable_size() double-counting. Equivalence
catches the case where a difference you report is actually a bug in one port.
"""

import pytest

from caches.radix import RadixCache, TreeNode
from caches.block_hash import BlockHashCache


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def to_block_hashes(tokens, blk=16):
    """Chained block hashes over full blocks only, mirroring vLLM."""
    h, out = None, []
    for i in range(0, len(tokens) - blk + 1, blk):
        h = hash((h, tuple(tokens[i:i + blk])))
        out.append(h)
    return out


def walk(cache):
    """Every node except the root."""
    stack, out = [cache.root], []
    while stack:
        n = stack.pop()
        for c in n.children.values():
            out.append(c)
            stack.append(c)
    return out


def tokens_in_tree(cache):
    return sum(len(n.key) for n in walk(cache))


def build(cache, *sequences):
    """Insert and immediately release each sequence."""
    for s in sequences:
        cache.release(cache.insert(s))
    return cache


# --------------------------------------------------------------------------

class TestCommonPrefixLen:
    """Pure two-sequence comparison. No tree involved."""

    def test_full_agreement(self):
        c = RadixCache(1000)
        assert c._common_prefix_len((1, 2, 3), (1, 2, 3)) == 3

    def test_partial(self):
        c = RadixCache(1000)
        assert c._common_prefix_len((1, 2, 3), (1, 2, 9)) == 2

    def test_no_agreement(self):
        c = RadixCache(1000)
        assert c._common_prefix_len((1, 2), (7, 8)) == 0

    def test_one_is_prefix_of_other(self):
        c = RadixCache(1000)
        assert c._common_prefix_len((1, 2, 3), (1, 2)) == 2
        assert c._common_prefix_len((1, 2), (1, 2, 3)) == 2

    def test_empty(self):
        c = RadixCache(1000)
        assert c._common_prefix_len((), (1, 2)) == 0

    def test_quantised_down(self):
        """The mechanism the project measures: 20 matched tokens report as 16."""
        c = RadixCache(1000, page_size=16)
        a = tuple(range(32))
        b = tuple(range(20)) + tuple(range(100, 112))
        assert c._common_prefix_len(a, b) == 16      # not 20


class TestMatch:
    """Tree descent. Three termination cases, all reachable."""

    def _tree(self, page_size=1):
        #  root -> (1,2) -> {3: (3,4,5), 9: (9,)}
        c = RadixCache(1000, page_size=page_size)
        a = TreeNode((1, 2), c.root)
        c.root.children[c._child_key((1, 2))] = a
        b = TreeNode((3, 4, 5), a)
        a.children[c._child_key((3, 4, 5))] = b
        d = TreeNode((9,), a)
        a.children[c._child_key((9,))] = d
        return c

    def test_full_path(self):
        assert self._tree().match([1, 2, 3, 4, 5]) == 5

    def test_ends_inside_a_node(self):
        """Divergence mid-node — the case block hashing cannot express."""
        assert self._tree().match([1, 2, 3, 4]) == 4

    def test_other_branch(self):
        assert self._tree().match([1, 2, 9]) == 3

    def test_no_matching_child(self):
        assert self._tree().match([1, 2, 7]) == 2

    def test_nothing_matches(self):
        assert self._tree().match([7, 8]) == 0

    def test_partial_first_node(self):
        """Half of node (1,2) is still a genuine hit."""
        assert self._tree().match([1]) == 1

    def test_longer_than_tree(self):
        assert self._tree().match([1, 2, 3, 4, 5, 6, 7]) == 5

    def test_empty_request(self):
        assert self._tree().match([]) == 0

    def test_match_does_not_mutate(self):
        """A router queries many workers; a query must not perturb state."""
        c = self._tree()
        before = tokens_in_tree(c)
        node_count = len(walk(c))
        c.match([1, 2, 3, 4])          # would split, if match split
        assert tokens_in_tree(c) == before
        assert len(walk(c)) == node_count


class TestSplit:
    """Where the bugs live: refcounts and rewiring across a split."""

    def test_shape_after_split(self):
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3, 4, 5]))
        c.release(c.insert([1, 2, 9]))

        assert c.match([1, 2, 3, 4, 5]) == 5      # original path intact
        assert c.match([1, 2, 9]) == 3            # new path reachable
        assert c.match([1, 2]) == 2               # shared prefix is now a node

    def test_split_preserves_token_count(self):
        """A split redistributes tokens; it does not create or destroy them."""
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3, 4, 5]))
        assert c.used == 5
        c.release(c.insert([1, 2, 9]))
        assert c.used == 6                        # only (9,) is new
        assert tokens_in_tree(c) == c.used

    def test_split_inherits_lock_ref(self):
        """A live request holds both halves. Losing this evicts under a request."""
        c = RadixCache(1000)
        held = c.insert([1, 2, 3, 4, 5])          # NOT released
        c.release(c.insert([1, 2, 9]))            # forces the split

        for n in walk(c):
            if n.key == (1, 2):
                assert n.lock_ref > 0, "shared prefix lost its refcount"
                break
        else:
            pytest.fail("expected a (1,2) node after the split")

    def test_child_object_identity_survives(self):
        """The truncated child is the same object; held references stay valid."""
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3, 4, 5]))
        original = c.root.children[c._child_key((1, 2, 3, 4, 5))]
        c.release(c.insert([1, 2, 9]))
        new_parent = c.root.children[c._child_key((1, 2))]
        assert new_parent.children[c._child_key((3, 4, 5))] is original

    def test_rejects_degenerate_split(self):
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3]))
        child = c.root.children[c._child_key((1, 2, 3))]
        with pytest.raises(AssertionError):
            c._split_node(child, 0)
        with pytest.raises(AssertionError):
            c._split_node(child, 3)


class TestInsert:
    def test_first_insert(self):
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3]))
        assert c.match([1, 2, 3]) == 3
        assert c.used == 3

    def test_duplicate_adds_nothing(self):
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3]))
        c.release(c.insert([1, 2, 3]))
        assert c.used == 3

    def test_extension_adds_only_the_suffix(self):
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3]))
        c.release(c.insert([1, 2, 3, 4, 5]))
        assert c.used == 5
        assert c.match([1, 2, 3, 4, 5]) == 5

    def test_disjoint_sequences(self):
        c = RadixCache(1000)
        build(c, [1, 2, 3], [7, 8, 9])
        assert c.used == 6
        assert c.match([1, 2, 3]) == 3
        assert c.match([7, 8, 9]) == 3

    def test_returns_terminal_node(self):
        c = RadixCache(1000)
        n = c.insert([1, 2, 3])
        assert isinstance(n, TreeNode)
        assert n.key == (1, 2, 3)

    def test_empty_insert_is_noop(self):
        c = RadixCache(1000)
        c.insert([])
        assert c.used == 0


class TestLockRef:
    def test_insert_locks_whole_path(self):
        c = RadixCache(1000)
        c.insert([1, 2, 3])                       # not released
        for n in walk(c):
            assert n.lock_ref > 0

    def test_release_unlocks_whole_path(self):
        c = RadixCache(1000)
        c.release(c.insert([1, 2, 3]))
        for n in walk(c):
            assert n.lock_ref == 0

    def test_two_holders(self):
        c = RadixCache(1000)
        a = c.insert([1, 2, 3])
        b = c.insert([1, 2, 3])
        node = c.root.children[c._child_key((1, 2, 3))]
        assert node.lock_ref == 2
        c.release(a)
        assert node.lock_ref == 1
        c.release(b)
        assert node.lock_ref == 0

    def test_shared_prefix_counts_both(self):
        c = RadixCache(1000)
        a = c.insert([1, 2, 3, 4])
        b = c.insert([1, 2, 9])
        shared = c.root.children[c._child_key((1, 2))]
        assert shared.lock_ref == 2, "shared prefix should be held by both"
        c.release(a); c.release(b)
        assert shared.lock_ref == 0

    def test_root_never_locked_by_walk(self):
        c = RadixCache(1000)
        before = c.root.lock_ref
        c.release(c.insert([1, 2, 3]))
        assert c.root.lock_ref == before


class TestEviction:
    def test_evicts_when_full(self):
        c = RadixCache(6)
        build(c, [1, 2, 3], [7, 8, 9])
        assert c.used <= 6
        build(c, [4, 5, 6])                       # must evict something
        assert c.used <= 6
        assert c.match([4, 5, 6]) == 3            # the newest survived

    def test_lru_order(self):
        c = RadixCache(6)
        build(c, [1, 2, 3])
        build(c, [7, 8, 9])
        c.match([1, 2, 3])                        # a query must NOT refresh
        build(c, [4, 5, 6])
        assert c.match([1, 2, 3]) == 0, "oldest should have been evicted"

    def test_locked_nodes_survive(self):
        c = RadixCache(6)
        held = c.insert([1, 2, 3])                # not released
        build(c, [7, 8, 9])
        build(c, [4, 5, 6])                       # forces eviction
        assert c.match([1, 2, 3]) == 3, "a held node was evicted"
        c.release(held)

    def test_all_locked_raises(self):
        c = RadixCache(3)
        c.insert([1, 2, 3])                       # fills and holds
        with pytest.raises(RuntimeError):
            c.insert([7, 8, 9])

    def test_used_drops_on_eviction(self):
        c = RadixCache(6)
        build(c, [1, 2, 3], [7, 8, 9], [4, 5, 6])
        assert c.used == tokens_in_tree(c)


class TestPageSize:
    """The sweep axis: how much match rate quantisation costs."""

    def test_page_1_is_exact(self):
        c = RadixCache(1000, page_size=1)
        c.release(c.insert(list(range(20))))
        query = list(range(20)) + [999]
        assert c.match(query) == 20

    def test_page_16_truncates_the_input(self):
        c = RadixCache(1000, page_size=16)
        c.release(c.insert(list(range(20))))      # only 16 are cacheable
        assert c.match(list(range(20))) == 16

    def test_page_16_quantises_the_match(self):
        """Identical through token 20; page 16 reports 16."""
        c = RadixCache(1000, page_size=16)
        c.release(c.insert(list(range(32))))
        query = list(range(20)) + list(range(100, 112))
        assert c.match(query) == 16

    def test_larger_page_never_matches_more(self):
        seq = list(range(64))
        query = list(range(50)) + list(range(200, 214))
        prev = None
        for ps in (1, 2, 4, 8, 16, 32):
            c = RadixCache(1000, page_size=ps)
            c.release(c.insert(seq))
            m = c.match(query)
            if prev is not None:
                assert m <= prev, f"page_size={ps} matched more than {prev}"
            prev = m


class TestInvariants:
    """Structural checks. These catch accounting drift, not logic errors."""

    SEQS = [
        [1, 2, 3, 4, 5],
        [1, 2, 9],
        [1, 2, 3, 4, 7],
        [7, 8],
        [1],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]

    def test_used_matches_tree_contents(self):
        c = RadixCache(1000)
        build(c, *self.SEQS)
        assert c.used == tokens_in_tree(c)

    def test_parent_child_pointers_agree(self):
        c = RadixCache(1000)
        build(c, *self.SEQS)
        for n in walk(c):
            assert n.parent is not None
            assert c._child_key(n.key) in n.parent.children
            assert n.parent.children[c._child_key(n.key)] is n

    def test_siblings_differ_at_first_page(self):
        """The invariant that makes single-page dict lookup correct."""
        c = RadixCache(1000)
        build(c, *self.SEQS)
        for n in [c.root] + walk(c):
            firsts = [c._child_key(ch.key) for ch in n.children.values()]
            assert len(firsts) == len(set(firsts))

    def test_no_empty_nodes(self):
        c = RadixCache(1000)
        build(c, *self.SEQS)
        for n in walk(c):
            assert len(n.key) > 0

    def test_all_refcounts_zero_after_release(self):
        c = RadixCache(1000)
        build(c, *self.SEQS)
        for n in walk(c):
            assert n.lock_ref == 0, f"leaked refcount on {n.key}"


class TestEquivalence:
    """Where the two ports must agree — and where they legitimately differ."""

    SEQS = [
        list(range(64)),
        list(range(32)) + list(range(100, 132)),
        list(range(16)) + list(range(200, 248)),
        list(range(64)) + list(range(300, 316)),
    ]

    def test_agree_at_page_16_unlimited_capacity(self):
        """With no eviction and matched granularity, the structures are
        equivalent. Any divergence here is a bug in one port, not a finding."""
        r = RadixCache(100_000, page_size=16)
        b = BlockHashCache(100_000)

        for seq in self.SEQS:
            hashes = to_block_hashes(seq)
            rm = r.match(seq) // 16          # tokens -> blocks
            bm = b.match(hashes)
            assert rm == bm, f"divergence on {len(seq)} tokens: {rm} vs {bm}"
            r.release(r.insert(seq))
            b.insert(hashes); b.release(hashes)

    def test_radix_never_matches_less_at_page_1(self):
        """Token granularity can only find at least as much as block
        granularity. This is the direction the hypothesis predicts."""
        r = RadixCache(100_000, page_size=1)
        b = BlockHashCache(100_000)

        for seq in self.SEQS:
            hashes = to_block_hashes(seq)
            rm = r.match(seq)
            bm = b.match(hashes) * 16        # blocks -> tokens
            assert rm >= bm, f"radix matched fewer tokens: {rm} < {bm}"
            r.release(r.insert(seq))
            b.insert(hashes); b.release(hashes)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))