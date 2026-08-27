import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from caches.block_hash import BlockHashCache

def test_exact_prefix_match():
    c = BlockHashCache(4)
    c.insert([1,2,3]); c.release([1,2,3])
    assert c.match([1,2,3,9]) == 3

def test_refcount_blocks_eviction():
    c = BlockHashCache(4)
    c.insert([1,2,3])          # refcount 1, not released
    c.insert([9]); c.release([9])              
    c.insert([8])              # must evict — only block 9 qualifies
    assert c.match([1,2,3]) == 3
    assert c.match([9]) == 0 
def test_eviction_is_lru():
    c = BlockHashCache(3)
    c.insert([1,2,3]); c.release([1,2,3])
    c.match([1])               # does NOT refresh — match is a pure query
    c.insert([9])              # evicts oldest unreferenced
    assert c.match([1,2,3]) == 0   # block 1 went first

test_exact_prefix_match()
test_refcount_blocks_eviction()
test_eviction_is_lru()
print("pass")