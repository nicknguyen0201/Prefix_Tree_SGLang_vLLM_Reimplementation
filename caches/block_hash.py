from collections import OrderedDict
from typing import List

class BlockHashCache:
    def __init__(self, capacity_blocks, blk_size=16):
        self.table={}
        self.ref_count={}
        self.lru_cache=OrderedDict()
        self.free_list=[i for i in range(capacity_blocks)]
    def match(self,hashes: List[str])->int:
        hits = 0
        for h in hashes:
            if h not in self.table:
                break
            hits += 1
        return hits
    def insert(self,hashes: List[str])->None:
        hits=self.match(hashes)
        
        #increase ref count to prevent eviction of req mid compute
        for h in hashes[:hits]:
            blk_id=self.table[h]
            self.ref_count[blk_id]+=1
            self.lru_cache.move_to_end(blk_id)

        for h in hashes[hits:]:
            blk_id=self._alloc()#alloc 1 block
            self.table[h]=blk_id
            self.ref_count[blk_id]=1
            self.lru_cache[blk_id]=h
            self.lru_cache.move_to_end(blk_id)

    def _alloc(self)-> int:
        if not self.free_list:
            self._evict()
        return self.free_list.pop()
    def _evict(self)->None:
        for blk_id in list(self.lru_cache):
            if self.ref_count.get(blk_id,0)==0:
                h=self.lru_cache.pop(blk_id)
                del self.table[h]
                del self.ref_count[blk_id]
                self.free_list.append(blk_id)
                return
        raise RuntimeError("no evictable block")
    
    def release(self,hashes)->None:
        for h in hashes:
            if h in self.table:
                self.ref_count[self.table[h]]-=1