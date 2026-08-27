# caches/radix.py
from typing import List, Sequence, Optional


class TreeNode:
    __slots__ = ('key', 'children', 'parent', 'lock_ref', 'last_access')

    def __init__(self, key=(), parent=None):
        self.key = tuple(key)      # token sequence this node covers
        self.children = {}         # first_page -> TreeNode
        #we doing a simulation on the algorithm so we ignore the value dict
        self.parent = parent
        self.lock_ref = 0
        self.last_access = 0.0


class RadixCache:
    """Token-indexed prefix cache, matching truncated to page_size."""

    def __init__(self, capacity_tokens: int, page_size: int = 1):
        self.capacity = capacity_tokens
        self.page_size = page_size
        self.root = TreeNode()
        self.root.lock_ref = 1          # root is never evictable
        self.used = 0
        self.clock = 0

    
    def _page_align(self, tokens):
        if self.page_size==1:
            return tokens
        n = (len(tokens) // self.page_size) * self.page_size
        return tokens[:n]

    def _child_key(self, tokens):
        return tokens[0] if self.page_size == 1 else tuple(tokens[:self.page_size])

    def _common_prefix_len(self, a, b) -> int:
        """Shared prefix length, rounded down to page_size."""
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return (n // self.page_size) * self.page_size

    def match(self, tokens: List[int]) -> int:
        """Walk the tree, report match token count. Does not mutate LRU"""
        tokens=self._page_align(tuple(tokens))
        if not tokens:
            return 0
        curr = self.root
        matched=0
        while tokens:
            child_key = self._child_key(tokens)
            if child_key not in curr.children:
                break # we found no more matches
            child = curr.children[child_key]
            cnt=self._common_prefix_len(child.key,tokens)
            if cnt<len(child.key):
                matched+=cnt
                break
            matched+=cnt
            tokens=tokens[cnt:]
            curr=child
        return matched
    def _split_node(self, child, split_len):
        """Split `child` into a prefix node of `split_len` tokens plus the
        original object holding the remainder. Returns the new prefix node."""
        assert 0 < split_len < len(child.key)
        parent=child.parent
        child_key=self._child_key(child.key)
        new_node = TreeNode(child.key[:split_len], parent=parent)
        new_node.lock_ref = child.lock_ref
        new_node.last_access = child.last_access
        child.key = child.key[split_len:]
        child.parent = new_node
        parent.children[child_key]=new_node
        new_node.children = {self._child_key(child.key): child}
        return new_node

    def insert(self, tokens: List[int]) -> Optional[TreeNode]:
        """Insert `tokens`, returning the terminal node with the path locked."""
        tokens = self._page_align(tuple(tokens))
        if not tokens:
            return None
        self.clock+=1
        self.root.last_access=self.clock
        curr=self.root

        while tokens:
            child_key = self._child_key(tokens)
            if child_key not in curr.children:
                break # no existing branch to follow; append below
            child = curr.children[child_key]
            cnt=self._common_prefix_len(child.key,tokens)
            if cnt==0:
                break
            child.last_access=self.clock
            if cnt<len(child.key):
                # request diverges mid-node: split so the shared part is its own node
                curr=self._split_node(child,cnt)
                curr.last_access=self.clock
                tokens=tokens[cnt:]
                break
            curr=child
            tokens=tokens[cnt:]

        # Lock the matched path *before* evicting: curr may itself be an
        # unlocked leaf, and eviction must not reclaim what we are extending.
        self._lock(curr)

        if tokens:
            try:
                self._evict_to_fit(len(tokens))
            except RuntimeError:
                self._unlock(curr)
                raise
            leaf=TreeNode(tokens,parent=curr)
            leaf.last_access=self.clock
            curr.children[self._child_key(tokens)]=leaf
            self.used+=len(tokens)
            curr=leaf
            self._lock(curr)
            self._unlock(curr.parent)

        return curr
    def _leaves(self):
        """Unlocked leaves — the only nodes that may be evicted."""
        stack, out = [self.root], []
        while stack:
            n = stack.pop()
            for c in n.children.values():
                stack.append(c)
                if not c.children and c.lock_ref == 0:
                    out.append(c)
        return out

    def _evict_to_fit(self, need: int) -> None:
        """Free space for `need` tokens by dropping least-recently-used leaves."""
        while self.used + need > self.capacity:
            leaves = self._leaves()
            if not leaves:
                raise RuntimeError("no evictable node")
            victim = min(leaves, key=lambda n: n.last_access)
            del victim.parent.children[self._child_key(victim.key)]
            self.used -= len(victim.key)
            victim.parent = None

    def _lock(self, node):
        while node is not self.root:
            node.lock_ref += 1
            node = node.parent

    def _unlock(self, node):
        while node is not None and node is not self.root:
            node.lock_ref -= 1
            node = node.parent

    def release(self, node: Optional[TreeNode]) -> None:
        self._unlock(node)