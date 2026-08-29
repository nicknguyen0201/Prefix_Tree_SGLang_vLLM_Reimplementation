def replay(cache, trace):
    """Feed a sequence of requests through a cache, record match counts."""
    rows = []
    for i, hashes in enumerate(trace):
        matched = cache.match(hashes)
        cache.insert(hashes)
        cache.release(hashes)          # single-shot: request completes immediately
        rows.append({
            'req': i,
            'blocks': len(hashes),
            'matched': matched,
            'missed': len(hashes) - matched,
        })
    return rows