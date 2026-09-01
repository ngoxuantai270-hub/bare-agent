def slice_page(items, page_index: int, page_size: int):
    """Return one page using a zero-based page index."""
    start = page_index * page_size
    return items[start : start + page_size]
