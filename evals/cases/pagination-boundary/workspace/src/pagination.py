def get_page(items, page: int, page_size: int):
    start = (page - 1) * page_size
    end = start + page_size - 1
    return items[start:end]
