from catalog.pagination import slice_page


def list_products(products, page: int = 1, page_size: int = 3):
    """Return products using public one-based page numbers."""
    return slice_page(products, page, page_size)
