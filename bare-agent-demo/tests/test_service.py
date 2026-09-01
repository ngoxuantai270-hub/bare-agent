from catalog.pagination import slice_page
from catalog.service import list_products

PRODUCTS = ["A", "B", "C", "D", "E", "F", "G"]


def test_internal_helper_uses_zero_based_index():
    assert slice_page(PRODUCTS, 0, 3) == ["A", "B", "C"]


def test_public_first_page():
    assert list_products(PRODUCTS, page=1, page_size=3) == ["A", "B", "C"]


def test_public_second_page():
    assert list_products(PRODUCTS, page=2, page_size=3) == ["D", "E", "F"]


def test_public_incomplete_last_page():
    assert list_products(PRODUCTS, page=3, page_size=3) == ["G"]
