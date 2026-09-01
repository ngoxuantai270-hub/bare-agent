from src.pagination import get_page


def test_exact_final_page():
    assert get_page([1, 2, 3, 4], 2, 2) == [3, 4]


def test_page_beyond_data():
    assert get_page([1, 2, 3], 3, 2) == []
