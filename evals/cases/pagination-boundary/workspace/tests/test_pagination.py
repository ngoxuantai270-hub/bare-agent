from src.pagination import get_page


def test_first_page():
    assert get_page([1, 2, 3, 4, 5], 1, 2) == [1, 2]


def test_second_page():
    assert get_page([1, 2, 3, 4, 5], 2, 2) == [3, 4]


def test_last_incomplete_page():
    assert get_page([1, 2, 3, 4, 5], 3, 2) == [5]
