from src.order import calculate_total


def test_without_discount():
    assert calculate_total([100, 50]) == 150


def test_vip_discount():
    assert calculate_total([100, 50], "VIP") == 120


def test_new_user_discount():
    assert calculate_total([100, 50], "NEW_USER") == 135


def test_unknown_discount():
    assert calculate_total([100, 50], "UNKNOWN") == 150
