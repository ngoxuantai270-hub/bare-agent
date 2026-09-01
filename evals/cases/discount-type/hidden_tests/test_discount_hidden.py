from src.order import calculate_total


def test_vip_discount_with_different_subtotal():
    assert calculate_total([80, 20], "VIP") == 80


def test_unknown_code_remains_undiscounted():
    assert calculate_total([25, 25], "OTHER") == 50
