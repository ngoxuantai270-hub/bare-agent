from calculator import inclusive_sum


def test_inclusive_sum_includes_both_endpoints() -> None:
    assert inclusive_sum(1, 5) == 15
