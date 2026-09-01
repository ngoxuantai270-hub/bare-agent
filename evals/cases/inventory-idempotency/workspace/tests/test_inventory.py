import pytest

from src.inventory import Inventory


def test_reserve_reduces_stock():
    inventory = Inventory(stock=10)
    assert inventory.reserve("order-1", 3) == 7
    assert inventory.stock == 7


def test_insufficient_stock_does_not_mutate_state():
    inventory = Inventory(stock=2)
    with pytest.raises(ValueError, match="insufficient stock"):
        inventory.reserve("order-1", 3)
    assert inventory.stock == 2
    assert inventory.reservations == {}
