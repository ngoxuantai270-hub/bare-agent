import pytest

from src.inventory import Inventory


def test_repeated_order_is_idempotent():
    inventory = Inventory(stock=10)
    inventory.reserve("order-1", 3)
    assert inventory.reserve("order-1", 3) == 7
    assert inventory.stock == 7


@pytest.mark.parametrize("quantity", [0, -1])
def test_non_positive_quantity_is_rejected_without_mutation(quantity: int):
    inventory = Inventory(stock=10)
    with pytest.raises(ValueError, match="positive"):
        inventory.reserve("order-1", quantity)
    assert inventory.stock == 10
    assert inventory.reservations == {}
