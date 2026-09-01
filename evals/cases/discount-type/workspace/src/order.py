from src.pricing import get_discount


def calculate_total(prices, discount_code=None):
    subtotal = sum(prices)
    if discount_code is None:
        return subtotal
    discount = get_discount(discount_code)
    return subtotal * (1 - discount / 100)
