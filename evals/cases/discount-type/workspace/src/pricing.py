DISCOUNTS = {
    "VIP": "20",
    "NEW_USER": "10",
}


def get_discount(code: str):
    return DISCOUNTS.get(code, 0)
