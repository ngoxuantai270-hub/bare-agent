class Inventory:
    def __init__(self, stock: int):
        self.stock = stock
        self.reservations = {}

    def reserve(self, order_id: str, quantity: int) -> int:
        self.stock -= quantity
        self.reservations[order_id] = quantity
        return self.stock
