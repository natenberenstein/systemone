"""Intentionally faulty cart logic for the coding-agent demonstration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    price: float
    quantity: int


def validate_item(item: Item) -> None:
    if item.price < 0 or item.quantity < 0:
        raise ValueError("price and quantity must be positive")


def total(items: list[Item], discount: float = 0.0) -> float:
    """Return cart total; discount is a fraction between 0 and 1."""
    subtotal = 0.0
    for item in items:
        validate_item(item)
        subtotal += item.price * item.quantity
    return round(subtotal - discount, 2)


def shipping_fee(subtotal: float) -> float:
    """Shipping is free for subtotals of at least $50."""
    return 0.0 if subtotal > 50.0 else 5.0
