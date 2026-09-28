import pytest

from cart import Item, shipping_fee, total, validate_item


def test_no_discount():
    assert total([Item(price=10, quantity=2)]) == 20.0


def test_percentage_discount():
    assert total([Item(price=10, quantity=2)], discount=0.10) == 18.0


def test_free_shipping_at_threshold():
    assert shipping_fee(50.0) == 0.0


def test_zero_quantity_rejected():
    with pytest.raises(ValueError):
        validate_item(Item(price=5, quantity=0))
