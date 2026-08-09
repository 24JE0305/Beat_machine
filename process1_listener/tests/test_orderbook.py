import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orderbook import OrderBook, Level, DEPTH


def test_new_book_is_zeroed():
    book = OrderBook("NIFTY")
    for level in range(DEPTH):
        assert book.get_level("bid", level) == Level(0.0, 0.0)
        assert book.get_level("ask", level) == Level(0.0, 0.0)


def test_set_level_writes_correct_slot_only():
    book = OrderBook("NIFTY")
    book.set_level("bid", 5, 22150.5, 300.0)
    assert book.get_level("bid", 5) == Level(22150.5, 300.0)
    assert book.get_level("bid", 4) == Level(0.0, 0.0)
    assert book.get_level("bid", 6) == Level(0.0, 0.0)
    for level in range(DEPTH):
        assert book.get_level("ask", level) == Level(0.0, 0.0)


def test_set_level_out_of_range_raises():
    book = OrderBook("NIFTY")
    with pytest.raises(ValueError):
        book.set_level("bid", 20, 100.0, 1.0)
    with pytest.raises(ValueError):
        book.set_level("bid", -1, 100.0, 1.0)


def test_invalid_side_raises():
    book = OrderBook("NIFTY")
    with pytest.raises(ValueError):
        book.set_level("buy", 0, 100.0, 1.0)


def test_best_bid_ask_reflect_level_zero():
    book = OrderBook("NIFTY")
    book.set_level("bid", 0, 22150.0, 500.0)
    book.set_level("ask", 0, 22151.0, 450.0)
    assert book.best_bid() == Level(22150.0, 500.0)
    assert book.best_ask() == Level(22151.0, 450.0)


def test_overwrite_same_level_replaces_not_accumulates():
    book = OrderBook("NIFTY")
    book.set_level("bid", 3, 100.0, 50.0)
    book.set_level("bid", 3, 101.0, 75.0)
    assert book.get_level("bid", 3) == Level(101.0, 75.0)


def test_seed_partial_levels_zeroes_remainder():
    book = OrderBook("NIFTY")
    bids = [(100.0, 10.0), (99.5, 20.0)]
    asks = [(100.5, 15.0)]
    book.seed(bids, asks, seq=42)

    assert book.get_level("bid", 0) == Level(100.0, 10.0)
    assert book.get_level("bid", 1) == Level(99.5, 20.0)
    assert book.get_level("bid", 2) == Level(0.0, 0.0)

    assert book.get_level("ask", 0) == Level(100.5, 15.0)
    assert book.get_level("ask", 1) == Level(0.0, 0.0)

    assert book.seq == 42


def test_seed_full_20_levels():
    book = OrderBook("NIFTY")
    bids = [(100.0 - i * 0.05, 10.0 + i) for i in range(20)]
    asks = [(100.5 + i * 0.05, 10.0 + i) for i in range(20)]
    book.seed(bids, asks, seq=1)

    for i in range(DEPTH):
        assert book.get_level("bid", i) == Level(*bids[i])
        assert book.get_level("ask", i) == Level(*asks[i])


def test_get_side_returns_all_20_in_order():
    book = OrderBook("NIFTY")
    for i in range(DEPTH):
        book.set_level("bid", i, 100.0 - i, float(i))
    side = book.get_side("bid")
    assert len(side) == DEPTH
    for i in range(DEPTH):
        assert side[i] == Level(100.0 - i, float(i))


def test_reseed_overwrites_previous_state_completely():
    """Simulates resync mode: a second seed() must fully replace old state,
    not merge with it — stale levels from before the gap must not survive."""
    book = OrderBook("NIFTY")
    book.seed([(100.0, 10.0)] * 20, [(101.0, 5.0)] * 20, seq=1)
    book.seed([(200.0, 1.0), (199.0, 2.0)], [(201.0, 1.0)], seq=99)

    assert book.get_level("bid", 0) == Level(200.0, 1.0)
    assert book.get_level("bid", 2) == Level(0.0, 0.0)
    assert book.get_level("ask", 1) == Level(0.0, 0.0)
    assert book.seq == 99