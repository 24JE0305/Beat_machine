"""
orderbook.py — In-memory L2 order book for a single instrument.

Spec reference: §6 of process1_websocket_listener_spec.md

    "Two fixed-size arrays (bids, asks), indexed by depth level 0-19,
    each holding (price, qty). Not a dict re-sorted on every update —
    that's O(n log n) per tick for no reason. A delta names an exact
    level, so it's an O(1) array write."

This module owns exactly one responsibility: hold current book state
and accept writes at a known index in O(1) time, zero allocation on
the hot path. It does NOT know about WebSockets, Redis, or sequence
numbers — those are Process B's (delta_loop's) job. Keeping this class
ignorant of everything upstream is deliberate: it makes the class
trivially unit-testable with fake data, and it means a bug in the
WebSocket layer can never corrupt book *logic*, only book *input*.
"""

from array import array
from typing import NamedTuple, Literal

DEPTH = 20  # levels 0..19, per spec
Side = Literal["bid", "ask"]


class Level(NamedTuple):
    price: float
    qty: float


class OrderBook:
    """
    Fixed-depth L2 book. One instance per instrument.

    Internal layout — this is the part worth understanding, not just
    reading past:

    self.bids and self.asks are each a flat `array('d', ...)` of length
    DEPTH * 2, laid out as [price0, qty0, price1, qty1, ..., price19, qty19].

    Why not a plain `list[Level]` (20 NamedTuples)?
    Because a NamedTuple is still a Python object. Replacing list[i] with
    a new Level on every delta means: allocate a tuple, incref/decref the
    old one, let the GC eventually reclaim it. That happens on every
    single tick, on every instrument you subscribe to, most heavily
    during exactly the volatile bursts where spoofing happens — the
    moment this code most needs to not be doing avoidable allocation.

    array('d', ...) is a contiguous C-level double buffer. Writing index
    i is a raw memory store — no allocation, no refcounting, no GC
    pressure. `set_level()` below is a true O(1) write in the sense the
    spec means it, not just "O(1) on average."

    The public API still returns Level namedtuples on reads (get_level,
    get_side) — reads are comparatively rare (once per delta, for
    serialization into the Redis blob) and the ergonomic win of a named
    (price, qty) tuple there is worth the one allocation. The hot path
    is WRITES, and writes never allocate.
    """

    __slots__ = ("instrument_id", "bids", "asks", "seq")

    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id
        self.bids = array("d", [0.0] * (DEPTH * 2))
        self.asks = array("d", [0.0] * (DEPTH * 2))
        self.seq = 0  # last applied sequence number; delta_loop owns writing this

    def _buf(self, side: Side) -> array:
        if side == "bid":
            return self.bids
        if side == "ask":
            return self.asks
        raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")

    def set_level(self, side: Side, level: int, price: float, qty: float) -> None:
        """O(1) write of a single depth level. This is what apply_delta() calls."""
        if not 0 <= level < DEPTH:
            raise ValueError(f"level {level} out of range 0-{DEPTH - 1}")
        buf = self._buf(side)
        idx = level * 2
        buf[idx] = price
        buf[idx + 1] = qty

    def get_level(self, side: Side, level: int) -> Level:
        if not 0 <= level < DEPTH:
            raise ValueError(f"level {level} out of range 0-{DEPTH - 1}")
        buf = self._buf(side)
        idx = level * 2
        return Level(price=buf[idx], qty=buf[idx + 1])

    def get_side(self, side: Side) -> list[Level]:
        """Full 20-level dump of one side, in level order. Used for Redis serialization."""
        buf = self._buf(side)
        return [Level(price=buf[i * 2], qty=buf[i * 2 + 1]) for i in range(DEPTH)]

    def best_bid(self) -> Level:
        return self.get_level("bid", 0)

    def best_ask(self) -> Level:
        return self.get_level("ask", 0)

    def seed(
        self,
        bid_levels: list[tuple[float, float]],
        ask_levels: list[tuple[float, float]],
        seq: int,
    ) -> None:
        """
        Bulk-load the book from a broker snapshot. Used in two places per
        spec: Phase 1 Step 3 (initial seed) and Resync Mode §5 (re-seed
        after a sequence gap). Levels beyond what the snapshot provides
        stay zeroed — a broker snapshot may legitimately have fewer than
        20 resting levels on one side during thin liquidity.
        """
        for i in range(DEPTH):
            if i < len(bid_levels):
                self.set_level("bid", i, *bid_levels[i])
            else:
                self.set_level("bid", i, 0.0, 0.0)
            if i < len(ask_levels):
                self.set_level("ask", i, *ask_levels[i])
            else:
                self.set_level("ask", i, 0.0, 0.0)
        self.seq = seq

    def __repr__(self) -> str:
        bb, ba = self.best_bid(), self.best_ask()
        return (
            f"OrderBook({self.instrument_id!r}, seq={self.seq}, "
            f"best_bid={bb.price}@{bb.qty}, best_ask={ba.price}@{ba.qty})"
        )