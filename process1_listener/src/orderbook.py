"""
orderbook.py — In-memory L2 order book for full-refresh feeds.
"""
from array import array
from typing import NamedTuple, Literal

DEPTH = 20
Side = Literal["bid", "ask"]

class Level(NamedTuple):
    price: float
    qty: float

class OrderBook:
    __slots__ = ("instrument_id", "bids", "asks", "has_bids", "has_asks")

    def __init__(self, instrument_id: str):
        self.instrument_id = instrument_id
        # Contiguous C-level double buffers for zero-allocation writes
        self.bids = array("d", [0.0] * (DEPTH * 2))
        self.asks = array("d", [0.0] * (DEPTH * 2))
        
        # State tracking for STALE lifecycle
        self.has_bids = False
        self.has_asks = False

    def _buf(self, side: Side) -> array:
        return self.bids if side == "bid" else self.asks

    def replace_side(self, side: Side, levels: list[tuple[float, float]]) -> None:
        """O(1) block overwrite of the entire 20-depth side."""
        buf = self._buf(side)
        for i in range(DEPTH):
            if i < len(levels):
                buf[i * 2] = levels[i][0]
                buf[i * 2 + 1] = levels[i][1]
            else:
                buf[i * 2] = 0.0
                buf[i * 2 + 1] = 0.0
        
        if side == "bid":
            self.has_bids = True
        elif side == "ask":
            self.has_asks = True

    def is_ready(self) -> bool:
        """True only if both sides have received at least one full refresh."""
        return self.has_bids and self.has_asks

    def reset_readiness(self) -> None:
        """Called upon connection loss to force STALE=True on reconnect."""
        self.has_bids = False
        self.has_asks = False

    def get_side(self, side: Side) -> list[Level]:
        buf = self._buf(side)
        return [Level(price=buf[i * 2], qty=buf[i * 2 + 1]) for i in range(DEPTH)]