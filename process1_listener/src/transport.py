"""
transport.py — The only broker-specific surface in this codebase.

Every other module (listener.py) talks to this Protocol, never to a raw
websocket or to Dhan directly. When real Dhan credentials are wired up,
only a DhanTransport implementation gets written — none of the
concurrency/resync/heartbeat logic changes, and none of it needs Dhan
credentials to be tested. FakeTransport below is what listener.py's
tests run against.
"""

import asyncio
from typing import Protocol, Any


class ConnectionClosed(Exception):
    """Raised by any Transport method when the underlying connection is
    dead or must be treated as dead. This is the ONE exception type the
    Global Supervisor (§2) watches for to trigger reconnect."""
    pass


class Transport(Protocol):
    async def connect(self) -> None: ...
    async def authenticate(self) -> None: ...
    async def subscribe(self, instruments: list[str]) -> None: ...

    async def recv(self) -> dict[str, Any]:
        """Wait for and return the next raw message: a delta, an ack,
        a pong, whatever the feed sends. Caller (delta_loop) is
        responsible for dispatching by type."""
        ...

    async def send(self, msg: dict[str, Any]) -> None: ...

    async def request_snapshot(self) -> dict[str, Any]:
        """Request a full L2 snapshot. Returns
        {"bids": [(price, qty), ...], "asks": [...], "seq": int}."""
        ...

    async def close(self) -> None: ...


class FakeTransport:
    """
    Scriptable Transport for testing delta_loop/resync_mode/heartbeat_loop
    without a real broker connection.

    Usage pattern in tests:
        t = FakeTransport()
        t.queue_messages([delta1, delta2, gap_delta, ...])
        t.set_snapshot({"bids": [...], "asks": [...], "seq": 10})
        # run delta_loop against it, then inspect t.sent (pings) etc.

    Key testing hooks:
    - recv() pulls from an internal asyncio.Queue — you can inject
      messages *while a test coroutine is mid-await*, which is exactly
      what's needed to simulate "deltas keep arriving while resync_mode
      awaits the snapshot" (the scenario Gemini's review flagged).
    - snapshot_delay lets a test hold up request_snapshot() for a
      controlled duration, so timeout behavior and concurrent-buffering
      behavior are both directly testable.
    - hang_recv, when set, makes recv() never return (simulates the
      silent half-open socket for heartbeat testing).
    """

    def __init__(self):
        self._inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self._snapshot: dict | None = None
        self.snapshot_delay: float = 0.0
        self.snapshot_raises_timeout: bool = False
        self.hang_recv: bool = False
        self.closed = False

    def queue_message(self, msg: dict) -> None:
        self._inbox.put_nowait(msg)

    def queue_messages(self, msgs: list[dict]) -> None:
        for m in msgs:
            self._inbox.put_nowait(m)

    def set_snapshot(self, snapshot: dict) -> None:
        self._snapshot = snapshot

    async def connect(self) -> None:
        pass

    async def authenticate(self) -> None:
        pass

    async def subscribe(self, instruments: list[str]) -> None:
        pass

    async def recv(self) -> dict:
        if self.hang_recv:
            # simulate a half-open socket: never resolves on its own.
            # Test must cancel this task to end it.
            await asyncio.Event().wait()
        return await self._inbox.get()

    async def send(self, msg: dict) -> None:
        self.sent.append(msg)

    async def request_snapshot(self) -> dict:
        if self.snapshot_delay:
            await asyncio.sleep(self.snapshot_delay)
        if self.snapshot_raises_timeout:
            raise asyncio.TimeoutError()
        if self._snapshot is None:
            raise RuntimeError("FakeTransport: no snapshot configured, call set_snapshot() first")
        return self._snapshot

    async def close(self) -> None:
        self.closed = True