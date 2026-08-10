import sys
import os
import asyncio
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orderbook import OrderBook
from transport import FakeTransport
from redis_writer import RedisWriter
from listener import run_process1

REDIS_URL = "redis://127.0.0.1:6379"


class CountingTransport(FakeTransport):
    """Extends FakeTransport to count how many times connect() is called,
    so the test can prove the supervisor actually restarted from the
    handshake, not just quietly retried something mid-stream."""

    def __init__(self):
        super().__init__()
        self.connect_count = 0

    async def connect(self) -> None:
        self.connect_count += 1
        await super().connect()


@pytest.mark.asyncio
async def test_supervisor_reconnects_on_silent_heartbeat_death_and_clears_redis():
    book = OrderBook("NIFTY")
    transport = CountingTransport()
    transport.hang_recv = True
    transport.set_snapshot({"bids": [(100.0, 1.0)], "asks": [(100.5, 1.0)], "seq": 1})

    writer = RedisWriter(redis_url=REDIS_URL, ttl_ms=30000)
    await writer.connect()

    import redis.asyncio as aioredis
    client = aioredis.from_url(REDIS_URL)
    await client.set("book:NIFTY", b"leftover-from-before-crash")
    await client.set("stale:NIFTY", b"0")

    import listener as listener_module
    original_heartbeat = listener_module.heartbeat_loop

    async def fast_heartbeat(transport, health, interval=5.0, silence_multiplier=2.0):
        return await original_heartbeat(transport, health, interval=0.05, silence_multiplier=2.0)

    listener_module.heartbeat_loop = fast_heartbeat
    try:
        await asyncio.wait_for(
            run_process1(
                transport,
                book,
                writer,
                instrument_id="NIFTY",
                instruments=["NIFTY"],
                backoff_base=0.05,
                backoff_max=0.1,
                max_restarts=1,
            ),
            timeout=5.0,
        )
    finally:
        listener_module.heartbeat_loop = original_heartbeat

    assert transport.connect_count == 2, "expected initial connect + exactly one restart"

    book_val = await client.get("book:NIFTY")
    assert book_val is None or book_val != b"leftover-from-before-crash"

    await client.aclose()
    await writer.close()

@pytest.mark.asyncio
async def test_redis_outage_does_not_trigger_broker_reconnect():
    """Core architectural claim behind the round-3 fix: a Redis outage
    must NOT cause the Global Supervisor to re-handshake the broker.
    connect_count stays at 1 even while every Redis write fails, and the
    book keeps updating correctly in memory the whole time."""
    book = OrderBook("NIFTY")
    transport = CountingTransport()
    transport.set_snapshot({"bids": [(100.0, 1.0)], "asks": [(100.5, 1.0)], "seq": 1})
    transport.queue_messages([
        {"type": "delta", "seq": 2, "side": "bid", "level": 0, "price": 101.0, "qty": 5.0},
        {"type": "delta", "seq": 3, "side": "bid", "level": 1, "price": 100.9, "qty": 6.0},
    ])

    dead_writer = RedisWriter(redis_url="redis://127.0.0.1:1", ttl_ms=5000)
    try:
        await dead_writer.connect()
    except Exception:
        pass

    run_task = asyncio.create_task(
        run_process1(
            transport,
            book,
            dead_writer,
            instrument_id="NIFTY",
            instruments=["NIFTY"],
            backoff_base=0.05,
            backoff_max=0.1,
        )
    )

    await asyncio.sleep(0.3)
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    assert transport.connect_count == 1, "Redis being down must not trigger any broker reconnect"
    assert book.get_level("bid", 0) == (101.0, 5.0), "book must keep updating even while Redis writes fail"
    assert book.get_level("bid", 1) == (100.9, 6.0)