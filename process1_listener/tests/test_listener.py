import sys
import os
import asyncio
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orderbook import OrderBook
from transport import FakeTransport, ConnectionClosed
from listener import (
    ConnectionHealth,
    heartbeat_loop,
    delta_loop,
    resync_mode,
    apply_delta,
    is_delta,
)


def make_delta(seq, side="bid", level=0, price=100.0, qty=10.0):
    return {"type": "delta", "seq": seq, "side": side, "level": level, "price": price, "qty": qty}


def test_is_delta_filters_non_delta_messages():
    assert is_delta({"type": "delta"}) is True
    assert is_delta({"type": "ack"}) is False
    assert is_delta({"type": "pong"}) is False
    assert is_delta({}) is False


def test_apply_delta_writes_book_and_advances_seq():
    book = OrderBook("NIFTY")
    apply_delta(book, make_delta(seq=5, side="bid", level=2, price=101.5, qty=20.0))
    assert book.get_level("bid", 2) == (101.5, 20.0)
    assert book.seq == 5


@pytest.mark.asyncio
async def test_delta_loop_applies_in_order_deltas_and_pushes_to_redis_queue():
    book = OrderBook("NIFTY")
    book.seq = 0
    transport = FakeTransport()
    transport.queue_messages([
        make_delta(seq=1, level=0, price=100.0, qty=5.0),
        make_delta(seq=2, level=1, price=99.5, qty=8.0),
        {"type": "ack"},
    ])
    health = ConnectionHealth()
    redis_queue = asyncio.Queue()

    task = asyncio.create_task(delta_loop(transport, book, health, redis_queue, "NIFTY"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert book.get_level("bid", 0) == (100.0, 5.0)
    assert book.get_level("bid", 1) == (99.5, 8.0)
    assert book.seq == 2
    assert redis_queue.qsize() == 2


@pytest.mark.asyncio
async def test_resync_mode_seeds_from_snapshot_and_replays_only_newer_deltas():
    book = OrderBook("NIFTY")
    transport = FakeTransport()
    transport.set_snapshot({
        "bids": [(100.0, 10.0)],
        "asks": [(100.5, 10.0)],
        "seq": 10,
    })
    health = ConnectionHealth()

    triggering_msg = make_delta(seq=12, side="bid", level=3, price=98.0, qty=1.0)

    last_seq = await resync_mode(transport, book, triggering_msg, health)

    assert book.get_level("bid", 0) == (100.0, 10.0)
    assert book.get_level("ask", 0) == (100.5, 10.0)
    assert book.get_level("bid", 3) == (98.0, 1.0)
    assert last_seq == 12


@pytest.mark.asyncio
async def test_resync_mode_drops_buffered_deltas_already_covered_by_snapshot():
    book = OrderBook("NIFTY")
    transport = FakeTransport()
    transport.set_snapshot({"bids": [(100.0, 10.0)], "asks": [(100.5, 10.0)], "seq": 20})
    health = ConnectionHealth()

    triggering_msg = make_delta(seq=15, side="bid", level=5, price=999.0, qty=999.0)

    last_seq = await resync_mode(transport, book, triggering_msg, health)

    assert book.get_level("bid", 5) == (0.0, 0.0)
    assert last_seq == 20


@pytest.mark.asyncio
async def test_resync_mode_buffers_deltas_arriving_while_snapshot_in_flight():
    book = OrderBook("NIFTY")
    transport = FakeTransport()
    transport.snapshot_delay = 0.15
    transport.set_snapshot({"bids": [(100.0, 10.0)], "asks": [(100.5, 10.0)], "seq": 5})
    health = ConnectionHealth()

    transport.queue_messages([
        make_delta(seq=6, side="ask", level=0, price=100.6, qty=3.0),
        make_delta(seq=7, side="ask", level=1, price=100.7, qty=4.0),
    ])

    triggering_msg = make_delta(seq=4, side="bid", level=0, price=999.0, qty=999.0)

    last_seq = await resync_mode(transport, book, triggering_msg, health)

    assert book.get_level("ask", 0) == (100.6, 3.0), "delta arriving mid-wait was lost"
    assert book.get_level("ask", 1) == (100.7, 4.0), "delta arriving mid-wait was lost"
    assert last_seq == 7


@pytest.mark.asyncio
async def test_resync_mode_times_out_and_raises_connection_closed():
    book = OrderBook("NIFTY")
    transport = FakeTransport()
    transport.snapshot_delay = 10.0
    transport.set_snapshot({"bids": [], "asks": [], "seq": 1})
    health = ConnectionHealth()

    triggering_msg = make_delta(seq=1)

    with pytest.raises(ConnectionClosed):
        await resync_mode(transport, book, triggering_msg, health, timeout=0.1)


@pytest.mark.asyncio
async def test_delta_loop_triggers_resync_on_sequence_gap():
    book = OrderBook("NIFTY")
    book.seq = 1
    transport = FakeTransport()
    transport.set_snapshot({"bids": [(50.0, 1.0)], "asks": [(50.5, 1.0)], "seq": 100})
    transport.queue_messages([make_delta(seq=5, side="bid", level=0, price=50.0, qty=1.0)])
    health = ConnectionHealth()
    redis_queue = asyncio.Queue()

    task = asyncio.create_task(delta_loop(transport, book, health, redis_queue, "NIFTY"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert book.seq == 100
    assert book.get_level("bid", 0) == (50.0, 1.0)


@pytest.mark.asyncio
async def test_heartbeat_raises_on_silence():
    transport = FakeTransport()
    health = ConnectionHealth()
    health.last_seen -= 100.0

    with pytest.raises(ConnectionClosed):
        await heartbeat_loop(transport, health, interval=0.05, silence_multiplier=2.0)


@pytest.mark.asyncio
async def test_heartbeat_stays_alive_when_health_is_touched():
    transport = FakeTransport()
    health = ConnectionHealth()

    async def toucher():
        for _ in range(5):
            await asyncio.sleep(0.02)
            health.touch()

    touch_task = asyncio.create_task(toucher())
    hb_task = asyncio.create_task(heartbeat_loop(transport, health, interval=0.05, silence_multiplier=2.0))

    await asyncio.sleep(0.15)
    assert not hb_task.done(), "heartbeat should not have died while health kept getting touched"

    hb_task.cancel()
    touch_task.cancel()
    for t in (hb_task, touch_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    assert len(transport.sent) >= 2, "heartbeat should have sent multiple pings by now"