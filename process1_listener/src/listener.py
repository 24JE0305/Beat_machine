import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from orderbook import OrderBook
from transport import Transport, ConnectionClosed
from redis_writer import RedisWriter, serialize_book

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
MARKET_POLL_SECONDS = 30.0


def is_market_open(now: datetime | None = None) -> bool:
    """NSE cash session gate: Mon-Fri, 09:15-15:30 IST."""
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

@dataclass
class ConnectionHealth:
    last_seen: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_seen = time.monotonic()


async def watchdog_loop(
    health: ConnectionHealth,
    silence_threshold: float = 15.0,
) -> None:
    """
    Monitors data liveness. Dhan pings automatically; we only care 
    if the actual data stream goes silent for too long.
    """
    while True:
        await asyncio.sleep(1.0)
        elapsed = time.monotonic() - health.last_seen
        if elapsed > silence_threshold:
            raise ConnectionClosed(
                f"Watchdog: no data received in {elapsed:.1f}s — connection dead."
            )


async def feed_loop(
    transport: Transport,
    book_map: dict[str, OrderBook],
    health: ConnectionHealth,
    redis_queue: asyncio.Queue,
) -> None:
    """
    Sole reader of the transport. Processes full-refresh packets.
    """
    while True:
        # transport.recv() must demultiplex the binary WS frame and 
        # yield a list of dicts: {"instrument_id": str, "side": "bid"|"ask", "levels": [(p,q)]}
        packets = await transport.recv()
        health.touch()

        for pkt in packets:
            inst_id = pkt["instrument_id"]
            if inst_id not in book_map:
                continue

            book = book_map[inst_id]
            book.replace_side(pkt["side"], pkt["levels"])

            # STALE logic: Only push to Redis if both sides are populated
            if book.is_ready():
                await _push_book_state(redis_queue, book, inst_id, stale=False)


async def _push_book_state(
    redis_queue: asyncio.Queue, book: OrderBook, instrument_id: str, stale: bool
) -> None:
    blob = serialize_book(
        instrument_id,
        0, # Sequence ignored
        [tuple(lv) for lv in book.get_side("bid")],
        [tuple(lv) for lv in book.get_side("ask")],
    )
    await redis_queue.put((instrument_id, blob, stale))


async def run_process1(
    transport: Transport,
    book_map: dict[str, OrderBook],
    redis_writer: RedisWriter,
    instruments: list[str],
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
) -> None:
    """
    Global Supervisor. Wraps connect, subscribe, and the TaskGroup.
    """
    backoff = backoff_base

    while True:
        if not is_market_open():
            await asyncio.sleep(MARKET_POLL_SECONDS)
            continue

        try:
            # Phase 1: Connect & Subscribe (Auth is in WSS URL now)
            await transport.connect()
            await transport.subscribe(instruments)

            # Reset readiness for all books to enforce STALE on reconnection
            for book in book_map.values():
                book.reset_readiness()

            health = ConnectionHealth()
            redis_queue: asyncio.Queue = asyncio.Queue()

            # Broadcast STALE=True for all tracked instruments immediately
            for inst_id in instruments:
                await redis_writer.write(inst_id, b"", stale=True)

            async with asyncio.TaskGroup() as tg:
                tg.create_task(watchdog_loop(health))
                tg.create_task(feed_loop(transport, book_map, health, redis_queue))
                tg.create_task(redis_writer.drain(redis_queue))

            backoff = backoff_base 

        except* ConnectionClosed as eg:
            print(f"Supervisor Triggered Reconnect: {eg.exceptions[0]}")
            # Broadcast STALE=True on disconnect, do not wait for TTL
            for inst_id in instruments:
                await redis_writer.write(inst_id, b"", stale=True)
            
            await transport.close()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, backoff_max)