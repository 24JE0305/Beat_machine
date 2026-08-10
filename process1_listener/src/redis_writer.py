"""
redis_writer.py — Batches book state from an in-memory queue into Redis,
per spec §7.

Correctness fix (round 2, Gemini review, validated real): the spec's §7
describes TWO separate keys — book blob and STALE flag. Written as two
separate Redis calls, there's a window where Process 2 can read a
cleared STALE flag against a not-yet-updated book. Fixed via a single
MULTI/EXEC pipeline transaction per write.

Resilience fix (round 3, Gemini review, validated real): drain() had no
error handling around write() — a transient Redis outage (connection
drop, timeout, Redis restart mid-session) would raise RedisError,
uncaught, crashing the task inside the Global Supervisor's TaskGroup.
Since RedisError doesn't subclass transport.ConnectionClosed, the
supervisor's `except* ConnectionClosed` wouldn't catch it, and the whole
process would die instead of entering backoff. This is spec §9's own
test harness requirement ("Redis unreachable mid-session -> book writes
fail gracefully, doesn't crash the listener") that had gone unimplemented.

Deliberately NOT fixed by converting RedisError -> ConnectionClosed (the
originally suggested fix): that would make the Global Supervisor tear
down and fully re-handshake the broker connection on every Redis blip,
conflating two independent failure domains. Instead, write()/clear()
catch RedisError internally, log it, and swallow it — drain() keeps
running, heartbeat_loop/delta_loop are completely undisturbed, the local
book keeps updating in memory, and Process 2 stays protected via the
book key's TTL expiring naturally while writes are failing.
"""

import asyncio
import logging
import msgpack
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisWriter:
    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379",
        book_key_prefix: str = "book",
        stale_key_prefix: str = "stale",
        ttl_ms: int = 750,
    ):
        self._url = redis_url
        self._client: aioredis.Redis | None = None
        self._book_key_prefix = book_key_prefix
        self._stale_key_prefix = stale_key_prefix
        self._ttl_ms = ttl_ms

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._url)
        await self._client.ping()

    async def write(self, instrument_id: str, book_blob: bytes, stale: bool) -> None:
        """Atomic write of book blob + stale flag. Single round trip,
        single transaction — Process 2 can never see a half-updated pair.

        Deliberately does NOT raise on Redis-level failures (connection
        drop, timeout, Redis restart mid-session). This is per spec §9's
        own test harness requirement: "Redis unreachable mid-session ->
        book writes fail gracefully, doesn't crash the listener." A
        RedisError here is a DIFFERENT failure domain than the broker
        WebSocket dying (transport.ConnectionClosed) — converting it to
        ConnectionClosed and letting the Global Supervisor catch it would
        tear down and fully re-handshake a perfectly healthy broker
        connection just because Redis blipped. Instead: log it, drop this
        one write, let drain() keep consuming the queue undisturbed.
        Process 2 stays protected regardless, via the book key's TTL
        expiring naturally while writes are failing."""
        if self._client is None:
            raise RuntimeError("RedisWriter.connect() must be called first")

        book_key = f"{self._book_key_prefix}:{instrument_id}"
        stale_key = f"{self._stale_key_prefix}:{instrument_id}"

        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.set(book_key, book_blob, px=self._ttl_ms)
                pipe.set(stale_key, b"1" if stale else b"0", px=self._ttl_ms)
                await pipe.execute()
        except aioredis.RedisError as e:
            logger.error("Redis write failed for %s, dropping this update: %s", instrument_id, e)

    async def clear(self, instrument_id: str) -> None:
        """Called by the Global Supervisor on reconnect — per spec §2,
        'ConnectionClosed anywhere -> clear Redis'. Deletes both keys
        outright rather than leaving stale data for the TTL to expire,
        since Process 2 should stop trading the instant, not TTL-later.

        Also does not raise on RedisError, for the same reason as
        write() — a failure here shouldn't interrupt the Global
        Supervisor's reconnect flow. If Redis is unreachable, the keys
        will naturally expire via TTL regardless."""
        if self._client is None:
            return
        book_key = f"{self._book_key_prefix}:{instrument_id}"
        stale_key = f"{self._stale_key_prefix}:{instrument_id}"
        try:
            await self._client.delete(book_key, stale_key)
        except aioredis.RedisError as e:
            logger.error("Redis clear() failed for %s (keys will expire via TTL): %s", instrument_id, e)

    async def drain(self, queue: asyncio.Queue) -> None:
        """
        Background coroutine: pulls (instrument_id, book_blob, stale)
        tuples off the queue and writes them. Per spec §7 — never call
        redis inline/synchronously from the hot delta path; this is the
        separate coroutine that does the actual network I/O, decoupled
        from delta application.

        Runs forever inside the Global Supervisor's TaskGroup; exits
        only when cancelled (sibling task raised) or the queue item is
        the sentinel None (used in tests to end the loop cleanly).
        """
        while True:
            item = await queue.get()
            if item is None:
                return
            instrument_id, book_blob, stale = item
            await self.write(instrument_id, book_blob, stale)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def serialize_book(instrument_id: str, seq: int, bids: list, asks: list) -> bytes:
    """MessagePack serialization per spec §7 (not JSON — avoids string
    parsing overhead at tick rate). bids/asks are lists of (price, qty)."""
    return msgpack.packb(
        {
            "instrument_id": instrument_id,
            "seq": seq,
            "bids": bids,
            "asks": asks,
        },
        use_bin_type=True,
    )