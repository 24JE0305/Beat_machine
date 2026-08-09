"""
redis_writer.py — Batches book state from an in-memory queue into Redis,
per spec §7.

Correctness fix applied here (from Gemini review, validated as a real
bug): the spec's §7 describes TWO separate keys — the book blob and the
STALE flag. Written as two separate Redis calls, there's a window where
Process 2 can read a cleared STALE flag against a not-yet-updated book,
or an updated book against a stale STALE=True that hasn't cleared yet.

Fix: every write is wrapped in a single MULTI/EXEC pipeline transaction,
so Process 2 can never observe a state where one changed and the other
didn't. This is the minimal correct fix — NOT the shared-memory seqlock
Gemini also proposed, which is real but premature (see reply to Gemini:
profile first, this is not yet a measured bottleneck).

TTL (§7): set on every write, so a hard-dead Process 1 (crashed without
raising a catchable exception) results in the key silently expiring
rather than Process 2 reading stale-but-present data forever.
"""

import asyncio
import msgpack
import redis.asyncio as aioredis


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
        single transaction — Process 2 can never see a half-updated pair."""
        if self._client is None:
            raise RuntimeError("RedisWriter.connect() must be called first")

        book_key = f"{self._book_key_prefix}:{instrument_id}"
        stale_key = f"{self._stale_key_prefix}:{instrument_id}"

        async with self._client.pipeline(transaction=True) as pipe:
            pipe.set(book_key, book_blob, px=self._ttl_ms)
            pipe.set(stale_key, b"1" if stale else b"0", px=self._ttl_ms)
            await pipe.execute()

    async def clear(self, instrument_id: str) -> None:
        """Called by the Global Supervisor on reconnect — per spec §2,
        'ConnectionClosed anywhere -> clear Redis'. Deletes both keys
        outright rather than leaving stale data for the TTL to expire,
        since Process 2 should stop trading the instant, not TTL-later."""
        if self._client is None:
            return
        book_key = f"{self._book_key_prefix}:{instrument_id}"
        stale_key = f"{self._stale_key_prefix}:{instrument_id}"
        await self._client.delete(book_key, stale_key)

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