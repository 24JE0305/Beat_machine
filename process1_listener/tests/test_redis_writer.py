import sys
import os
import asyncio
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redis_writer import RedisWriter, serialize_book
import redis.asyncio as aioredis


REDIS_URL = "redis://127.0.0.1:6379"


@pytest.mark.asyncio
async def test_write_sets_both_book_and_stale_keys_atomically():
    writer = RedisWriter(redis_url=REDIS_URL, ttl_ms=5000)
    await writer.connect()

    blob = serialize_book("NIFTY", seq=42, bids=[(100.0, 1.0)], asks=[(100.5, 1.0)])
    await writer.write("NIFTY", blob, stale=False)

    client = aioredis.from_url(REDIS_URL)
    book_val = await client.get("book:NIFTY")
    stale_val = await client.get("stale:NIFTY")

    assert book_val == blob
    assert stale_val == b"0"

    await client.aclose()
    await writer.close()


@pytest.mark.asyncio
async def test_write_stale_true_sets_flag_correctly():
    writer = RedisWriter(redis_url=REDIS_URL, ttl_ms=5000)
    await writer.connect()

    blob = serialize_book("NIFTY", seq=1, bids=[], asks=[])
    await writer.write("NIFTY", blob, stale=True)

    client = aioredis.from_url(REDIS_URL)
    stale_val = await client.get("stale:NIFTY")
    assert stale_val == b"1"

    await client.aclose()
    await writer.close()


@pytest.mark.asyncio
async def test_ttl_expires_keys_as_a_dead_mans_switch():
    writer = RedisWriter(redis_url=REDIS_URL, ttl_ms=150)
    await writer.connect()

    blob = serialize_book("NIFTY", seq=1, bids=[(1.0, 1.0)], asks=[(1.0, 1.0)])
    await writer.write("NIFTY", blob, stale=False)

    client = aioredis.from_url(REDIS_URL)
    assert await client.get("book:NIFTY") is not None

    await asyncio.sleep(0.3)

    assert await client.get("book:NIFTY") is None
    assert await client.get("stale:NIFTY") is None

    await client.aclose()
    await writer.close()


@pytest.mark.asyncio
async def test_clear_deletes_keys_immediately_not_waiting_for_ttl():
    writer = RedisWriter(redis_url=REDIS_URL, ttl_ms=30000)
    await writer.connect()

    blob = serialize_book("NIFTY", seq=1, bids=[(1.0, 1.0)], asks=[(1.0, 1.0)])
    await writer.write("NIFTY", blob, stale=False)

    await writer.clear("NIFTY")

    client = aioredis.from_url(REDIS_URL)
    assert await client.get("book:NIFTY") is None
    assert await client.get("stale:NIFTY") is None

    await client.aclose()
    await writer.close()


@pytest.mark.asyncio
async def test_drain_processes_queue_until_sentinel():
    writer = RedisWriter(redis_url=REDIS_URL, ttl_ms=5000)
    await writer.connect()

    queue: asyncio.Queue = asyncio.Queue()
    blob1 = serialize_book("NIFTY", seq=1, bids=[(1.0, 1.0)], asks=[])
    blob2 = serialize_book("NIFTY", seq=2, bids=[(2.0, 2.0)], asks=[])
    await queue.put(("NIFTY", blob1, False))
    await queue.put(("NIFTY", blob2, False))
    await queue.put(None)

    await asyncio.wait_for(writer.drain(queue), timeout=2.0)

    client = aioredis.from_url(REDIS_URL)
    val = await client.get("book:NIFTY")
    assert val == blob2

    await client.aclose()
    await writer.close()