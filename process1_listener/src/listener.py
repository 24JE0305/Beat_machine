"""
listener.py — Process 1's actual concurrency core: heartbeat, delta
application, resync, and the Global Supervisor that wraps all of it.

Spec references: §1-5 of process1_websocket_listener_spec.md

Concurrency model note (why heartbeat doesn't call transport.recv()):
Most WebSocket client libraries — including whatever wraps Dhan's feed —
do not support two coroutines independently calling recv() on the same
connection; only one reader is safe. So delta_loop is the SOLE reader.
It sees every message type, not just deltas, and simply skips non-delta
messages. Liveness is tracked via a shared ConnectionHealth timestamp
that delta_loop touches on every message it receives, delta or not.
heartbeat_loop only sends pings and checks that shared timestamp.

resync_mode concurrency fix (Gemini review round 1, validated real):
A single coroutine cannot simultaneously await a snapshot AND keep
looping on wait_for_message() to buffer incoming deltas — fixed by
racing the snapshot request against repeated recv() calls using
asyncio.wait(FIRST_COMPLETED) inside resync_mode itself.

Round 2 fixes (Gemini review, all three validated real before
implementing):

1. STALE=True was never pushed to Redis when a gap was detected — the
   spec's own pseudocode has this as the literal first line of
   resync_mode, and it was missing entirely. Fixed: resync_mode now
   takes redis_queue/instrument_id and pushes STALE=True as its first
   action, before anything else, and STALE=False + fresh book state as
   its last action before returning. This makes resync_mode fully own
   the STALE flag's lifecycle, matching the spec's structure, rather
   than relying on the 750ms TTL to eventually catch it.

2. heartbeat_loop's `await transport.send(...)` had no timeout. On a
   half-open TCP connection with a full OS send buffer, this can hang
   indefinitely — which is exactly the coroutine responsible for
   detecting that scenario, defeating its own purpose. Fixed: wrapped
   in asyncio.wait_for with a short send_timeout; a timeout there is
   itself treated as proof of death and raises ConnectionClosed.

3. apply_delta did unguarded dict access (msg["side"], etc.) — a
   malformed message from the feed would raise KeyError, which is NOT
   caught by `except* ConnectionClosed` and would crash the whole
   TaskGroup / process. Fixed: is_valid_delta() validates required
   fields BEFORE any dict access happens, both in delta_loop's main
   path and inside resync_mode's buffering loop. A malformed message
   is treated conservatively as an untrustworthy position in the
   sequence — logged and forced into a resync rather than silently
   skipped (skipping could hide a real, silent gap) — but is NOT
   included in the replay buffer, since we can't trust its seq for the
   "already covered by snapshot" comparison.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from orderbook import OrderBook
from transport import Transport, ConnectionClosed
from redis_writer import RedisWriter, serialize_book

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

REQUIRED_DELTA_FIELDS = ("side", "level", "price", "qty", "seq")


def is_delta(msg: dict) -> bool:
    return msg.get("type") == "delta"


def is_valid_delta(msg: dict) -> bool:
    """True only if msg is a delta AND has every field apply_delta() needs.
    Checked BEFORE any dict access — never rely on try/except KeyError as
    the primary guard for data arriving off a live network feed."""
    return is_delta(msg) and all(f in msg for f in REQUIRED_DELTA_FIELDS)


def apply_delta(book: OrderBook, msg: dict) -> None:
    """O(1) write per spec §6. Caller MUST have already validated msg via
    is_valid_delta() — this function assumes the shape is correct."""
    book.set_level(msg["side"], msg["level"], msg["price"], msg["qty"])
    book.seq = msg["seq"]


# ---------------------------------------------------------------------------
# Liveness tracking (shared between heartbeat_loop and delta_loop)
# ---------------------------------------------------------------------------

@dataclass
class ConnectionHealth:
    last_seen: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_seen = time.monotonic()


# ---------------------------------------------------------------------------
# Process A — Heartbeat (spec §4)
# ---------------------------------------------------------------------------

async def heartbeat_loop(
    transport: Transport,
    health: ConnectionHealth,
    interval: float = 5.0,
    silence_multiplier: float = 2.0,
    send_timeout: float = 1.0,
) -> None:
    """Sends an application-level ping every `interval` seconds. Does NOT
    read the socket itself (see module docstring). Raises ConnectionClosed
    if:
      - send() itself doesn't complete within send_timeout (half-open
        TCP connection with a stuck OS send buffer), or
      - no message of any kind has been seen within
        interval * silence_multiplier.
    """
    while True:
        try:
            await asyncio.wait_for(transport.send({"type": "ping"}), timeout=send_timeout)
        except asyncio.TimeoutError:
            raise ConnectionClosed(
                f"heartbeat: send() did not complete within {send_timeout}s — "
                f"likely a half-open connection with a stuck send buffer"
            )

        await asyncio.sleep(interval)
        elapsed = time.monotonic() - health.last_seen
        if elapsed > interval * silence_multiplier:
            raise ConnectionClosed(
                f"heartbeat: no message received in {elapsed:.1f}s "
                f"(threshold {interval * silence_multiplier:.1f}s) — silent death"
            )


# ---------------------------------------------------------------------------
# Resync Mode (spec §5, concurrency-fixed, STALE-flag-fixed)
# ---------------------------------------------------------------------------

async def resync_mode(
    transport: Transport,
    book: OrderBook,
    health: ConnectionHealth,
    redis_queue: asyncio.Queue,
    instrument_id: str,
    triggering_msg: dict | None = None,
    timeout: float = 2.5,
) -> int:
    """
    Buffer-and-replay resync. Owns the full STALE flag lifecycle per
    spec: pushes STALE=True as the first action (Process 2 stops trading
    NOW), and STALE=False + fresh book state as the last action before
    returning. Returns the new last_seq on success. Raises
    ConnectionClosed on timeout.

    triggering_msg: the delta that revealed the gap, if it was itself a
    valid, well-formed delta (has a trustworthy seq). Pass None when
    resync was instead forced by a malformed message — we can't trust
    an untrustworthy message's seq enough to replay it.
    """
    await _push_book_state(redis_queue, book, instrument_id, stale=True)

    delta_buffer: list[dict] = []
    if triggering_msg is not None and is_valid_delta(triggering_msg):
        delta_buffer.append(triggering_msg)

    snapshot_task = asyncio.ensure_future(transport.request_snapshot())
    deadline = time.monotonic() + timeout

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                snapshot_task.cancel()
                raise ConnectionClosed("resync timeout, forcing full reconnect")

            recv_task = asyncio.ensure_future(transport.recv())
            done, pending = await asyncio.wait(
                {snapshot_task, recv_task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if snapshot_task in done:
                if recv_task in pending:
                    recv_task.cancel()
                break  # snapshot arrived -- exit the buffering loop

            if recv_task in done:
                health.touch()
                msg = recv_task.result()
                if is_valid_delta(msg):
                    delta_buffer.append(msg)
                elif is_delta(msg):
                    logger.warning("malformed delta during resync buffering, dropped: %r", msg)
                continue  # keep waiting on snapshot_task

            snapshot_task.cancel()
            recv_task.cancel()
            raise ConnectionClosed("resync timeout, forcing full reconnect")

    except asyncio.CancelledError:
        snapshot_task.cancel()
        raise
    except asyncio.TimeoutError:
        raise ConnectionClosed("resync timeout, forcing full reconnect")

    snapshot = snapshot_task.result()

    replay_batch, delta_buffer = delta_buffer, []

    book.seed(snapshot["bids"], snapshot["asks"], seq=snapshot["seq"])
    last_seq = snapshot["seq"]
    for d in replay_batch:
        if d["seq"] <= snapshot["seq"]:
            continue
        apply_delta(book, d)
        last_seq = d["seq"]

    await _push_book_state(redis_queue, book, instrument_id, stale=False)
    return last_seq


# ---------------------------------------------------------------------------
# Process B — Delta Loop (spec §4)
# ---------------------------------------------------------------------------

async def delta_loop(
    transport: Transport,
    book: OrderBook,
    health: ConnectionHealth,
    redis_queue: asyncio.Queue,
    instrument_id: str,
) -> None:
    """Sole reader of the transport. Applies in-order deltas, hands off
    to resync_mode on a sequence gap OR a malformed message, and pushes
    updated book state onto redis_queue for the normal in-order path
    (resync_mode pushes its own STALE transitions internally)."""
    last_seq = book.seq

    while True:
        msg = await transport.recv()
        health.touch()

        if not is_delta(msg):
            continue

        if not is_valid_delta(msg):
            logger.warning("malformed delta received, forcing resync: %r", msg)
            last_seq = await resync_mode(
                transport, book, health, redis_queue, instrument_id, triggering_msg=None
            )
            continue

        if msg["seq"] != last_seq + 1:
            last_seq = await resync_mode(
                transport, book, health, redis_queue, instrument_id, triggering_msg=msg
            )
            continue

        apply_delta(book, msg)
        last_seq = msg["seq"]
        await _push_book_state(redis_queue, book, instrument_id, stale=False)


async def _push_book_state(
    redis_queue: asyncio.Queue, book: OrderBook, instrument_id: str, stale: bool
) -> None:
    blob = serialize_book(
        instrument_id,
        book.seq,
        [tuple(lv) for lv in book.get_side("bid")],
        [tuple(lv) for lv in book.get_side("ask")],
    )
    await redis_queue.put((instrument_id, blob, stale))


# ---------------------------------------------------------------------------
# Global Supervisor (spec §2)
# ---------------------------------------------------------------------------

async def run_process1(
    transport: Transport,
    book: OrderBook,
    redis_writer: RedisWriter,
    instrument_id: str,
    instruments: list[str],
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    max_restarts: int | None = None,
) -> None:
    """
    Wraps handshake, auth, snapshot seed, and both coroutines. Any
    ConnectionClosed anywhere in scope -> clear Redis, log, exponential
    backoff, restart from handshake. This is the ONLY place reconnection
    logic lives (spec §2).

    max_restarts: None = run forever (production). Set to an int in
    tests to bound the loop so a test can actually finish.
    """
    backoff = backoff_base
    restarts = 0

    while max_restarts is None or restarts <= max_restarts:
        should_stop = False
        had_error = False

        try:
            await transport.connect()
            await transport.authenticate()
            await transport.subscribe(instruments)

            snapshot = await transport.request_snapshot()
            book.seed(snapshot["bids"], snapshot["asks"], seq=snapshot["seq"])

            health = ConnectionHealth()
            redis_queue: asyncio.Queue = asyncio.Queue()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(heartbeat_loop(transport, health))
                tg.create_task(delta_loop(transport, book, health, redis_queue, instrument_id))
                tg.create_task(redis_writer.drain(redis_queue))

            backoff = backoff_base

        except* ConnectionClosed:
            had_error = True
            await redis_writer.clear(instrument_id)
            await transport.close()
            restarts += 1
            if max_restarts is not None and restarts > max_restarts:
                should_stop = True

        if should_stop:
            return

        if had_error:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, backoff_max)