"""
listener.py — Process 1's actual concurrency core: heartbeat, delta
application, resync, and the Global Supervisor that wraps all of it.

Spec references: §1-5 of process1_websocket_listener_spec.md

Concurrency model note (why heartbeat doesn't call transport.recv()):
Most WebSocket client libraries — including whatever wraps Dhan's feed —
do not support two coroutines independently calling recv() on the same
connection; only one reader is safe. So delta_loop is the SOLE reader.
It sees every message type, not just deltas, and simply skips non-delta
messages (per spec §4: "if not is_delta(msg): continue # other message
type (ack, info, etc.)"). Liveness is tracked via a shared ConnectionHealth
timestamp that delta_loop touches on every message it receives, delta or
not. heartbeat_loop only sends pings and checks that shared timestamp —
this is exactly what spec §4 means by "no pong / no message at all"
being the death signal, not pong specifically.

resync_mode concurrency fix (from Gemini review, validated real):
The spec's pseudocode has delta_loop `await resync_mode(msg)` — a single
coroutine cannot simultaneously await a snapshot AND keep looping on
wait_for_message() to buffer incoming deltas; that requires two
concurrently-scheduled things. Fixed here by racing the snapshot request
against repeated recv() calls using asyncio.wait(FIRST_COMPLETED) inside
resync_mode itself. This preserves the spec's "single coroutine = no
lock needed" invariant (delta_buffer is only ever touched by this one
coroutine, just not by a *single blocking await* like the original
pseudocode implied) and its "one rejoin point" requirement (resync_mode
does its own reading, then hands control back to delta_loop's normal
loop on return).
"""

import asyncio
import time
from dataclasses import dataclass, field

from orderbook import OrderBook
from transport import Transport, ConnectionClosed
from redis_writer import RedisWriter, serialize_book


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def is_delta(msg: dict) -> bool:
    return msg.get("type") == "delta"


def apply_delta(book: OrderBook, msg: dict) -> None:
    """O(1) write per spec §6. msg shape: {"type": "delta", "side": "bid"|"ask",
    "level": int, "price": float, "qty": float, "seq": int}."""
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
) -> None:
    """Sends an application-level ping every `interval` seconds. Does NOT
    read the socket itself (see module docstring). Raises ConnectionClosed
    if no message of any kind has been seen within interval * silence_multiplier,
    which the Global Supervisor catches to trigger reconnect."""
    while True:
        await transport.send({"type": "ping"})
        await asyncio.sleep(interval)
        elapsed = time.monotonic() - health.last_seen
        if elapsed > interval * silence_multiplier:
            raise ConnectionClosed(
                f"heartbeat: no message received in {elapsed:.1f}s "
                f"(threshold {interval * silence_multiplier:.1f}s) — silent death"
            )


# ---------------------------------------------------------------------------
# Resync Mode (spec §5, concurrency-fixed)
# ---------------------------------------------------------------------------

async def resync_mode(
    transport: Transport,
    book: OrderBook,
    triggering_msg: dict,
    health: ConnectionHealth,
    timeout: float = 2.5,
) -> int:
    """
    Buffer-and-replay resync. Returns the new last_seq on success.
    Raises ConnectionClosed on timeout (per spec: "don't stay stuck
    buffering forever" -> force full reconnect via Global Supervisor).
    """
    delta_buffer: list[dict] = [triggering_msg]

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
                if is_delta(msg):
                    delta_buffer.append(msg)
                continue  # keep waiting on snapshot_task

            # asyncio.wait's own timeout fired -- neither completed in time
            snapshot_task.cancel()
            recv_task.cancel()
            raise ConnectionClosed("resync timeout, forcing full reconnect")

    except asyncio.CancelledError:
        snapshot_task.cancel()
        raise
    except asyncio.TimeoutError:
        # request_snapshot() itself raised TimeoutError (e.g. FakeTransport
        # simulating a broker-side timeout rather than our deadline firing)
        raise ConnectionClosed("resync timeout, forcing full reconnect")

    snapshot = snapshot_task.result()

    # Atomic swap -- per spec, happens the instant the snapshot arrives,
    # before seeding, so nothing arriving after this point mixes into
    # the batch we're about to replay.
    replay_batch, delta_buffer = delta_buffer, []

    book.seed(snapshot["bids"], snapshot["asks"], seq=snapshot["seq"])
    last_seq = snapshot["seq"]
    for d in replay_batch:
        if d["seq"] <= snapshot["seq"]:
            continue  # already baked into the snapshot
        apply_delta(book, d)
        last_seq = d["seq"]

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
    to resync_mode on a sequence gap, and pushes updated book state onto
    redis_queue (spec §7: never write to Redis inline from this loop)."""
    last_seq = book.seq

    while True:
        msg = await transport.recv()
        health.touch()

        if not is_delta(msg):
            continue

        if msg["seq"] != last_seq + 1:
            last_seq = await resync_mode(transport, book, msg, health)
            await _push_book_state(redis_queue, book, instrument_id, stale=False)
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
    logic lives (spec §2) -- heartbeat_loop and delta_loop/resync_mode
    only ever raise ConnectionClosed, they never retry themselves.

    max_restarts: None = run forever (production). Set to an int in
    tests to bound the loop so a test can actually finish.
    """
    backoff = backoff_base
    restarts = 0

    while max_restarts is None or restarts <= max_restarts:
        # except* groups disallow break/continue/return directly inside
        # them (PEP 654) -- so the exit decision is made via this flag
        # and acted on after the try/except* block, not inside it.
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

            backoff = backoff_base  # unreachable in practice (TaskGroup runs forever
            # until a member raises), kept for completeness/documentation of intent.

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