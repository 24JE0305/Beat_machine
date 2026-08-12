import asyncio
import os
import time
import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
import redis.asyncio as aioredis
from datetime import datetime

TARGET_STOCKS = [
    "1333", "2885", "11536", "1594", "4963", 
    "3045", "3456", "1922", "11483", "11915"
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data_capture")
BATCH_SIZE = 1000

class DepthLogger:
    def __init__(self, instrument_ids: list):
        self.instrument_ids = instrument_ids
        self.redis = aioredis.from_url("redis://127.0.0.1:6379")
        self.buffer = []
        self.last_blob = {sec_id: None for sec_id in instrument_ids}
        self.writer = None
        self.current_file_path = None
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def flatten_snapshot(self, sec_id: str, book_data: dict, timestamp: float) -> dict:
        row = {
            "timestamp": timestamp,
            "security_id": int(sec_id),
            "seq": book_data.get("seq", 0)
        }
        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])

        for i in range(20):
            row[f"bid_px_{i}"] = bids[i][0] if i < len(bids) else 0.0
            row[f"bid_qty_{i}"] = bids[i][1] if i < len(bids) else 0.0
            row[f"ask_px_{i}"] = asks[i][0] if i < len(asks) else 0.0
            row[f"ask_qty_{i}"] = asks[i][1] if i < len(asks) else 0.0

        return row

    def _write_batch_sync(self, batch_data: list):
        if not batch_data:
            return

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(OUTPUT_DIR, f"orderbook_depth_{date_str}.parquet")

        table = pa.Table.from_pylist(batch_data)

        if self.writer is None or self.current_file_path != file_path:
            if self.writer is not None:
                self.writer.close()
            self.current_file_path = file_path
            self.writer = pq.ParquetWriter(file_path, table.schema, compression="snappy")

        self.writer.write_table(table)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Appended {len(batch_data)} snapshots -> {file_path}")

    async def flush_to_disk(self):
        if not self.buffer:
            return
        batch_to_write = list(self.buffer)
        self.buffer.clear()
        # Non-blocking async thread dispatch
        await asyncio.to_thread(self._write_batch_sync, batch_to_write)

    async def start_recording(self):
        print(f"Depth Logger running. Monitoring {len(self.instrument_ids)} stocks...")
        try:
            while True:
                for sec_id in self.instrument_ids:
                    book_key = f"book:{sec_id}"
                    stale_key = f"stale:{sec_id}"

                    keys = await self.redis.mget([stale_key, book_key])
                    stale_flag, book_blob = keys[0], keys[1]

                    if stale_flag == b"1" or not book_blob:
                        continue

                    if book_blob != self.last_blob[sec_id]:
                        self.last_blob[sec_id] = book_blob
                        book_data = msgpack.unpackb(book_blob, raw=False)
                        row = self.flatten_snapshot(sec_id, book_data, time.time())
                        self.buffer.append(row)

                        if len(self.buffer) >= BATCH_SIZE:
                            await self.flush_to_disk()

                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            pass
        finally:
            await self.flush_to_disk()
            if self.writer is not None:
                self.writer.close()
            await self.redis.aclose()

if __name__ == "__main__":
    logger = DepthLogger(TARGET_STOCKS)
    try:
        asyncio.run(logger.start_recording())
    except KeyboardInterrupt:
        print("\nDepth Logger stopped.")