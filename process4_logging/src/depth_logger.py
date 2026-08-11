import asyncio
import os
import time
import msgpack
import pandas as pd
import redis.asyncio as aioredis
from datetime import datetime

# Must match target stocks in Process 2
TARGET_STOCKS = [
    "1333",   # HDFC BANK
    "2885",   # RELIANCE
    "11536",  # TCS
    "1594",   # INFOSYS
    "4963",   # ICICI BANK
    "3045",   # SBI
    "3456",   # TATA MOTORS
    "1922",   # KOTAK MAHINDRA BANK
    "11483",  # LT
    "11915",  # YES BANK
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data_capture")
BATCH_SIZE = 1000  # Number of ticks to buffer before writing to disk

class DepthLogger:
    def __init__(self, instrument_ids: list):
        self.instrument_ids = instrument_ids
        self.redis = aioredis.from_url("redis://127.0.0.1:6379")
        self.buffer = []
        self.last_seq = {sec_id: -1 for sec_id in instrument_ids}
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def flatten_snapshot(self, sec_id: str, book_data: dict, timestamp: float) -> dict:
        row = {
            "timestamp": timestamp,
            "security_id": sec_id,
            "seq": book_data.get("seq", 0)
        }
        
        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])

        # Flatten 20 Bid levels
        for i in range(20):
            if i < len(bids):
                row[f"bid_px_{i}"] = bids[i][0]
                row[f"bid_qty_{i}"] = bids[i][1]
            else:
                row[f"bid_px_{i}"] = 0.0
                row[f"bid_qty_{i}"] = 0.0

        # Flatten 20 Ask levels
        for i in range(20):
            if i < len(asks):
                row[f"ask_px_{i}"] = asks[i][0]
                row[f"ask_qty_{i}"] = asks[i][1]
            else:
                row[f"ask_px_{i}"] = 0.0
                row[f"ask_qty_{i}"] = 0.0

        return row

    def flush_to_disk(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(OUTPUT_DIR, f"orderbook_depth_{date_str}.parquet")

        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(file_path, engine="pyarrow", compression="snappy")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Flushed {len(self.buffer)} snapshots -> {file_path}")
        self.buffer.clear()

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

                    book_data = msgpack.unpackb(book_blob, raw=False)
                    seq = book_data.get("seq", 0)

                    # Only record if orderbook state has updated
                    if seq != self.last_seq[sec_id]:
                        self.last_seq[sec_id] = seq
                        row = self.flatten_snapshot(sec_id, book_data, time.time())
                        self.buffer.append(row)

                        if len(self.buffer) >= BATCH_SIZE:
                            self.flush_to_disk()

                await asyncio.sleep(0.005)  # Scan every 5ms
        except asyncio.CancelledError:
            pass
        finally:
            self.flush_to_disk()
            await self.redis.aclose()

if __name__ == "__main__":
    logger = DepthLogger(TARGET_STOCKS)
    try:
        asyncio.run(logger.start_recording())
    except KeyboardInterrupt:
        print("\nDepth Logger stopped.")