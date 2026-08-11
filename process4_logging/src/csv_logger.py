import asyncio
import csv
import os
from datetime import datetime
import redis.asyncio as aioredis
import msgpack

class OrderflowLogger:
    def __init__(self, instrument_ids: list, log_dir: str = "logs"):
        self.instrument_ids = instrument_ids
        self.log_dir = log_dir
        self.redis = aioredis.from_url("redis://127.0.0.1:6379")
        
        os.makedirs(self.log_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        # One master file for all stocks
        self.filename = os.path.join(self.log_dir, f"orderflow_batch_{date_str}.csv")
        
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "SecurityId", "Best_Bid_Price", "Best_Bid_Qty", "Best_Ask_Price", "Best_Ask_Qty"])

    async def record_stream(self):
        print(f"Process 4 Logger started. Tracking {len(self.instrument_ids)} stocks into {self.filename}...")
        
        while True:
            batch_rows = []
            
            # Scan the Redis queue for all requested stocks
            for sec_id in self.instrument_ids:
                book_key = f"book:{sec_id}"
                book_blob = await self.redis.get(book_key)
                
                if book_blob:
                    book_data = msgpack.unpackb(book_blob, raw=False)
                    best_bid = book_data["bids"][0] if book_data["bids"] else (0.0, 0)
                    best_ask = book_data["asks"][0] if book_data["asks"] else (0.0, 0)
                    
                    batch_rows.append([
                        datetime.now().isoformat(),
                        sec_id,
                        best_bid[0], best_bid[1],
                        best_ask[0], best_ask[1]
                    ])
            
            # Write the entire batch to the CSV at once for maximum I/O speed
            if batch_rows:
                with open(self.filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(batch_rows)
            
            await asyncio.sleep(1.0)

async def main():
    # Make sure this list exactly matches Process 2!
    target_stocks = [
        "1333",   # HDFC BANK
        "2885",   # RELIANCE
        "11536",  # TCS
        "1594",   # INFOSYS
        "4963",   # ICICI BANK
        "3045",   # SBI (STATE BANK OF INDIA)
        "3456",   # TATA MOTORS
        "1922",   # KOTAK MAHINDRA BANK
        "11483",  # LT (LARSEN & TOUBRO)
        "11915",  # YES BANK
    ]
    logger = OrderflowLogger(instrument_ids=target_stocks)
    await logger.record_stream()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess 4 Logger shut down safely.")