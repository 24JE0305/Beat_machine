import asyncio
import os
import sys
from dotenv import load_dotenv

from listener import run_process1
from orderbook import OrderBook
from redis_writer import RedisWriter
from dhan_transport import DhanTransport

async def main():
    # 1. Load the environment variables from the .env file
    load_dotenv()
    
    # 2. Fetch the credentials securely
    client_id = os.getenv("DHAN_CLIENT_ID")
    access_token = os.getenv("DHAN_ACCESS_TOKEN")
    
    # Failsafe: Crash immediately if the .env file is missing or misspelled
    if not client_id or not access_token:
        print("ERROR: Missing DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN in .env file.")
        sys.exit(1)
    
    # 3. The 20 instruments you want to track
    # Replace these examples with the actual Security IDs from the Dhan Scrip Master
    instruments = [
        "1333",   # HDFC Bank
        "2885",   # Reliance
        "3456",   # Tata Motors
        "11915",  # YES Bank
        # ... add all 20 of your target stocks here
    ] 
    
    # 4. Initialize the architecture
    transport = DhanTransport(client_id, access_token)
    redis_writer = RedisWriter() 
    
    # 5. Create an OrderBook for every instrument we are tracking
    # This automatically spawns 20 independent memory queues
    book_map = {inst: OrderBook(inst) for inst in instruments}
    
    # 6. Connect Redis and start the system
    print("Connecting to Redis...")
    await redis_writer.connect()
    
    print(f"Starting Process 1 Global Supervisor for {len(instruments)} stocks...")
    await run_process1(transport, book_map, redis_writer, instruments)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess 1 shut down manually.")