import asyncio
import msgpack
import numpy as np
import redis.asyncio as aioredis
import onnxruntime as ort

from memory_bridge import MemoryBridge
from tensor_transformer import TensorTransformer

class InferenceEngine:
    def __init__(self, model_path: str, instrument_id: str):
        self.instrument_id = instrument_id
        
        # 1. Initialize Infrastructure
        self.redis = aioredis.from_url("redis://127.0.0.1:6379")
        self.transformer = TensorTransformer()
        self.bridge = MemoryBridge(create=True)
        
        # 2. Load the YOLO ONNX Model
        print(f"Loading ONNX model from {model_path}...")
        # self.session = ort.InferenceSession(model_path)
        # self.input_name = self.session.get_inputs()[0].name
        
    async def run_loop(self):
        book_key = f"book:{self.instrument_id}"
        stale_key = f"stale:{self.instrument_id}"
        
        print(f"Process 2 ONNX Engine listening for {self.instrument_id}...")
        
        while True:
            # 1. Fetch both keys in a single network call for speed
            keys = await self.redis.mget([stale_key, book_key])
            stale_flag, book_blob = keys[0], keys[1]
            
            # If connection dropped or no data, do not trade
            if stale_flag == b"1" or not book_blob:
                await asyncio.sleep(0.001)
                continue
                
            # 2. Unpack the binary data
            book_data = msgpack.unpackb(book_blob, raw=False)
            
            # 3. Transform to 2D Spatial Heatmap
            heatmap = self.transformer.create_heatmap(book_data["bids"], book_data["asks"])
            
            # 4. Reshape for YOLO (Batch=1, Channel=1, Height=40, Width=2)
            input_tensor = heatmap.reshape(1, 1, 40, 2).astype(np.float32)
            
            # 5. Execute YOLO Inference (Uncomment when model is ready)
            # outputs = self.session.run(None, {self.input_name: input_tensor})
            # spoof_detected = self._parse_yolo_output(outputs)
            
            spoof_detected = False # Placeholder
            
            # 6. The Sniper Trigger
            if spoof_detected:
                best_ask_price = book_data["asks"][0][0]
                print(f"TRAP DETECTED! Firing sniper at {best_ask_price}")
                
                # Instantly flip the byte in physical RAM. C++ sees this in 1 nanosecond.
                self.bridge.arm_and_fire(price=best_ask_price, qty=100)
                
                # Sleep to prevent firing 10,000 times a second on the same signal
                await asyncio.sleep(1.0)
                self.bridge.disarm()
            
            # Yield to the event loop so we don't lock up a single CPU core at 100%
            await asyncio.sleep(0.001) 

async def main():
    # We will point this to your actual .onnx file later
    engine = InferenceEngine(model_path="yolov8_orderbook.onnx", instrument_id="1333")
    await engine.run_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess 2 shut down manually.")