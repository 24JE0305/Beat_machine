import asyncio
import msgpack
import numpy as np
import redis.asyncio as aioredis
import onnxruntime as ort

from memory_bridge import MemoryBridge
from tensor_transformer import TensorTransformer

class InferenceEngine:
    def __init__(self, model_path: str, instrument_ids: list):
        self.instrument_ids = instrument_ids
        
        # 1. Initialize Infrastructure
        self.redis = aioredis.from_url("redis://127.0.0.1:6379")
        self.transformer = TensorTransformer()
        self.bridge = MemoryBridge(create=True)
        
        # 2. Load the YOLO ONNX Model
        print(f"Loading ONNX model from {model_path}...")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def _parse_yolov8_output(self, predictions: np.ndarray, conf_threshold: float = 0.85) -> bool:
        """
        Parses the raw tensor output from a YOLOv8 ONNX model.
        Assumes a single-class model where class 0 is 'spoof_trap'.
        """
        # YOLOv8 standard output shape: (batch_size, 4 + num_classes, num_anchors)
        # For 1 class, it typically looks like (1, 5, 8400)
        preds = predictions[0] 
        
        # Rows 0-3 are bounding box coordinates (x, y, w, h)
        # Row 4 contains the confidence scores for Class 0 across all anchors
        class_scores = preds[4, :] 
        
        # If the highest confidence detection beats our threshold, trigger the sniper
        if np.max(class_scores) > conf_threshold:
            return True
            
        return False

    async def run_loop(self):
        print(f"Process 2 ONNX Engine listening for {len(self.instrument_ids)} stocks...")
        
        while True:
            # Scan all stocks continuously 
            for sec_id in self.instrument_ids:
                book_key = f"book:{sec_id}"
                stale_key = f"stale:{sec_id}"
                
                keys = await self.redis.mget([stale_key, book_key])
                stale_flag, book_blob = keys[0], keys[1]
                
                if stale_flag == b"1" or not book_blob:
                    continue
                    
                # Unpack and transform...
                book_data = msgpack.unpackb(book_blob, raw=False)
                heatmap = self.transformer.create_heatmap(book_data["bids"], book_data["asks"])
                input_tensor = heatmap.reshape(1, 1, 40, 2).astype(np.float32)
                
                # Run ONNX Inference
                ort_inputs = {self.input_name: input_tensor}
                ort_outs = self.session.run(None, ort_inputs)
                
                # Parse the tensor array
                spoof_detected = self._parse_yolov8_output(ort_outs[0], conf_threshold=0.85)
                
                if spoof_detected:
                    best_ask_price = book_data["asks"][0][0]
                    print(f"TRAP DETECTED on {sec_id}! Firing paper trade...")
                    
                    # Pass the specific security_id to physical RAM
                    self.bridge.arm_and_fire(security_id=int(sec_id), price=best_ask_price, qty=100)
                    
                    # Pause briefly to prevent the sniper from double-firing on the exact same frame
                    await asyncio.sleep(1.0)
                    self.bridge.disarm()
            
            await asyncio.sleep(0.001)

async def main():
    # Pick your Security IDs from the Dhan Scrip Master CSV
    target_stocks = ["1333", "11915", "3456", "13538"] 
    engine = InferenceEngine(model_path="yolov8_orderbook.onnx", instrument_ids=target_stocks)
    await engine.run_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess 2 shut down manually.")