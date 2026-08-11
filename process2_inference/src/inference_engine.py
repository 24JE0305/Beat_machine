import asyncio
import msgpack
import os
import time
import numpy as np
import redis.asyncio as aioredis
import onnxruntime as ort

from memory_bridge import MemoryBridge
from tensor_transformer import TensorTransformer

class InferenceEngine:
    def __init__(self, model_path: str, instrument_ids: list):
        self.instrument_ids = instrument_ids
        
        # Cooldown management (in seconds)
        self.cooldowns = {sec_id: 0.0 for sec_id in instrument_ids}
        self.cooldown_seconds = 10.0  # 10 second lock per stock
        
        # >>> NEW: Position State Tracker
        # Stores None if flat, or dict: {"entry_price": float, "entry_time": float, "side": "BUY"}
        self.active_positions = {sec_id: None for sec_id in instrument_ids}
        
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
        preds = predictions[0] 
        class_scores = preds[4, :] 
        
        if np.max(class_scores) > conf_threshold:
            return True
            
        return False

    # >>> NEW: Dedicated Spoof Exit Logic Method
    def check_exit_logic(self, sec_id: str, book_data: dict, current_time: float, spoof_detected: bool) -> tuple[bool, str | None]:
        """
        Evaluates whether an active trade should be closed on this tick.
        """
        pos = self.active_positions[sec_id]
        if pos is None:
            return False, None

        time_in_trade = current_time - pos["entry_time"]
        bids = book_data["bids"]
        asks = book_data["asks"]

        # RULE 1: Time-Decay Timeout (If spoof hasn't moved price within 1.5s, exit)
        if time_in_trade > 1.5:
            return True, "TIMEOUT_1500MS"

        # RULE 2: Calculate Top-5 Order Book Imbalance (OBI)
        top_5_bid_vol = sum(b[1] for b in bids[:5])
        top_5_ask_vol = sum(a[1] for a in asks[:5])
        total_vol = (top_5_bid_vol + top_5_ask_vol) or 1
        obi = (top_5_bid_vol - top_5_ask_vol) / total_vol

        # If OBI flips heavily against our Long position, exit immediately
        if pos["side"] == "BUY" and obi < -0.30:
            return True, "OBI_REVERSAL"

        # RULE 3: Wall Cancellation vs. Absorption
        # If YOLOv8 no longer detects the spoof wall, the spoofer either pulled it or it was eaten
        if not spoof_detected:
            current_mid = (bids[0][0] + asks[0][0]) / 2.0
            if current_mid > pos["entry_price"]:
                return True, "SPOOF_WALL_PULLED_PROFIT"
            else:
                return True, "SPOOF_WALL_ABSORBED_STOP"

        return False, None

    async def run_loop(self):
        print(f"Process 2 ONNX Engine listening for {len(self.instrument_ids)} stocks...")
        
        while True:
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
                
                # 1. Create a 64x64 canvas padded with zeros
                padded_heatmap = np.zeros((64, 64), dtype=np.float32)
                # 2. Place the 40x2 heatmap into the top-left corner
                padded_heatmap[:40, :2] = heatmap
                # 3. Shape tensor to (1, 3, 64, 64) for ONNX batch input
                input_tensor = np.tile(padded_heatmap[None, None, :, :], (1, 3, 1, 1))
                
                # Run ONNX Inference
                ort_inputs = {self.input_name: input_tensor}
                ort_outs = self.session.run(None, ort_inputs)
                
                current_time = time.time()
                spoof_detected = self._parse_yolov8_output(ort_outs[0], conf_threshold=0.85)
                
                # >>> NEW: BRANCHING EXECUTION (Check Exits FIRST, then Entries)
                
                # --- BRANCH A: WE HAVE AN ACTIVE TRADE IN THIS STOCK -> CHECK EXIT ---
                if self.active_positions[sec_id] is not None:
                    should_exit, reason = self.check_exit_logic(sec_id, book_data, current_time, spoof_detected)
                    if should_exit:
                        best_bid_price = book_data["bids"][0][0]
                        print(f"[{time.strftime('%H:%M:%S')}] EXIT TRAP on {sec_id}! Reason: {reason} | Exiting at {best_bid_price}")
                        
                        # Fire Exit Order (For short-duration paper trading / closing the leg)
                        self.bridge.arm_and_fire(security_id=int(sec_id), price=best_bid_price, qty=2000)
                        await asyncio.sleep(0.1)
                        self.bridge.disarm()
                        
                        # Clear position state
                        self.active_positions[sec_id] = None
                        self.cooldowns[sec_id] = current_time  # Start cooldown after exit
                        
                # --- BRANCH B: WE ARE FLAT -> CHECK ENTRY ---
                else:
                    if spoof_detected and (current_time - self.cooldowns[sec_id] > self.cooldown_seconds):
                        best_ask_price = book_data["asks"][0][0]
                        print(f"[{time.strftime('%H:%M:%S')}] TRAP DETECTED on {sec_id}! Firing BUY entry...")
                        
                        # Record open trade state in RAM
                        self.active_positions[sec_id] = {
                            "entry_price": best_ask_price,
                            "entry_time": current_time,
                            "side": "BUY"
                        }
                        
                        # Pass the specific security_id to physical RAM with 2000 shares
                        self.bridge.arm_and_fire(security_id=int(sec_id), price=best_ask_price, qty=2000)
                        await asyncio.sleep(0.1)
                        self.bridge.disarm()
            
            await asyncio.sleep(0.001)

async def main():
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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../yolov8_orderbook.onnx")
    
    engine = InferenceEngine(model_path=model_path, instrument_ids=target_stocks)
    try:
        await engine.run_loop()
    except asyncio.CancelledError:
        pass
    finally:
        engine.bridge.deactivate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess 2 shut down manually.")