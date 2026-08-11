import os
import pandas as pd
import numpy as np
import onnxruntime as ort
from tensor_transformer import OrderbookTensorTransformer  # Your existing transformer
from dhan_fee_simulator import calculate_pnl_and_fees      # Your fee simulator

class SpoofBacktester:
    def __init__(self, model_path, parquet_file):
        self.session = ort.InferenceSession(model_path)
        self.transformer = OrderbookTensorTransformer()
        self.data = pd.read_parquet(parquet_file)
        self.position = 0  # 1 for Long, -1 for Short, 0 for Flat
        self.entry_price = 0.0
        self.entry_time = 0
        self.trades = []

    def reconstruct_book(self, row):
        """Reconstructs 20-level bids and asks from a flattened row."""
        bids = [[row[f"bid_px_{i}"], row[f"bid_qty_{i}"], row[f"bid_ord_{i}"]] for i in range(20)]
        asks = [[row[f"ask_px_{i}"], row[f"ask_qty_{i}"], row[f"ask_ord_{i}"]] for i in range(20)]
        return bids, asks

    def run_backtest(self):
        print(f"Running backtest over {len(self.data)} frames...")
        
        for idx, row in self.data.iterrows():
            timestamp = row["timestamp"]
            bids, asks = self.reconstruct_book(row)
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            # 1. Convert book to tensor format for YOLOv8
            tensor = self.transformer.transform(bids, asks)
            
            # 2. Run inference
            inputs = {self.session.get_inputs()[0].name: tensor}
            outputs = self.session.run(None, inputs)
            
            # Extract spoof detections: [confidence, side, level_index, spoof_volume]
            detections = outputs[0]
            
            # 3. Check Exit Logic first if we are in a position
            if self.position != 0:
                exit_signal, exit_reason = self.check_exit_logic(
                    self.position, bids, asks, timestamp, detections
                )
                if exit_signal:
                    exit_px = best_bid if self.position == 1 else best_ask
                    pnl = calculate_pnl_and_fees(self.entry_price, exit_px, self.position, qty=100)
                    self.trades.append({
                        "entry_time": self.entry_time,
                        "exit_time": timestamp,
                        "pnl": pnl,
                        "reason": exit_reason
                    })
                    self.position = 0
                    continue
            
            # 4. Entry Logic (Only if flat)
            if self.position == 0 and len(detections) > 0:
                # Example: If spoof detected on Ask side -> Buy momentum before wall is pulled
                conf, side, level, _ = detections[0]
                if conf > 0.85:
                    if side == 1: # Spoof sell wall detected -> enter Long
                        self.position = 1
                        self.entry_price = best_ask
                        self.entry_time = timestamp
                    elif side == 0: # Spoof buy wall detected -> enter Short
                        self.position = -1
                        self.entry_price = best_bid
                        self.entry_time = timestamp

        print(f"Backtest Complete. Total Trades: {len(self.trades)}")
        return pd.DataFrame(self.trades)

    def check_exit_logic(self, position, bids, asks, current_time, detections):
        # Implementation of structured spoof exit rules (detailed in Section 3)
        return False, None