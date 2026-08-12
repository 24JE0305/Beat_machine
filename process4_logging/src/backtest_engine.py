import os
import sys
import pandas as pd
import numpy as np
import onnxruntime as ort

# Add process2_inference/src to path for TensorTransformer import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../process2_inference/src")))
from tensor_transformer import TensorTransformer

class OfflineBacktester:
    def __init__(self, model_path: str, parquet_file: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.transformer = TensorTransformer()
        self.df = pd.read_parquet(parquet_file)
        self.trades = []

    def unflatten_row(self, row):
        bids = [[row[f"bid_px_{i}"], row[f"bid_qty_{i}"]] for i in range(20) if row[f"bid_px_{i}"] > 0]
        asks = [[row[f"ask_px_{i}"], row[f"ask_qty_{i}"]] for i in range(20) if row[f"ask_px_{i}"] > 0]
        return bids, asks

    def run(self):
        # Key positions and cooldowns per security_id to avoid multi-stock cross-contamination
        positions = {}
        cooldowns = {}

        print(f"Replaying {len(self.df)} recorded orderbook ticks...")

        for idx, row in self.df.iterrows():
            timestamp = row["timestamp"]
            sec_id = str(int(row["security_id"]))
            bids, asks = self.unflatten_row(row)

            # Guard against empty orderbook sides
            if not bids or not asks:
                continue

            # Initialize tracking dicts for new instruments
            if sec_id not in positions:
                positions[sec_id] = None
                cooldowns[sec_id] = 0.0

            # Transform into 64x64 canvas input format
            heatmap = self.transformer.create_heatmap(bids, asks)
            padded_heatmap = np.zeros((64, 64), dtype=np.float32)
            padded_heatmap[:40, :2] = heatmap
            input_tensor = np.tile(padded_heatmap[None, None, :, :], (1, 3, 1, 1))

            # Run ONNX inference
            ort_outs = self.session.run(None, {self.input_name: input_tensor})
            class_scores = ort_outs[0][0][4, :]
            spoof_detected = bool(np.max(class_scores) > 0.85)

            position = positions[sec_id]

            # 1. Check Exit Logic for THIS specific stock
            if position is not None:
                time_in_trade = timestamp - position["entry_time"]
                current_mid = (bids[0][0] + asks[0][0]) / 2.0
                
                should_exit = False
                reason = None

                if time_in_trade > 1.5:
                    should_exit, reason = True, "TIMEOUT_1500MS"
                elif not spoof_detected:
                    if current_mid > position["entry_price"]:
                        should_exit, reason = True, "WALL_PULLED_PROFIT"
                    else:
                        should_exit, reason = True, "WALL_ABSORBED_STOP"

                if should_exit:
                    exit_price = bids[0][0]
                    pnl = (exit_price - position["entry_price"]) * position["qty"]
                    self.trades.append({
                        "security_id": sec_id,
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "hold_time_ms": int((timestamp - position["entry_time"]) * 1000),
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "reason": reason
                    })
                    positions[sec_id] = None
                    cooldowns[sec_id] = timestamp + 10.0
                    continue

            # 2. Check Entry Logic for THIS specific stock
            if positions[sec_id] is None and spoof_detected and timestamp > cooldowns[sec_id]:
                positions[sec_id] = {
                    "entry_price": asks[0][0],
                    "entry_time": timestamp,
                    "qty": 2000
                }

        results_df = pd.DataFrame(self.trades)
        print(f"Backtest completed. Total Trades: {len(results_df)}")
        if not results_df.empty:
            print(f"Total Gross PnL: ₹{results_df['pnl'].sum():.2f}")
            print(f"Win Rate: {(results_df['pnl'] > 0).mean() * 100:.1f}%")
        return results_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../../process2_inference/yolov8_orderbook.onnx")
    
    # Example usage once a parquet capture file exists in process4_logging/data_capture/
    capture_dir = os.path.join(base_dir, "../data_capture")
    if os.path.exists(capture_dir):
        files = [os.path.join(capture_dir, f) for f in os.listdir(capture_dir) if f.endswith(".parquet")]
        if files:
            backtester = OfflineBacktester(model_path, files[0])
            df_results = backtester.run()
            print(df_results.head())
        else:
            print("No .parquet depth recordings found. Run depth_logger.py during market hours first.")