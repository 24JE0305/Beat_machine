import numpy as np

class TensorTransformer:
    def __init__(self, depth: int = 20):
        self.depth = depth

    def create_heatmap(self, bids: list, asks: list) -> np.ndarray:
        """
        Transforms 1D order book arrays into a 2D normalized spatial heatmap for YOLO.
        Returns a numpy array of shape (40, 2) normalized between 0.0 and 1.0.
        """
        # 1. Pad to ensure exactly 20 levels
        bids_padded = bids[:self.depth] + [(0.0, 0.0)] * (self.depth - len(bids))
        asks_padded = asks[:self.depth] + [(0.0, 0.0)] * (self.depth - len(asks))
        
        ask_array = np.array(asks_padded, dtype=np.float32)
        bid_array = np.array(bids_padded, dtype=np.float32)
        
        # 2. Extract Prices and Quantities
        ask_prices, ask_qtys = ask_array[:, 0], ask_array[:, 1]
        bid_prices, bid_qtys = bid_array[:, 0], bid_array[:, 1]
        
        # 3. Calculate Mid-Price for Price Normalization
        best_bid = bid_prices[0] if bid_prices[0] > 0 else 1.0
        best_ask = ask_prices[0] if ask_prices[0] > 0 else 1.0
        mid_price = (best_bid + best_ask) / 2.0
        
        # 4. Normalize Prices: Percentage distance from mid-price
        # This keeps price features small and centered around 0.0
        norm_ask_prices = (ask_prices - mid_price) / mid_price
        norm_bid_prices = (bid_prices - mid_price) / mid_price
        
        # 5. Normalize Quantities: Scale between 0.0 and 1.0 based on max book volume
        # A massive spoof wall will become 1.0, normal orders will be 0.1 - 0.3
        max_qty = max(np.max(ask_qtys), np.max(bid_qtys), 1.0)
        norm_ask_qtys = ask_qtys / max_qty
        norm_bid_qtys = bid_qtys / max_qty
        
        # 6. Rebuild normalized arrays
        norm_asks = np.column_stack((norm_ask_prices, norm_ask_qtys))
        norm_bids = np.column_stack((norm_bid_prices, norm_bid_qtys))
        
        # 7. Reverse asks so best ask touches best bid in the center of the matrix
        norm_asks = norm_asks[::-1]
        
        # Stack vertically -> Top 20 = Asks, Bottom 20 = Bids (Shape: 40x2)
        heatmap = np.vstack((norm_asks, norm_bids))
        
        return heatmap