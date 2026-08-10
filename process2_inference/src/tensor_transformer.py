import numpy as np

class TensorTransformer:
    def __init__(self, depth: int = 20):
        self.depth = depth

    def create_heatmap(self, bids: list, asks: list) -> np.ndarray:
        """
        Transforms 1D order book arrays into a 2D spatial heatmap for YOLO.
        Returns a numpy array of shape (40, 2) -> 40 levels, 2 features (price, qty).
        """
        # Ensure we have exactly 20 levels padded with zeros if the book is thin
        bids_padded = bids[:self.depth] + [(0.0, 0.0)] * (self.depth - len(bids))
        asks_padded = asks[:self.depth] + [(0.0, 0.0)] * (self.depth - len(asks))
        
        ask_array = np.array(asks_padded, dtype=np.float32)
        bid_array = np.array(bids_padded, dtype=np.float32)
        
        # Reverse asks so the lowest ask (best ask) is at the bottom of the ask block,
        # physically touching the highest bid (best bid) in the center of the matrix.
        ask_array = ask_array[::-1] 
        
        # Stack them vertically to create a 40x2 spatial "image"
        # Top 20 rows = Asks, Bottom 20 rows = Bids
        heatmap = np.vstack((ask_array, bid_array))
        
        return heatmap