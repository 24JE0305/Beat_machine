import asyncio
import struct
import websockets
import json
from typing import Any

from transport import ConnectionClosed

class DhanTransport:
    def __init__(self, client_id: int, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.ws = None
        # Dhan's production 20-depth URL
        self.url = f"wss://depth-api-feed.dhan.co/twentydepth?token={self.access_token}&clientId={self.client_id}&authType=2"

    async def connect(self) -> None:
        try:
            self.ws = await websockets.connect(self.url)
        except Exception as e:
            raise ConnectionClosed(f"Failed to connect to Dhan: {e}")

    async def subscribe(self, instruments: list[str]) -> None:
        if not self.ws:
            raise ConnectionClosed("Cannot subscribe, websocket not connected")
        
        # Dhan requires us to map instruments to their exchange segment
        # For now, we will assume "NSE_EQ" for everything to keep it simple.
        instrument_list = [{"ExchangeSegment": "NSE_EQ", "SecurityId": inst} for inst in instruments]
        
        req = {
            "RequestCode": 23,
            "InstrumentCount": len(instruments),
            "InstrumentList": instrument_list
        }
        
        try:
            await self.ws.send(json.dumps(req))
        except Exception as e:
            raise ConnectionClosed(f"Failed to send subscription: {e}")

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()
    
    async def recv(self) -> list[dict]:
        if not self.ws:
            raise ConnectionClosed("Websocket not connected")

        try:
            frame = await self.ws.recv()
        except Exception as e:
            raise ConnectionClosed(f"Websocket read failed: {e}")

        # Ensure we actually received binary data
        if not isinstance(frame, bytes):
            return []

        packets = []
        offset = 0
        total_length = len(frame)

        # A single frame can contain multiple packets stacked together
        while offset < total_length:
            # 1. Unpack Header (first 12 bytes)
            # < = little-endian, h = int16, b = byte, i = int32, I = uint32
            header_format = "<h b b i I"
            header_size = struct.calcsize(header_format)
            
            if offset + header_size > total_length:
                break # Frame is cut off, stop parsing
                
            header_bytes = frame[offset : offset + header_size]
            msg_len, feed_code, exch_seg, security_id, seq = struct.unpack(header_format, header_bytes)

            # Safeguard against malformed/zero-length packets
            if msg_len <= 0:
                break

            # Determine side from Feed Code (41 = Bid, 51 = Ask)
            if feed_code == 41:
                side = "bid"
            elif feed_code == 51:
                side = "ask"
            else:
                # If it's not a Bid or Ask packet, skip to the next packet in the frame
                offset += msg_len
                continue

            # 2. Unpack Payload (20 levels)
            # Move offset past the header to read the 20 levels
            payload_offset = offset + header_size
            levels = []
            
            for _ in range(20):
                # <dII = little-endian float64 (8 bytes), uint32 (4 bytes), uint32 (4 bytes)
                level_format = "<d I I"
                level_size = struct.calcsize(level_format)
                
                if payload_offset + level_size > offset + msg_len:
                    break
                    
                level_bytes = frame[payload_offset : payload_offset + level_size]
                price, quantity, num_orders = struct.unpack(level_format, level_bytes)
                
                # We only append if price > 0 to skip empty/zero levels
                if price > 0:
                    levels.append((price, float(quantity)))
                    
                payload_offset += level_size
            
            # 3. Add to our parsed packets list
            packets.append({
                "instrument_id": str(security_id),
                "side": side,
                "levels": levels
            })
            
            # 4. Advance the offset to the next packet in the stacked frame
            offset += msg_len

        return packets