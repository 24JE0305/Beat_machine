import ctypes
from multiprocessing import shared_memory

# This is the exact C-struct memory layout that our C++ sniper will read.
# 1 byte for the trigger flag, 8 bytes for the execution price, 4 bytes for quantity.
class ExecutionSignal(ctypes.Structure):
    _fields_ = [
        ("fire_order", ctypes.c_bool),      # 1 byte: 0 = Wait, 1 = FIRE!
        ("target_price", ctypes.c_double),  # 8 bytes: Limit price to shoot at
        ("target_qty", ctypes.c_uint32),    # 4 bytes: How many shares
        ("is_active", ctypes.c_bool)        # 1 byte: Is Process 2 alive?
    ]

class MemoryBridge:
    def __init__(self, name="dhan_sniper_bridge", create=True):
        self.name = name
        self.size = ctypes.sizeof(ExecutionSignal)
        
        if create:
            # Create a brand new block of RAM in the OS
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            except FileExistsError:
                # If it crashed previously, attach to the old one and clear it
                self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        else:
            # Process 3 (or test scripts) will use create=False to just attach to it
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            
        # Map our C-struct onto the physical memory buffer
        self.signal = ExecutionSignal.from_buffer(self.shm.buf)
        
        if create:
            self.disarm()

    def arm_and_fire(self, price: float, qty: int):
        """Called by YOLOv8 when a spoofing trap is detected."""
        self.signal.target_price = price
        self.signal.target_qty = qty
        self.signal.fire_order = True  # C++ sniper sees this instantly

    def disarm(self):
        self.signal.fire_order = False
        self.signal.target_price = 0.0
        self.signal.target_qty = 0
        self.signal.is_active = True

    def close(self):
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass