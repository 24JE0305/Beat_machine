import ctypes
from multiprocessing import shared_memory

SIDE_BUY = 0
SIDE_SELL = 1

# This is the exact C-struct memory layout that our C++ sniper will read.
# Field ORDER and SIZES must match sniper.cpp's ExecutionSignal exactly --
# this is raw shared memory, not a serialized format.
class ExecutionSignal(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("fire_order", ctypes.c_bool),
        ("side", ctypes.c_uint8),           # 0 = BUY, 1 = SELL
        ("security_id", ctypes.c_uint32),
        ("target_price", ctypes.c_double),  
        ("target_qty", ctypes.c_uint32),    
        ("is_active", ctypes.c_bool)        
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
            self.signal.is_active = True

    def arm_and_fire(self, security_id: int, price: float, qty: int, side: int = SIDE_BUY):
        self.signal.security_id = security_id
        self.signal.target_price = price
        self.signal.target_qty = qty
        self.signal.side = side
        self.signal.fire_order = True

    def disarm(self):
        self.signal.fire_order = False
        self.signal.target_price = 0.0
        self.signal.target_qty = 0
        self.signal.side = SIDE_BUY

    def deactivate(self):
        """Call on shutdown so the C++ sniper exits cleanly."""
        self.signal.is_active = False

    def close(self):
        # 1. Destroy the reference to the C-pointer
        self.signal = None 
        
        # 2. Force Python's Garbage Collector to instantly wipe it from RAM
        import gc
        gc.collect() 
        
        # 3. Now it is safe to close the memory block
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass