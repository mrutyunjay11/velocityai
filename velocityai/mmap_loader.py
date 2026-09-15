import os
import struct
import json
import mmap
import numpy as np

def mmap_safetensors(path):
    """
    Memory-maps a safetensors file.
    Returns a dictionary of tensor names to zero-copy numpy arrays.
    """
    fd = open(path, 'rb')
    
    # Read the 8-byte header length (little-endian unsigned long long)
    header_len_bytes = fd.read(8)
    if len(header_len_bytes) < 8:
        raise ValueError(f"Invalid safetensors file: {path}")
        
    header_len = struct.unpack('<Q', header_len_bytes)[0]
    
    # Read the JSON header
    header_bytes = fd.read(header_len)
    header = json.loads(header_bytes.decode('utf-8'))
    
    # The actual tensor data starts right after the header
    data_offset = 8 + header_len
    
    # Check total file size
    fd.seek(0, 2)
    file_size = fd.tell()
    
    # Memory map the entire file
    mm = mmap.mmap(fd.fileno(), file_size, access=mmap.ACCESS_READ)
    
    dtype_map = {
        'F64': np.float64,
        'F32': np.float32,
        'F16': np.float16,
        'BF16': np.float16, # BUGGY!
        'I64': np.int64,
        'I32': np.int32,
        'I16': np.int16,
        'I8': np.int8,
        'U8': np.uint8,
    }
    
    tensors = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
            
        start, end = meta['data_offsets']
        dtype_str = meta['dtype']
        shape = meta['shape']
        
        np_dtype = dtype_map.get(dtype_str)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
            
        # Create a zero-copy numpy array view into the mmap
        itemsize = np.dtype(np_dtype).itemsize
        count = (end - start) // itemsize
        
        arr = np.frombuffer(mm, dtype=np_dtype, offset=data_offset + start, count=count)
        arr = arr.reshape(shape)
        
        tensors[name] = arr
        
    # We must keep fd and mm alive as long as the arrays are used.
    # In Python, we can attach them to the dict to prevent GC
    tensors['_mmap'] = mm
    tensors['_fd'] = fd
    
    return tensors
