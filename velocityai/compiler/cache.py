import hashlib
import pickle

class Cache:
    def __init__(self):
        self.memory_cache = {}
        
    def _compute_key(self, fn, input_shapes, input_dtypes):
        h = hashlib.sha256()
        h.update(fn.__name__.encode())
        for shape in input_shapes:
            h.update(str(shape).encode())
        for dtype in input_dtypes:
            h.update(str(dtype).encode())
        return h.hexdigest()
        
    def get(self, fn, input_shapes, input_dtypes):
        key = self._compute_key(fn, input_shapes, input_dtypes)
        return self.memory_cache.get(key)
        
    def put(self, fn, input_shapes, input_dtypes, compiled_fn):
        key = self._compute_key(fn, input_shapes, input_dtypes)
        self.memory_cache[key] = compiled_fn

_GLOBAL_CACHE = Cache()
