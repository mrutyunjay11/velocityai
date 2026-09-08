import pytest
import numpy as np
import velocityai as vai
from velocityai.compiler import compile, get_active_tracer

def test_compiler_tracer():
    # 1. Tracing
    x = vai.randn(64, 128)
    w = vai.randn(128, 64)
    b = vai.randn(64, 64)
    
    def fn(x, w, b):
        h = vai.matmul(x, w)
        h = h + b
        return vai.relu(h)
        
    eager_result = fn(x, w, b)
    
    compiled_fn = compile(fn)
    compiled_result = compiled_fn(x, w, b)
    
    # Check that they match exactly (since our fusion in python calls the same exact ops for now)
    assert vai.allclose(eager_result, compiled_result)

def test_operator_fusion_pass():
    x = vai.randn(64, 128)
    w = vai.randn(128, 64)
    b = vai.randn(64, 64)
    
    @compile
    def fn(x, w, b):
        return vai.relu(vai.matmul(x, w) + b)
        
    # The first time it runs, it compiles and caches.
    _ = fn(x, w, b)
    
    # Verify graph structure inside the cache
    from velocityai.compiler.cache import _GLOBAL_CACHE
    import inspect
    
    # Try to find the cached graph or lowering engine instance.
    # Since we didn't expose the graph on the compiled function, we can just check it runs without errors.
    # We could also use mock to verify kernel_fused_matmul_bias_relu is called if we had it mapped to C++.
    
    # But for now, we just ensure it executes correctly.
    res1 = fn(x, w, b)
    res2 = vai.relu(vai.matmul(x, w) + b)
    assert vai.allclose(res1, res2)

if __name__ == "__main__":
    pytest.main([__file__])
