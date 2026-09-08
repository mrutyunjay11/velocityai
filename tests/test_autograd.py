import pytest
import numpy as np
import velocityai as vai

def test_autograd_basic():
    # Setup
    x = vai.randn(1, 3)
    x.requires_grad = True
    
    w = vai.randn(3, 1)
    w.requires_grad = True
    
    b = vai.randn(1, 1)
    b.requires_grad = True
    
    # Forward pass
    h = vai.matmul(x, w)
    out = vai.add(h, b)
    
    # Backward pass
    out.backward()
    
    assert x.grad is not None
    assert w.grad is not None
    assert b.grad is not None
    
    # Check shapes
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape
    assert b.grad.shape == b.shape

def test_autograd_loss():
    logits = vai.randn(2, 4)
    logits.requires_grad = True
    
    targets = vai.zeros((2,)) # Assume class 0
    # Wait, targets needs to be int for cross entropy, but since we didn't implement 
    # int casting fully, let's assume it accepts float and casts in numpy inside
    
    loss = vai.ops.cross_entropy_loss(logits, targets)
    loss.backward()
    
    assert logits.grad is not None
    assert logits.grad.shape == (2, 4)
    
if __name__ == "__main__":
    pytest.main([__file__])
