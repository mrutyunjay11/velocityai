import pytest
import numpy as np
import velocityai as vai

def test_tensor_creation():
    t = vai.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t.shape == (2, 3)
    assert t.ndim == 2
    assert t.numel == 6
    assert t.is_contiguous
    assert t.dtype == vai.float32

    z = vai.zeros(3, 4)
    assert z.shape == (3, 4)
    assert np.all(z.numpy() == 0.0)

    o = vai.ones(2, 2)
    assert np.all(o.numpy() == 1.0)

    f = vai.full((2, 3), 42.0)
    assert np.all(f.numpy() == 42.0)

    r = vai.randn(10, 10)
    assert r.shape == (10, 10)
    assert not np.all(r.numpy() == 0.0)

def test_numpy_zero_copy_interop():
    # VelocityAI -> NumPy
    t = vai.tensor([[1.0, 2.0], [3.0, 4.0]])
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)
    assert arr[0, 1] == 2.0

    # NumPy -> VelocityAI (zero-copy)
    np_arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    t_from_np = vai.from_numpy(np_arr, copy=False)
    assert t_from_np.shape == (2, 2)
    assert t_from_np.numpy()[1, 0] == 30.0

    # Verify zero-copy: mutate np_arr and verify t_from_np reflects it
    np_arr[0, 0] = 999.0
    assert t_from_np.numpy()[0, 0] == 999.0

def test_strides_and_views():
    t = vai.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t.strides == (3, 1)

    # Reshape (zero-copy view)
    reshaped = t.reshape(3, 2)
    assert reshaped.shape == (3, 2)
    assert reshaped.is_contiguous
    assert np.array_equal(reshaped.numpy(), t.numpy().reshape(3, 2))

    # Transpose (zero-copy strided view)
    transposed = t.transpose(0, 1)
    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert not transposed.is_contiguous
    assert np.array_equal(transposed.numpy(), t.numpy().T)

    # .T property
    assert np.array_equal(t.T.numpy(), t.numpy().T)

    # contiguous() copy
    c = transposed.contiguous()
    assert c.is_contiguous
    assert c.strides == (2, 1)
    assert np.array_equal(c.numpy(), t.numpy().T)

def test_slicing():
    t = vai.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    # 1D slice along dim 0
    row1 = t[1]
    assert row1.shape == (3,)
    assert np.array_equal(row1.numpy(), np.array([4.0, 5.0, 6.0], dtype=np.float32))

    # Slice range
    rows = t[0:2]
    assert rows.shape == (2, 3)
    assert np.array_equal(rows.numpy(), t.numpy()[0:2])

def test_arithmetic_operations():
    a = vai.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = vai.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Addition
    c_add = a + b
    assert np.allclose(c_add.numpy(), a.numpy() + b.numpy())

    # Subtraction
    c_sub = b - a
    assert np.allclose(c_sub.numpy(), b.numpy() - a.numpy())

    # Multiplication (Hadamard / element-wise)
    c_mul = a * b
    assert np.allclose(c_mul.numpy(), a.numpy() * b.numpy())

    # Division
    c_div = b / a
    assert np.allclose(c_div.numpy(), b.numpy() / a.numpy())

    # Scalar arithmetic
    c_scalar_add = a + 10.0
    assert np.allclose(c_scalar_add.numpy(), a.numpy() + 10.0)

    c_scalar_mul = a * 3.0
    assert np.allclose(c_scalar_mul.numpy(), a.numpy() * 3.0)

def test_matmul():
    # 2x2
    a = vai.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = vai.tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    np_expected = a.numpy() @ b.numpy()
    assert np.allclose(c.numpy(), np_expected)

    # Larger matrix (128x128)
    np_A = np.random.randn(128, 64).astype(np.float32)
    np_B = np.random.randn(64, 128).astype(np.float32)
    A = vai.from_numpy(np_A)
    B = vai.from_numpy(np_B)

    C = vai.matmul(A, B)
    assert np.allclose(C.numpy(), np_A @ np_B, rtol=1e-4, atol=1e-4)

def test_reductions_and_relu():
    x = vai.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
    
    # ReLU
    r = vai.relu(x)
    assert np.allclose(r.numpy(), np.maximum(0.0, x.numpy()))

    # Total Sum
    s = x.sum()
    assert np.isclose(s.item(), np.sum(x.numpy()))

    # Sum along dim 0
    s0 = x.sum(dim=0)
    assert np.allclose(s0.numpy(), np.sum(x.numpy(), axis=0))

    # Sum along dim 1
    s1 = x.sum(dim=1)
    assert np.allclose(s1.numpy(), np.sum(x.numpy(), axis=1))

    # Mean
    m = x.mean()
    assert np.isclose(m.item(), np.mean(x.numpy()))

def test_memory_pool_stats():
    # Allocate tensors and check stats
    vai.memory.empty_cache()
    t1 = vai.randn(1000, 1000) # ~4MB
    assert vai.memory.allocated_bytes() > 0
    peak = vai.memory.peak_bytes()
    assert peak >= vai.memory.allocated_bytes()

    del t1
    vai.memory.empty_cache()
    assert vai.memory.cached_bytes() == 0
