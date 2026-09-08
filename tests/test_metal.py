import velocityai as vai
import time

def test_metal_matmul():
    print("Creating tensors...")
    # Try creating large tensors to see speedup
    M, K, N = 1024, 1024, 1024
    a = vai.randn(M, K)
    b = vai.randn(K, N)
    
    print("Testing CPU Matmul...")
    start = time.time()
    out_cpu = vai.matmul(a, b)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f}s")
    
    print("Testing Metal Matmul...")
    try:
        a_metal = a.to("metal")
        b_metal = b.to("metal")
    except Exception as e:
        print(f"Failed to move to metal: {e}")
        return
        
    start = time.time()
    out_metal = vai.matmul(a_metal, b_metal)
    # Move back to CPU to compare
    out_metal_cpu = out_metal.to("cpu")
    metal_time = time.time() - start
    
    print(f"Metal time: {metal_time:.4f}s")
    print(f"Speedup: {cpu_time / metal_time:.2f}x")
    
    # Check correctness
    diff = abs(out_cpu.numpy() - out_metal_cpu.numpy()).max()
    print(f"Max difference between CPU and Metal: {diff}")
    assert diff < 1e-4, "Metal computation differs too much!"
    print("Metal backend test PASSED!")

if __name__ == "__main__":
    test_metal_matmul()
