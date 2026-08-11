# test_cupy.py
import sys

def test_cupy():
    print("=" * 60)
    print("CUPY INSTALLATION TEST")
    print("=" * 60)
    
    try:
        import cupy as cp
        print(f"✅ CuPy imported successfully!")
        print(f"   Version: {cp.__version__}")
    except ImportError as e:
        print(f"❌ CuPy is not installed!")
        print(f"   Error: {e}")
        return False
    
    try:
        # Check GPU
        gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"\n✅ Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\n   GPU {i}:")
            print(f"   - Name: {props['name'].decode()}")
            print(f"   - Memory: {props['totalGlobalMem'] / 1024**3:.2f} GB")
            print(f"   - Compute Capability: {props['major']}.{props['minor']}")
        
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False
    
    # Test basic operations
    try:
        print("\nTesting CuPy operations...")
        
        # Create arrays on GPU
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([10, 20, 30, 40, 50])
        c = a + b
        
        print(f"   a: {a}")
        print(f"   b: {b}")
        print(f"   a + b: {c}")
        print("✅ Basic operations work!")
        
        # Test random generation
        random_array = cp.random.randint(0, 1000, size=(5, 5))
        print(f"\n   Random 5x5 array:\n{random_array}")
        print("✅ Random generation works!")
        
        # Performance test
        import time
        import numpy as np
        
        size = 10000000
        print(f"\n   Performance test with {size:,} elements:")
        
        # CPU (NumPy)
        start = time.time()
        a_cpu = np.random.random(size)
        b_cpu = np.random.random(size)
        c_cpu = a_cpu + b_cpu
        cpu_time = time.time() - start
        
        # GPU (CuPy)
        start = time.time()
        a_gpu = cp.random.random(size)
        b_gpu = cp.random.random(size)
        c_gpu = a_gpu + b_gpu
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start
        
        print(f"   CPU time: {cpu_time:.3f}s")
        print(f"   GPU time: {gpu_time:.3f}s")
        print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
        print("✅ Performance test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! CuPy is ready to use.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_cupy()
    sys.exit(0 if success else 1)
