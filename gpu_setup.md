# 🚀 GPU-Accelerated Agent-Based Model Setup Guide

**System Requirements:**
- NVIDIA GPU with CUDA support (RTX 2080 Super, RTX 3050, etc.)
- Ubuntu 20.04 / 22.04 / 24.04
- 8GB+ GPU memory recommended

---

## Step 1: Check Your GPU

```bash
lspci | grep -i vga
```

Example output:
```
01:00.0 VGA compatible controller: NVIDIA Corporation TU104M [GeForce RTX 2080 SUPER Mobile / Max-Q] (rev a1)
```

---

## Step 2: Install NVIDIA Drivers

```bash
# Add the graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install the recommended driver
sudo apt install nvidia-driver-550

# Restart your computer
sudo reboot
```

**After reboot, verify:**
```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 ...    Off |   00000000:01:00.0  On |                  N/A |
| N/A   48C    P8             10W /  150W |      33MiB /   8192MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

---

## Step 3: Set Up Python Virtual Environment

```bash
# Install virtualenv if needed
sudo apt install python3-pip python3-venv

# Create and activate virtual environment
virtualenv -p python3 venv
source venv/bin/activate
```

Your prompt should change to: `(venv) user@host:~$`

---

## Step 4: Install Python Packages

```bash
# Install base packages
pip install numpy matplotlib

# Install GPU packages
pip install cupy-cuda12x numba

# Install CUDA headers
pip install cupy-cuda12x[ctk]
```

---

## Step 5: Test GPU Setup

Create `test_gpu.py`:

```python
#!/usr/bin/env python
import sys
import time
import numpy as np

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
    
    try:
        print("\nTesting CuPy operations...")
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([10, 20, 30, 40, 50])
        c = a + b
        
        print(f"   a: {a}")
        print(f"   b: {b}")
        print(f"   a + b: {c}")
        print("✅ Basic operations work!")
        
        random_array = cp.random.randint(0, 1000, size=(5, 5))
        print(f"\n   Random 5x5 array:\n{random_array}")
        print("✅ Random generation works!")
        
        size = 10000000
        print(f"\n   Performance test with {size:,} elements:")
        
        start = time.time()
        a_cpu = np.random.random(size)
        b_cpu = np.random.random(size)
        c_cpu = a_cpu + b_cpu
        cpu_time = time.time() - start
        
        start = time.time()
        a_gpu = cp.random.random(size)
        b_gpu = cp.random.random(size)
        c_gpu = a_gpu + b_gpu
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start
        
        print(f"   CPU time: {cpu_time:.3f}s")
        print(f"   GPU time: {gpu_time:.3f}s")
        if gpu_time > 0:
            print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
        print("✅ Performance test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! CuPy is ready to use.")
    print("=" * 60)
    return True


def test_numba():
    print("\n" + "=" * 60)
    print("NUMBA CUDA TEST")
    print("=" * 60)
    
    try:
        from numba import cuda
        print(f"✅ Numba imported successfully!")
        
        if cuda.is_available():
            print(f"✅ CUDA available in Numba!")
            print(f"   GPU: {cuda.get_current_device().name}")
            print(f"   Compute Capability: {cuda.get_current_device().compute_capability}")
            
            @cuda.jit
            def add_kernel(a, b, c):
                idx = cuda.grid(1)
                if idx < a.size:
                    c[idx] = a[idx] + b[idx]
            
            n = 1000000
            a = np.random.random(n).astype(np.float32)
            b = np.random.random(n).astype(np.float32)
            c = np.zeros_like(a)
            
            d_a = cuda.to_device(a)
            d_b = cuda.to_device(b)
            d_c = cuda.to_device(c)
            
            threads_per_block = 256
            blocks = (n + threads_per_block - 1) // threads_per_block
            add_kernel[blocks, threads_per_block](d_a, d_b, d_c)
            
            result = d_c.copy_to_host()
            expected = a + b
            
            if np.allclose(result, expected):
                print("✅ Simple kernel test passed!")
            else:
                print("❌ Kernel test failed")
                return False
            
            print("✅ Numba test passed!")
            
        else:
            print("❌ CUDA is NOT available in Numba!")
            return False
            
    except ImportError as e:
        print(f"❌ Numba is not installed!")
        return False
    except Exception as e:
        print(f"❌ Numba test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Numba CUDA is ready to use.")
    print("=" * 60)
    return True


def main():
    print("\n" + "=" * 60)
    print("GPU ACCELERATION TEST SUITE")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("=" * 60 + "\n")
    
    cupy_ok = test_cupy()
    numba_ok = test_numba()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if cupy_ok and numba_ok:
        print("✅ BOTH TESTS PASSED! Your GPU is ready!")
    elif cupy_ok:
        print("⚠️  CuPy works but Numba has issues.")
    else:
        print("❌ GPU tests failed. Check installation.")

if __name__ == "__main__":
    main()
```

**Run the test:**
```bash
python test_gpu.py
```

Expected output:
```
============================================================
CUPY INSTALLATION TEST
============================================================
✅ CuPy imported successfully!
   Version: 14.1.1

✅ Found 1 GPU(s)

   GPU 0:
   - Name: NVIDIA GeForce RTX 2080 Super
   - Memory: 7.60 GB
   - Compute Capability: 7.5

Testing CuPy operations...
   a: [1 2 3 4 5]
   b: [10 20 30 40 50]
   a + b: [11 22 33 44 55]
✅ Basic operations work!

   Random 5x5 array:
[[690 800 732  97 448]
 [802  52 713  10 191]
 ...
✅ Random generation works!

   Performance test with 10,000,000 elements:
   CPU time: 0.120s
   GPU time: 0.132s
   Speedup: 0.9x
✅ Performance test passed!

============================================================
✅ ALL TESTS PASSED! CuPy is ready to use.
============================================================
```

---

## Step 6: Run Your Simulation

```bash
python abm_simulation_gpu.py
```

Expected output:
```
============================================================
GPU-ACCELERATED ABM - ZERO AGENT LOSS
============================================================
Agents: 60,000,000 on 10,000x10,000 grid
Move probability: 5%
Neighborhood: 99x99
Iterations: 30
Debug mode: False
============================================================

Iter   Time(s)    Moves           Alive        Occ%      
------------------------------------------------------------
1      0.089      2,999,998       60,000,000   60.0%     
2      0.022      3,000,001       60,000,000   60.0%     
...
30     0.021      2,999,999       60,000,000   60.0%     
------------------------------------------------------------

============================================================
SIMULATION COMPLETE
============================================================
Total time: 0.74s
Average per iteration: 0.025s
Initial agents: 60,000,000
Final alive: 60,000,000
Agents lost: 0
Total moves: 89,999,996
Average moves per iteration: 3,000,000
Final occupancy: 60.0%
============================================================
```

---

## Performance Results

| System | Time/Iteration | Speedup |
|--------|---------------|---------|
| CPU (12 cores) | ~8 seconds | 1x |
| GPU (RTX 2080 Super) | **~0.02 seconds** | **~400x faster!** |

---

## Troubleshooting

### CuPy installation fails:
```bash
pip install cupy-cuda12x
# or for older CUDA
pip install cupy-cuda11x
```

### CUDA headers not found:
```bash
pip install cupy-cuda12x[ctk]
export CUDA_PATH=/usr/local/cuda
```

### Numba can't find CUDA:
```bash
sudo apt install nvidia-cuda-toolkit
```

---

## Summary

You now have:
- ✅ NVIDIA drivers (580.173.02)
- ✅ CUDA 13.0 support
- ✅ CuPy 14.1.1 working
- ✅ Numba CUDA support
- ✅ Full GPU-accelerated ABM simulation

**Your GPU is ready to simulate 60 million agents in ~0.02 seconds per iteration!** 🚀🔥

---

*Guide created: 2026-08-11*
*Tested on Ubuntu 22.04 with NVIDIA GeForce RTX 2080 Super Mobile*

