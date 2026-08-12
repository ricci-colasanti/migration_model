"""
SIMPLE GPU ABM - COMMAND LINE VERSION
Minimal implementation showing GPU acceleration concepts
No visualization - just prints statistics
"""

import numpy as np
from numba import cuda
import time
import math

# ============================================================
# GPU KERNEL - Moves agents on the grid
# ============================================================

@cuda.jit  # CONCEPT 1: @cuda.jit decorator compiles this for GPU
def move_agents_kernel(agent_x, agent_y, cell_grid, width, height, move_prob, iteration):
    """
    Each GPU thread handles ONE agent.
    Moves agents to random adjacent cells if empty.
    
    Parameters:
    - agent_x, agent_y: GPU arrays of agent positions
    - cell_grid: GPU array tracking which cell each agent occupies (-1 = empty)
    - width, height: Grid dimensions
    - move_prob: Probability an agent tries to move (0.0 to 1.0)
    - iteration: Current step number (for random number generation)
    """
    
    # CONCEPT 2: Thread Indexing
    # Each thread gets a unique ID. We use 1D grid.
    idx = cuda.grid(1)
    
    # CONCEPT 2: Guard Pattern
    # We launch MORE threads than agents (for GPU efficiency)
    # Threads without agents exit immediately
    if idx >= agent_x.shape[0]:
        return
    
    # Each thread handles ONE agent (thread N → agent N)
    x = agent_x[idx]
    y = agent_y[idx]
    
    # CONCEPT 3: Pseudo-Random Number Generation on GPU
    # Can't use Python's random() - CPU only!
    # We use LCG (Linear Congruential Generator)
    # Include iteration so randomness changes each step
    seed = (idx * 1103515245 + 12345 + iteration * 1000000) & 0x7fffffff
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    rand_val = (seed & 0x7fffffff) / 2147483647.0
    
    # Check if agent should move this step
    if rand_val > move_prob:
        return  # Agent stays put
    
    # Try up to 8 random directions to find an empty cell
    max_attempts = 8
    moved = False
    
    for attempt in range(max_attempts):
        # Generate random direction (-1, 0, or 1 for dx and dy)
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        dx = (seed % 3) - 1
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        dy = (seed % 3) - 1
        
        # Skip if no movement (dx=0 and dy=0)
        if dx == 0 and dy == 0:
            continue
        
        # Calculate new position
        nx = x + dx
        ny = y + dy
        
        # Skip if out of bounds
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        
        # Convert (x,y) to 1D array index
        flat_idx = ny * width + nx
        current_flat = y * width + x
        
        # CONCEPT 4: Atomic Compare-And-Swap (CAS)
        # Thread-safe way to claim an empty cell
        # atomically: "if cell is -1 (empty), set it to idx (agent ID)"
        old_val = cuda.atomic.cas(cell_grid, flat_idx, -1, idx)
        
        if old_val == -1:
            # SUCCESS! We claimed the empty cell
            # 1. Clear our old cell (mark as empty)
            cell_grid[current_flat] = -1
            # 2. Update agent position
            agent_x[idx] = nx
            agent_y[idx] = ny
            moved = True
            break  # Exit loop - we've moved
    
    # If moved is False, all attempts failed
    # Agent stays in place - no action needed

# ============================================================
# SIMULATION CLASS
# ============================================================

class SimpleGPUABM:
    """Agent-Based Model running on GPU"""
    
    def __init__(self, width, height, num_agents, move_prob=0.3, random_start=True):
        """
        Initialize the simulation.
        
        Parameters:
        - width, height: Grid dimensions
        - num_agents: Number of agents
        - move_prob: Probability an agent tries to move each step
        - random_start: If True, agents start randomly distributed
                       If False, agents start in a compact block at center
        """
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.move_prob = move_prob
        self.step_count = 0
        self.total_cells = width * height
        self.random_start = random_start
        
        print(f"\n{'='*70}")
        print(f"GPU ABM - COMMAND LINE VERSION")
        print(f"{'='*70}")
        print(f"GPU: {cuda.get_current_device().name}")
        print(f"Grid: {width} x {height} = {self.total_cells:,} cells")
        print(f"Agents: {num_agents:,} ({num_agents/self.total_cells:.2%} occupancy)")
        print(f"Move probability: {move_prob*100:.0f}%")
        print(f"Start type: {'Random' if random_start else 'Compact block'}")
        print(f"{'='*70}\n")
        
        # CONCEPT 5: GPU Memory Allocation
        # Allocate arrays directly on GPU (VRAM)
        # Data lives on GPU for fast access by kernels
        print("Allocating GPU memory...")
        start_time = time.perf_counter()
        
        self.cell_grid_gpu = cuda.device_array(self.total_cells, dtype=np.int32)
        self.agent_x_gpu = cuda.device_array(num_agents, dtype=np.int32)
        self.agent_y_gpu = cuda.device_array(num_agents, dtype=np.int32)
        
        # Initialize the grid to all -1 (empty)
        self._fill_grid()
        
        # Place agents (either random or block)
        self._place_agents()
        
        # CONCEPT 7: CPU copies for visualization/stats
        # We keep CPU copies to show data transfer concept
        self.agent_x_cpu = np.zeros(num_agents, dtype=np.int32)
        self.agent_y_cpu = np.zeros(num_agents, dtype=np.int32)
        
        # CONCEPT 7: GPU → CPU Data Transfer
        # Copy initial positions back to CPU
        self._sync_from_gpu()
        
        elapsed = time.perf_counter() - start_time
        print(f"✓ Memory allocated and initialized in {elapsed:.2f}s\n")
    
    def _fill_grid(self):
        """Fill the cell grid with -1 (empty) on the GPU"""
        @cuda.jit
        def fill_kernel(grid, size):
            idx = cuda.grid(1)
            if idx < size:
                grid[idx] = -1
        
        # CONCEPT 6: Launch Configuration
        # Calculate blocks needed to cover all cells
        threads_per_block = 256
        blocks = (self.total_cells + threads_per_block - 1) // threads_per_block
        
        # Launch the kernel
        fill_kernel[blocks, threads_per_block](self.cell_grid_gpu, self.total_cells)
        
        # CONCEPT 8: cuda.synchronize()
        # Wait for GPU to finish before continuing
        cuda.synchronize()
    
    def _place_agents(self):
        """Place agents according to random_start setting"""
        if self.random_start:
            self._place_agents_random()
        else:
            self._place_agents_block()
    
    def _place_agents_random(self):
        """Place agents randomly across the entire grid"""
        print(f"Generating random positions for {self.num_agents:,} agents...")
        start_time = time.perf_counter()
        
        # Generate random positions on CPU
        # Use np.random for efficiency on CPU
        x_cpu = np.random.randint(0, self.width, self.num_agents, dtype=np.int32)
        y_cpu = np.random.randint(0, self.height, self.num_agents, dtype=np.int32)
        
        elapsed = time.perf_counter() - start_time
        print(f"  Generated positions in {elapsed:.2f}s")
        
        # CONCEPT 7: CPU → GPU Data Transfer
        print("  Transferring to GPU...")
        start_time = time.perf_counter()
        
        self.agent_x_gpu = cuda.to_device(x_cpu)
        self.agent_y_gpu = cuda.to_device(y_cpu)
        
        elapsed = time.perf_counter() - start_time
        print(f"  Transferred in {elapsed:.2f}s")
        
        # Register agents in the grid
        print("  Registering agents on grid...")
        start_time = time.perf_counter()
        
        @cuda.jit
        def place_kernel(x_arr, y_arr, grid, num_agents, width):
            idx = cuda.grid(1)
            if idx < num_agents:
                flat_idx = y_arr[idx] * width + x_arr[idx]
                grid[flat_idx] = idx  # Store agent ID in cell
        
        # CONCEPT 6: Launch Configuration
        threads_per_block = 256
        blocks = (self.num_agents + threads_per_block - 1) // threads_per_block
        
        # Launch the kernel
        place_kernel[blocks, threads_per_block](
            self.agent_x_gpu, self.agent_y_gpu,
            self.cell_grid_gpu, self.num_agents, self.width
        )
        
        # CONCEPT 8: cuda.synchronize()
        cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
        print(f"  Registered in {elapsed:.2f}s")
    
    def _place_agents_block(self):
        """Place agents in a compact block at the center of the grid"""
        # Calculate block size to fit all agents in a square
        block_size = int(math.ceil(math.sqrt(self.num_agents)))
        center_x = self.width // 2
        center_y = self.height // 2
        start_x = center_x - block_size // 2
        start_y = center_y - block_size // 2
        
        # CONCEPT 7: CPU-GPU Data Movement (CPU side)
        # Create CPU arrays with initial positions
        x_cpu = np.zeros(self.num_agents, dtype=np.int32)
        y_cpu = np.zeros(self.num_agents, dtype=np.int32)
        
        # Fill positions (simple square pattern)
        agent_idx = 0
        for dy in range(block_size):
            for dx in range(block_size):
                if agent_idx >= self.num_agents:
                    break
                x = start_x + dx
                y = start_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    x_cpu[agent_idx] = x
                    y_cpu[agent_idx] = y
                    agent_idx += 1
            if agent_idx >= self.num_agents:
                break
        
        # CONCEPT 7: CPU → GPU Data Transfer
        # cuda.to_device() copies from CPU RAM to GPU VRAM
        self.agent_x_gpu = cuda.to_device(x_cpu)
        self.agent_y_gpu = cuda.to_device(y_cpu)
        
        # GPU kernel to register agents in the grid
        @cuda.jit
        def place_kernel(x_arr, y_arr, grid, num_agents, width):
            idx = cuda.grid(1)
            if idx < num_agents:
                flat_idx = y_arr[idx] * width + x_arr[idx]
                grid[flat_idx] = idx  # Store agent ID in cell
        
        # CONCEPT 6: Launch Configuration
        threads_per_block = 256
        blocks = (self.num_agents + threads_per_block - 1) // threads_per_block
        
        # Launch the kernel
        place_kernel[blocks, threads_per_block](
            self.agent_x_gpu, self.agent_y_gpu,
            self.cell_grid_gpu, self.num_agents, self.width
        )
        
        # CONCEPT 8: cuda.synchronize()
        cuda.synchronize()
    
    def _sync_from_gpu(self):
        """
        CONCEPT 7: GPU → CPU Data Transfer
        Copy data from GPU back to CPU.
        This is SLOW - only do when necessary!
        """
        self.agent_x_cpu = self.agent_x_gpu.copy_to_host()
        self.agent_y_cpu = self.agent_y_gpu.copy_to_host()
    
    def step(self):
        """
        Advance simulation by one step.
        Returns statistics about the current state.
        """
        # CONCEPT 6: Launch Configuration
        # Calculate blocks needed for all agents
        threads_per_block = 256
        blocks = (self.num_agents + threads_per_block - 1) // threads_per_block
        
        # Launch the movement kernel
        # This runs on GPU - ALL agents move in parallel!
        move_agents_kernel[blocks, threads_per_block](
            self.agent_x_gpu, self.agent_y_gpu, self.cell_grid_gpu,
            self.width, self.height, self.move_prob, self.step_count
        )
        
        # CONCEPT 8: cuda.synchronize()
        # CRITICAL: Wait for GPU to finish before reading results!
        # Without this, we'd read incomplete/old data
        cuda.synchronize()
        
        # CONCEPT 7: GPU → CPU Data Transfer
        # Copy updated positions back to CPU for stats
        # This is slow but necessary for our statistics
        self._sync_from_gpu()
        
        # Compute statistics on CPU (now that we have the data)
        stats = self._compute_stats()
        
        self.step_count += 1
        return stats
    
    def _compute_stats(self):
        """
        Compute statistics about agent distribution.
        Shows we can analyze data after transferring to CPU.
        """
        # Mean position
        mean_x = np.mean(self.agent_x_cpu)
        mean_y = np.mean(self.agent_y_cpu)
        
        # Standard deviation (spread)
        std_x = np.std(self.agent_x_cpu)
        std_y = np.std(self.agent_y_cpu)
        
        # Min/Max positions (bounding box)
        min_x = np.min(self.agent_x_cpu)
        max_x = np.max(self.agent_x_cpu)
        min_y = np.min(self.agent_y_cpu)
        max_y = np.max(self.agent_y_cpu)
        
        # Count agents in each quadrant
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        q1 = np.sum((self.agent_x_cpu < mid_x) & (self.agent_y_cpu < mid_y))
        q2 = np.sum((self.agent_x_cpu >= mid_x) & (self.agent_y_cpu < mid_y))
        q3 = np.sum((self.agent_x_cpu < mid_x) & (self.agent_y_cpu >= mid_y))
        q4 = np.sum((self.agent_x_cpu >= mid_x) & (self.agent_y_cpu >= mid_y))
        
        return {
            'step': self.step_count + 1,
            'mean_x': mean_x,
            'mean_y': mean_y,
            'std_x': std_x,
            'std_y': std_y,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'q1': q1,
            'q2': q2,
            'q3': q3,
            'q4': q4
        }

# ============================================================
# MAIN
# ============================================================

def main():
    """Run the simulation and print statistics"""
    
    # Simulation parameters - BIG SIMULATION!
    WIDTH, HEIGHT = 10000, 10000
    NUM_AGENTS = 60_000_000  # 60 million agents!
    MOVE_PROB = 1.0          # 100% move probability
    STEPS = 100              # Test with 100 steps first
    RANDOM_START = True      # Random distribution for immediate coverage
    
    print("\n" + "="*70)
    print("GPU ABM - MASSIVE SIMULATION")
    print("="*70)
    print(f"Grid: {WIDTH:,} x {HEIGHT:,} = {WIDTH*HEIGHT:,} cells")
    print(f"Agents: {NUM_AGENTS:,} ({NUM_AGENTS/(WIDTH*HEIGHT):.2%} occupancy)")
    print(f"Move probability: {MOVE_PROB*100:.0f}%")
    print(f"Steps: {STEPS}")
    print(f"Start type: {'Random' if RANDOM_START else 'Compact block'}")
    print("="*70)
    
    # Estimate memory usage
    grid_mb = (WIDTH * HEIGHT * 4) / (1024 * 1024)
    agents_mb = (NUM_AGENTS * 4 * 2) / (1024 * 1024)  # x and y arrays
    total_mb = grid_mb + agents_mb
    
    print(f"\nEstimated GPU Memory Usage:")
    print(f"  Grid: {grid_mb:.1f} MB")
    print(f"  Agent arrays: {agents_mb:.1f} MB")
    print(f"  Total: {total_mb:.1f} MB")
    print(f"  (Plus overhead for other operations)")
    
    # Create simulation
    print(f"\nInitializing simulation...")
    sim = SimpleGPUABM(WIDTH, HEIGHT, NUM_AGENTS, move_prob=MOVE_PROB, 
                       random_start=RANDOM_START)
    
    print("Running simulation...\n")
    print(f"{'Step':>6} {'Mean X':>10} {'Mean Y':>10} {'Std X':>8} {'Std Y':>8} "
          f"{'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8} {'Time (s)':>10}")
    print("-" * 95)
    
    # Track timing
    step_times = []
    total_start = time.perf_counter()
    
    # Run simulation
    for i in range(STEPS):
        step_start = time.perf_counter()
        stats = sim.step()
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        # Print statistics
        print(f"{stats['step']:>6} {stats['mean_x']:>10.2f} {stats['mean_y']:>10.2f} "
              f"{stats['std_x']:>8.2f} {stats['std_y']:>8.2f} "
              f"{stats['q1']:>8,} {stats['q2']:>8,} {stats['q3']:>8,} {stats['q4']:>8,} "
              f"{step_time:>10.3f}")
        
        # Print progress every 10 steps
        if (i + 1) % 10 == 0:
            avg_time = np.mean(step_times[-10:])
            print(f"\n--- Step {i+1} completed ---")
            print(f"  Avg step time (last 10): {avg_time:.3f}s")
            print(f"  Agents spread: X: {stats['min_x']:,}-{stats['max_x']:,}, "
                  f"Y: {stats['min_y']:,}-{stats['max_y']:,}")
            print(f"  Quadrants: Q1={stats['q1']:,}, Q2={stats['q2']:,}, "
                  f"Q3={stats['q3']:,}, Q4={stats['q4']:,}")
            
            # Check balance
            expected = NUM_AGENTS // 4
            max_dev = max(abs(stats['q1'] - expected), abs(stats['q2'] - expected),
                         abs(stats['q3'] - expected), abs(stats['q4'] - expected))
            print(f"  Quadrant balance: max deviation = {max_dev:,} ({max_dev/NUM_AGENTS*100:.4f}%)\n")
    
    total_time = time.perf_counter() - total_start
    avg_step_time = np.mean(step_times)
    
    # Final summary
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    
    # Get final stats
    final_stats = sim._compute_stats()
    
    print(f"\nPerformance Summary:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average step time: {avg_step_time:.3f}s")
    print(f"  Steps: {STEPS}")
    print(f"  Agents per second: {NUM_AGENTS / avg_step_time:,.0f}")
    
    print(f"\nFinal agent distribution:")
    print(f"  Center of mass: ({final_stats['mean_x']:.2f}, {final_stats['mean_y']:.2f})")
    print(f"  Spread: X σ={final_stats['std_x']:.2f}, Y σ={final_stats['std_y']:.2f}")
    print(f"  Bounding box: X [{final_stats['min_x']:,}, {final_stats['max_x']:,}], "
          f"Y [{final_stats['min_y']:,}, {final_stats['max_y']:,}]")
    print(f"  Bounding box width: {final_stats['max_x'] - final_stats['min_x']:,} cells")
    print(f"  Bounding box height: {final_stats['max_y'] - final_stats['min_y']:,} cells")
    
    print(f"\nQuadrant distribution:")
    expected = NUM_AGENTS // 4
    print(f"    Q1 (top-left):  {final_stats['q1']:>10,} agents  (diff: {final_stats['q1']-expected:>+8,})")
    print(f"    Q2 (top-right): {final_stats['q2']:>10,} agents  (diff: {final_stats['q2']-expected:>+8,})")
    print(f"    Q3 (bottom-left): {final_stats['q3']:>10,} agents  (diff: {final_stats['q3']-expected:>+8,})")
    print(f"    Q4 (bottom-right): {final_stats['q4']:>10,} agents  (diff: {final_stats['q4']-expected:>+8,})")
    
    # Calculate and display coverage
    coverage_x = (final_stats['max_x'] - final_stats['min_x']) / WIDTH * 100
    coverage_y = (final_stats['max_y'] - final_stats['min_y']) / HEIGHT * 100
    print(f"\nGrid coverage:")
    print(f"  X-axis: {coverage_x:.1f}% of grid width")
    print(f"  Y-axis: {coverage_y:.1f}% of grid height")
    
    # Estimate time to full coverage
    if not RANDOM_START:
        expansion_rate = (final_stats['max_x'] - final_stats['min_x']) / STEPS
        steps_to_full = (WIDTH - (final_stats['max_x'] - final_stats['min_x'])) / expansion_rate
        print(f"\nEstimated steps to full coverage: {steps_to_full:.0f}")

if __name__ == "__main__":
    main()