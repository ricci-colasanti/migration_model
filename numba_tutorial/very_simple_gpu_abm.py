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
    
    def __init__(self, width, height, num_agents, move_prob=0.3):
        """
        Initialize the simulation.
        
        Parameters:
        - width, height: Grid dimensions
        - num_agents: Number of agents
        - move_prob: Probability an agent tries to move each step
        """
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.move_prob = move_prob
        self.step_count = 0
        self.total_cells = width * height
        
        print(f"\n{'='*60}")
        print(f"GPU ABM - COMMAND LINE VERSION")
        print(f"{'='*60}")
        print(f"GPU: {cuda.get_current_device().name}")
        print(f"Grid: {width} x {height} = {self.total_cells} cells")
        print(f"Agents: {num_agents} ({num_agents/self.total_cells:.1%} occupancy)")
        print(f"Move probability: {move_prob*100:.0f}%")
        print(f"{'='*60}\n")
        
        # CONCEPT 5: GPU Memory Allocation
        # Allocate arrays directly on GPU (VRAM)
        # Data lives on GPU for fast access by kernels
        self.cell_grid_gpu = cuda.device_array(self.total_cells, dtype=np.int32)
        self.agent_x_gpu = cuda.device_array(num_agents, dtype=np.int32)
        self.agent_y_gpu = cuda.device_array(num_agents, dtype=np.int32)
        
        # Initialize the grid to all -1 (empty)
        self._fill_grid()
        
        # Place agents in center block
        self._place_agents()
        
        # CONCEPT 7: CPU copies for visualization/stats
        # We keep CPU copies to show data transfer concept
        self.agent_x_cpu = np.zeros(num_agents, dtype=np.int32)
        self.agent_y_cpu = np.zeros(num_agents, dtype=np.int32)
        
        # CONCEPT 7: GPU → CPU Data Transfer
        # Copy initial positions back to CPU
        self._sync_from_gpu()
        
        print("✓ Simulation ready!\n")
    
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
        """Place agents in a block at the center of the grid"""
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
        # Wait for all agents to be placed
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
    
    # Simulation parameters
    WIDTH, HEIGHT = 50, 50
    NUM_AGENTS = 500
    MOVE_PROB = 0.3
    STEPS = 20  # Fewer steps for demonstration
    
    # Create simulation
    sim = SimpleGPUABM(WIDTH, HEIGHT, NUM_AGENTS, move_prob=MOVE_PROB)
    
    print("Running simulation...\n")
    print(f"{'Step':>6} {'Mean X':>8} {'Mean Y':>8} {'Std X':>8} {'Std Y':>8} "
          f"{'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6}")
    print("-" * 75)
    
    # Run simulation
    for i in range(STEPS):
        stats = sim.step()
        
        # Print statistics
        print(f"{stats['step']:>6} {stats['mean_x']:>8.2f} {stats['mean_y']:>8.2f} "
              f"{stats['std_x']:>8.2f} {stats['std_y']:>8.2f} "
              f"{stats['q1']:>6} {stats['q2']:>6} {stats['q3']:>6} {stats['q4']:>6}")
        
        # Print progress every 5 steps
        if (i + 1) % 5 == 0:
            print(f"\n--- Step {i+1} completed ---")
            print(f"  Agents spread: X: {stats['min_x']}-{stats['max_x']}, "
                  f"Y: {stats['min_y']}-{stats['max_y']}")
            print(f"  Quadrants: Q1={stats['q1']}, Q2={stats['q2']}, "
                  f"Q3={stats['q3']}, Q4={stats['q4']}\n")
    
    # Final summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Get final stats
    final_stats = sim._compute_stats()
    print(f"\nFinal agent distribution:")
    print(f"  Center of mass: ({final_stats['mean_x']:.2f}, {final_stats['mean_y']:.2f})")
    print(f"  Spread: X σ={final_stats['std_x']:.2f}, Y σ={final_stats['std_y']:.2f}")
    print(f"  Bounding box: X [{final_stats['min_x']}, {final_stats['max_x']}], "
          f"Y [{final_stats['min_y']}, {final_stats['max_y']}]")
    print(f"\n  Quadrant distribution:")
    print(f"    Q1 (top-left):  {final_stats['q1']:>4} agents")
    print(f"    Q2 (top-right): {final_stats['q2']:>4} agents")
    print(f"    Q3 (bottom-left): {final_stats['q3']:>4} agents")
    print(f"    Q4 (bottom-right): {final_stats['q4']:>4} agents")

if __name__ == "__main__":
    main()