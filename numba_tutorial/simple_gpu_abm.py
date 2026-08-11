"""
SIMPLE GPU ABM – TRUE RANDOM DIFFUSION (iteration-based seed)
Each agent picks a different random direction every step.
"""

import numpy as np
from numba import cuda
import tkinter as tk
import time
import math

# ============================================================
# GPU KERNEL (random direction per step)
# ============================================================

@cuda.jit  # CONCEPT 1: This decorator compiles the function for GPU
# - Transforms Python into CUDA machine code for thousands of GPU cores
# - Kernel must be "pure" - takes all inputs as parameters, returns nothing
# - Only uses operations supported on GPU (no Python lists, strings, etc.)
def move_agents_kernel(agent_x, agent_y, cell_grid, width, height, move_prob, iteration):
    # CONCEPT 2: Thread Indexing - each thread gets a unique ID
    idx = cuda.grid(1)  # Returns unique thread index (0, 1, 2, ...)
    # The (1) means 1-dimensional grid of threads
    
    # CONCEPT 2: The Guard Pattern - prevents out-of-bounds errors
    # We launch MORE threads than agents (for GPU efficiency)
    # Threads without agents to process must exit immediately
    if idx >= agent_x.shape[0]:  # If this thread has no agent to handle
        return  # Thread exits - like a worker going home if no box to paint

    # Each thread handles ONE agent (mapping: thread N → agent N)
    x = agent_x[idx]  # Get this agent's current x position
    y = agent_y[idx]  # Get this agent's current y position

    # CONCEPT 3: Pseudo-Random Number Generation on GPU
    # We can't use Python's random module (CPU-only), so we implement our own LCG
    # Seed includes iteration so each step gives new randomness
    seed = (idx * 1103515245 + 12345 + iteration * 1000000) & 0x7fffffff
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    rand_val = (seed & 0x7fffffff) / 2147483647.0
    
    # Check if agent should move this step
    if rand_val > move_prob:
        return  # stays put

    # CONCEPT: Multiple Attempts with Random Directions
    # Try up to 8 different random directions to find an empty cell
    # This reduces the chance of an agent failing to move due to occupied cells
    max_attempts = 8
    moved = False
    
    for attempt in range(max_attempts):
        # Generate a NEW random direction each attempt
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        dx = (seed % 3) - 1  # Gives -1, 0, or 1
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        dy = (seed % 3) - 1  # Gives -1, 0, or 1
        
        # If direction is (0,0), skip this attempt and try another
        if dx == 0 and dy == 0:
            continue  # Try another direction
        
        nx = x + dx
        ny = y + dy
        
        # Check if within grid bounds
        if not (0 <= nx < width and 0 <= ny < height):
            continue  # Try another direction
        
        flat_idx = ny * width + nx
        current_flat = y * width + x
        
        # CONCEPT 4: Atomic Compare-And-Swap (CAS) for thread safety
        # When multiple agents try to claim the same cell, CAS ensures only one wins
        # atomically: "if cell is empty (-1), set it to this agent's ID"
        old_val = cuda.atomic.cas(cell_grid, flat_idx, -1, idx)
        
        if old_val == -1:
            # SUCCESS! We claimed the empty cell
            # 1. Clear our old cell (mark it as empty)
            cell_grid[current_flat] = -1
            # 2. Update agent position in GPU memory
            agent_x[idx] = nx
            agent_y[idx] = ny
            moved = True
            break  # Exit the loop - we've successfully moved
    
    # If moved is False, all attempts failed (all neighbors occupied or out of bounds)
    # Agent simply stays in place - no action needed

# ============================================================
# SIMULATION CLASS
# ============================================================

class SimpleGPUABM:
    def __init__(self, width, height, num_agents, move_prob=0.3, delay=0.1):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.move_prob = move_prob
        self.delay = delay
        self.step_count = 0

        print(f"\n{'='*50}")
        print(f"GPU ABM – RANDOM DIFFUSION")
        print(f"{'='*50}")
        print(f"GPU: {cuda.get_current_device().name}")
        print(f"Grid: {width} x {height} = {width*height} cells")
        print(f"Agents: {num_agents} ({num_agents/(width*height):.1%} occupancy)")
        print(f"Move probability: {move_prob*100:.0f}%")
        print(f"Delay: {delay}s")
        print(f"{'='*50}\n")

        # CONCEPT 5: GPU Memory Allocation
        # Allocate memory on the GPU for the cell grid and agent positions
        # cuda.device_array() creates an array directly in GPU memory
        # This is where data lives during computation - CPU can't access it directly
        self.cell_grid_gpu = cuda.device_array(width * height, dtype=np.int32)
        self._fill_cell_grid()

        self.agent_x_gpu = cuda.device_array(num_agents, dtype=np.int32)
        self.agent_y_gpu = cuda.device_array(num_agents, dtype=np.int32)

        # Place agents in a centre block
        self._place_agents_center()

        # CPU copies for drawing
        # We keep CPU copies so we can visualize the simulation
        self.agent_x_cpu = np.zeros(num_agents, dtype=np.int32)
        self.agent_y_cpu = np.zeros(num_agents, dtype=np.int32)
        self._sync_from_gpu()

        print("Ready to simulate!\n" + "="*50 + "\n")

    def _fill_cell_grid(self):
        """Fill the cell grid with -1 (empty)"""
        @cuda.jit
        def fill_kernel(cell_grid, size):
            idx = cuda.grid(1)  # Each thread gets unique index
            if idx < size:      # Guard pattern - only threads within bounds
                cell_grid[idx] = -1

        # CONCEPT 6: Launch Configuration
        # Calculate blocks needed to cover all cells (width * height)
        threads = 256  # Good number for GPU efficiency
        # Ceiling division ensures we have ENOUGH threads
        blocks = (self.width * self.height + threads - 1) // threads
        fill_kernel[blocks, threads](self.cell_grid_gpu, self.width * self.height)
        cuda.synchronize()  # CONCEPT 8: Wait for GPU to finish

    def _place_agents_center(self):
        """Place all agents in a compact block at the center of the grid"""
        # Calculate block size to fit all agents in a square
        block_size = int(math.ceil(math.sqrt(self.num_agents)))
        center_x = self.width // 2
        center_y = self.height // 2
        start_x = center_x - block_size // 2
        start_y = center_y - block_size // 2

        # CONCEPT 7: CPU-GPU Data Movement (CPU side)
        # Create CPU arrays with initial agent positions
        x_cpu = np.zeros(self.num_agents, dtype=np.int32)
        y_cpu = np.zeros(self.num_agents, dtype=np.int32)

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
        # cuda.to_device() copies data from CPU memory to GPU memory
        # This is necessary because the GPU can only access its own memory
        self.agent_x_gpu = cuda.to_device(x_cpu)
        self.agent_y_gpu = cuda.to_device(y_cpu)

        # GPU kernel to place agents on the grid
        @cuda.jit
        def place_kernel(x_arr, y_arr, cell_grid, num_agents, width):
            idx = cuda.grid(1)  # Each thread gets unique index
            if idx < num_agents:  # Guard pattern
                flat_idx = y_arr[idx] * width + x_arr[idx]
                cell_grid[flat_idx] = idx  # Store agent ID in cell

        # CONCEPT 6: Launch Configuration for place_kernel
        threads = 256
        blocks = (self.num_agents + threads - 1) // threads
        place_kernel[blocks, threads](self.agent_x_gpu, self.agent_y_gpu,
                                      self.cell_grid_gpu, self.num_agents, self.width)
        cuda.synchronize()  # CONCEPT 8: Wait for GPU to finish

    def _sync_from_gpu(self):
        """Copy agent positions from GPU back to CPU for visualization"""
        # CONCEPT 7: GPU → CPU Data Transfer
        # .copy_to_host() copies data from GPU memory back to CPU memory
        # This is slow but necessary because Tkinter (visualization) runs on CPU
        self.agent_x_cpu = self.agent_x_gpu.copy_to_host()
        self.agent_y_cpu = self.agent_y_gpu.copy_to_host()

    def step(self):
        """Advance the simulation by one time step"""
        # CONCEPT 6: Launch Configuration
        # Calculate the number of thread blocks needed
        threads_per_block = 256  # Optimal for most GPUs (multiple of 32)
        # Ceiling division: (num_agents + threads - 1) // threads
        # Ensures we launch ENOUGH threads to cover all agents
        blocks = (self.num_agents + threads_per_block - 1) // threads_per_block

        # CONCEPT 6: Kernel Launch
        # This launches the kernel with (blocks, threads_per_block) configuration
        # Total threads = blocks * threads_per_block (e.g., 2 * 256 = 512 threads)
        # Some threads will be "extra" and hit the guard pattern
        move_agents_kernel[blocks, threads_per_block](
            self.agent_x_gpu, self.agent_y_gpu, self.cell_grid_gpu,
            self.width, self.height, self.move_prob, self.step_count
        )

        # CONCEPT 8: cuda.synchronize() - Wait for GPU to finish
        # This ensures all GPU work completes before we continue
        # Without this, we might try to read data that's still being processed!
        cuda.synchronize()
        
        # CONCEPT 7: GPU → CPU Data Transfer
        # Copy updated positions back to CPU for visualization
        self._sync_from_gpu()
        time.sleep(self.delay)  # Slow down simulation for viewing

        self.step_count += 1
        return self.step_count

# ============================================================
# TKINTER VISUALIZER
# ============================================================

class ABMVisualizer:
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents

        self.root = tk.Tk()
        self.root.title(f"GPU ABM – {num_agents} agents")

        self.canvas_size = 600
        self.canvas = tk.Canvas(self.root, width=self.canvas_size,
                                height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.scale = self.canvas_size / max(width, height)
        self.agent_size = max(2, self.scale * 0.8)

    def draw(self, x_positions, y_positions):
        """Draw agents as blue squares on the canvas"""
        self.canvas.delete("all")
        for i in range(len(x_positions)):
            x = x_positions[i]
            y = y_positions[i]
            cx = x * self.scale + self.scale / 2
            cy = y * self.scale + self.scale / 2
            s = self.agent_size
            self.canvas.create_rectangle(
                cx - s/2, cy - s/2,
                cx + s/2, cy + s/2,
                fill='blue', outline=''
            )
        self.root.update()

    def update_title(self, step):
        self.root.title(f"GPU ABM – Step {step} – {self.num_agents} agents")

    def close(self):
        self.root.destroy()

# ============================================================
# MAIN
# ============================================================

def main():
    # Simulation parameters
    WIDTH, HEIGHT = 50, 50
    NUM_AGENTS = 500
    MOVE_PROB = 0.3      # 30% chance to move each step
    STEPS = 200
    DELAY = 0.1

    print("\n" + "="*50)
    print("GPU ABM – RANDOM DIFFUSION")
    print("="*50)
    print(f"Grid: {WIDTH}x{HEIGHT} = {WIDTH*HEIGHT} cells")
    print(f"Agents: {NUM_AGENTS} ({NUM_AGENTS/(WIDTH*HEIGHT):.1%} occupancy)")
    print(f"Move probability: {MOVE_PROB*100:.0f}%")
    print(f"Steps: {STEPS}")
    print(f"Delay: {DELAY}s")
    print("Starting: Agents in a block at the centre")
    print("Each agent picks a random direction every step")
    print("="*50 + "\n")

    # Create simulation and visualizer
    sim = SimpleGPUABM(WIDTH, HEIGHT, NUM_AGENTS, move_prob=MOVE_PROB, delay=DELAY)
    viz = ABMVisualizer(WIDTH, HEIGHT, NUM_AGENTS)

    try:
        print("Running simulation... Close the window to stop\n")
        for i in range(STEPS):
            step = sim.step()
            viz.draw(sim.agent_x_cpu, sim.agent_y_cpu)
            viz.update_title(step)
            if (i+1) % 20 == 0:
                print(f"Step {i+1}/{STEPS} completed")

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        viz.close()
        print("\nDone!")

if __name__ == "__main__":
    main()