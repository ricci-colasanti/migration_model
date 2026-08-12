"""
MASSIVE GPU ABM - 60M agents on 15,000x15,000 grid
View a 200x200 window of the grid (zoomed in)
"""

import numpy as np
from numba import cuda
import tkinter as tk
import math

# ============================================================
# GPU KERNEL (random direction per step)
# ============================================================

@cuda.jit
def move_agents_kernel(agent_x, agent_y, cell_grid, width, height, move_prob, iteration):
    idx = cuda.grid(1)
    if idx >= agent_x.shape[0]:
        return

    x = agent_x[idx]
    y = agent_y[idx]

    # Seed includes iteration so each step gives new randomness
    seed = (idx * 1103515245 + 12345 + iteration * 1000000) & 0x7fffffff
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    rand_val = (seed & 0x7fffffff) / 2147483647.0
    if rand_val > move_prob:
        return

    # Pick one random direction
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    dx = (seed % 3) - 1
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    dy = (seed % 3) - 1

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


# ============================================================
# VRAM UTILITY
# ============================================================

def get_vram_usage():
    """Get current VRAM usage in MB using CUDA"""
    try:
        mem_info = cuda.current_context().get_memory_info()
        free = mem_info.free / (1024 * 1024)
        total = mem_info.total / (1024 * 1024)
        used = total - free
        return used, total, free
    except:
        return 0, 0, 0

# ============================================================
# SIMULATION CLASS
# ============================================================

class MassiveGPUABM:
    def __init__(self, width, height, num_agents, move_prob=0.3, view_size=200):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.move_prob = move_prob
        self.step_count = 0
        self.view_size = view_size

        print(f"\n{'='*60}")
        print(f"MASSIVE GPU ABM")
        print(f"{'='*60}")
        print(f"GPU: {cuda.get_current_device().name}")
        
        # Show GPU memory info
        used, total, free = get_vram_usage()
        print(f"VRAM: {used:.1f} MB used / {total:.1f} MB total ({free:.1f} MB free)")
        print(f"{'='*60}")
        
        print(f"Grid: {width:,} x {height:,} = {width*height:,} cells")
        print(f"Agents: {num_agents:,} ({num_agents/(width*height):.1%} occupancy)")
        print(f"Move probability: {move_prob*100:.0f}%")
        print(f"Viewport: {view_size}x{view_size} cells")
        print(f"{'='*60}\n")

        # Calculate memory requirements
        grid_mb = (width * height * 4) / (1024 * 1024)
        agent_mb = (num_agents * 2 * 4) / (1024 * 1024)
        total_mb = grid_mb + agent_mb
        
        print(f"Memory requirements:")
        print(f"  Cell Grid: {grid_mb:.1f} MB")
        print(f"  Agents: {agent_mb:.1f} MB")
        print(f"  Total: {total_mb:.1f} MB")
        print(f"{'='*60}\n")

        # Pick a random viewport position
        max_x = width - view_size
        max_y = height - view_size
        self.view_x = np.random.randint(0, max_x)
        self.view_y = np.random.randint(0, max_y)
        print(f"Random viewport position: ({self.view_x}, {self.view_y})")
        print(f"Viewport area: ({self.view_x}, {self.view_y}) to ({self.view_x + view_size}, {self.view_y + view_size})")

        # Allocate GPU memory
        print("\nAllocating GPU memory...")
        self.cell_grid_gpu = cuda.device_array(width * height, dtype=np.int32)
        self._fill_cell_grid()

        self.agent_x_gpu = cuda.device_array(num_agents, dtype=np.int32)
        self.agent_y_gpu = cuda.device_array(num_agents, dtype=np.int32)

        # Place agents randomly
        print("Placing agents randomly...")
        self._place_agents_random()
        print("Agents placed!\n")

        # Show VRAM after allocation
        used, total, free = get_vram_usage()
        print(f"VRAM after allocation: {used:.1f} MB used / {total:.1f} MB total ({free:.1f} MB free)")
        print(f"{'='*60}\n")

        # CPU copies for visualization (only the viewport)
        self.view_x_cpu = np.zeros(view_size * view_size, dtype=np.int32)
        self.view_y_cpu = np.zeros(view_size * view_size, dtype=np.int32)
        
        # Now sync viewport
        visible = self._sync_viewport()
        print(f"Initial visible agents in viewport: {visible:,}")

        print("\nReady to simulate!\n" + "="*60 + "\n")

    def _fill_cell_grid(self):
        """Fill the cell grid with -1 (empty)"""
        @cuda.jit
        def fill_kernel(cell_grid, size):
            idx = cuda.grid(1)
            if idx < size:
                cell_grid[idx] = -1

        threads = 256
        blocks = (self.width * self.height + threads - 1) // threads
        fill_kernel[blocks, threads](self.cell_grid_gpu, self.width * self.height)
        cuda.synchronize()

    def _place_agents_random(self):
        """Place agents randomly across the grid"""
        # Generate random positions on CPU
        positions = np.random.choice(self.width * self.height, 
                                     self.num_agents, replace=False)
        
        x_cpu = (positions % self.width).astype(np.int32)
        y_cpu = (positions // self.width).astype(np.int32)

        self.agent_x_gpu = cuda.to_device(x_cpu)
        self.agent_y_gpu = cuda.to_device(y_cpu)

        # Place agents in cell grid using kernel
        @cuda.jit
        def place_kernel(x_arr, y_arr, cell_grid, num_agents, width):
            idx = cuda.grid(1)
            if idx < num_agents:
                flat_idx = y_arr[idx] * width + x_arr[idx]
                cell_grid[flat_idx] = idx

        threads = 256
        blocks = (self.num_agents + threads - 1) // threads
        place_kernel[blocks, threads](self.agent_x_gpu, self.agent_y_gpu,
                                      self.cell_grid_gpu, self.num_agents, self.width)
        cuda.synchronize()

    def _sync_viewport(self):
        """Copy only the viewport area from GPU to CPU"""
        # Get positions from GPU
        x_cpu = self.agent_x_gpu.copy_to_host()
        y_cpu = self.agent_y_gpu.copy_to_host()
        
        # Filter agents in viewport
        mask = (x_cpu >= self.view_x) & (x_cpu < self.view_x + self.view_size) & \
               (y_cpu >= self.view_y) & (y_cpu < self.view_y + self.view_size)
        
        view_agents = np.where(mask)[0]
        self.view_x_cpu = x_cpu[view_agents]
        self.view_y_cpu = y_cpu[view_agents]
        
        # Convert viewport coordinates to local coordinates (0 to view_size)
        self.view_x_cpu = self.view_x_cpu - self.view_x
        self.view_y_cpu = self.view_y_cpu - self.view_y
        
        return len(view_agents)

    def step(self):
        """Run one simulation step"""
        threads_per_block = 256
        blocks = (self.num_agents + threads_per_block - 1) // threads_per_block

        move_agents_kernel[blocks, threads_per_block](
            self.agent_x_gpu, self.agent_y_gpu, self.cell_grid_gpu,
            self.width, self.height, self.move_prob, self.step_count
        )

        cuda.synchronize()
        self.step_count += 1
        
        # Sync viewport for display
        visible = self._sync_viewport()
        
        return visible

# ============================================================
# TKINTER VISUALIZER (Zoomed in - 4x bigger)
# ============================================================

class ABMVisualizer:
    def __init__(self, view_size, num_agents):
        self.view_size = view_size
        self.num_agents = num_agents

        self.root = tk.Tk()
        self.root.title(f"GPU ABM – {num_agents:,} agents")

        # Make canvas 4x bigger (1200x1200 instead of 600x600)
        self.canvas_size = 1200
        self.canvas = tk.Canvas(self.root, width=self.canvas_size,
                                height=self.canvas_size, bg='white')
        self.canvas.pack()

        # Scale: viewport -> canvas (bigger = more zoomed in)
        self.scale = self.canvas_size / view_size
        # Make agents large enough to see clearly
        self.agent_size = max(4, self.scale * 1.5)

        print(f"Visualization: {self.canvas_size}x{self.canvas_size} canvas")
        print(f"Scale: {self.scale:.2f} pixels per cell")
        print(f"Agent size: {self.agent_size:.1f} pixels")

    def draw(self, x_positions, y_positions):
        self.canvas.delete("all")
        
        # Draw grid lines (every 10 cells)
        for i in range(0, self.view_size + 1, 10):
            self.canvas.create_line(i * self.scale, 0, i * self.scale, self.canvas_size, fill='lightgray', width=0.5)
            self.canvas.create_line(0, i * self.scale, self.canvas_size, i * self.scale, fill='lightgray', width=0.5)
        
        # Draw each agent as a visible square
        for i in range(len(x_positions)):
            x = x_positions[i]
            y = y_positions[i]
            cx = x * self.scale
            cy = y * self.scale
            s = self.agent_size
            self.canvas.create_rectangle(
                cx - s/2, cy - s/2,
                cx + s/2, cy + s/2,
                fill='blue', outline='darkblue', width=0.5
            )
        
        
        self.root.update()

    def update_title(self, step, visible_count):
        self.root.title(f"GPU ABM – Step {step} – Visible: {visible_count:,} agents")

    def close(self):
        self.root.destroy()

# ============================================================
# MAIN
# ============================================================

def main():
    # MASSIVE PARAMETERS - 15,000 x 15,000 grid
    WIDTH, HEIGHT = 15000, 15000     # 225 million cells
    NUM_AGENTS = 60000000            # 60 million agents (26.7% occupancy)
    MOVE_PROB = 1.0                  # 30% move chance
    VIEW_SIZE = 200                  # Viewport size (200x200 - zoomed in)
    STEPS = 1000

    print("\n" + "="*60)
    print("MASSIVE GPU ABM - 15,000x15,000 GRID")
    print("="*60)
    print(f"Grid: {WIDTH:,}x{HEIGHT:,} = {WIDTH*HEIGHT:,} cells")
    print(f"Agents: {NUM_AGENTS:,} ({NUM_AGENTS/(WIDTH*HEIGHT):.1%} occupancy)")
    print(f"Move probability: {MOVE_PROB*100:.0f}%")
    print(f"Viewport: {VIEW_SIZE}x{VIEW_SIZE} cells")
    print(f"Canvas: 1200x1200 pixels (4x zoom)")
    print(f"Steps: {STEPS}")
    print("="*60 + "\n")

    # Create simulation
    sim = MassiveGPUABM(WIDTH, HEIGHT, NUM_AGENTS, MOVE_PROB, VIEW_SIZE)
    
    # Create visualizer
    viz = ABMVisualizer(VIEW_SIZE, NUM_AGENTS)

    try:
        print("Running simulation... Close the window to stop\n")
        
        for i in range(STEPS):
            visible = sim.step()
            
            # Draw viewport
            viz.draw(sim.view_x_cpu, sim.view_y_cpu)
            viz.update_title(i+1, visible)
            
            # Print progress every 10 steps
            if (i+1) % 10 == 0:
                print(f"Step {i+1}/{STEPS} completed – {visible:,} agents in viewport")

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        viz.close()
        print("\nDone!")

if __name__ == "__main__":
    main()
