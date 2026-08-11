import numpy as np
from multiprocessing import Process, Lock, shared_memory, cpu_count
import random
import time
import tkinter as tk

class SimpleABM:
    """Super simple ABM with Tkinter visualization - agents start in center block"""
    
    def __init__(self, width, height, num_agents, num_cores=2):
        self.width = width
        self.height = height
        self.num_cores = num_cores
        self.num_agents = num_agents
        
        # Shared memory for grid
        grid_bytes = width * height * 4
        self.shm_grid = shared_memory.SharedMemory(create=True, size=grid_bytes)
        self.grid = np.ndarray((height, width), dtype=np.int32, buffer=self.shm_grid.buf)
        self.grid.fill(-1)
        
        # Shared memory for agents
        agent_bytes = num_agents * 3 * 4
        self.shm_agents = shared_memory.SharedMemory(create=True, size=agent_bytes)
        self.agents = np.ndarray((num_agents, 3), dtype=np.int32, buffer=self.shm_agents.buf)
        self.agents[:, 0] = np.arange(num_agents)
        
        # Place agents in a block at center
        self.place_agents_in_block()
        
        self.lock = Lock()
        
        # Tkinter setup
        self.root = tk.Tk()
        self.root.title(f"ABM - {num_agents} agents on {width}x{height}")
        
        # Canvas
        canvas_size = 600
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg='white')
        self.canvas.pack()
        
        # Scale factor
        self.scale = canvas_size / max(width, height)
        
        # Draw initial state
        self.draw_agents()
    
    def place_agents_in_block(self):
        """Place all agents in a block at the center of the grid"""
        # Calculate block size (square block)
        block_size = int(np.ceil(np.sqrt(self.num_agents)))
        
        # Center of grid
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Start position (top-left of block)
        start_x = center_x - block_size // 2
        start_y = center_y - block_size // 2
        
        # Place agents
        agent_idx = 0
        for dy in range(block_size):
            for dx in range(block_size):
                if agent_idx >= self.num_agents:
                    break
                
                x = start_x + dx
                y = start_y + dy
                
                # Make sure within bounds
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.agents[agent_idx, 1] = x
                    self.agents[agent_idx, 2] = y
                    self.grid[y, x] = agent_idx
                    agent_idx += 1
            
            if agent_idx >= self.num_agents:
                break
        
        print(f"Placed {agent_idx} agents in block at center")
        print(f"Block size: {block_size}x{block_size}")
        print(f"Start position: ({start_x}, {start_y})")
    
    def worker(self, start_idx, end_idx, shm_grid_name, shm_agents_name, 
               width, height, lock):
        """Worker process"""
        shm_grid = shared_memory.SharedMemory(name=shm_grid_name)
        grid = np.ndarray((height, width), dtype=np.int32, buffer=shm_grid.buf)
        
        shm_agents = shared_memory.SharedMemory(name=shm_agents_name)
        agents = np.ndarray((self.num_agents, 3), dtype=np.int32, buffer=shm_agents.buf)
        
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for idx in range(start_idx, end_idx):
            agent = agents[idx]
            aid = int(agent[0])
            x = int(agent[1])
            y = int(agent[2])
            
            empty = []
            for dx, dy in neighbors:
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx] == -1:
                        empty.append((nx, ny))
            
            if not empty:
                continue
            
            nx, ny = random.choice(empty)
            
            with lock:
                if grid[ny, nx] == -1 and grid[y, x] == aid:
                    grid[ny, nx] = aid
                    grid[y, x] = -1
                    agent[1] = nx
                    agent[2] = ny
        
        shm_grid.close()
        shm_agents.close()
    
    def step(self):
        """Run one iteration"""
        chunk_size = self.num_agents // self.num_cores
        chunks = []
        for i in range(self.num_cores):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_cores-1 else self.num_agents
            chunks.append((start, end))
        
        processes = []
        for start, end in chunks:
            p = Process(target=self.worker, 
                       args=(start, end, self.shm_grid.name, self.shm_agents.name,
                             self.width, self.height, self.lock))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        # Update visualization
        self.draw_agents()
        self.root.update()
    
    def draw_agents(self):
        """Draw all agents on canvas"""
        self.canvas.delete("all")
        
        # Draw each agent as a small dot
        for i in range(self.num_agents):
            x = int(self.agents[i, 1])
            y = int(self.agents[i, 2])
            
            # Scale to canvas coordinates
            cx = x * self.scale + self.scale/2
            cy = y * self.scale + self.scale/2
            
            # Draw small circle
            radius = max(2, self.scale * 0.4)
            self.canvas.create_oval(cx - radius, cy - radius, 
                                   cx + radius, cy + radius,
                                   fill='blue', outline='')
    
    def run(self, iterations):
        """Run simulation for N iterations"""
        for i in range(iterations):
            self.step()
            self.root.title(f"ABM - Iteration {i+1}/{iterations} - {self.num_agents} agents")
            self.root.update()
            time.sleep(0.05)  # Slow down so we can see movement
    
    def cleanup(self):
        """Clean up shared memory"""
        self.shm_grid.close()
        self.shm_grid.unlink()
        self.shm_agents.close()
        self.shm_agents.unlink()


# Run it
if __name__ == "__main__":
    # Parameters
    WIDTH, HEIGHT = 100, 100
    NUM_AGENTS = 2000
    NUM_CORES = 2
    ITERATIONS = 200
    
    print(f"\n{'='*50}")
    print(f"ABM with Tkinter Visualization")
    print(f"{'='*50}")
    print(f"Grid: {WIDTH}x{HEIGHT} = {WIDTH*HEIGHT:,} cells")
    print(f"Agents: {NUM_AGENTS} ({NUM_AGENTS/(WIDTH*HEIGHT):.1%} occupancy)")
    print(f"Cores: {NUM_CORES}")
    print(f"Iterations: {ITERATIONS}")
    print("Agents start in a block at the center")
    print("Close the window to stop\n")
    
    # Create and run model
    model = SimpleABM(WIDTH, HEIGHT, NUM_AGENTS, NUM_CORES)
    
    try:
        model.run(ITERATIONS)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        model.cleanup()
        print("Done!")
