import numpy as np
from multiprocessing import Process, Lock, shared_memory, cpu_count
import random
import time
import psutil
import os

class SuperSimpleABM:
    """ABM with both grid and agents in shared memory - with agent tracking"""
    
    def __init__(self, width, height, num_agents, num_cores=None):
        # Auto-detect cores if not specified
        if num_cores is None:
            num_cores = cpu_count()
        
        self.width = width
        self.height = height
        self.num_cores = num_cores
        self.num_agents = num_agents
        
        # Pick a random agent to track
        self.tracked_agent_id = random.randint(0, num_agents - 1)
        self.tracked_positions = []  # Store (x, y) for each iteration
        
        # Track memory before allocation
        self.initial_memory = self.get_memory_usage()
        
        # === SHARED MEMORY: Grid ===
        grid_bytes = width * height * 4  # int32 = 4 bytes
        self.shm_grid = shared_memory.SharedMemory(create=True, size=grid_bytes)
        self.grid = np.ndarray((height, width), dtype=np.int32, buffer=self.shm_grid.buf)
        self.grid.fill(-1)
        
        # === SHARED MEMORY: Agent table ===
        # Agent table: [id, x, y] - all int32
        agent_bytes = num_agents * 3 * 4
        self.shm_agents = shared_memory.SharedMemory(create=True, size=agent_bytes)
        self.agents = np.ndarray((num_agents, 3), dtype=np.int32, buffer=self.shm_agents.buf)
        self.agents[:, 0] = np.arange(num_agents)  # Set IDs
        
        # Place agents randomly
        positions = np.random.choice(width*height, num_agents, replace=False)
        for i, pos in enumerate(positions):
            y = pos // width
            x = pos % width
            self.agents[i, 1] = x
            self.agents[i, 2] = y
            self.grid[y, x] = i
        
        # Record initial position of tracked agent
        tracked_x = int(self.agents[self.tracked_agent_id, 1])
        tracked_y = int(self.agents[self.tracked_agent_id, 2])
        self.tracked_positions.append((tracked_x, tracked_y))
        
        # Global lock
        self.lock = Lock()
        
        # Track final memory
        self.final_memory = self.get_memory_usage()
        
        # Print initialization info
        print(f"\n{'='*60}")
        print(f"INITIALIZATION")
        print(f"{'='*60}")
        print(f"Grid: {width:,}x{height:,} = {width*height:,} cells")
        print(f"Grid memory: {grid_bytes / (1024*1024):.2f} MB")
        print(f"Agents: {num_agents:,} ({num_agents/(width*height):.1%} occupancy)")
        print(f"Agent table memory: {agent_bytes / (1024*1024):.2f} MB")
        print(f"Total memory used: {(self.final_memory - self.initial_memory):.2f} MB")
        print(f"Using {num_cores} CPU cores")
        print(f"Tracking agent ID: {self.tracked_agent_id}")
        print(f"Initial position: ({tracked_x}, {tracked_y})")
        print(f"{'='*60}\n")
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def worker(self, start_idx, end_idx, shm_grid_name, shm_agents_name, 
               width, height, lock):
        """Worker process: moves agents from start_idx to end_idx"""
        # Attach to shared memory
        shm_grid = shared_memory.SharedMemory(name=shm_grid_name)
        grid = np.ndarray((height, width), dtype=np.int32, buffer=shm_grid.buf)
        
        shm_agents = shared_memory.SharedMemory(name=shm_agents_name)
        agents = np.ndarray((self.num_agents, 3), dtype=np.int32, buffer=shm_agents.buf)
        
        # Moore neighborhood
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Work on our chunk of agents
        for idx in range(start_idx, end_idx):
            agent = agents[idx]
            aid = int(agent[0])
            x = int(agent[1])
            y = int(agent[2])
            
            # Find empty neighbors
            empty = []
            for dx, dy in neighbors:
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx] == -1:
                        empty.append((nx, ny))
            
            if not empty:
                continue
            
            nx, ny = random.choice(empty)
            
            # Lock and move
            with lock:
                if grid[ny, nx] == -1 and grid[y, x] == aid:
                    grid[ny, nx] = aid
                    grid[y, x] = -1
                    agent[1] = nx
                    agent[2] = ny
        
        # Clean up
        shm_grid.close()
        shm_agents.close()
    
    def step(self):
        """Run one movement iteration"""
        start_time = time.time()
        
        # Calculate chunks
        chunk_size = self.num_agents // self.num_cores
        chunks = []
        for i in range(self.num_cores):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_cores-1 else self.num_agents
            chunks.append((start, end))
        
        # Start worker processes
        processes = []
        for start, end in chunks:
            p = Process(target=self.worker, 
                       args=(start, end, self.shm_grid.name, self.shm_agents.name,
                             self.width, self.height, self.lock))
            processes.append(p)
            p.start()
        
        # Wait for all workers
        for p in processes:
            p.join()
        
        # Record tracked agent position
        tracked_x = int(self.agents[self.tracked_agent_id, 1])
        tracked_y = int(self.agents[self.tracked_agent_id, 2])
        self.tracked_positions.append((tracked_x, tracked_y))
        
        elapsed = time.time() - start_time
        return elapsed
    
    def get_stats(self):
        """Get current stats including memory"""
        alive = np.sum(self.grid != -1)
        occupancy = alive / (self.width * self.height)
        current_memory = self.get_memory_usage()
        
        return {
            'alive': int(alive),
            'occupancy': occupancy,
            'total_agents': self.num_agents,
            'memory_mb': current_memory,
            'memory_delta': current_memory - self.initial_memory
        }
    
    def verify_positions(self):
        """Verify agent positions match grid"""
        errors = 0
        for i in range(self.num_agents):
            aid = int(self.agents[i, 0])
            x = int(self.agents[i, 1])
            y = int(self.agents[i, 2])
            if self.grid[y, x] != aid:
                errors += 1
        return errors
    
    def get_tracked_agent_path(self):
        """Get the full path of the tracked agent"""
        return self.tracked_positions
    
    def cleanup(self):
        """Clean up shared memory"""
        self.shm_grid.close()
        self.shm_grid.unlink()
        self.shm_agents.close()
        self.shm_agents.unlink()


# Test it
if __name__ == "__main__":
    # Parameters
    WIDTH, HEIGHT = 1000, 1000  # 1,000,000 cells
    NUM_AGENTS = 200000         # 200,000 agents (20% occupancy)
    ITERATIONS = 10
    
    print(f"\n{'='*60}")
    print(f"STARTING LARGE-SCALE ABM SIMULATION")
    print(f"{'='*60}")
    print(f"Grid: {WIDTH:,}x{HEIGHT:,} = {WIDTH*HEIGHT:,} cells")
    print(f"Agents: {NUM_AGENTS:,} ({NUM_AGENTS/(WIDTH*HEIGHT):.1%} occupancy)")
    print(f"Iterations: {ITERATIONS}")
    print(f"{'='*60}\n")
    
    # Create model with auto-detected cores
    model = SuperSimpleABM(WIDTH, HEIGHT, NUM_AGENTS, num_cores=None)
    
    print(f"{'='*60}")
    print(f"RUNNING SIMULATION")
    print(f"{'='*60}\n")
    
    # Run iterations
    for i in range(ITERATIONS):
        elapsed = model.step()
        stats = model.get_stats()
        errors = model.verify_positions()
        
        # Get tracked agent position
        tracked_pos = model.agents[model.tracked_agent_id]
        tracked_x, tracked_y = int(tracked_pos[1]), int(tracked_pos[2])
        
        print(f"Iter {i:2d}: {elapsed:.4f}s | "
              f"alive: {stats['alive']:,} | "
              f"occupancy: {stats['occupancy']:.1%} | "
              f"memory: {stats['memory_mb']:.1f} MB | "
              f"errors: {errors} | "
              f"tracked agent: ({tracked_x:4d}, {tracked_y:4d})")
    
    print(f"\n{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Final stats:")
    stats = model.get_stats()
    print(f"  Alive agents: {stats['alive']:,}")
    print(f"  Occupancy: {stats['occupancy']:.1%}")
    print(f"  Memory usage: {stats['memory_mb']:.1f} MB")
    print(f"  Memory delta: +{stats['memory_delta']:.1f} MB")
    
    # Show tracked agent path
    print(f"\nTracked Agent ID: {model.tracked_agent_id}")
    print(f"Movement path (start -> end):")
    path = model.get_tracked_agent_path()
    for i, (x, y) in enumerate(path):
        print(f"  Iter {i:2d}: ({x:4d}, {y:4d})")
    
    # Calculate total distance moved
    if len(path) > 1:
        total_distance = 0
        for i in range(1, len(path)):
            dx = abs(path[i][0] - path[i-1][0])
            dy = abs(path[i][1] - path[i-1][1])
            total_distance += dx + dy
        print(f"\nTotal Manhattan distance moved: {total_distance} cells")
    
    # Clean up
    model.cleanup()
    print(f"{'='*60}\n")
