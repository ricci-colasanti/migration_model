import numpy as np
from multiprocessing import Process, Lock, shared_memory
import random

class ABMSimulation:
    """Agent-Based Model Simulation - No visualization, pure simulation logic"""
    
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
        
        # Place agents in center block
        self.place_agents_in_block()
        
        # Global lock
        self.lock = Lock()
        
        print(f"Simulation initialized: {num_agents} agents on {width}x{height} grid")
    
    def place_agents_in_block(self):
        """Place all agents in a block at the center of the grid"""
        block_size = int(np.ceil(np.sqrt(self.num_agents)))
        center_x = self.width // 2
        center_y = self.height // 2
        start_x = center_x - block_size // 2
        start_y = center_y - block_size // 2
        
        agent_idx = 0
        for dy in range(block_size):
            for dx in range(block_size):
                if agent_idx >= self.num_agents:
                    break
                x = start_x + dx
                y = start_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.agents[agent_idx, 1] = x
                    self.agents[agent_idx, 2] = y
                    self.grid[y, x] = agent_idx
                    agent_idx += 1
            if agent_idx >= self.num_agents:
                break
    
    def worker(self, start_idx, end_idx, shm_grid_name, shm_agents_name, 
               width, height, lock):
        """Worker process: moves agents in chunk"""
        shm_grid = shared_memory.SharedMemory(name=shm_grid_name)
        grid = np.ndarray((height, width), dtype=np.int32, buffer=shm_grid.buf)
        
        shm_agents = shared_memory.SharedMemory(name=shm_agents_name)
        agents = np.ndarray((self.num_agents, 3), dtype=np.int32, buffer=shm_agents.buf)
        
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Create list of indices and shuffle them
        indices = list(range(start_idx, end_idx))
        random.shuffle(indices)  # ← Randomize order
        
        # Process agents in shuffled order
        for idx in indices:
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
        """Run one iteration of the simulation"""
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
    
    def get_agent_positions(self):
        """Return all agent positions as list of (x, y) tuples"""
        positions = []
        for i in range(self.num_agents):
            positions.append((int(self.agents[i, 1]), int(self.agents[i, 2])))
        return positions
    
    def get_agent_count(self):
        """Return number of agents"""
        return self.num_agents
    
    def get_grid_size(self):
        """Return grid dimensions"""
        return self.width, self.height
    
    def cleanup(self):
        """Clean up shared memory"""
        self.shm_grid.close()
        self.shm_grid.unlink()
        self.shm_agents.close()
        self.shm_agents.unlink()
