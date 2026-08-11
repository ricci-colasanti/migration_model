import numpy as np
from multiprocessing import Process, Lock, shared_memory
import random

class ABMSimulation:
    """Agent-Based Model Simulation with integer gradient field (0-1000)"""
    
    def __init__(self, width, height, num_agents, num_cores=2, 
                 exploration_rate=0.10, neighborhood_size=5):
        self.width = width
        self.height = height
        self.num_cores = num_cores
        self.num_agents = num_agents
        self.exploration_rate = exploration_rate
        self.neighborhood_size = neighborhood_size
        
        # === GRID: 2 values per cell ===
        # First value: agent ID (-1 = empty)
        # Second value: gradient value (integer 0-1000)
        grid_bytes = width * height * 4 * 2  # 2 int32 values per cell
        self.shm_grid = shared_memory.SharedMemory(create=True, size=grid_bytes)
        
        # Create view with 2 columns: [agent_id, gradient_value]
        self.grid = np.ndarray((height, width, 2), dtype=np.int32, buffer=self.shm_grid.buf)
        self.grid[:, :, 0] = -1  # Initialize all cells to empty
        self.grid[:, :, 1] = 0   # Initialize gradient values
        
        # === AGENTS: 4 values per agent ===
        # [id, x, y, value] - value is integer 0-1000
        agent_bytes = num_agents * 4 * 4  # 4 int32 values per agent
        self.shm_agents = shared_memory.SharedMemory(create=True, size=agent_bytes)
        self.agents = np.ndarray((num_agents, 4), dtype=np.int32, buffer=self.shm_agents.buf)
        self.agents[:, 0] = np.arange(num_agents)  # Agent IDs
        
        # Place agents randomly with random integer values
        self.place_agents_random()
        
        # Generate gradient field (1000 at top, 0 at bottom)
        self.generate_gradient()
        
        # Generate neighbors with configurable size
        self.neighbors = self.generate_neighbors(neighborhood_size)
        
        # Global lock
        self.lock = Lock()
        
        print(f"Simulation initialized: {num_agents} agents on {width}x{height} grid")
        print(f"Gradient: 1000 at top, 0 at bottom")
        print(f"Agent values: random integers between 0 and 1000")
        print(f"Agents placed randomly across the grid")
        print(f"Neighborhood: {neighborhood_size}x{neighborhood_size} ({len(self.neighbors)} neighbors per agent)")
        print(f"Exploration rate: {exploration_rate*100:.0f}% (random search center)")
        print(f"Movement: only if better match found")
    
    def generate_gradient(self):
        """Generate gradient field: 1000 at top, 0 at bottom"""
        for y in range(self.height):
            gradient = int(((self.height - 1 - y) / (self.height - 1)) * 1000) if self.height > 1 else 1000
            for x in range(self.width):
                self.grid[y, x, 1] = gradient
    
    def generate_neighbors(self, size):
        """Generate neighbor offsets for given neighborhood size"""
        neighbors = []
        offset = size // 2
        
        for dy in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((dx, dy))
        
        return neighbors
    
    def place_agents_random(self):
        """Place all agents randomly across the grid with random integer values"""
        positions = np.random.choice(self.width * self.height, self.num_agents, replace=False)
        
        for i, pos in enumerate(positions):
            y = pos // self.width
            x = pos % self.width
            
            self.agents[i, 1] = x
            self.agents[i, 2] = y
            self.agents[i, 3] = random.randint(0, 1000)
            self.grid[y, x, 0] = i
    
    def worker(self, start_idx, end_idx, shm_grid_name, shm_agents_name, 
               width, height, lock, neighbors, exploration_rate):
        """Worker process: moves agents to cells matching their value"""
        # Attach to shared memory
        shm_grid = shared_memory.SharedMemory(name=shm_grid_name)
        grid = np.ndarray((height, width, 2), dtype=np.int32, buffer=shm_grid.buf)
        
        shm_agents = shared_memory.SharedMemory(name=shm_agents_name)
        agents = np.ndarray((self.num_agents, 4), dtype=np.int32, buffer=shm_agents.buf)
        
        # Create list of indices and shuffle them
        indices = list(range(start_idx, end_idx))
        random.shuffle(indices)
        
        # Process agents in shuffled order
        for idx in indices:
            agent = agents[idx]
            aid = int(agent[0])
            x = int(agent[1])
            y = int(agent[2])
            agent_value = agent[3]
            
            # === EXPLORATION ===
            if random.random() < exploration_rate:
                rand_pos = random.randint(0, width * height - 1)
                search_x = rand_pos % width
                search_y = rand_pos // width
            else:
                search_x = x
                search_y = y
            
            # Get current cell gradient
            current_gradient = grid[y, x, 1]
            current_diff = abs(agent_value - current_gradient)
            
            # Shuffle neighbors for this agent
            shuffled_neighbors = neighbors.copy()
            random.shuffle(shuffled_neighbors)
            
            # Find better empty cells
            better_cells = []
            for dx, dy in shuffled_neighbors:
                nx, ny = search_x + dx, search_y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx, 0] == -1:
                        cell_gradient = grid[ny, nx, 1]
                        diff = abs(agent_value - cell_gradient)
                        if diff < current_diff:
                            better_cells.append((nx, ny, diff))
            
            if not better_cells:
                continue
            
            better_cells.sort(key=lambda cell: cell[2])
            nx, ny, _ = better_cells[0]
            
            # Lock and move
            with lock:
                if grid[ny, nx, 0] == -1 and grid[y, x, 0] == aid:
                    grid[ny, nx, 0] = aid
                    grid[y, x, 0] = -1
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
                             self.width, self.height, self.lock, self.neighbors,
                             self.exploration_rate))
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
    
    def get_agent_values(self):
        """Return all agent values as list of integers (0-1000)"""
        values = []
        for i in range(self.num_agents):
            values.append(int(self.agents[i, 3]))
        return values
    
    def get_gradient_value(self, x, y):
        """Get gradient value at position (0-1000)"""
        return int(self.grid[y, x, 1])
    
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
