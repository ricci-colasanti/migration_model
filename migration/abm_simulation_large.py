import numpy as np
from multiprocessing import Process, Lock, shared_memory, cpu_count
import random
import time

class ABMSimulationLarge:
    """Large-scale ABM with integer gradient field (0-1000)"""
    
    def __init__(self, width, height, num_agents, move_probability=1.0, 
                 neighborhood_size=5, num_cores=None):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.move_probability = move_probability
        
        # Auto-detect cores if not specified
        if num_cores is None:
            num_cores = cpu_count()
        self.num_cores = num_cores
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING LARGE-SCALE ABM")
        print(f"{'='*60}")
        print(f"Grid: {width:,}x{height:,} = {width*height:,} cells")
        print(f"Agents: {num_agents:,} ({num_agents/(width*height):.1%} occupancy)")
        print(f"Move probability: {move_probability*100:.0f}%")
        print(f"Neighborhood: {neighborhood_size}x{neighborhood_size}")
        print(f"Cores: {num_cores}")
        print(f"{'='*60}\n")
        
        # Calculate memory requirements
        grid_mb = (width * height * 4 * 2) / (1024 * 1024)  # 2 values per cell
        agent_mb = (num_agents * 4 * 4) / (1024 * 1024)     # 4 values per agent
        total_mb = grid_mb + agent_mb
        
        print(f"Memory requirements:")
        print(f"  Grid (agent_id + gradient): {grid_mb:.2f} MB")
        print(f"  Agents (id + x + y + value): {agent_mb:.2f} MB")
        print(f"  Total: {total_mb:.2f} MB")
        print(f"{'='*60}\n")
        
        # === SHARED MEMORY: Grid with 2 values per cell ===
        grid_bytes = width * height * 4 * 2
        self.shm_grid = shared_memory.SharedMemory(create=True, size=grid_bytes)
        self.grid = np.ndarray((height, width, 2), dtype=np.int32, buffer=self.shm_grid.buf)
        self.grid[:, :, 0] = -1  # Initialize all cells to empty
        self.grid[:, :, 1] = 0   # Initialize gradient values
        
        # === SHARED MEMORY: Agents with 4 values ===
        agent_bytes = num_agents * 4 * 4
        self.shm_agents = shared_memory.SharedMemory(create=True, size=agent_bytes)
        self.agents = np.ndarray((num_agents, 4), dtype=np.int32, buffer=self.shm_agents.buf)
        self.agents[:, 0] = np.arange(num_agents)  # Agent IDs
        
        # Place agents randomly with random values
        print("Placing agents randomly...")
        self.place_agents_random()
        print("Agents placed!\n")
        
        # Generate gradient field (1000 at top, 0 at bottom)
        print("Generating gradient field...")
        self.generate_gradient()
        print("Gradient generated!\n")
        
        # Generate neighbors
        self.neighbors = self.generate_neighbors(neighborhood_size)
        print(f"Generated {len(self.neighbors)} neighbor offsets")
        print(f"{'='*60}\n")
        
        # Global lock
        self.lock = Lock()
        
        print("Simulation ready!")
        print(f"{'='*60}\n")
    
    def generate_gradient(self):
        """Generate gradient field: 1000 at top, 0 at bottom"""
        for y in range(self.height):
            # Top = 1000, Bottom = 0
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
        # Generate random positions without replacement
        positions = np.random.choice(self.width * self.height, self.num_agents, replace=False)
        
        # Place agents
        for i, pos in enumerate(positions):
            y = pos // self.width
            x = pos % self.width
            
            # Set agent position
            self.agents[i, 1] = x
            self.agents[i, 2] = y
            
            # Set random integer value (0 to 1000)
            self.agents[i, 3] = random.randint(0, 1000)
            
            # Place in grid
            self.grid[y, x, 0] = i
    
    def worker(self, start_idx, end_idx, shm_grid_name, shm_agents_name, 
               width, height, lock, neighbors, move_probability):
        """Worker process: moves agents to better matching cells"""
        # Attach to shared memory
        shm_grid = shared_memory.SharedMemory(name=shm_grid_name)
        grid = np.ndarray((height, width, 2), dtype=np.int32, buffer=shm_grid.buf)
        
        shm_agents = shared_memory.SharedMemory(name=shm_agents_name)
        agents = np.ndarray((self.num_agents, 4), dtype=np.int32, buffer=shm_agents.buf)
        
        # Create list of indices and shuffle them
        indices = list(range(start_idx, end_idx))
        random.shuffle(indices)
        
        # Track moves
        moves = 0
        
        # Process agents in shuffled order
        for idx in indices:
            # Check if agent moves this turn
            if random.random() > move_probability:
                continue
            
            agent = agents[idx]
            aid = int(agent[0])
            x = int(agent[1])
            y = int(agent[2])
            agent_value = agent[3]  # Integer 0-1000
            
            # Get current cell gradient
            current_gradient = grid[y, x, 1]
            current_diff = abs(agent_value - current_gradient)
            
            # Shuffle neighbors for THIS agent to avoid directional bias
            shuffled_neighbors = neighbors.copy()
            random.shuffle(shuffled_neighbors)
            
            # Find all empty neighboring cells that are BETTER than current
            better_cells = []
            for dx, dy in shuffled_neighbors:
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny, nx, 0] == -1:  # Empty cell
                        cell_gradient = grid[ny, nx, 1]
                        diff = abs(agent_value - cell_gradient)
                        
                        # ONLY consider cells that are BETTER
                        if diff < current_diff:
                            better_cells.append((nx, ny, diff))
            
            # If no better cells found, agent stays put
            if not better_cells:
                continue
            
            # Sort by difference (closest match first)
            better_cells.sort(key=lambda cell: cell[2])
            
            # Pick the best matching cell
            nx, ny, _ = better_cells[0]
            
            # Lock and move with double-check
            with lock:
                if grid[ny, nx, 0] == -1 and grid[y, x, 0] == aid:
                    grid[ny, nx, 0] = aid
                    grid[y, x, 0] = -1
                    agent[1] = nx
                    agent[2] = ny
                    moves += 1
        
        shm_grid.close()
        shm_agents.close()
        return moves
    
    def step(self):
        """Run one iteration of the simulation"""
        # Split agents into chunks
        chunk_size = self.num_agents // self.num_cores
        chunks = []
        for i in range(self.num_cores):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_cores-1 else self.num_agents
            chunks.append((start, end))
        
        # Start workers
        processes = []
        for start, end in chunks:
            p = Process(target=self.worker, 
                       args=(start, end, self.shm_grid.name, self.shm_agents.name,
                             self.width, self.height, self.lock, self.neighbors,
                             self.move_probability))
            processes.append(p)
            p.start()
        
        # Wait for workers
        for p in processes:
            p.join()
        
        # Count moves (alive agents)
        total_moves = np.sum(self.grid[:, :, 0] != -1)
        
        return total_moves
    
    def get_stats(self):
        """Get current statistics"""
        alive = np.sum(self.grid[:, :, 0] != -1)
        occupancy = alive / (self.width * self.height)
        return {
            'alive': int(alive),
            'occupancy': occupancy,
            'total_agents': self.num_agents
        }
    
    def cleanup(self):
        """Clean up shared memory"""
        self.shm_grid.close()
        self.shm_grid.unlink()
        self.shm_agents.close()
        self.shm_agents.unlink()


def main():
    # Parameters
    WIDTH, HEIGHT = 10000, 10000  # 100 million cells
    NUM_AGENTS = 60000000         # 60 million agents (60% occupancy)
    MOVE_PROBABILITY = 0.05       # 5% chance to move per iteration
    NEIGHBORHOOD_SIZE = 5         # 5x5 neighborhood
    ITERATIONS = 10
    NUM_CORES = None              # Auto-detect all cores
    
    print(f"\n{'='*60}")
    print(f"LARGE-SCALE ABM SIMULATION")
    print(f"{'='*60}")
    print(f"Starting with {NUM_AGENTS:,} agents on {WIDTH:,}x{HEIGHT:,} grid")
    print(f"Move probability: {MOVE_PROBABILITY*100:.0f}%")
    print(f"Neighborhood: {NEIGHBORHOOD_SIZE}x{NEIGHBORHOOD_SIZE}")
    print(f"Iterations: {ITERATIONS}")
    print(f"{'='*60}\n")
    
    # Create simulation
    sim = ABMSimulationLarge(WIDTH, HEIGHT, NUM_AGENTS, MOVE_PROBABILITY, 
                            NEIGHBORHOOD_SIZE, NUM_CORES)
    
    try:
        # Run iterations
        total_time = 0
        print("Running simulation...\n")
        print(f"{'Iteration':<12} {'Time (s)':<12} {'Occupancy':<12}")
        print("-" * 60)
        
        for i in range(ITERATIONS):
            start_time = time.time()
            
            # Step the simulation
            moves = sim.step()
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Get stats
            stats = sim.get_stats()
            
            print(f"{i+1:<12} {elapsed:<12.3f} {stats['occupancy']:<12.2%}")
        
        print("-" * 60)
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Neighborhood: {NEIGHBORHOOD_SIZE}x{NEIGHBORHOOD_SIZE}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per iteration: {total_time/ITERATIONS:.3f}s")
        print(f"Final occupancy: {stats['occupancy']:.2%}")
        print(f"Final agents: {stats['alive']:,}")
        
        # Calculate performance
        neighbor_count = len(sim.neighbors)
        print(f"\nPerformance estimate:")
        print(f"  Agents: {NUM_AGENTS:,} per iteration")
        print(f"  Neighbors per agent: {neighbor_count}")
        print(f"  Estimated moves per second: {(NUM_AGENTS * MOVE_PROBABILITY) / (total_time/ITERATIONS):,.0f}")
        print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
    
    finally:
        # Clean up
        sim.cleanup()
        print("Done!")

if __name__ == "__main__":
    main()
