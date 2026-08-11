import numpy as np
from numba import cuda
import time
import warnings
import os
import sys

# ===== AGGRESSIVE WARNING SUPPRESSION =====
# Suppress all Numba warnings
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['NUMBA_OPTIMIZATION'] = '1'
os.environ['NUMBA_DEBUG'] = '0'

# Filter all UserWarning from numba
warnings.filterwarnings('ignore', module='numba')
warnings.filterwarnings('ignore', category=UserWarning, module='numba.cuda.dispatcher')
warnings.filterwarnings('ignore', message='.*Grid size.*')
warnings.filterwarnings('ignore', message='.*occupancy.*')

# Also suppress through the warnings system
if not sys.warnoptions:
    import subprocess
    # This is a hack but it works

# Redirect stderr temporarily to suppress compilation warnings
import contextlib
import io

# Numba's internal logging
import logging
logging.getLogger('numba').setLevel(logging.ERROR)

# Disable numba's performance warnings specifically
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', NumbaPerformanceWarning)
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Monkey patch the dispatcher to not show warnings
from numba.cuda.dispatcher import Dispatcher
_original_call = Dispatcher.__call__
def _patched_call(self, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        return _original_call(self, *args, **kwargs)
# Apply patch
Dispatcher.__call__ = _patched_call
# ============================================

class ABMSimulationGPU:
    """GPU-accelerated ABM using Numba CUDA - ZERO AGENT LOSS with Debug Tracking"""
    
    def __init__(self, width, height, num_agents, move_probability=0.05, 
                 neighborhood_size=5, debug=False):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.move_probability = move_probability
        self.neighborhood_size = neighborhood_size
        self.debug = debug
        
        # Track a random agent if debug is enabled
        if debug:
            self.tracked_agent_id = np.random.randint(0, num_agents)
            self.tracked_positions = []
            self.tracked_moves = 0
            self.tracked_iterations = []
        
        if not cuda.is_available():
            raise RuntimeError("CUDA not available!")
        
        print(f"\n{'='*60}")
        print(f"GPU-ACCELERATED ABM - ZERO AGENT LOSS")
        print(f"{'='*60}")
        print(f"GPU: {cuda.get_current_device().name}")
        print(f"Grid: {width:,}x{height:,} = {width*height:,} cells")
        print(f"Agents: {num_agents:,} ({num_agents/(width*height):.1%} occupancy)")
        print(f"Move probability: {move_probability*100:.0f}%")
        print(f"Neighborhood: {neighborhood_size}x{neighborhood_size}")
        if debug:
            print(f"DEBUG: Tracking agent ID: {self.tracked_agent_id}")
        print(f"{'='*60}\n")
        
        # Allocate on GPU
        print("Allocating GPU memory...")
        self.grid_flat = cuda.device_array(width * height, dtype=np.int32)
        self._fill_grid()
        
        self.agents = cuda.device_array((num_agents, 3), dtype=np.int32)
        
        print("Placing agents randomly...")
        self.place_agents_random()
        print("Agents placed!\n")
        
        # Compile kernel
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.move_kernel = self.create_move_kernel()
        print("CUDA kernel compiled!")
        print(f"{'='*60}\n")
        
        self.iterations = 0
        self.total_moves = 0
        print("Simulation ready on GPU!")
        print(f"{'='*60}\n")
    
    def _fill_grid(self):
        """Fill grid with -1"""
        @cuda.jit
        def fill_kernel(grid_flat, size):
            idx = cuda.grid(1)
            if idx < size:
                grid_flat[idx] = -1
        
        threads = 256
        blocks = (self.width * self.height + threads - 1) // threads
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            fill_kernel[blocks, threads](self.grid_flat, self.width * self.height)
        cuda.synchronize()
    
    def place_agents_random(self):
        """Place agents randomly"""
        positions = np.random.choice(self.width * self.height, 
                                     self.num_agents, replace=False)
        
        agents_cpu = np.zeros((self.num_agents, 3), dtype=np.int32)
        agents_cpu[:, 0] = np.arange(self.num_agents)
        agents_cpu[:, 1] = positions % self.width
        agents_cpu[:, 2] = positions // self.width
        
        self.agents = cuda.to_device(agents_cpu)
        
        @cuda.jit
        def place_kernel(agents, grid_flat, num_agents, width):
            idx = cuda.grid(1)
            if idx < num_agents:
                aid = agents[idx, 0]
                x = agents[idx, 1]
                y = agents[idx, 2]
                flat_idx = y * width + x
                grid_flat[flat_idx] = aid
        
        threads = 256
        blocks = (self.num_agents + threads - 1) // threads
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            place_kernel[blocks, threads](self.agents, self.grid_flat, self.num_agents, self.width)
        cuda.synchronize()
        
        # Record initial tracked position if debug is enabled
        if self.debug:
            agents_cpu = self.agents.copy_to_host()
            x = agents_cpu[self.tracked_agent_id, 1]
            y = agents_cpu[self.tracked_agent_id, 2]
            self.tracked_positions.append((int(x), int(y)))
            self.tracked_iterations.append(0)
    
    def create_move_kernel(self):
        """Create movement kernel with atomic operations and move counter"""
        
        @cuda.jit
        def move_kernel(agents, grid_flat, width, height, move_probability, 
                        neighborhood_size, iteration, move_counter):
            idx = cuda.grid(1)
            
            if idx >= agents.shape[0]:
                return
            
            # Get agent data
            aid = agents[idx, 0]
            x = agents[idx, 1]
            y = agents[idx, 2]
            
            # Generate random number
            seed = (idx * 1103515245 + 12345 + iteration * 1000000) & 0x7fffffff
            seed = (seed * 1103515245 + 12345) & 0x7fffffff
            rand_val = (seed & 0x7fffffff) / 2147483647.0
            
            # Check if agent moves this turn
            if rand_val > move_probability:
                return
            
            offset = neighborhood_size // 2
            max_attempts = neighborhood_size * neighborhood_size * 2
            
            # Current flat index
            current_flat = y * width + x
            
            for attempt in range(max_attempts):
                # Generate random neighbor
                seed = (seed * 1103515245 + 12345) & 0x7fffffff
                dx = (seed % neighborhood_size) - offset
                seed = (seed * 1103515245 + 12345) & 0x7fffffff
                dy = (seed % neighborhood_size) - offset
                
                if dx == 0 and dy == 0:
                    continue
                
                nx = x + dx
                ny = y + dy
                
                if 0 <= nx < width and 0 <= ny < height:
                    target_flat = ny * width + nx
                    
                    # Use atomic compare-and-swap
                    old_val = cuda.atomic.cas(grid_flat, target_flat, -1, aid)
                    
                    # If we successfully claimed the target cell
                    if old_val == -1:
                        # Free the old cell
                        if grid_flat[current_flat] == aid:
                            grid_flat[current_flat] = -1
                            # Update agent position
                            agents[idx, 1] = nx
                            agents[idx, 2] = ny
                            # Increment move counter atomically
                            cuda.atomic.add(move_counter, 0, 1)
                            return
        
        return move_kernel
    
    def step(self):
        """Run one iteration"""
        start_time = time.time()
        self.iterations += 1
        
        # Get old tracked position if debug is enabled
        if self.debug:
            agents_cpu = self.agents.copy_to_host()
            old_x = int(agents_cpu[self.tracked_agent_id, 1])
            old_y = int(agents_cpu[self.tracked_agent_id, 2])
        
        # Create move counter on GPU
        move_counter = cuda.device_array(1, dtype=np.int64)
        move_counter[0] = 0
        
        # Launch kernel
        threads_per_block = 256
        blocks = max(1, (self.num_agents + threads_per_block - 1) // threads_per_block)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.move_kernel[blocks, threads_per_block](
                self.agents, self.grid_flat, self.width, self.height, 
                self.move_probability, self.neighborhood_size, self.iterations,
                move_counter)
        
        cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        # Get move count from GPU
        moves = int(move_counter[0])
        self.total_moves += moves
        
        # Update tracked position if debug is enabled
        if self.debug:
            agents_cpu = self.agents.copy_to_host()
            new_x = int(agents_cpu[self.tracked_agent_id, 1])
            new_y = int(agents_cpu[self.tracked_agent_id, 2])
            
            if new_x != old_x or new_y != old_y:
                self.tracked_moves += 1
            
            self.tracked_positions.append((new_x, new_y))
            self.tracked_iterations.append(self.iterations)
        
        # Count alive agents
        grid_cpu = self.grid_flat.copy_to_host()
        alive = int(np.sum(grid_cpu != -1))
        
        return elapsed, alive, moves
    
    def get_stats(self):
        """Get current statistics"""
        grid_cpu = self.grid_flat.copy_to_host()
        alive = int(np.sum(grid_cpu != -1))
        stats = {
            'alive': alive,
            'occupancy': alive / (self.width * self.height),
            'total_agents': self.num_agents,
            'iterations': self.iterations,
            'total_moves': self.total_moves
        }
        
        # Add debug info if enabled
        if self.debug:
            stats['tracked_agent_id'] = self.tracked_agent_id
            stats['tracked_moves'] = self.tracked_moves
            stats['tracked_positions'] = self.tracked_positions
            stats['tracked_iterations'] = self.tracked_iterations
        
        return stats
    
    def print_tracked_path(self):
        """Print the movement path of the tracked agent"""
        if not self.debug:
            print("Debug mode not enabled. Set debug=True to track agents.")
            return
        
        print(f"\n{'='*60}")
        print(f"TRACKED AGENT {self.tracked_agent_id}")
        print(f"{'='*60}")
        print(f"Total moves: {self.tracked_moves}")
        print(f"Path length: {len(self.tracked_positions)}")
        print("\nPath (Iteration: (x, y)):")
        
        # Show all positions (or last 20 if too many)
        if len(self.tracked_positions) <= 20:
            for i, (x, y) in enumerate(self.tracked_positions):
                iter_num = self.tracked_iterations[i]
                print(f"  Iter {iter_num:4d}: ({x:6d}, {y:6d})")
        else:
            # Show first 10 and last 10
            for i in range(10):
                x, y = self.tracked_positions[i]
                iter_num = self.tracked_iterations[i]
                print(f"  Iter {iter_num:4d}: ({x:6d}, {y:6d})")
            print("  ...")
            for i in range(-10, 0):
                x, y = self.tracked_positions[i]
                iter_num = self.tracked_iterations[i]
                print(f"  Iter {iter_num:4d}: ({x:6d}, {y:6d})")
        
        if len(self.tracked_positions) > 1:
            total_dist = 0
            for i in range(1, len(self.tracked_positions)):
                dx = abs(self.tracked_positions[i][0] - self.tracked_positions[i-1][0])
                dy = abs(self.tracked_positions[i][1] - self.tracked_positions[i-1][1])
                total_dist += dx + dy
            print(f"\nTotal Manhattan distance: {total_dist:,}")
            print(f"Average distance per move: {total_dist / max(1, self.tracked_moves):.1f}")
        print(f"{'='*60}\n")
    
    def cleanup(self):
        """Clean up GPU memory"""
        del self.grid_flat
        del self.agents
        print("GPU memory cleaned up!")


def main():
    # Parameters
    WIDTH, HEIGHT = 10000, 10000  # 100 million cells
    NUM_AGENTS = 60000000         # 60 million agents (60% occupancy)
    MOVE_PROBABILITY = 0.5       # 5% chance to move per iteration
    NEIGHBORHOOD_SIZE = 99        # 99x99 neighborhood
    ITERATIONS = 100
    DEBUG = True                  # Set to True to track an agent
    
    print(f"\n{'='*60}")
    print(f"GPU-ACCELERATED ABM - ZERO AGENT LOSS")
    print(f"{'='*60}")
    print(f"Agents: {NUM_AGENTS:,} on {WIDTH:,}x{HEIGHT:,} grid")
    print(f"Move probability: {MOVE_PROBABILITY*100:.0f}%")
    print(f"Neighborhood: {NEIGHBORHOOD_SIZE}x{NEIGHBORHOOD_SIZE}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Debug mode: {DEBUG}")
    print(f"{'='*60}\n")
    
    # Create simulation
    sim = ABMSimulationGPU(WIDTH, HEIGHT, NUM_AGENTS, MOVE_PROBABILITY, 
                          NEIGHBORHOOD_SIZE, debug=DEBUG)
    
    try:
        print(f"{'Iter':<6} {'Time(s)':<10} {'Moves':<15} {'Alive':<12} {'Occ%':<10}")
        print("-" * 60)
        
        total_time = 0
        for i in range(ITERATIONS):
            elapsed, alive, moves = sim.step()
            total_time += elapsed
            stats = sim.get_stats()
            
            # Show tracked position if debug is enabled
            if DEBUG:
                pos = stats['tracked_positions'][-1]
                print(f"{i+1:<6} {elapsed:<10.3f} {moves:<15,} {alive:<12,} {stats['occupancy']*100:<10.1f}% ({pos[0]},{pos[1]})")
            else:
                print(f"{i+1:<6} {elapsed:<10.3f} {moves:<15,} {alive:<12,} {stats['occupancy']*100:<10.1f}%")
        
        print("-" * 60)
        print(f"\n{'='*60}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per iteration: {total_time/ITERATIONS:.3f}s")
        print(f"Initial agents: {NUM_AGENTS:,}")
        print(f"Final alive: {stats['alive']:,}")
        print(f"Agents lost: {NUM_AGENTS - stats['alive']:,}")
        print(f"Total moves: {stats['total_moves']:,}")
        print(f"Average moves per iteration: {stats['total_moves']/ITERATIONS:,.0f}")
        print(f"Final occupancy: {stats['occupancy']*100:.1f}%")
        
        # Print tracked agent path if debug is enabled
        if DEBUG:
            sim.print_tracked_path()
        
        print(f"{'='*60}\n")
    
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
    
    finally:
        sim.cleanup()
        print("Done!")

if __name__ == "__main__":
    main()
