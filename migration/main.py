import time
import sys
import argparse
from multiprocessing import cpu_count
from abm_simulation import ABMSimulation
from abm_visualizer import ABMVisualizer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Agent-Based Model Simulation with Gradient Field')
    parser.add_argument('--exploration', '-e', 
                       type=float, 
                       default=0.00,
                       help='Exploration rate (0.0 to 1.0). Default: 0.00 (10%%)')
    parser.add_argument('--neighborhood', '-n',
                       type=int,
                       default=5,
                       help='Neighborhood size (odd number: 3, 5, 7, 9, etc.). Default: 5')
    parser.add_argument('--width', '-w',
                       type=int,
                       default=200,
                       help='Grid width. Default: 200')
    parser.add_argument('--height', '-H',
                       type=int,
                       default=200,
                       help='Grid height. Default: 200')
    parser.add_argument('--agents', '-a',
                       type=int,
                       default=10000,
                       help='Number of agents. Default: 10000')
    parser.add_argument('--iterations', '-i',
                       type=int,
                       default=100,
                       help='Number of iterations. Default: 100')
    parser.add_argument('--delay', '-d',
                       type=float,
                       default=0.005,
                       help='Delay between iterations in seconds. Default: 0.005')
    parser.add_argument('--cores', '-c',
                       type=int,
                       default=None,
                       help='Number of CPU cores to use. Default: all available')
    
    args = parser.parse_args()
    
    # Validate exploration rate
    if args.exploration < 0.0 or args.exploration > 1.0:
        print("Error: Exploration rate must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Validate neighborhood size (must be odd)
    if args.neighborhood < 3 or args.neighborhood % 2 == 0:
        print("Error: Neighborhood size must be an odd number >= 3 (3, 5, 7, 9, etc.)")
        sys.exit(1)
    
    # Parameters
    WIDTH = args.width
    HEIGHT = args.height
    NUM_AGENTS = args.agents
    NUM_CORES = args.cores if args.cores is not None else cpu_count()
    ITERATIONS = args.iterations
    DELAY = args.delay
    EXPLORATION_RATE = args.exploration
    NEIGHBORHOOD_SIZE = args.neighborhood
    
    # Calculate neighbors count
    neighbors_count = NEIGHBORHOOD_SIZE * NEIGHBORHOOD_SIZE - 1
    
    print(f"\n{'='*50}")
    print(f"ABM with Gradient Field and Exploration")
    print(f"{'='*50}")
    print(f"Grid: {WIDTH}x{HEIGHT} = {WIDTH*HEIGHT:,} cells")
    print(f"Agents: {NUM_AGENTS} ({NUM_AGENTS/(WIDTH*HEIGHT):.1%} occupancy)")
    print(f"Cores: {NUM_CORES} (all available)")
    print(f"Iterations: {ITERATIONS}")
    print(f"Delay: {DELAY}s between iterations")
    print(f"Gradient: 1000 at top, 0 at bottom")
    print(f"Agent values: random integers between 0 and 1000")
    print(f"Neighborhood: {NEIGHBORHOOD_SIZE}x{NEIGHBORHOOD_SIZE} ({neighbors_count} neighbors per agent)")
    print(f"Exploration rate: {EXPLORATION_RATE*100:.0f}% (random search center)")
    print("Agents move to empty cells matching their value (only if better)")
    print("Agents placed randomly across the grid")
    print("Close the window to stop\n")
    
    # Create simulation with configurable neighborhood
    sim = ABMSimulation(WIDTH, HEIGHT, NUM_AGENTS, NUM_CORES, 
                       EXPLORATION_RATE, NEIGHBORHOOD_SIZE)
    
    # Create visualizer (pass simulation for value access)
    viz = ABMVisualizer(WIDTH, HEIGHT, NUM_AGENTS, sim)
    
    # Show initial state
    positions = sim.get_agent_positions()
    viz.update_positions(positions)
    
    try:
        # Run simulation
        for i in range(ITERATIONS):
            # Step the simulation
            sim.step()
            
            # Get updated positions
            positions = sim.get_agent_positions()
            
            # Update visualization
            viz.update_positions(positions)
            viz.update_title(i + 1, ITERATIONS)
            
            # Delay so we can see movement
            time.sleep(DELAY)
            
            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{ITERATIONS} completed")
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    finally:
        # Clean up
        sim.cleanup()
        print("\n" + "="*50)
        print("SIMULATION COMPLETE!")
        print("="*50)
        
        # Wait for user to press Enter before closing
        input("Press Enter to close the window...")
        
        viz.close()
        print("Done!")
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
