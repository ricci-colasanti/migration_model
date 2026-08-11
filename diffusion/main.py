import time
from multiprocessing import cpu_count
from abm_simulation import ABMSimulation
from abm_visualizer import ABMVisualizer

def main():
    # Parameters
    WIDTH, HEIGHT = 400, 400
    NUM_AGENTS = 20000
    NUM_CORES = cpu_count()  # Auto-detect all cores
    ITERATIONS = 100
    DELAY = 0  # Seconds between iterations
    
    print(f"\n{'='*50}")
    print(f"ABM with Tkinter Visualization")
    print(f"{'='*50}")
    print(f"Grid: {WIDTH}x{HEIGHT} = {WIDTH*HEIGHT:,} cells")
    print(f"Agents: {NUM_AGENTS} ({NUM_AGENTS/(WIDTH*HEIGHT):.1%} occupancy)")
    print(f"Cores: {NUM_CORES} (all available)")
    print(f"Iterations: {ITERATIONS}")
    print(f"Delay: {DELAY}s between iterations")
    
    # Create simulation
    sim = ABMSimulation(WIDTH, HEIGHT, NUM_AGENTS, NUM_CORES)
    
    # Create visualizer
    viz = ABMVisualizer(WIDTH, HEIGHT, NUM_AGENTS)
    
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
        


if __name__ == "__main__":
    main()
