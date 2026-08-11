import tkinter as tk
import time

class ABMVisualizer:
    """Tkinter visualizer for ABM - just the canvas"""
    
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        
        # Tkinter setup
        self.root = tk.Tk()
        self.root.title(f"ABM - {num_agents} agents on {width}x{height}")
        
        # Canvas
        canvas_size = 800
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg='black')
        self.canvas.pack()
        
        # Scale factor
        self.scale = canvas_size / max(width, height)
        
        # Track current positions
        self.positions = []
    
    def update_positions(self, positions):
        """Update agent positions and redraw"""
        self.positions = positions
        self.draw()
    
    def draw(self):
        """Draw all agents as squares on canvas"""
        self.canvas.delete("all")
        
        # Size of square
        size = max(2, self.scale * 0.8)  # ← 'size' is defined here
        
        for x, y in self.positions:
            cx = x * self.scale + self.scale/2
            cy = y * self.scale + self.scale/2
            
            # Draw square - using 'size' variable
            self.canvas.create_rectangle(
                cx - size/2, cy - size/2,  # ← 'size' not 's'
                cx + size/2, cy + size/2,  # ← 'size' not 's'
                fill='blue',
                outline=''
            )        
        self.root.update()
    
    def update_title(self, iteration, total_iterations):
        """Update window title with progress"""
        self.root.title(f"ABM - Iteration {iteration}/{total_iterations} - {self.num_agents} agents")
    
    def close(self):
        """Close the window"""
        self.root.destroy()
