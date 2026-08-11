import tkinter as tk

class ABMVisualizer:
    """Tkinter visualizer with color gradient based on agent values (0-1000)"""
    
    def __init__(self, width, height, num_agents, simulation):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.simulation = simulation  # Reference to get values
        
        # Tkinter setup
        self.root = tk.Tk()
        self.root.title(f"ABM - {num_agents} agents on {width}x{height}")
        
        # Canvas
        canvas_size = 1600
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg='black')
        self.canvas.pack()
        
        # Scale factor
        self.scale = canvas_size / max(width, height)
        
        # Track current positions and values
        self.positions = []
        self.values = []
    
    def update_positions(self, positions):
        """Update agent positions and redraw"""
        self.positions = positions
        self.values = self.simulation.get_agent_values()  # Get values as integers
        self.draw()
    
    def value_to_color(self, value):
        """Convert value (0-1000) to a color"""
        # Normalize to 0-1 range
        normalized = value / 1000.0
        
        # Blue (low) -> Purple -> Red (high)
        r = int(normalized * 255)
        g = int((1 - abs(normalized - 0.5) * 2) * 128)  # Peak green in middle
        b = int((1 - normalized) * 255)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def draw(self):
        """Draw all agents as colored squares based on their values"""
        self.canvas.delete("all")
        
        size = max(2, self.scale * 0.8)
        
        for i, (x, y) in enumerate(self.positions):
            cx = x * self.scale + self.scale/2
            cy = y * self.scale + self.scale/2
            
            # Get agent value and convert to color
            value = self.values[i] if i < len(self.values) else 500
            color = self.value_to_color(value)
            
            # Draw square with color
            self.canvas.create_rectangle(
                cx - size/2, cy - size/2,
                cx + size/2, cy + size/2,
                fill=color,
                outline=''
            )
        
        self.root.update()
    
    def update_title(self, iteration, total_iterations):
        """Update window title with progress"""
        self.root.title(f"ABM - Iteration {iteration}/{total_iterations} - {self.num_agents} agents")
    
    def close(self):
        """Close the window"""
        self.root.destroy()
