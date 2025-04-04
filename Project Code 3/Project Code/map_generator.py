import numpy as np
import random
from collections import deque

# Map Elements
EMPTY = 0
WALL = 1
AGENT = 2
ENEMY = 3

def create_empty_grid(size=10):
    """Creates an empty grid with only the agent placed."""
    if size < 3:
        raise ValueError("Grid size must be at least 3x3")
    grid = np.full((size, size), WALL)
    # Place agent away from edges
    start_x, start_y = random.randint(1, size-2), random.randint(1, size-2)
    grid[start_x, start_y] = AGENT
    return grid, (start_x, start_y)

def carve_passages(grid, start, corridor_width=1, open_area_chance=0.2):
    """Carves paths with chances to create open areas."""
    size = grid.shape[0]
    stack = [start]
    visited = set([start])
    
    directions = [
        ((-2, 0), (-1, 0)),  # North
        ((2, 0), (1, 0)),     # South
        ((0, -2), (0, -1)),   # West
        ((0, 2), (0, 1))      # East
    ]
    
    while stack:
        x, y = stack.pop()
        random.shuffle(directions)
        
        # Occasionally create an open area instead of a corridor
        if random.random() < open_area_chance and corridor_width == 1:
            # Create a 2x2 or 3x3 open area
            area_size = random.choice([2, 3])
            for dx in range(area_size):
                for dy in range(area_size):
                    nx, ny = x + dx - 1, y + dy - 1  # Center the area
                    if 0 <= nx < size and 0 <= ny < size:
                        grid[nx, ny] = EMPTY
                        visited.add((nx, ny))
            continue
        
        for (dx, dy), (wx, wy) in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                grid[nx, ny] = EMPTY
                grid[x + wx, y + wy] = EMPTY
                visited.add((nx, ny))
                stack.append((nx, ny))

def is_fully_accessible(grid, start):
    """BFS to verify all empty tiles are reachable from agent."""
    size = grid.shape[0]
    queue = deque([start])
    visited = set([start])
    reachable = 0
    total_empty = np.sum(grid == EMPTY) + 1  # +1 for agent
    
    while queue:
        x, y = queue.popleft()
        reachable += 1
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < size and 0 <= ny < size and 
                (nx, ny) not in visited and 
                grid[nx, ny] in {EMPTY, AGENT}):
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    return reachable == total_empty

def place_enemies_safely(grid, start, max_enemies=5):
    """Places 0-5 enemies without blocking access."""
    size = grid.shape[0]
    agent_x, agent_y = start
    
    # Get all empty cells EXCEPT agent's position
    empty_cells = [
        (x,y) for x in range(size) for y in range(size) 
        if grid[x,y] == EMPTY and (x,y) != (agent_x, agent_y)
    ]
    
    num_enemies = random.randint(0, min(max_enemies, len(empty_cells)))
    enemies_placed = 0
    
    for _ in range(num_enemies):
        if not empty_cells:
            break
            
        ex, ey = random.choice(empty_cells)
        grid[ex, ey] = ENEMY
        
        if is_fully_accessible(grid, start):
            enemies_placed += 1
            empty_cells.remove((ex, ey))
        else:
            grid[ex, ey] = EMPTY  # Revert if blocks paths
    
    return enemies_placed

def generate_valid_map(size=10, max_enemies=5, corridor_width=1, open_area_chance=0.2):
    """Generates a valid map with guaranteed accessibility."""
    for _ in range(100):  # Max attempts
        grid, start = create_empty_grid(size)
        carve_passages(grid, start, corridor_width, open_area_chance)
        
        if not is_fully_accessible(grid, start):
            continue
            
        place_enemies_safely(grid, start, max_enemies)
        return grid, start
    
    raise RuntimeError("Failed to generate valid map after 100 attempts")

def print_grid(grid):
    """Prints the grid in a readable format."""
    symbols = {EMPTY: '0', WALL: '2', AGENT: '3', ENEMY: '4'}
    for row in grid:
        print(' '.join(symbols.get(cell, str(cell)) for cell in row))
    print()

# Generate and print sample maps with open areas
print("=== Maps with Open Areas ===")
for i in range(5):
    try:
        print(f"\nSample Map {i+1} (with open areas):")
        grid, start = generate_valid_map(size=12, max_enemies=5, corridor_width=1, open_area_chance=0.3)
        print(f"Agent starts at: {start}")
        print_grid(grid)
    except Exception as e:
        print(f"Error generating map: {e}")

# Generate some traditional maze-like maps for comparison
print("\n=== Traditional Maze-like Maps ===")
for i in range(5):
    try:
        print(f"\nTraditional Map {i+1}:")
        grid, start = generate_valid_map(size=12, max_enemies=3, corridor_width=1, open_area_chance=0)
        print(f"Agent starts at: {start}")
        print_grid(grid)
    except Exception as e:
        print(f"Error generating map: {e}")