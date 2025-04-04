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
    grid = np.full((size, size), WALL)  # Start with all walls
    # Pick a random starting point for the agent
    start_x, start_y = random.randint(1, size - 2), random.randint(1, size - 2)
    grid[start_x, start_y] = AGENT
    return grid, (start_x, start_y)

def carve_passages(grid, start, corridor_width=1):
    """Carves paths using a randomized DFS maze generation with adjustable corridor width."""
    size = grid.shape[0]
    stack = [start]
    visited = set()
    visited.add(start)

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    while stack:
        x, y = stack[-1]
        random.shuffle(directions)  # Randomize carving direction

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                # Carve out the passage with specified width
                grid[nx, ny] = EMPTY
                grid[x + dx//2, y + dy//2] = EMPTY  # Remove wall between cells
                
                # Create wider corridors if needed
                if corridor_width > 1:
                    for i in range(1, corridor_width):
                        # Horizontal corridors
                        if dx != 0 and ny + i < size:
                            grid[nx, ny + i] = EMPTY
                            grid[x + dx//2, y + dy//2 + i] = EMPTY
                        # Vertical corridors
                        if dy != 0 and nx + i < size:
                            grid[nx + i, ny] = EMPTY
                            grid[x + dx//2 + i, y + dy//2] = EMPTY
                
                visited.add((nx, ny))
                stack.append((nx, ny))
                break
        else:
            stack.pop()

def is_fully_accessible(grid, start):
    """Checks if all empty tiles are reachable from the agent."""
    size = grid.shape[0]
    queue = deque([start])
    visited = set()
    visited.add(start)

    total_empty = sum(1 for x in range(size) for y in range(size) if grid[x, y] in {EMPTY, AGENT})

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                if grid[nx, ny] in {EMPTY, AGENT}:  
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return len(visited) == total_empty  # True if all empty spaces are reachable

def place_enemies(grid, start, max_enemies=5):
    """Places 0 to max_enemies enemies while ensuring accessibility."""
    size = grid.shape[0]
    empty_spaces = [(x, y) for x in range(size) for y in range(size) if grid[x, y] == EMPTY]
    num_enemies = random.randint(0, max_enemies)
    enemy_positions = []
    
    for _ in range(num_enemies):
        if not empty_spaces:
            break
        random.shuffle(empty_spaces)
        for ex, ey in empty_spaces:
            grid[ex, ey] = ENEMY
            if is_fully_accessible(grid, start):
                enemy_positions.append((ex, ey))
                empty_spaces.remove((ex, ey))
                break
            else:
                grid[ex, ey] = EMPTY
    return enemy_positions

def generate_valid_map(size=10, max_enemies=5, max_attempts=100, corridor_width=1):
    """Generates a valid map with wider corridors and 0-max_enemies enemies."""
    for _ in range(max_attempts):
        grid, start = create_empty_grid(size)
        carve_passages(grid, start, corridor_width)

        if not is_fully_accessible(grid, start):
            continue

        place_enemies(grid, start, max_enemies)
        return grid

    raise ValueError("Failed to generate a valid map after max attempts")

def print_grid(grid):
    """Prints the grid in a readable format."""
    symbols = {EMPTY: ' ', WALL: '▓', AGENT: 'A', ENEMY: 'E'}
    for row in grid:
        print(' '.join(symbols.get(cell, str(cell)) for cell in row))
    print()

# Generate sample maps with different characteristics
sample_maps = []

# 1. Open map with few walls and 2 enemies
grid, _ = create_empty_grid()
carve_passages(grid, (0,0), corridor_width=3)
place_enemies(grid, (0,0), max_enemies=2)
sample_maps.append(grid.copy())

# 2. Maze-like with wide corridors and random enemies
grid, start = create_empty_grid()
carve_passages(grid, start, corridor_width=2)
place_enemies(grid, start, max_enemies=random.randint(0,5))
sample_maps.append(grid.copy())

# 3. Complex map with varying corridor widths
grid, start = create_empty_grid()
# First carve wide main paths
carve_passages(grid, start, corridor_width=2)
# Then add some narrow branches
for _ in range(10):
    x, y = random.choice([(x,y) for x in range(10) for y in range(10) if grid[x,y] == EMPTY])
    carve_passages(grid, (x,y), corridor_width=1)
place_enemies(grid, start, max_enemies=4)
sample_maps.append(grid.copy())

# 4. Mostly open with clusters of walls
grid = np.full((10,10), EMPTY)
# Add agent
start = (random.randint(0,9), random.randint(0,9))
grid[start[0], start[1]] = AGENT
# Add some wall clusters
for _ in range(5):
    x, y = random.randint(0,7), random.randint(0,7)
    grid[x:x+3, y:y+3] = WALL
    grid[x+1,y+1] = EMPTY  # Leave center open
place_enemies(grid, start, max_enemies=3)
sample_maps.append(grid.copy())

# 5. Classic maze with 1-2 wide paths and few enemies
grid, start = create_empty_grid()
carve_passages(grid, start, corridor_width=1)
place_enemies(grid, start, max_enemies=2)
sample_maps.append(grid.copy())

# Print all sample maps
for i, map in enumerate(sample_maps, 1):
    print(f"Map {i}:")
    print_grid(map)