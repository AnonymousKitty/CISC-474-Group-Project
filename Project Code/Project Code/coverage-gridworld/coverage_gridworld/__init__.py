from gymnasium.envs.registration import register
from coverage_gridworld.env import CoverageGridworld
from coverage_gridworld.map_generator import generate_valid_map
import numpy as np
register(
    id="standard",
    entry_point="coverage_gridworld:CoverageGridworld"
)

register(
    id="just_go",   # very easy difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    }
)

register(
    id="safe",   # easy difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
            [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
        ]
    }
)

register(
    id="maze",   # medium difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
            [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
        ]
    }
)

register(
    id="chokepoint",   # hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
        ]
    }
)

register(
    id="sneaky_enemies",   # very hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
        ]
    }
)

# To create a predefined map, just add walls and enemies. The agent always starts in the top-left corner.
# The enemy's orientation is randomly defined and the cells under surveillance will be spawned automatically
maps = [
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    ],
    [
        [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
    ]
]


register(
    id="all_maps",   # very hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map_list": maps
    }
)

training_maps = []
for i in range(100):
    grid, start = generate_valid_map(size=10, max_enemies=5, corridor_width=1, open_area_chance=0.3)
    #print(np.shape(grid))
    training_maps.append(list(grid))
    training_maps.append(None)
    training_maps.append(None)
    grid, start = generate_valid_map(size=10, max_enemies=5, corridor_width=1, open_area_chance=0)
    #print(np.shape(grid))
    training_maps.append(list(grid))
    training_maps.append(None)

register(
    id="training_maps",   # very hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map_list": training_maps
    }
)