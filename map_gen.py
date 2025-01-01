# cave_generation.py

import random

def generate_cave_map(width, height, fill_prob=0.45, smoothing_steps=5):
    map = [[1 if random.random() < fill_prob else 0 for _ in range(width)] for _ in range(height)]
    for _ in range(smoothing_steps):
        map = smooth_map(map)
    return map

def smooth_map(map):
    new_map = [[0 for _ in row] for row in map]
    for y in range(len(map)):
        for x in range(len(map[0])):
            wall_count = count_walls_around(map, x, y)
            if map[y][x] == 1:
                new_map[y][x] = 1 if wall_count >= 4 else 0
            else:
                new_map[y][x] = 1 if wall_count >= 5 else 0
    return new_map

def count_walls_around(map, x, y):
    wall_count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            neighbor_x = x + j
            neighbor_y = y + i
            if i == 0 and j == 0:
                continue
            elif neighbor_x < 0 or neighbor_x >= len(map[0]) or neighbor_y < 0 or neighbor_y >= len(map):
                wall_count += 1  # Treat out-of-bounds as walls
            elif map[neighbor_y][neighbor_x] == 1:
                wall_count += 1
    return wall_count

def place_colonies(map, num_colonies, max_colony_size=4):
    colonies = []
    free_cells = [(x, y) for y in range(len(map)) for x in range(len(map[0])) if map[y][x] == 0]

    for _ in range(num_colonies):
        if not free_cells:
            break
        value = random.randint(1, max_colony_size)
        size = min(value, max_colony_size)  # Ensure size doesn't exceed max_colony_size

        # Attempt to place colony of this size
        placed = False
        attempts = 0
        max_attempts = 100
        while not placed and attempts < max_attempts:
            attempts += 1
            x, y = random.choice(free_cells)
            # Check if the colony fits
            if can_place_structure(map, x, y, size):
                place_structure(map, x, y, size, 2)  # 2 represents colony
                colonies.append({'position': (x, y), 'size': size, 'value': value})
                # Remove occupied cells from free_cells
                for dy in range(size):
                    for dx in range(size):
                        if (x + dx, y + dy) in free_cells:
                            free_cells.remove((x + dx, y + dy))
                placed = True
        if not placed:
            print(f"Failed to place colony of size {size} after {max_attempts} attempts.")
    return colonies

def place_resources(map, num_resources, max_resource_value=100):
    resources = []
    free_cells = [(x, y) for y in range(len(map)) for x in range(len(map[0])) if map[y][x] == 0]

    for _ in range(num_resources):
        if not free_cells:
            break
        value = random.randint(1, max_resource_value)
        size = max(1, value // 10)  # Physical size is 1/10 the resource value, minimum size is 1

        # Attempt to place resource of this size
        placed = False
        attempts = 0
        max_attempts = 100
        while not placed and attempts < max_attempts:
            attempts += 1
            x, y = random.choice(free_cells)
            # Check if the resource fits
            if can_place_structure(map, x, y, size):
                place_structure(map, x, y, size, 3)  # 3 represents resource
                resources.append({'position': (x, y), 'size': size, 'value': value})
                # Remove occupied cells from free_cells
                for dy in range(size):
                    for dx in range(size):
                        if (x + dx, y + dy) in free_cells:
                            free_cells.remove((x + dx, y + dy))
                placed = True
        if not placed:
            print(f"Failed to place resource of size {size} after {max_attempts} attempts.")
    return resources

def can_place_structure(map, x, y, size):
    width = len(map[0])
    height = len(map)
    if x + size > width or y + size > height:
        return False
    for dy in range(size):
        for dx in range(size):
            if map[y + dy][x + dx] != 0:
                return False
    return True

def place_structure(map, x, y, size, value):
    for dy in range(size):
        for dx in range(size):
            map[y + dy][x + dx] = value

def flood_fill_iterative(map, x, y, target_values, replacement_value):
    width = len(map[0])
    height = len(map)
    stack = [(x, y)]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if map[y][x] not in target_values:
            continue

        map[y][x] = replacement_value

        stack.append((x+1, y))
        stack.append((x-1, y))
        stack.append((x, y+1))
        stack.append((x, y-1))

def ensure_connectivity(map, colony_positions, resource_positions):
    # Flood fill from the first colony
    temp_map = [row[:] for row in map]
    x, y = colony_positions[0]['position']
    flood_fill_iterative(temp_map, x, y, target_values=[0, 2, 3], replacement_value=-1)

    # Connect colonies
    for colony in colony_positions[1:]:
        cx, cy = colony['position']
        if temp_map[cy][cx] != -1:
            carve_path(map, (cx, cy), colony_positions[0]['position'])
            flood_fill_iterative(temp_map, cx, cy, target_values=[0, 2, 3], replacement_value=-1)

    # Connect resources
    for resource in resource_positions:
        rx, ry = resource['position']
        if temp_map[ry][rx] != -1:
            carve_path(map, (rx, ry), colony_positions[0]['position'])
            flood_fill_iterative(temp_map, rx, ry, target_values=[0, 2, 3], replacement_value=-1)

def carve_path(map, start, end):
    x0, y0 = start
    x1, y1 = end

    while (x0, y0) != (x1, y1):
        if x0 < x1:
            x0 += 1
        elif x0 > x1:
            x0 -= 1
        elif y0 < y1:
            y0 += 1
        elif y0 > y1:
            y0 -= 1
        # Only clear walls; do not overwrite colonies or resources
        if map[y0][x0] == 1:
            map[y0][x0] = 0  # Clear wall

def generate_cave(width, height, num_colonies=1, num_resources=1, max_colony_size=4, max_resource_value=100):
    map = generate_cave_map(width, height)
    colony_positions = place_colonies(map, num_colonies, max_colony_size)
    resource_positions = place_resources(map, num_resources, max_resource_value)
    ensure_connectivity(map, colony_positions, resource_positions)
    return map, colony_positions, resource_positions
