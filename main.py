# main.py

import pygame
import sys
from map_gen import generate_cave
from ant import Ant, WorkerAnt, SoldierAnt  # Import the new classes
import random
import cupy as cp
import noise  # Ensure you have the 'noise' library installed
import numpy as np  # Added import for NumPy
from numba import njit  # Added import for Numba

# Define some colors
DARK_BROWN = (87, 47, 13)
BROWN = (101, 67, 33)
RESOURCE_COLOR = (250, 250, 255)

# Add functions to generate color variations
def get_floor_color():
    base_color = DARK_BROWN
    variation = random.randint(-2, 0)
    return tuple(max(0, min(255, c + variation)) for c in base_color)

def get_wall_color():
    base_color = BROWN
    variation = random.randint(-15, 15)
    return tuple(max(0, min(255, c + variation)) for c in base_color)

def main():
    pygame.init()

    # Check for a valid GPU once
    try:
        cp.cuda.Device(0).compute_capability
        diffuse_pheromones_func = diffuse_pheromones_gpu
    except cp.cuda.runtime.CUDARuntimeError:
        diffuse_pheromones_func = diffuse_pheromones_cpu
        print("No compatible GPU found, falling back to CPU diffusion.")

    # Map dimensions
    width = 120
    height = 80
    cell_size = 11

    num_colonies = 4  
    num_resources = 16  

    simulation_speed = 5.0 

    screen_width = width * cell_size
    screen_height = height * cell_size

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Ant Simulation")

    clock = pygame.time.Clock()

    # Generate the cave map
    cave_map, colony_positions, resource_positions = generate_cave(
        width, height, num_colonies, num_resources
    )

    # Create noise maps for terrain colors
    scale = 11.0  
    seed = random.randint(0, 100)

    # First noise map for color variations
    noise_map = []
    for y in range(height):
        row = []
        for x in range(width):
            nx = x / width - 0.5
            ny = y / height - 0.5
            noise_value = noise.pnoise2(
                nx * scale,
                ny * scale,
                octaves=4,
                persistence=0.5,
                lacunarity=0.6,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            row.append(noise_value)
        noise_map.append(row)

    # Normalize noise_map to range [0, 1]
    min_noise = min(min(row) for row in noise_map)
    max_noise = max(max(row) for row in noise_map)
    noise_map = [
        [(value - min_noise) / (max_noise - min_noise) for value in row]
        for row in noise_map
    ]

    # Convert noise_map to NumPy array for better performance
    noise_map = np.array(noise_map, dtype=np.float32)

    # Second noise map for material selection (grey or brown) for walls only
    material_noise_map = []
    seed2 = random.randint(0, 100)
    for y in range(height):
        row = []
        for x in range(width):
            nx = x / width - 0.5
            ny = y / height - 0.5
            noise_value = noise.pnoise2(
                nx * scale,
                ny * scale,
                octaves=4,
                persistence=0.1,
                lacunarity=2.0,
                repeatx=width,
                repeaty=height,
                base=seed2
            )
            row.append(noise_value)
        material_noise_map.append(row)

    # Normalize material_noise_map to range [0, 1]
    min_m_noise = min(min(row) for row in material_noise_map)
    max_m_noise = max(max(row) for row in material_noise_map)
    material_noise_map = [
        [(value - min_m_noise) / (max_m_noise - min_m_noise) for value in row]
        for row in material_noise_map
    ]

    material_noise_map = np.array(material_noise_map, dtype=np.float32)

    # Define colors for grey (stone) and brown (soil)
    GREY = (90, 70, 70)
    BROWN = (101, 67, 33)
    DARK_BROWN = (87, 47, 13)

    # Create a background Surface to draw the terrain with noise-based colors
    background = pygame.Surface((screen_width, screen_height))

    for y in range(height):
        for x in range(width):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            # Determine base color based on cave_map
            if cave_map[y][x] == 1:  # Wall
                # Determine if the wall is grey or brown based on material noise
                material_value = material_noise_map[y, x]
                if material_value > 0.5:
                    base_color = GREY  
                else:
                    base_color = BROWN  
            else:
                base_color = DARK_BROWN 

            # Apply noise-based variation to the color
            noise_value = noise_map[y, x]
            variation = int(noise_value * 30) - 15  # Variation from -15 to +15
            color = tuple(
                max(0, min(255, c + variation)) for c in base_color
            )
            pygame.draw.rect(background, color, rect)

    # Generate unique colors for colonies
    def generate_unique_colors(num_colors):
        colors = []
        while len(colors) < num_colors:
            color = (
                random.randint(20, 255),
                random.randint(0, 100),
                random.randint(0, 255)
            )
            if color not in colors:
                colors.append(color)
        return colors

    colony_colors = generate_unique_colors(num_colonies)

    for idx, colony in enumerate(colony_positions):
        colony['id'] = idx
        colony['color'] = colony_colors[idx]
        colony['parameters'] = {
            'sensor_angle': random.uniform(30, 90),
            'sensor_distance': 2,
            'deposit_amount': random.randint(25, 60),
            'can_steal': random.choice([True, False]),
            'resources_stolen': 0
        }
        colony['resources_returned'] = 2
        colony['ants_captured'] = 0  
        colony['ants'] = [] 
        colony['worker_ratio'] = random.uniform(0.5, 0.9)  # Worker:Soldier ratio
        # Set a default value for colony size or initial ant count
        colony['value'] = 1 

    # Initialize pheromone maps for each colony as NumPy arrays
    pheromone_maps = []
    for _ in colony_positions:
        pheromone_map = np.zeros((height, width), dtype=np.float32)
        pheromone_maps.append(pheromone_map)

    # Initialize ants
    ants = []
    for colony in colony_positions:
        x, y = colony['position']
        size = colony['size']
        num_workers = 12  # Number of worker ants per colony
        num_soldiers = 4  # Number of soldier ants per colony
        for _ in range(num_workers):
            ant_x = x + size / 2
            ant_y = y + size / 2
            direction = random.uniform(0, 360)
            ant = WorkerAnt(
                ant_x, ant_y, direction,
                colony=colony,
                speed=simulation_speed
            )
            ants.append(ant)
            colony['ants'].append(ant)
        for _ in range(num_soldiers):
            ant_x = x + size / 2
            ant_y = y + size / 2
            direction = random.uniform(0, 360)
            ant = SoldierAnt(
                ant_x, ant_y, direction,
                colony=colony,
                speed=simulation_speed,
                ants=ants  # Pass the global ants list to the SoldierAnt
            )
            ants.append(ant)
            colony['ants'].append(ant) 

    # Button to toggle text display
    button_rect = pygame.Rect(10, screen_height - 40, 100, 30)
    show_text = True

    # Button to toggle pheromone visualization
    pheromone_button_rect = pygame.Rect(120, screen_height - 40, 150, 30)
    show_pheromones = True

    # Main loop
    running = True
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    simulation_speed += 0.1  
                    
                    for ant in ants:
                        ant.speed = simulation_speed
                elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                    simulation_speed = max(0.1, simulation_speed - 0.1)  

                    for ant in ants:
                        ant.speed = simulation_speed

            # Toggle text display
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    show_text = not show_text
                elif pheromone_button_rect.collidepoint(event.pos):
                    show_pheromones = not show_pheromones


        for ant in ants:
            colony_id = ant.colony['id']
            pheromone_map = pheromone_maps[colony_id]
            rival_colonies = [colony for colony in colony_positions if colony['id'] != ant.colony['id']]
            ant.update(cave_map, pheromone_maps[ant.colony['id']], resource_positions, rival_colonies)
        
        # Spawn new ants if resources returned reach threshold
        for colony in colony_positions:
            while colony['resources_returned'] >= 5:
                colony['resources_returned'] -= 5
                # Determine the type of new ant based on the worker-to-soldier ratio
                if random.random() < colony['worker_ratio']:
                    # Spawn a worker ant
                    ant_x = colony['position'][0] + colony['size'] / 2
                    ant_y = colony['position'][1] + colony['size'] / 2
                    direction = random.uniform(0, 360)
                    ant = WorkerAnt(
                        ant_x, ant_y, direction,
                        colony=colony,
                        speed=simulation_speed
                    )
                else:
                    # Spawn a soldier ant
                    ant_x = colony['position'][0] + colony['size'] / 2
                    ant_y = colony['position'][1] + colony['size'] / 2
                    direction = random.uniform(0, 360)
                    ant = SoldierAnt(
                        ant_x, ant_y, direction,
                        colony=colony,
                        speed=simulation_speed,
                        ants=ants  # Pass the global ants list to the SoldierAnt
                    )
                ants.append(ant)
                colony['ants'].append(ant)  # Add the ant to the colony's list of ants

        # Evaporate and diffuse pheromones
        for idx, pheromone_map in enumerate(pheromone_maps):
            pheromone_map = diffuse_pheromones_func(pheromone_map, diffusion_rate=0.032)
            pheromone_map = pheromone_map - 0.2 * simulation_speed  # Adjusted evaporation rate
            pheromone_map = np.maximum(pheromone_map, 0)
            pheromone_maps[idx] = pheromone_map

        # Draw the background onto the screen
        screen.blit(background, (0, 0))

        # Draw colonies
        for colony in colony_positions:
            x, y = colony['position']
            size = colony['size']
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size * size, cell_size * size)
            pygame.draw.rect(screen, colony['color'], rect)

        # Draw resources
        for resource in resource_positions:
            x, y = resource['position']
            size = resource['size']
            center = ((x + size / 2) * cell_size, (y + size / 2) * cell_size)
            radius = (size * cell_size) // 2
            pygame.draw.circle(screen, RESOURCE_COLOR, center, radius)

        # Draw pheromones if show_pheromones is True
        if show_pheromones:
            for idx, pheromone_map in enumerate(pheromone_maps):
                color = colony_positions[idx]['color']
                for y in range(height):
                    for x in range(width):
                        pheromone_level = pheromone_map[y, x]
                        if pheromone_level > 0:
                            alpha = min(int(pheromone_level * 2), 255)
                            pheromone_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                            pheromone_surface.fill((*color, alpha))
                            screen.blit(pheromone_surface, (x * cell_size, y * cell_size))

        # Draw ants
        for ant in ants:
            ant_color = darken_color(ant.colony['color'])  # Darken the colony color for the ant
            if isinstance(ant, WorkerAnt):
                ant_rect = pygame.Rect(
                    int(ant.x * cell_size), int(ant.y * cell_size), cell_size // 2, cell_size // 2
                )
                pygame.draw.rect(screen, ant_color, ant_rect)
            elif isinstance(ant, SoldierAnt):
                ant_rect = pygame.Rect(
                    int(ant.x * cell_size), int(ant.y * cell_size), cell_size // 1.2, cell_size // 1.2
                )
                pygame.draw.ellipse(screen, ant_color, ant_rect)  # Draw soldier ants as ellipses

        # Draw resource values
        font = pygame.font.SysFont(None, cell_size)
        for resource in resource_positions:
            x, y = resource['position']
            size = resource['size']
            value = resource['value']
            # Calculate the center of the resource area
            center_x = (x + size / 2) * cell_size
            center_y = (y + size / 2) * cell_size
            # Render the value
            if show_text:
                large_font = pygame.font.SysFont(None, cell_size * 2)
                text = large_font.render(str(value), True, (0, 0, 0))
                text_rect = text.get_rect(center=(center_x, center_y))
                screen.blit(text, text_rect)

        # Count ants per colony
        ants_per_colony = {colony['id']: 0 for colony in colony_positions}
        for ant in ants:
            colony_id = ant.colony['id']
            ants_per_colony[colony_id] += 1

        # Draw colony parameters and resources returned if show_text is True
        if show_text:
            info_font = pygame.font.SysFont(None, 20)
            text_y = 10
            for colony in colony_positions:
                color = colony['color']
                colony_id = colony['id']
                sensor_angle = colony['parameters']['sensor_angle']
                sensor_distance = colony['parameters']['sensor_distance']
                deposit_amount = colony['parameters']['deposit_amount']
                resources_returned = colony['resources_returned']
                num_ants = ants_per_colony[colony_id]
                ants_captured = colony['ants_captured']
                text = (f"Colony {colony_id}: Ants {num_ants}, Resources {resources_returned}, "
                        f"Sensor Angle {sensor_angle:.1f}, Sensor Distance {sensor_distance}, "
                        f"Deposit Amount {deposit_amount}, "
                        f"Can Steal {colony['parameters']['can_steal']}, "
                        f"Resources Stolen {colony['parameters']['resources_stolen']}, "
                        f"Ants Captured {ants_captured}")
                text_surface = info_font.render(text, True, color)
                screen.blit(text_surface, (10, text_y))
                text_y += 20  # Increment y position for next text

        # Draw the toggle buttons
        pygame.draw.rect(screen, (200, 200, 200), button_rect)
        button_text = font.render("Text", True, (0, 0, 0))
        button_text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, button_text_rect)

        pygame.draw.rect(screen, (200, 200, 200), pheromone_button_rect)
        pheromone_button_text = pygame.font.SysFont(None, 26).render("Pheromones", True, (0, 0, 0))
        pheromone_button_text_rect = pheromone_button_text.get_rect(center=pheromone_button_rect.center)
        screen.blit(pheromone_button_text, pheromone_button_text_rect)

        # Update display
        pygame.display.flip()
        clock.tick(30 * simulation_speed)  # Adjust frame rate based on simulation speed

    pygame.quit()
    sys.exit()

@njit
def diffuse_pheromones_cpu(pheromone_map, diffusion_rate=0.02):
    height, width = pheromone_map.shape
    new_map = pheromone_map.copy()

    for y in range(height):
        for x in range(width):
            pheromone = pheromone_map[y, x]
            diffused_amount = pheromone * diffusion_rate
            if diffused_amount > 0:
                shared_amount = diffused_amount / 16  # Distribute to 8 neighbors
                new_map[y, x] -= diffused_amount
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx != 0 or dy != 0:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                new_map[ny, nx] += shared_amount
    return new_map

def diffuse_pheromones_gpu(pheromone_map, diffusion_rate=0.04):
    pheromone_map_gpu = cp.array(pheromone_map, dtype=cp.float32)
    height, width = pheromone_map_gpu.shape
    new_map_gpu = cp.copy(pheromone_map_gpu)

    # Create shifted versions of the pheromone map for diffusion
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx != 0 or dy != 0:
                shifted_map = cp.roll(pheromone_map_gpu, shift=(dy, dx), axis=(0, 1))
                new_map_gpu += (shifted_map - pheromone_map_gpu) * diffusion_rate / 64

    # Ensure no negative values
    new_map_gpu = cp.maximum(new_map_gpu, 0)
    
    return cp.asnumpy(new_map_gpu)

def darken_color(color, factor=0.7):
    """Darken a color by a given factor."""
    return tuple(max(0, int(c * factor)) for c in color)

if __name__ == "__main__":
    main()
