# ant.py

import random
import math
from heapq import heappush, heappop
from numba import njit  # Added import for Numba
import numpy as np

@njit
def numba_heuristic(a_x, a_y, b_x, b_y):
    """Manhattan distance heuristic optimized with Numba."""
    return abs(a_x - b_y) + abs(a_y - b_y)

class Ant:
    def __init__(self, x, y, direction, colony, speed=1.0):
        self.x = x
        self.y = y
        self.direction = direction  # Angle in degrees
        self.colony = colony  # Reference to the colony
        self.colony_position = colony['position']
        self.colony_size = colony['size']
        self.colony_center = (
            self.colony_position[0] + self.colony_size // 2,
            self.colony_position[1] + self.colony_size // 2
        )
        self.state = 'searching'  # 'searching', 'transporting', or 'raiding'
        self.has_resource = False
        self.path = []
        self.speed = speed
        self.sensor_angle = colony['parameters']['sensor_angle']
        self.sensor_distance = colony['parameters']['sensor_distance']
        self.deposit_amount = colony['parameters']['deposit_amount']
        self.target_colony = None  # Target colony to raid

    def update(self, cave_map, pheromone_map, resource_positions, rival_colonies):
        # Ant behavior based on state
        if self.state == 'searching':
            self.search(cave_map, pheromone_map, resource_positions, rival_colonies)
        elif self.state == 'transporting':
            self.transport(cave_map, pheromone_map)
        elif self.state == 'raiding':
            self.transport(cave_map, pheromone_map, stealing=True)

    def search(self, cave_map, pheromone_map, resource_positions, rival_colonies):
        # Check for resources or rival colony within sensor range
        resource_in_range = self.check_for_resource_in_range(resource_positions)
        rival_colony = self.detect_rival_colony(rival_colonies)

        if resource_in_range:
            # Regular resource collection behavior
            self.move_towards(resource_in_range)
        elif rival_colony and not self.has_resource and not rival_colony['resources_returned'] == 0 and self.colony['parameters']['can_steal'] == True:
            # Switch to raiding state if a rival colony is detected
            self.state = 'raiding'
            self.target_colony = rival_colony
            self.steal_resource(rival_colony, cave_map)
            self.colony['parameters']['resources_stolen'] += 1
        else:
            # Move according to pheromones if no resource or rival colony detected
            self.move(cave_map, pheromone_map)

        # Check if the ant is currently on a resource tile
        if self.check_for_resource(cave_map, resource_positions):
            self.has_resource = True
            self.state = 'transporting'
            # Find path back to colony
            self.find_path_to_colony(cave_map)

    def detect_rival_colony(self, rival_colonies):
        # Check if there’s a rival colony in range with resources
        for rival in rival_colonies:
            if rival['id'] != self.colony['id'] and rival['resources_returned'] > 0:
                # Check if within range (using sensor distance)
                if self.distance_to(rival['position']) <= self.sensor_distance:
                    return rival  # Return rival colony if in range and has resources
        return None

    def distance_to(self, position):
        dx = position[0] - self.x
        dy = position[1] - self.y
        return math.sqrt(dx ** 2 + dy ** 2)
    
    def move(self, cave_map, pheromone_map):
        # Sensors positions (left, center, right)
        sensor_distance = self.sensor_distance
        sensor_angle = self.sensor_angle  # Use colony-specific sensor angle
        angles = [-sensor_angle, 0, sensor_angle]
        readings = []

        for angle_offset in angles:
            sensor_dir = math.radians(self.direction + angle_offset)
            sensor_x = int(self.x + sensor_distance * math.cos(sensor_dir))
            sensor_y = int(self.y + sensor_distance * math.sin(sensor_dir))
            pheromone_level = self.get_pheromone_level(pheromone_map, sensor_x, sensor_y)
            readings.append((pheromone_level, angle_offset))

        # Choose the direction with the highest pheromone concentration
        readings.sort(reverse=True)
        max_pheromone, best_angle = readings[0]

        if max_pheromone > 0:
            self.direction += best_angle
        else:
            # Random movement when no pheromone detected
            self.direction += random.uniform(-10, 10) * self.speed

        # Normalize direction
        self.direction %= 360

        # Calculate new position
        dx = math.cos(math.radians(self.direction)) * self.speed * 0.1
        dy = math.sin(math.radians(self.direction)) * self.speed * 0.1
        new_x = self.x + dx
        new_y = self.y + dy

        # Collision detection with walls
        if self.can_move_to(int(new_x), int(new_y), cave_map):
            self.x = new_x
            self.y = new_y
        else:
            # Bounce back by changing direction
            self.direction += 180
            self.direction %= 360

    def follow_path(self, cave_map):
        if not self.path:
            # No path found; stay in place
            return

        next_node = self.path[0]
        new_x, new_y = next_node
        if self.can_move_to(new_x, new_y, cave_map):
            # Move towards next node with speed adjustment
            dx = (new_x + 0.5 - self.x) * self.speed * 0.1
            dy = (new_y + 0.5 - self.y) * self.speed * 0.1
            self.x += dx
            self.y += dy

            # Check if close enough to the next node
            if abs(self.x - (new_x + 0.5)) < 0.1 and abs(self.y - (new_y + 0.5)) < 0.1:
                self.x = new_x + 0.5
                self.y = new_y + 0.5
                self.path.pop(0)
        else:
            # Recalculate path if blocked
            self.find_path_to_colony(cave_map)

    def steal_resource(self, rival_colony, cave_map):
        # Steal one resource from the rival colony
        if rival_colony['resources_returned'] > 0:
            rival_colony['resources_returned'] -= 1
            self.has_resource = True
            self.state = 'transporting'  # Set state to transporting
            # Find path back to home colony after stealing
            self.find_path_to_colony(cave_map)

    def move_towards(self, target_position):
        # Adjust direction towards the target position
        dx = target_position[0] - self.x
        dy = target_position[1] - self.y
        self.direction = math.degrees(math.atan2(dy, dx))

        # Calculate new position
        distance = math.hypot(dx, dy)
        step = min(distance, self.speed * 0.1)
        ratio = step / distance if distance != 0 else 0
        self.x += dx * ratio
        self.y += dy * ratio

    def transport(self, cave_map, pheromone_map, raiding=False):
        # Leave pheromone trail
        if not raiding:
            self.leave_pheromone(pheromone_map)

        # Check if reached colony
        if self.check_for_colony():
            # Deposit resource and switch back to searching
            self.colony['resources_returned'] += 1
            self.has_resource = False
            self.state = 'searching'
            self.path = []
            return

        # Move along the path to the colony
        self.follow_path(cave_map)

    def find_path_to_colony(self, cave_map):
        start = (int(self.x), int(self.y))
        goal = (int(self.colony_center[0]), int(self.colony_center[1]))
        self.path = self.a_star(cave_map, start, goal)
        # Remove the starting position from the path
        if self.path and self.path[0] == start:
            self.path.pop(0)

    def a_star(self, cave_map, start, goal):
        width = len(cave_map[0])
        height = len(cave_map)

        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: numba_heuristic(start[0], start[1], goal[0], goal[1])}  # Use Numba-optimized heuristic

        while open_set:
            current = heappop(open_set)[1]

            if current == goal:
                return self.reconstruct_path(came_from, current)

            neighbors = self.get_neighbors(current, cave_map)
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + 1  # Distance between nodes is 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + numba_heuristic(neighbor[0], neighbor[1], goal[0], goal[1])  # Use Numba-optimized heuristic
                    heappush(open_set, (f_score[neighbor], neighbor))

        # Path not found
        return []

    # def heuristic(self, a, b):
    #     # Manhattan distance heuristic
    #     return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        return total_path

    def get_neighbors(self, node, cave_map):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if self.can_move_to(nx, ny, cave_map):
                neighbors.append((nx, ny))
        return neighbors

    def can_move_to(self, x, y, cave_map):
        width = len(cave_map[0])
        height = len(cave_map)
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        if cave_map[int(y)][int(x)] == 1:
            return False
        return True

    def get_pheromone_level(self, pheromone_map, x, y):
        if pheromone_map is None:
            return 0
        width = pheromone_map.shape[1]
        height = pheromone_map.shape[0]
        if x < 0 or x >= width or y < 0 or y >= height:
            return 0
        return pheromone_map[int(y), int(x)]

    def leave_pheromone(self, pheromone_map):
        x = int(self.x)
        y = int(self.y)
        width = pheromone_map.shape[1]
        height = pheromone_map.shape[0]
        if 0 <= x < width and 0 <= y < height:
            pheromone_map[y, x] += self.deposit_amount  # Deposit pheromone

    def check_for_resource(self, cave_map, resource_positions):
        x = int(self.x)
        y = int(self.y)
        for resource in resource_positions:
            rx, ry = resource['position']
            size = resource['size']
            if rx <= x < rx + size and ry <= y < ry + size:
                # Remove one unit of resource
                resource['value'] -= 1
                if resource['value'] <= 0:
                    # Remove resource from map
                    for dy in range(size):
                        for dx in range(size):
                            cave_map[ry + dy][rx + dx] = 0
                    resource_positions.remove(resource)
                return True
        return False

    def check_for_resource_in_range(self, resource_positions):
        sensor_distance = self.sensor_distance
        sensor_range_squared = sensor_distance ** 2
        for resource in resource_positions:
            rx, ry = resource['position']
            size = resource['size']
            # Calculate the center of the resource
            resource_center_x = rx + size / 2
            resource_center_y = ry + size / 2
            dx = resource_center_x - self.x
            dy = resource_center_y - self.y
            distance_squared = dx ** 2 + dy ** 2
            if distance_squared <= sensor_range_squared:
                return (resource_center_x, resource_center_y)
        return None

    def check_for_colony(self):
        x = int(self.x)
        y = int(self.y)
        cx, cy = self.colony_position
        size = self.colony_size
        if cx <= x < cx + size and cy <= y < cy + size:
            return True
        return False

class WorkerAnt(Ant):
    def update(self, cave_map, pheromone_map, resource_positions, rival_colonies):
        if self.state == 'searching':
            self.search(cave_map, pheromone_map, resource_positions, rival_colonies)
        elif self.state == 'transporting':
            self.transport(cave_map, pheromone_map)
        elif self.state == 'raiding':
            self.transport(cave_map, pheromone_map, stealing=True)

class SoldierAnt(Ant):
    def __init__(self, x, y, direction, colony, speed=1.0, ants=None):
        super().__init__(x, y, direction, colony, speed)
        self.state = 'hunting'  # Soldiers start in hunting state
        self.ants = ants  # Store the reference to the global ants list

    def update(self, cave_map, pheromone_map, resource_positions, rival_colonies):
        if self.state == 'hunting':
            self.hunt(cave_map, rival_colonies, pheromone_map)
        elif self.state == 'transporting':
            self.transport(cave_map, pheromone_map, raiding=True)

    def hunt(self, cave_map, rival_colonies, pheromone_map=None):
        rival_ant = self.detect_rival_ant(rival_colonies)
        if rival_ant:
            if isinstance(rival_ant, SoldierAnt):
                self.duel(rival_ant, cave_map)
            else:
                self.move_towards((rival_ant.x, rival_ant.y))
                if self.distance_to((rival_ant.x, rival_ant.y)) < 1:
                    self.capture_rival_ant(rival_ant, cave_map)
        else:
            self.move(cave_map, pheromone_map)

    def detect_rival_ant(self, rival_colonies):
        for rival_colony in rival_colonies:
            for ant in rival_colony['ants']:
                if self.distance_to((ant.x, ant.y)) <= self.sensor_distance:
                    return ant
        return None

    def duel(self, rival_ant, cave_map):
        if random.random() < 0.5:
            # This ant wins the duel
            self.capture_rival_ant(rival_ant, cave_map)
        else:
            # Rival ant wins the duel
            rival_ant.capture_rival_ant(self, cave_map)

    def capture_rival_ant(self, rival_ant, cave_map):
        rival_ant.colony['ants'].remove(rival_ant)
        self.ants.remove(rival_ant)  # Remove the ant from the global ants list
        self.colony['ants_captured'] += 1  # Increment the ants captured counter
        self.has_resource = True
        self.state = 'transporting'
        # Find path back to colony after capturing
        self.find_path_to_colony(cave_map)

    def transport(self, cave_map, pheromone_map, raiding=False):
        # Leave pheromone trail
        if not raiding:
            self.leave_pheromone(pheromone_map)

        # Check if reached colony
        if self.check_for_colony():
            # Deposit resource and switch back to hunting
            self.colony['resources_returned'] += 4  # Award 4 resources for capturing a soldier ant
            self.has_resource = False
            self.state = 'hunting'  # Switch back to hunting state
            self.path = []
            return

        # Move along the path to the colony
        self.follow_path(cave_map)
