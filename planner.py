import numpy as np
from typing import List, Tuple, Optional
import scipy
import heapq

# def dfs(grid, start, end):
#     """A DFS example"""
#     rows, cols = len(grid), len(grid[0])
#     stack = [start]
#     visited = set()
#     parent = {start: None}

#     # Consider all 8 possible moves (up, down, left, right, and diagonals)
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
#                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal moves

#     while stack:
#         x, y = stack.pop()
#         if (x, y) == end:
#             # Reconstruct the path
#             path = []
#             while (x, y) is not None:
#                 path.append((x, y))
#                 if parent[(x, y)] is None:
#                     break  # Stop at the start node
#                 x, y = parent[(x, y)]
#             return path[::-1]  # Return reversed path

#         if (x, y) in visited:
#             continue
#         visited.add((x, y))

#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0 and (nx, ny) not in visited:
#                 stack.append((nx, ny))
#                 parent[(nx, ny)] = (x, y)

#     return None  # Return None if no path is found

def heuristic(a:Tuple[int, int], b:Tuple[int, int])->float:

    return np.sqrt((b[0] - a[0])**2 + (b[1]-a[1])**2)

def get_neighbors(current: Tuple[int, int], world: np.ndarray) -> List[Tuple[int, int]]:
    rows, cols = world.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = current[0] + dx, current[1] + dy
        if 0 <= nx < rows and 0 <= ny < cols and world[nx, ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def plan_path(world: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Computes a path from the start position to the end position 
    using a certain planning algorithm (DFS is provided as an example).

    Parameters:
    - world (np.ndarray): A 2D numpy array representing the grid environment.
      - 0 represents a walkable cell.
      - 1 represents an obstacle.
    - start (Tuple[int, int]): The (row, column) coordinates of the starting position.
    - end (Tuple[int, int]): The (row, column) coordinates of the goal position.

    Returns:
    - np.ndarray: A 2D numpy array where each row is a (row, column) coordinate of the path.
      The path starts at 'start' and ends at 'end'. If no path is found, returns None.
    """
    # Ensure start and end positions are tuples of integers
    # start = (int(start[0]), int(start[1]))
    # end = (int(end[0]), int(end[1]))

    # Convert the numpy array to a list of lists for compatibility with the example DFS function
    # world_list: List[List[int]] = world.tolist()

    # Perform DFS pathfinding and return the result as a numpy array
    # path = dfs(world_list, start, end)

    start = tuple(map(int, start))
    end = tuple(map(int, end))
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return np.array(path[::-1])
        
        for neighbor in get_neighbors(current, world):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None
    # return np.array(path) if path else None
