# import numpy as np
# from typing import List, Tuple, Optional

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

#     return None

# class PlannerAgent:
	
# 	def __init__(self):
# 		pass
	
# 	def plan_action(self, world: np.ndarray, current: Tuple[int, int], pursued: Tuple[int, int], pursuer: Tuple[int, int]) -> Optional[np.ndarray]:
# 		"""
# 		Computes a path from the start position to the end position 
# 		using a certain planning algorithm (DFS is provided as an example).

# 		Parameters:
# 		- world (np.ndarray): A 2D numpy array representing the grid environment.
# 		- 0 represents a walkable cell.
# 		- 1 represents an obstacle.
# 		- start (Tuple[int, int]): The (row, column) coordinates of the starting position.
# 		- end (Tuple[int, int]): The (row, column) coordinates of the goal position.

# 		Returns:
# 		- np.ndarray: A 2D numpy array where each row is a (row, column) coordinate of the path.
# 		The path starts at 'start' and ends at 'end'. If no path is found, returns None.
# 		"""
		
# 		directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
#                   	  		   [-1, -1], [-1, 1], [1, -1], [1, 1]]) 

# 		return directions[np.random.choice(9)]
     

import numpy as np
import heapq
from typing import Tuple, Optional, List

class PlannerAgent:
    def __init__(self):
        # Depth of minimax lookahead.This tells how many steps ahead to simulate
        self.lookahead_depth = 3

    def plan_action(
        self,
        world: np.ndarray,
        current: Tuple[int, int],
        pursued: Tuple[int, int],
        pursuer: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        action = self._minimax(world, current, pursued, pursuer, self.lookahead_depth)
        return action

    def _minimax(self, world, current, pursued, pursuer, depth) -> np.ndarray:
        
        # This will be the root of the minimax search. We will simulate the outcome for each 
        # possible move. This function returns the action that maximizes the agent's
        # minimum score.

        # stay in the same place and all 8 directions
        directions = np.array([
            [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
            [-1, -1], [-1, 1], [1, -1], [1, 1]
        ])

        # We will start with the lowest possible score
        best_score = -np.inf
        # We will keep the default to be staying at same place if there is no better move.
        best_action = np.array([0, 0])
        # Trying all possible moves
        for action in directions:
            next_pos = (current[0] + action[0], current[1] + action[1])
            if not self._is_valid(world, next_pos):
                # here we will skip the moves that go into obstacles or out of bounds
                continue
            score = self._min_value(world, next_pos, pursued, pursuer, depth - 1)
            if score > best_score:
                best_score = score
                # If this move is better then update the best action
                best_action = action
        return best_action

    def _min_value(self, world, agent_pos, pursued, pursuer, depth):
        
        # In this function we will perform the below:
        # For each possible move of the pursuer we will assume they minimize the agent's score.
        # We will return the minimal score the agent can be forced into.
        
        # If the depth limit is reached or agent caught or catches the target.
        if depth == 0 or np.array_equal(agent_pos, pursued) or np.array_equal(agent_pos, pursuer):
            return self._evaluate(agent_pos, pursued, pursuer)
        # We will start with the highest possible value. This is because the adversary would minimize.
        min_score = np.inf
        for pursuer_action in self._get_valid_moves(world, pursuer):
            pursuer_next = (pursuer[0] + pursuer_action[0], pursuer[1] + pursuer_action[1])
            # Agent's turn again
            score = self._max_value(world, agent_pos, pursued, pursuer_next, depth - 1)
            if score < min_score:
                # Keep the lowest score
                min_score = score
        return min_score

    def _max_value(self, world, agent_pos, pursued, pursuer, depth):
        
        # In this function we will do the below:
        # For each possible move of the agent we will assume they maximize the agent's score.
        # We will return the maximal score the agent can get.
        
        # If the depth limit is reached or the agent is caught or catches the target.
        if depth == 0 or np.array_equal(agent_pos, pursued) or np.array_equal(agent_pos, pursuer):
            return self._evaluate(agent_pos, pursued, pursuer)
        # We will start with the lowest possible value. This is because the agent wants to maximize.
        max_score = -np.inf
        for agent_action in self._get_valid_moves(world, agent_pos):
            agent_next = (agent_pos[0] + agent_action[0], agent_pos[1] + agent_action[1])
            # Pursuer's turn again
            score = self._min_value(world, agent_next, pursued, pursuer, depth - 1)
            if score > max_score:
                # keeping the highest score
                max_score = score
        return max_score

    def _evaluate(self, agent_pos, pursued, pursuer):

        # This is the heuristic evaluation function for the states that do not terminate.
        # This function returns a score based on the distance to target and the distance from the pursuer.
        # If agent catches the target, a large positive.
        # Large negative if the agent is caught by its pursuer.
        # Otherwise closer to the target and farther from pursuer is better.
        dist_to_pursued = np.linalg.norm(np.array(agent_pos) - np.array(pursued))
        dist_from_pursuer = np.linalg.norm(np.array(agent_pos) - np.array(pursuer))
        if np.array_equal(agent_pos, pursued):
            return 1000  # Win
        if np.array_equal(agent_pos, pursuer):
            return -1000  # Loss
        return -dist_to_pursued + 1.5 * dist_from_pursuer

    def _get_valid_moves(self, world, pos):

        # This function returns a list of valid moves from a given location.
        # Valid if it doesn't hit obstacles and is within bounds.
        directions = [
            [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
            [-1, -1], [-1, 1], [1, -1], [1, 1]
        ]
        valid = []
        for d in directions:
            next_pos = (pos[0] + d[0], pos[1] + d[1])
            if self._is_valid(world, next_pos):
                valid.append(np.array(d))
        return valid

    def _is_valid(self, world, pos):
        
        # Checks if a position is within the grid bounds and not a obstacle.
        rows, cols = world.shape
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols and world[r, c] == 0

    def _a_star(self, world, start, goal, avoid=None) -> Optional[List[Tuple[int, int]]]:
        
        # This is an A* path finding function. This function finds the shortest path from start to goal.
        # It returns the paths as a list of positions.
        rows, cols = world.shape
        heap = []
        # (priority, position)
        heapq.heappush(heap, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        while heap:
            _, current = heapq.heappop(heap)
            if current == goal:
                # Reconstruct the path from goal to start
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                # now return the path from start to goal
                return path[::-1]
            
            # Exploring all possible moves from the current position
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                neighbor = (current[0] + d[0], current[1] + d[1])
                if not self._is_valid(world, neighbor):
                    # skip the invalid moves
                    continue
                # add a penalty for being close to avoid pursuer.
                penalty = 0
                if avoid is not None:
                    penalty = 10 / (1 + np.linalg.norm(np.array(neighbor) - np.array(avoid)))
                tentative_g = g_score[current] + 1 + penalty
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    # Euclidean distance to goal
                    f_score = tentative_g + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(heap, (f_score, neighbor))
                    came_from[neighbor] = current
        # if no path is found
        return None


