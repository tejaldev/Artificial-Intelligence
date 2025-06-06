
import numpy as np
from typing import Tuple, Optional

class PlannerAgent:
    def __init__(self):
        # Default probabilities are [p_left, p_straight, p_right]. This was given to us in the project requirements.
        # These are used to find out how often an action stays as it is or gets rotated.
        self.current_prob = [0.3, 0.3, 0.4]  # Initialized with the same values from main.py

    def update_probabilities(self, prob):
        # Here we will update probabilities dynamically before each game
        # These values will be recieved from main.py
        self.current_prob = prob  

    def plan_action(
        self,
        world: np.ndarray,  
        current: Tuple[int, int],  
        pursued: Tuple[int, int],  
        pursuer: Tuple[int, int]   
    ) -> Optional[np.ndarray]:
        
        # This function decides the best action to take based on what the current state is. To do so it
        # will use a heuristic function and uncertainty awareness. 
        # This function returns a numpy array [dx, dy]. It represents the selected move direction.

        # All possible actions: 8 directions and staying still
        directions = [
            np.array([0, 0]),  # Staying still
            np.array([-1, 0]), np.array([1, 0]),  # Up, Down
            np.array([0, -1]), np.array([0, 1]),  # Left, Right
            np.array([-1, -1]), np.array([-1, 1]),  # Diagonals
            np.array([1, -1]), np.array([1, 1])
        ]

        best_score = -np.inf  # This is initialized to a very low value to find max.
        best_action = np.array([0, 0])  # Here we will set our default action to be staying in place

        # Here we will evaluate each possible action to find the optimal one.
        for action in directions:
            # Considering uncertainty we will calculate expected action. We will apply uncertainty to
            # calculate the expected effect of this action.
            expected_action = self._expected_action(action)

            # Now based on the expected action we will predict the next position
            next_pos = (current[0] + expected_action[0], current[1] + expected_action[1])

            # Check to see if the move is valid. It should not be outside the grid or in obstacles.
            if not self._is_valid(world, next_pos):
                continue  # Skip the invalid moves.

            # Now we will evaluate the heuristic score for this potential move
            score = self._heuristic(next_pos, pursued, pursuer)

            # We should keep the move with the highest heuristic score. We will update the best action 
            # if this move has a higher heuristic score. 
            if score > best_score:
                best_score = score
                best_action = action  # Here we will choose the original action and not the expected one

        return best_action  # We will return the final selected move.

    def _expected_action(self, action):
        # In this function we will compute the expected action outcome under uncertainty. 
        # This function returns numpy array representing the expected action after taking into 
        # consideration the uncertainty. 
        p1, p2, p3 = self.current_prob  # Probabilities for left, straight, right

        # Define the rotated versions of the action
        straight = action  # No rotation in this one
        left = np.array([-action[1], action[0]])  # Rotate Left by 90°
        right = np.array([action[1], -action[0]])  # Rotate Right by 90°

        # We will compute the expected action using the probabilities
        expected = p1 * left + p2 * straight + p3 * right

        # Since it is a grid, we will round to the nearest valid grid direction
        return np.round(expected).astype(int)

    def _heuristic(self, agent_pos, pursued, pursuer):
        
        # This is the heuristic function to evaluate the quality of a move. We encourage getting closer to
        # the pursued agent, staying further from the pursuer, and we will weight the distance to the 
        # pursuer more heavily. 
        dist_to_pursued = np.linalg.norm(np.array(agent_pos) - np.array(pursued))
        dist_from_pursuer = np.linalg.norm(np.array(agent_pos) - np.array(pursuer))

        # Here we will calculate the final heuristic score.
        return -dist_to_pursued + 2.0 * dist_from_pursuer

    def _is_valid(self, world, pos):
        """Check if a position is within grid bounds and not blocked by an obstacle."""
        # This function will check if a position is within our grid bounds and is not blocked by an
        #obstacle.
        rows, cols = world.shape
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols and world[r, c] == 0  
