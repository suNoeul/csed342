import heapq
from typing import Tuple, List


############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return whether |state| is an end state or not.
    def isEnd(self, state) -> bool: raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state): raise NotImplementedError("Override me")


class SearchAlgorithm:
    # First, call solve on the desired SearchProblem |problem|.
    # Then it should set two things:
    # - self.actions: list of actions that takes one from the start state to an end
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem: SearchProblem): raise NotImplementedError("Override me")


############################################################
# Uniform cost search algorithm (Dijkstra's algorithm).

class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    # TMI : UCS Algorithm은 h(s) = 0인 A* Algorithm의 특수한 Case이다.
    def solve(self, problem: SearchProblem, heuristic=lambda x: 0):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.startState()
        frontier.update(startState, 0)

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, pastCost = frontier.removeMin()
            if state == None: break
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print(("Exploring %s with pastCost %s" % (state, pastCost)))

            # Check if we've reached an end state; if so, extract solution.
            if problem.isEnd(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print(("numStatesExplored = %d" % self.numStatesExplored))
                    print(("totalCost = %s" % self.totalCost))
                    print(("actions = %s" % self.actions))
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.succAndCost(state):
                if self.verbose >= 3:
                    print(("  Action %s => %s with cost %s + %s" % (action, newState, pastCost, cost)))
                if frontier.update(newState, pastCost + cost + heuristic(newState) - heuristic(state)):
                    # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)
        if self.verbose >= 1:
            print("No path found")


# Data structure for supporting uniform cost search.
class PriorityQueue:
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority: int) -> bool:
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return state, priority
        return None, None  # Nothing left...


# -----------------------
# For Maze search problem
# -----------------------

directions = {'LEFT': (0,-1), 'RIGHT': (0, 1), 'UP': (-1, 0), 'DOWN': (1, 0)}

class CalculateMoveCost():
    def __init__(self, map, moveCost={'LEFT': 1, 'RIGHT': 1, 'UP': 1, 'DOWN': 1}):
        self.map = map
        self.moveCost = moveCost

    def __call__(self, state, direction):
        assert direction in self.moveCost.keys(), f"Move should be one of {self.moveCost.keys()}! Got {direction}."
        
        action = directions[direction]

        new_state = (state[0] + action[0], state[1] + action[1])

        if self.map[new_state[0]][new_state[1]] == 1:
            return 99999999
        return self.moveCost[direction]

class FindPossibleMoves():
    def __init__(self, map, directions=['LEFT', 'RIGHT', 'UP', 'DOWN']):
        self.map = map
        self.directions = directions

    def __call__(self, state):
        result = set()

        for direction in self.directions:
            action = directions[direction]
            next_state = (state[0] + action[0], state[1] + action[1])

            if next_state[0] < len(self.map) and next_state[0] >= 0:
                if next_state[1] < len(self.map[0]) and next_state[1] >= 0:
                    result.add((direction, next_state))

        return result