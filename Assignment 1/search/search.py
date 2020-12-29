# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def genericSearch(problem, OPEN, cycle_checking):

    # Add Start State Path
    START = [(problem.getStartState(), "START", 0)]
    OPEN.push(START)

    # To Keep track of all previously expanded states if we are using cycle checking
    expanded = set()

    while not OPEN.isEmpty():
        # Remove Path from OPEN
        path = OPEN.pop()
        # Get Current State
        cur_state = path[-1][0]
        # Check if the State is a Goal State and return the actions (Exclude the "START" direction)
        if problem.isGoalState(cur_state):
            return [state[1] for state in path[1:]]
        # Full Cycle Checking
        if cycle_checking:
            # Expand Iff the Current State has not been expanded
            if cur_state not in expanded:
                # Expand and Add Successors
                for succ in problem.getSuccessors(cur_state):
                    # Expand iff if the Successor state has not been expanded
                    if succ[0] not in expanded:
                        OPEN.push(path + [succ])
                # Mark the state as expanded
                expanded.add(cur_state)
        # Path Checking
        else:
            # Add Successors
            for succ in problem.getSuccessors(cur_state):
                # Check if any ancestors contain the successor state
                if succ[0] not in [state[0] for state in path]:
                    # Push the Successor path to the OPEN set
                    OPEN.push(path + [succ])

    # if No solution found
    return []

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Construct a Stack
    OPEN = util.Stack()
    return genericSearch(problem, OPEN, False)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # Construct a Queue
    OPEN = util.Queue()
    return genericSearch(problem, OPEN, True)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # Define a cost function to calculate the cost of all past actions
    def costFunc(path):
        return problem.getCostOfActions([state[1] for state in path[1:]])

    # Construct a Priority Queue that uses Action Cost as priority
    OPEN = util.PriorityQueueWithFunction(costFunc)

    return genericSearch(problem, OPEN, True)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Define a cost function to calculate the cost of all past actions plus the heuristic
    # f(x) = g(x) * 0.9999 + h(x) (the coefficient is used to break ties effectively)
    def costFunc(path):
        return problem.getCostOfActions([state[1] for state in path[1:]]) * 0.9999 + heuristic(path[-1][0], problem)

    # Construct a Priority Queue that uses Action Cost as priority
    OPEN = util.PriorityQueueWithFunction(costFunc)

    return genericSearch(problem, OPEN, True)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
