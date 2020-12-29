# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # print("Score: ", scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # If PacMan is too close to ghost
        if min([manhattanDistance(newPos, gState.getPosition()) for gState in newGhostStates]) <= 3:
            return -1/min([manhattanDistance(newPos, gState.getPosition()) for gState in newGhostStates])
        # Find minimum path to all foods
        else:
            curPos = newPos
            total_distance = 1
            while newFood:
                min_distance, min_pos, min_idx = min([(manhattanDistance(curPos, newF), newF, idx)
                                                      for idx, newF in enumerate(newFood)])
                total_distance += min_distance
                curPos = min_pos
                newFood.pop(min_idx)
            return 1/total_distance            
                


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        best_move, best_move_score = self.minimax(gameState, 0, 0, gameState.getNumAgents())
        return best_move

    def minimax(self, cur_state, cur_depth, cur_agent, num_agents):

        best_move = None

        # Counter for search depth
        if cur_agent >= num_agents:
            cur_agent = 0
            cur_depth += 1

        # Terminate at depth X or if the game is over
        if cur_depth >= self.depth or cur_state.isWin() or cur_state.isLose():
            return best_move, self.evaluationFunction(cur_state)

        # Denotes Pacman (Max)
        if cur_agent == 0:
            best_score = -float("inf")
        # Denotes Ghosts (Min)
        else:
            best_score = float("inf")

        # Iterate through all possible actions
        for move in cur_state.getLegalActions(cur_agent):

            # Find the successor state
            next_state = cur_state.generateSuccessor(cur_agent, move)

            # Recursively find the minimax score of the state
            next_move, next_score = self.minimax(next_state, cur_depth, cur_agent + 1, num_agents)

            # Update the best move and score for Pacman
            if cur_agent == 0 and next_score > best_score:
                best_move, best_score = move, next_score

            # Update the best move and score for Ghost
            if cur_agent != 0 and next_score < best_score:
                best_move, best_score = move, next_score
        return best_move, best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_move, best_move_score = self.AlphaBeta(gameState, -float("inf"), float("inf"), 0, 0, gameState.getNumAgents())
        return best_move

    def AlphaBeta(self, cur_state, alpha, beta, cur_depth, cur_agent, num_agents):


        best_move = None
        # Counter for search depth
        if cur_agent >= num_agents:
            cur_agent = 0
            cur_depth += 1
        # Terminate at depth X or if the game is over
        if cur_depth >= self.depth or cur_state.isWin() or cur_state.isLose():
            return best_move, self.evaluationFunction(cur_state)

        # Denotes Pacman (Max)
        if cur_agent == 0:
            best_score = -float("inf")
        # Denotes Ghosts (Min)
        else:
            best_score = float("inf")

        # Iterate through all possible actions
        for move in cur_state.getLegalActions(cur_agent):

            # Find the successor state
            next_state = cur_state.generateSuccessor(cur_agent, move)

            # Recursively find the minimax score of the state
            next_move, next_score = self.AlphaBeta(next_state, alpha, beta, cur_depth, cur_agent + 1, num_agents)

            # Update the best move, score for Pacman, and alpha
            if cur_agent == 0:
                if next_score > best_score:
                    best_move, best_score = move, next_score

                if best_score >= beta:
                    return best_move, best_score

                alpha = max(alpha, best_score)

            # Update the best move and score for Ghost, and eta
            if cur_agent != 0:
                if next_score < best_score:
                    best_move, best_score = move, next_score

                if best_score <= alpha:
                    return best_move, best_score

                beta = min(beta, best_score)
        return best_move, best_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        best_move, best_move_score = self.ExpectiMax(gameState, 0, 0, gameState.getNumAgents())
        return best_move

    def ExpectiMax(self, cur_state, cur_depth, cur_agent, num_agents):

        best_move = None
        # Counter for search depth
        if cur_agent >= num_agents:
            cur_agent = 0
            cur_depth += 1

        # Terminate at depth X or if the game is over
        if cur_depth >= self.depth or cur_state.isWin() or cur_state.isLose():
            return best_move, self.evaluationFunction(cur_state)

        # Denotes Pacman (Max)
        if cur_agent == 0:
            best_score = -float("inf")

        # Denotes Ghosts (Min)
        else:
            best_score = 0

        # Iterate through all possible actions
        for move in cur_state.getLegalActions(cur_agent):

            # Find the successor state
            next_state = cur_state.generateSuccessor(cur_agent, move)

            # Recursively find the minimax score of the state
            next_move, next_score = self.ExpectiMax(next_state, cur_depth, cur_agent + 1, num_agents)

            # Update the best move, score for Pacman
            if cur_agent == 0:
                if next_score > best_score:
                    best_move, best_score = move, next_score

            # Update the best score for Ghost using probability
            if cur_agent != 0:
                best_score += next_score * 1.0/len(cur_state.getLegalActions(cur_agent))
        return best_move, best_score


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # (x, y) of current position
    # (x, y) position collection of food
    # (x, y) position collection of capsules
    # (x, y) position collection of ghosts
    # collection of scared ghosts time

    cur_Pos = currentGameState.getPacmanPosition()
    remain_Food = currentGameState.getFood().asList()
    remain_Capsules = currentGameState.getCapsules()

    remain_GhostStates = currentGameState.getGhostStates()
    remain_GhostPos = [ghostState.getPosition() for ghostState in remain_GhostStates]
    remain_GhostTime = [ghostState.scaredTimer for ghostState in remain_GhostStates]

    # If win return score
    if currentGameState.isWin():
        return currentGameState.getScore()
    # If lose return -score
    elif currentGameState.isLose():
        return currentGameState.getScore()

    # Evaluation Function
    else:

        # Assign weights for features (food, capsule, ghost, scared_ghost)
        food_weight = len(remain_Food)
        capsule_weight = 10
        ghost_weight = 150
        scared_ghost_weight = 100

        ghost_score = 0
        scared_ghost_score = 0

        # Ghost
        for idx, ghost_Pos in enumerate(remain_GhostPos):
            # Non-Scared Ghost
            if manhattanDistance(cur_Pos, ghost_Pos) <= 3 and remain_GhostTime[idx] == 0:
                ghost_score += manhattanDistance(cur_Pos, ghost_Pos)
            # Feasible to eat Scared Ghost (Manhattan Distance less than time)
            elif remain_GhostTime[idx] >= manhattanDistance(cur_Pos, ghost_Pos):
                scared_ghost_score += manhattanDistance(cur_Pos, ghost_Pos)

        # Calculate ghost and scared ghost score based on weight
        ghost_score = ghost_weight / ghost_score if ghost_score > 0 else 0
        scared_ghost_score = scared_ghost_weight / scared_ghost_score if scared_ghost_score > 0 else 0

        # Capsule
        new_Pos = cur_Pos
        capsule_score = 0

        # Find the closest capsule
        if remain_Capsules and max(remain_GhostTime) == 0:
            min_distance, min_pos, min_idx = min([(manhattanDistance(new_Pos, newC), newC, idx)
                                                  for idx, newC in enumerate(remain_Capsules)])
            capsule_score += min_distance

        capsule_score = capsule_weight/capsule_score if capsule_score > 0 else 0
        # To implement the logic that eating a capsule is better than having a distance of 1 close to a capsule
        capsule_score += (len([t for t in remain_GhostTime if t != 0]) * 1.5 * capsule_weight)

        # Food
        new_Pos = cur_Pos
        food_score = 0

        # Eat all the foods
        # while remain_Food:
        #     min_distance, min_pos, min_idx = min([(manhattanDistance(new_Pos, newF), newF, idx)
        #                                           for idx, newF in enumerate(remain_Food)])
        #     food_score += min_distance
        #     new_Pos = min_pos
        #     remain_Food.pop(min_idx)

        food_score += mstHeuristic(new_Pos, remain_Food)

        food_score = food_weight / food_score if food_score > 0 else 0

        # Return a linear combination of the scores
        return capsule_score + food_score - ghost_score + scared_ghost_score + currentGameState.getScore()

def mstHeuristic(position, foodGridList):

    # Create a dictionary with <position, index> pair
    foodGridDict = dict(zip(foodGridList, range(len(foodGridList))))

    # Construct a Matrix that stores the position's index and the manhattan distance
    foodGridMatrix = [[0 for i in range(len(foodGridList))] for j in range(len(foodGridList))]

    # Populate the Matrix
    for i in range(len(foodGridMatrix)):
        for j in range(len(foodGridMatrix)):
            if i <= j:
                foodGridMatrix[i][j] = manhattanDistance(foodGridList[i], foodGridList[j])
                foodGridMatrix[j][i] = foodGridMatrix[i][j]

    # Use Kruskal's algorithm to construct a minimum spanning tree
    # Get all the vertices (current position and food positions)
    fGList = foodGridList + [position]
    fGMatrix = foodGridMatrix
    fGDict = foodGridDict

    # Construct a list [v1, v2, w] that stores two vertices and the weight of the edge
    food_graph = []
    for i in fGList:
        for j in fGList:
            if i >= j:
                continue
            else:
                if j != position and i != position:
                    food_graph.append([i, j, fGMatrix[fGDict[i]][fGDict[j]]])
                else:
                    food_graph.append([i,j, manhattanDistance(i,j)])

    # Sort all edges by weight (manhattan distance)
    food_graph = sorted(food_graph, key=lambda x:x[2])

    # Parent and Rank to Keep track of cycles
    reprSet = {}
    rank = {}

    # Make Set
    for pos in fGList:
        reprSet[pos] = pos
        rank[pos] = 0

    num_edge = 0
    i = 0
    total_distance = 0

    # Minimum Spanning Tree has V - 1 (# of vertices - 1) edges
    while num_edge < len(fGList) - 1:
        v1, v2, weight = food_graph[i]

        # Track the set the vertices are located in
        v1repr = findSetRepr(reprSet, v1)
        v2repr = findSetRepr(reprSet, v2)

        # If they do not belong to the same set(a.k.a forming a cycle)
        if v1repr != v2repr:
            # Add the weight to total distance
            total_distance += weight

            # Union Set (Bring two vertices together)
            if rank[v1repr] < rank[v2repr]:
                reprSet[v1repr] = v2repr
                rank[v2repr] += 1
            else:
                reprSet[v2repr] = v1repr
                rank[v1repr] += 1
            num_edge += 1
        i += 1
    return total_distance


def findSetRepr(ReprSet, node):
    if ReprSet[node] == node:
        return node
    return findSetRepr(ReprSet, ReprSet[node])

# Abbreviation
better = betterEvaluationFunction




