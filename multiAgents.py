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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def getAction(self, gameState: GameState):
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
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # no point evaluating if the game is already decided
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        # Get all remaining food positions
        foodList = newFood.asList()

        # reciprocal so that closer food gives a bigger boost
        # tried just using -distance but this scales better when combining features
        closestFood = min(manhattanDistance(newPos, food) for food in foodList)
        foodScore = 10.0 / closestFood

        # Handle ghosts differently based on wheather theyre scared or not
        ghostPenalty = 0
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            dist = manhattanDistance(newPos, ghost.getPosition())
            
            if scaredTime > 0:
                # scared ghost is free points, so we nudge towards it
                ghostPenalty += 100.0 / (dist + 1)
            elif dist < 5:
                # only care about ghosts that are actually close
                # Far ghosts shouldnt affect decision making as they dont matter
                ghostPenalty -= 10.0 / (dist + 1)
        # Combine everything, base score handles food eaten/time penalties already
        return successorGameState.getScore() + foodScore + ghostPenalty

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        #util.raiseNotDefined()
        def minimax(state, depth, agentIndex):
                # Base case - stop if game is over or we've reached our depth limit
                if state.isWin() or state.isLose() or depth == self.depth:
                    return self.evaluationFunction(state)
                
                actions = state.getLegalActions(agentIndex)
                
                # only ++ depth once everyone has taken a turn
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                
                # recursively generate values
                successorValues = [minimax(state.generateSuccessor(agentIndex, a), nextDepth, nextAgent) for a in actions]
                
                if agentIndex == 0:
                    return max(successorValues)
                else:
                    return min(successorValues)
            # root needs to return an action, not a score
            # pacman moves at root so recursive calls start at ghost (index 1)
        actions = gameState.getLegalActions(0)
        return max(actions, key=lambda a: minimax(gameState.generateSuccessor(0, a), 0, 1))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphabeta(state, depth, agentIndex, alpha, beta):
                # Same base case as minimax
                if state.isWin() or state.isLose() or depth == self.depth:
                    return self.evaluationFunction(state)
                
                actions = state.getLegalActions(agentIndex)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                
                if agentIndex == 0:  # Pacman MAX node
                    value = float('-inf')
                    for action in actions:
                        successor = state.generateSuccessor(agentIndex, action)
                        value = max(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                        
                        # If we found something better than what ghost allows, stop
                        if value > beta:
                            return value
                        alpha = max(alpha, value)
                    return value
                
                else:  # Ghost MIN node
                    value = float('inf')
                    for action in actions:
                        successor = state.generateSuccessor(agentIndex, action)
                        value = min(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                        
                        # If we found something worse than what pacman already has, stop
                        if value < alpha:
                            return value
                        beta = min(beta, value)
                    return value
            
        # Root call - start with worst possible alpha/beta
        actions = gameState.getLegalActions(0)
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')
        
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 0, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def expectimax(state, depth, agentIndex):
                # Base case identical to minimax
                if state.isWin() or state.isLose() or depth == self.depth:
                    return self.evaluationFunction(state)
                
                actions = state.getLegalActions(agentIndex)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                
                successorValues = [expectimax(state.generateSuccessor(agentIndex, a),
                                            nextDepth, nextAgent) for a in actions]
                
                if agentIndex == 0:
                    # Pacman still maximizes - he plays optimally
                    return max(successorValues)
                else:
                    # Ghosts are random so we take the average instead of minimum
                    # Uniform distribution means each action equally likely
                    return sum(successorValues) / len(successorValues)
        # Root call identical to minimax
        actions = gameState.getLegalActions(0)
        return max(actions, key=lambda a: expectimax(gameState.generateSuccessor(0, a), 0, 1))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Extract useful information from current state
    pos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # Terminal states
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    # --- Food Feature ---
    # Closer to nearest food is better, reciprocal so closer = higher value
    foodDistances = [manhattanDistance(pos, f) for f in foodList]
    closestFoodScore = 10.0 / min(foodDistances)

    # Penalize having lots of food remaining - we want to eat it all
    remainingFoodPenalty = -4 * len(foodList)

    # --- Ghost Features ---
    ghostScore = 0
    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # Scared ghost = opportunity, get closer for more points
            ghostScore += 200.0 / (dist + 1)
        else:
            # Increased radius from 3 to 5, and stronger penalty
            if dist < 5:
                ghostScore -= 1000.0 / (dist + 1)

    # --- Capsule Feature ---
    # Capsules let us eat ghosts, reward being close to them
    capsuleScore = 0
    if capsules:
        closestCapsule = min(manhattanDistance(pos, c) for c in capsules)
        capsuleScore = 5.0 / closestCapsule

    return score + closestFoodScore + remainingFoodPenalty + ghostScore + capsuleScore


better = betterEvaluationFunction