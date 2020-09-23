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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #Total score determined by a hierarchy of factors:
        # Lowest: How far away the closest ghost is, we want the ghost to be far
        # Low: How far away the nearest food is, we want it to be close
        # High: How many food are left, we want it to be as low as possible
        # Highest: If a ghost is right on top of us
        #By combining these four we get the currentScore
        currentScore = 0
        timeScore = 0

        if min(newScaredTimes) > 0:
            timeScore = min(newScaredTimes) * -10

        #Find the distance to the nearest ghost and initialie ghost values
        ghostPositions = []
        ghostScore = 0
        for ghost in newGhostStates:
            ghostPositions.append(manhattanDistance(newPos, ghost.getPosition()))
        minGhostPosition = min(ghostPositions)

        #Set the ghost score
        #if the ghost will be on top of pacman, RUN AWAY
        if (minGhostPosition == 0):
            #Highest - Set to the lowest possible value in the code to denote highest significance
            return -1000000
        #else set the ghost score to the minimum ghost position
        else:
            #Lowest - Nothing is changed about it to denote Lowest in the hierarchy
            ghostScore = minGhostPosition
        
        #Set the food score
        foodPositions = []
        foodSize = 0
        foodScore = 0
        for food in newFood.asList():
            foodPositions.append(manhattanDistance(newPos, food))
            foodSize += 1

        #IF there is food remaining, find the closest one
        if foodSize > 0:
            foodScore = min(foodPositions)

        #Low - Multiplying this value by -2 makes it higher in priority than the ghostPosition by 2
        #This allows the foodScore to be more important than the ghostScore, but low enough to be
        #affected by it if ghostScore is large enough (tried this with different values, but -2 seems to work the best)
        foodScore *= -2
        #High - Multiplying this value by 1000 makes it higher in priority than the foodPosition by a lot and
        #enough to offset subtracting the ghostScore
        foodScore -= 1000 * foodSize

        # (Negative timeScore) + (Negative food score) - (Negative ghostScore) = Negative current score (most of the time), positive when ghostScore is massive
        #Don't know exactly why, but 10/ghostScore resulted in a better final result
        currentScore = timeScore + foodScore - 10/ghostScore
        return currentScore

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

        # get available actions for pacman in this state
        legalPacmanActions = gameState.getLegalActions(0)

        # get successor states for each of those actions
        sucessorStates = []
        for action in legalPacmanActions:
            sucessorStates.append((action, gameState.generateSuccessor(0, action)))

        # set default best action value, we want this to be an unachievable low number
        bestAction = ("Left", -1000000)


        # for each of those successor states we need it's value
        for state in sucessorStates:
            actionValue = self.minimax(state[1], 1, 0)
            if actionValue > bestAction[1]:
                bestAction = (state[0], actionValue)
        return bestAction[0]

    def minimax(self, gameState, agentIndex, currentDepth):
        """Get the minimax value of this state"""
        # If the gameState is a terminal state then we can find it's score
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        # If the agentIdx is pacman we need it's maximum value of the agent states
        if agentIndex == 0:
            return self.maxStateValue(gameState, currentDepth)
        # If it's not pacman then we need it's minimum values
        else:
            return self.minStateValue(gameState, agentIndex, currentDepth)


    def maxStateValue(self, gameState, currentDepth):
        # Again get all legalActions in the given gameState for the Agent ( always pacman )
        legalPacmanActions = gameState.getLegalActions(0)

        # Again get all of the successor states for each action
        sucessorStates = []
        for action in legalPacmanActions:
            sucessorStates.append((action, gameState.generateSuccessor(0, action)))

        # find the maximum minimax value of the successor states agents
        # we want to initialize this return value as unachievable low so it is
        # set in first comparison
        returnValue = -10000.0
        for successor in sucessorStates:
            returnValue = max(returnValue, self.minimax(successor[1], 1, currentDepth))
        return returnValue


    def minStateValue(self, gameState, agentIdx, currentDepth):
        # Again get all legalActions in the given gameState for the Agent ( always pacman )
        legalActions = gameState.getLegalActions(agentIdx)

        # Again get all of the successor states for each action
        sucessorStates = []
        for action in legalActions:
            sucessorStates.append((action, gameState.generateSuccessor(agentIdx, action)))


        # find the minimum minimax value of the successor states agents or increment depth
        # we want to initialize this return value as unachievable high so it is
        # set in first comparison as well
        returnValue = 10000.1
        for successor in sucessorStates:
            if agentIdx == gameState.getNumAgents() - 1:
                returnValue = min(returnValue, self.minimax(successor[1], 0, currentDepth + 1))
            else:
                returnValue = min(returnValue, self.minimax(successor[1], agentIdx + 1, currentDepth))
        return returnValue




#The below section utilizes both the pseudocode from the lecture and ideas from:
#https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python/ and
#https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agentwith alpha-beta pruning (question 3)
    """
    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        
        def alphaBeta(gameState,agent,depth, alpha, beta):
            v = []

            # Terminate state #
            if not gameState.getLegalActions(agent) or depth == self.depth:
                return self.evaluationFunction(gameState),0

            # All ghosts have finised one round: increase depth #
            if agent == gameState.getNumAgents() - 1:
                depth += 1
                nextAgent = self.index
            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if v == []:
                    successor = alphaBeta(gameState.generateSuccessor(agent, action), nextAgent, depth, alpha, beta)

                    v.append(successor[0])
                    v.append(action)

                    if agent == self.index:
                        alpha = max(v[0], alpha)
                    else:
                        beta = min(v[0], beta)
                else:
                    #max-value method from lecture
                    if v[0] > beta and agent == self.index:
                        return v
                    #min-value method from lecture
                    elif v[0] < alpha and agent != self.index:
                        return v
                    
                    successor = alphaBeta(gameState.generateSuccessor(agent, action), nextAgent, depth, alpha, beta)

                    #current position equals pacman's original position
                    if agent == self.index:
                        if successor[0] > v[0]:
                            v[0] = successor[0]
                            v[1] = action
                            #double checking the max alpha value
                            alpha = max(v[0],alpha)
                    else:
                        if successor[0] < v[0]:
                            v[0] = successor[0]
                            v[1] = action
                            #double checking the mind beta value
                            beta = min(v[0],beta)
            return v
        # Call AB with initial depth = 0 and -inf and +inf (a,b) values
        return alphaBeta(gameState,self.index,0, -float("inf"), float("inf"))[1]


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
