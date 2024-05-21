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
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

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

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER

    # Collect legal moves and successor states 
    legalMoves = gameState.getLegalActions(0)

    # Choose one of the best actions
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # print("Best minimax value at depth {}: {}".format(self.depth, bestScore))
    return legalMoves[chosenIndex]

    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER

    # Define minimix
    def minimax(agentIndex, depth, state):
      # Case 1. IsEnd(s) : return Utility(s)
      if state.isWin() or state.isLose() :
        return state.getScore()
           
      # Case 2. d = 0 : return Eval(s)
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agentIndex)
      nextIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextIndex == 0 else depth

      # Case 3. Player(s) = Agent : return max V(Succ(s,a), d) 
      if agentIndex == self.index : 
        return max(minimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions)
      
      # Case 4. Player(s) = Opp : return min V(Succ(s,a), d)
      else : 
        return min(minimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions)
    
    # Start minimax from the first ghost
    return minimax(1, self.depth, gameState.generateSuccessor(0, action))

    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    
    # Collect legal moves and successor states 
    legalMoves = gameState.getLegalActions(0)

    # Choose one of the best actions
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # print("Best minimax value at depth {}: {}".format(self.depth, bestScore))
    return legalMoves[chosenIndex]
    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    
    # Define minimix
    def expectimax(agentIndex, depth, state):
      # Case 1. IsEnd(s) : return Utility(s)
      if state.isWin() or state.isLose() :
        return state.getScore()
           
      # Case 2. d = 0 : return Eval(s)
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agentIndex)
      nextIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextIndex == 0 else depth

      # Case 3. Player(s) = Agent : return max V(Succ(s,a), d) 
      if agentIndex == self.index : 
        return max(expectimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions)
      
      # Case 4. Player(s) = Opp : return expected value
      else : 
        return sum(expectimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions) / len(actions)
    
    # Start minimax from the first ghost
    return expectimax(1, self.depth, gameState.generateSuccessor(0, action))

    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing stop-biasedly from their legal moves.
    """

    # BEGIN_YOUR_ANSWER
    legalMoves = gameState.getLegalActions(0)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def biasedexpectimax(agentIndex, depth, state):
      # Case 1. IsEnd(s) : return Utility(s)
      if state.isWin() or state.isLose() :
        return state.getScore()
           
      # Case 2. d = 0 : return Eval(s)
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agentIndex)
      nextIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextIndex == 0 else depth

      # Case 3. Player(s) = Agent : return max V(Succ(s,a), d) 
      if agentIndex == self.index : 
        return max(biasedexpectimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions)
      
      # Case 4. Player(s) = Opp : return Biased expected value
      else : 
        Weight = [0.5 * (1/len(actions)) if action != Directions.STOP else 0.5 + 0.5 * (1/len(actions)) for action in actions]
        return sum(biasedexpectimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) * w for action, w in zip(actions, Weight)) 
    
    # Start minimax from the first ghost
    return biasedexpectimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
      The even-numbered ghost should be modeled as choosing uniformly at random from their legal moves.
    """

    # BEGIN_YOUR_ANSWER
    legalMoves = gameState.getLegalActions(0)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    # print("Best minimax value at depth {}: {}".format(self.depth, bestScore))
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectiminimax(agentIndex, depth, state):
      # Case 1. IsEnd(s) : return Utility(s)
      if state.isWin() or state.isLose() :
        return state.getScore()
           
      # Case 2. d = 0 : return Eval(s)
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agentIndex)
      nextIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextIndex == 0 else depth

      # Case 3. Player(s) = Agent : return max V(Succ(s,a), d) 
      if agentIndex == self.index : 
        return max(expectiminimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions)
      
      # Case 4. Player(s) = Opp : return expected minmax value
      else : 
        if agentIndex % 2 == 0 : # Even case
          return sum(expectiminimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions) / len(actions)
        else : # Odd case
          return min(expectiminimax(nextIndex, nextDepth, state.generateSuccessor(agentIndex, action)) for action in actions)
            
    return expectiminimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    legalMoves = gameState.getLegalActions(0)
    scores = [self.getQ(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    # print("Best minimax value at depth {}: {}".format(self.depth, bestScore))
    return legalMoves[chosenIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def alphabeta(agentIndex, depth, alpha, beta, state) :
      # Case 1. IsEnd(s) : return Utility(s)
      if state.isWin() or state.isLose() :
        return state.getScore()
           
      # Case 2. d = 0 : return Eval(s)
      if depth == 0:
        return self.evaluationFunction(state)
      
      actions = state.getLegalActions(agentIndex)
      nextIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextIndex == 0 else depth
      

      # Case 3. Player(s) = Agent : return max V(Succ(s,a), d) 
      if agentIndex == self.index : 
        value = float('-inf')
        for action in actions :
          value = max(value, alphabeta(nextIndex, nextDepth, alpha, beta, state.generateSuccessor(agentIndex, action)))
          alpha = max(alpha, value)
          if beta <= alpha : 
            break
        return value
      # Case 4. Player(s) = Opp : return expected minmax value
      else :
        value = float('inf')        
        for action in actions:
          if agentIndex % 2 == 0 : # Even case
            return sum(alphabeta(nextIndex, nextDepth, alpha, beta, state.generateSuccessor(agentIndex, action)) for action in actions) / len(actions)
          else : # Odd case   
            value = min(value, alphabeta(nextIndex, nextDepth, alpha, beta, state.generateSuccessor(agentIndex, action)))
            beta = min(beta, value)
            if beta <= alpha:
                break  # Alpha prune
        return value
      
    alpha = float('-inf')
    beta  = float('inf')
    return alphabeta(1, self.depth, alpha, beta, gameState.generateSuccessor(0, action))
    
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER

  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  scared = [ghost for ghost in newGhostStates if ghost.scaredTimer]
  if scared :
    minghostDist = min(map(lambda x : manhattanDistance(newPos, x.getPosition()), scared))
  else :
    minghostDist = 0

  anyGhostScared = any(time > 0 for time in newScaredTimes)
  capsules = currentGameState.getCapsules()
  foods = newFood.asList()
  capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in capsules]
  foodDistances = [manhattanDistance(newPos, food) for food in foods]
  
  score = currentGameState.getScore()
  # if part : newScaredTimes가 0 이상이 경우 -> 음식이나 캡슐 향해 공격적 진행
  if anyGhostScared :
    # 1-1. 캡슐이 1개 이상 있는 경우 : 
    if capsules :
      nearestCapsuleDistance = min(capsuleDistances)
      score += 50 / (nearestCapsuleDistance + 1) 
    # 1-2. 캡슐을 다 먹어서 맵에 없는 경우 : 음식을 향해 이동
    elif foods :
      nearestFoodDistance = min(foodDistances) 
      score += 100 / (nearestFoodDistance + 1) 
    score += 200 / (minghostDist + 1)
  # else part : newScaredTimes가 활성화 되지 않은 경우
  else :
    # 2. foodDistance와 CapsuleDistance를 반영
    if capsules:
      nearestCapsuleDistance = min(capsuleDistances)
      score += 200 / (nearestCapsuleDistance + 1)  
    nearestFoodDistance = min(foodDistances)
    score += 50 / (nearestFoodDistance + 1)  
  
  return score
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  
  return 'AlphaBetaAgent'

  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

# PS C:\Users\PC\Desktop\AI(CSED342)\HW5> python pacman.py -l smallClassic -p ExpectiminimaxAgent -a evalFn=better -c -n 20 -q
# Pacman emerges victorious! Score: 966
# Pacman died! Score: 413
# Pacman emerges victorious! Score: 1628
# Pacman emerges victorious! Score: 1557
# Pacman emerges victorious! Score: 1024
# Pacman emerges victorious! Score: 1448
# Pacman emerges victorious! Score: 1245
# Pacman died! Score: 129
# Pacman emerges victorious! Score: 1366
# Pacman emerges victorious! Score: 1532
# Pacman emerges victorious! Score: 1408
# Pacman emerges victorious! Score: 1631
# Pacman emerges victorious! Score: 1457
# Pacman died! Score: 54
# Pacman died! Score: -23
# Pacman emerges victorious! Score: 1235
# Pacman emerges victorious! Score: 1326
# Pacman emerges victorious! Score: 1445
# Pacman emerges victorious! Score: 1319
# Pacman emerges victorious! Score: 1327
# Average Score: 1124.35
# Scores:        966, 413, 1628, 1557, 1024, 1448, 1245, 129, 1366, 1532, 1408, 1631, 1457, 54, -23, 1235, 1326, 1445, 1319, 1327
# Win Rate:      16/20 (0.80)
# Record:        Win, Loss, Win, Win, Win, Win, Win, Loss, Win, Win, Win, Win, Win, Loss, Loss, Win, Win, Win, Win, Win
