from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1a: Solve the segmentation problem under a unigram model
# python submission.py --model seg --text-corpus leo-will.txt
class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.query   
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == ''    
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)

        # action : state[:i], remain_part : state[i:], cost : unigramCost(action)
        return [(state[:i], state[i:], self.unigramCost(state[:i])) for i in range(1, len(state)+1)]
    
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    # TMI : Callable[[Arg1Type, Arg2Type, ...], ReturnType]
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    
    # 최소 totalCost로 end State에 도달하는 경로 출력    
    return ' '.join(ucs.actions)

    # END_YOUR_CODE


############################################################
# Problem 1b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        # state 정의 : (처리된 index, 처리된 단어)
        return (-1, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0]+1 >= len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)

        index, lastword = state
        queryWord = self.queryWords[index+1] 
        fillset = self.possibleFills(queryWord) or {queryWord}
        return [(word, (index+1, word), self.bigramCost(lastword, word)) for word in fillset]

        # END_YOUR_CODE


def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions) if ucs.actions else ''

    # END_YOUR_CODE


############################################################
# Problem 1c: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (self.query, wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == ''
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        
        query_sentence, lastword = state
        results = []
        for i in range(1, len(query_sentence) + 1) :
            word = query_sentence[:i]
            remain = query_sentence[i:]
            fillset = self.possibleFills(word) or {word}
            for action in fillset:
                results.append((action, (remain, action), self.bigramCost(lastword, action)))
        return results

        # END_YOUR_CODE


def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_CODE


############################################################
# Problem 2a: Solve the maze search problem with uniform cost search

class MazeProblem(util.SearchProblem):
    def __init__(self, start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
        self.start = start
        self.goal = goal
        self.moveCost = moveCost
        self.possibleMoves = possibleMoves

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.goal
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)            
        return [(direction, next, self.moveCost(state, direction)) for direction, next in self.possibleMoves(state)]
        # END_YOUR_CODE
            

def UCSMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves))
    
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    return ucs.totalCost
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the maze search problem with A* search

def consistentHeuristic(goal: tuple):
    def _consistentHeuristic(state: tuple) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return abs(goal[0]-state[0]) + abs(goal[1]-state[1])
        # END_YOUR_CODE
    return _consistentHeuristic

def AStarMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves), heuristic=consistentHeuristic(goal))
    
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    total_cost = 0
    for action in ucs.actions :
        total_cost += moveCost.moveCost[action]
    return total_cost
    # END_YOUR_CODE

############################################################


if __name__ == '__main__':
    shell.main()
