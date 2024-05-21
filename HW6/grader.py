#!/usr/bin/env python
"""
Grader for template assignment
Optionally run as grader.py [basic|all] to run a subset of tests
"""

import random

import graderUtil
import util
import collections
import copy
grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False

############################################################
# Problem 1a: Simple Chain CSP

def test1a():
    solver = submission.BacktrackingSearch()
    solver.solve(submission.create_chain_csp(4))
    grader.requireIsEqual(1, solver.optimalWeight)
    grader.requireIsEqual(2, solver.numOptimalAssignments)
    grader.requireIsEqual(9, solver.numOperations)

grader.addBasicPart('1a-1-basic', test1a, 2, maxSeconds=1, description="Basic test for create_chain_csp")

def get_csp_result(csp, BacktrackingSearch=None, **kargs):
    if BacktrackingSearch is None:
        BacktrackingSearch = (solution.BacktrackingSearch if solution_exist else
                              submission.BacktrackingSearch)
    solver = BacktrackingSearch()
    solver.solve(csp, **kargs)
    return (solver.optimalWeight,
            solver.numOptimalAssignments,
            solver.numOperations)

def test1a():
    pred = get_csp_result(submission.create_chain_csp(6))
    if solution_exist:
        grader.requireIsEqual(get_csp_result(solution.create_chain_csp(6)), pred)

grader.addHiddenPart('1a-1-hidden', test1a, 1, maxSeconds=1, description="Hidden test for create_chain_csp")

############################################################
# Problem 2a: N-Queens

def test2a_1():
    nQueensSolver = submission.BacktrackingSearch()
    nQueensSolver.solve(submission.create_nqueens_csp(8))
    grader.requireIsEqual(1.0, nQueensSolver.optimalWeight)
    grader.requireIsEqual(92, nQueensSolver.numOptimalAssignments)
    grader.requireIsEqual(2057, nQueensSolver.numOperations)

grader.addBasicPart('2a-1-basic', test2a_1, 2, maxSeconds=1, description="Basic test for create_nqueens_csp for n=8")

def test2a_hidden(n):
    pred = get_csp_result(submission.create_nqueens_csp(n))
    if solution_exist:
        grader.requireIsEqual(get_csp_result(solution.create_nqueens_csp(n)), pred)

grader.addHiddenPart('2a-2-hidden', lambda: test2a_hidden(3), 1, maxSeconds=1, description="Test create_nqueens_csp with n=3")

grader.addHiddenPart('2a-3-hidden', lambda: (test2a_hidden(4), test2a_hidden(7)) , 1, maxSeconds=1, description="Test create_nqueens_csp with different n")

############################################################
# Problem 2b: Most constrained variable


def test2b_1():
    mcvSolver = submission.BacktrackingSearch()
    mcvSolver.solve(submission.create_nqueens_csp(8), mcv = True)
    grader.requireIsEqual(1.0, mcvSolver.optimalWeight)
    grader.requireIsEqual(92, mcvSolver.numOptimalAssignments)
    grader.requireIsEqual(1361, mcvSolver.numOperations)

grader.addBasicPart('2b-1-basic', test2b_1, 2, maxSeconds=1, description="Basic test for MCV with n-queens CSP")

def test2b_2():
    def get_csp_result_with_mcv(BacktrackingSearch):
        return get_csp_result(util.create_map_coloring_csp(), BacktrackingSearch, mcv=True)
    pred = get_csp_result_with_mcv(submission.BacktrackingSearch)
    if solution_exist:
        answer = get_csp_result_with_mcv(solution.BacktrackingSearch)
        grader.requireIsEqual(answer, pred)

grader.addHiddenPart('2b-2-hidden', test2b_2, 1, maxSeconds=1, description="Test MCV with different CSPs")

def test2b_3():
    # We will use our implementation of n-queens csp
    # mcvSolver.solve(our_nqueens_csp(8), mcv = True)
    create_nqueens_csp = (solution.create_nqueens_csp if solution_exist else
                          submission.create_nqueens_csp)
    def get_csp_result_with_mcv(BacktrackingSearch):
        return get_csp_result(create_nqueens_csp(8), BacktrackingSearch, mcv=True)
    pred = get_csp_result_with_mcv(submission.BacktrackingSearch)
    if solution_exist:
        answer = get_csp_result_with_mcv(solution.BacktrackingSearch)
        grader.requireIsEqual(answer, pred)

grader.addHiddenPart('2b-3-hidden', test2b_3, 1, maxSeconds=1, description="Test for MCV with n-queens CSP")

############################################################
# Problem 3a: Sum factor

def test3a_1():
    csp = util.CSP()
    csp.add_variable('A', [0, 1, 2, 3])
    csp.add_variable('B', [0, 6, 7])
    csp.add_variable('C', [0, 5])

    sumVar = submission.get_sum_variable(csp, 'sum-up-to-15', ['A', 'B', 'C'], 15)
    # print(csp.variables)
    # for key, word in csp.binaryFactors.items():
    #     # print("\n", "-"*30, key, "\n" , word, "\n\n")
    csp.add_unary_factor(sumVar, lambda n: n in [12, 13])
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(4, sumSolver.numOptimalAssignments)
    
    csp.add_unary_factor(sumVar, lambda n: n == 12)
    sumSolver = submission.BacktrackingSearch()
    sumSolver.solve(csp)
    grader.requireIsEqual(2, sumSolver.numOptimalAssignments)

grader.addBasicPart('3a-1-basic', test3a_1, 2, maxSeconds=1, description="Basic test for get_sum_variable")

def test3a_2():
    BacktrackingSearch = (solution.BacktrackingSearch if solution_exist else
                          submission.BacktrackingSearch)

    def get_result(get_sum_variable):
        csp = util.CSP()
        sumVar = get_sum_variable(csp, 'zero', [], 15)
        sumSolver = BacktrackingSearch()
        sumSolver.solve(csp)
        out1 = sumSolver.numOptimalAssignments

        csp = util.CSP()
        sumVar = get_sum_variable(csp, 'zero', [], 15)
        csp.add_unary_factor(sumVar, lambda n: n > 0)
        sumSolver = BacktrackingSearch()
        sumSolver.solve(csp)
        out2 = sumSolver.numOptimalAssignments

        return out1, out2

    pred = get_result(submission.get_sum_variable)
    if solution_exist:
        grader.requireIsEqual(get_result(solution.get_sum_variable), pred)

grader.addHiddenPart('3a-2-hidden', test3a_2, 1, maxSeconds=1, description="Test get_sum_variable with empty list of variables")

def test3a_3():
    def get_result(get_sum_variable):
        csp = util.CSP()
        csp.add_variable('A', [0, 1, 2])
        csp.add_variable('B', [0, 1, 2])
        csp.add_variable('C', [0, 1, 2])

        sumVar = submission.get_sum_variable(csp, 'sum-up-to-7', ['A', 'B', 'C'], 7)
        sumSolver = submission.BacktrackingSearch()
        sumSolver.solve(csp)
        out1 = sumSolver.numOptimalAssignments

        csp.add_unary_factor(sumVar, lambda n: n == 6)
        sumSolver = submission.BacktrackingSearch()
        sumSolver.solve(csp)
        out2 = sumSolver.numOptimalAssignments

        return out1, out2

    pred = get_result(submission.get_sum_variable)
    if solution_exist:
        grader.requireIsEqual(get_result(solution.get_sum_variable), pred)

grader.addHiddenPart('3a-3-hidden', test3a_3, 2, maxSeconds=1, description="Test get_sum_variable with different variables")


############################################################
# Problem 3b: Light-bulb problem

def test3b_1():
    numBulbs = 3
    numButtons = 3
    maxNumRelations = 2
    buttonSets=({0, 2}, {1, 2}, {1, 2})

    csp = submission.create_lightbulb_csp(buttonSets, numButtons)
    solver = submission.BacktrackingSearch()
    solver.solve(csp)
    pred = solver.numOptimalAssignments
    answer = 2
    grader.requireIsEqual(answer, pred)

grader.addBasicPart('3b-1-basic', test3b_1, 2, maxSeconds=1, description="Basic test for light-bulb problem")

def test3b_2():
    numBulbs = 10
    numButtons = 10
    maxNumRelations = 7
    all_buttons = list(range(numButtons))

    random.seed(SEED)
    buttonSets = tuple(set(random.sample(all_buttons, maxNumRelations))
                       for bulbIndex in range(numBulbs))

    pred = get_csp_result(submission.create_lightbulb_csp(buttonSets, numButtons))
    if solution_exist:
        answer = get_csp_result(solution.create_lightbulb_csp(buttonSets, numButtons))
        grader.requireIsEqual(answer[0:2], pred[0:2])

grader.addHiddenPart('3b-2-hidden', test3b_2, 2, maxSeconds=1, description="Test light-bulb problem arguments")


grader.grade()
