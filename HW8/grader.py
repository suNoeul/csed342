#!/usr/bin/env python

from logic import *

import pickle, os
import graderUtil

grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    solution_exist = True
except ModuleNotFoundError:
    solution_exist = False

# name: name of this formula (used to load the models)
# predForm: the formula predicted in the submission
def checkFormula(name, predForm):
    predModels = performModelChecking([predForm], findAll=True)

    if solution_exist:
        filename = os.path.join('models', name + '.pklz')
        targetModels = pickle.load(open(filename, 'rb'))

        def hashkey(model): return tuple(sorted(str(atom) for atom in model))
        targetModelSet = set(hashkey(model) for model in targetModels)
        predModelSet = set(hashkey(model) for model in predModels)

        for model in targetModels:
            if hashkey(model) not in predModelSet:
                return
        for model in predModels:
            if hashkey(model) not in targetModelSet:
                return
        grader.assignFullCredit()

############################################################
# Problem 1: propositional logic

grader.addHiddenPart('1a-1-hidden', lambda : checkFormula('1a', submission.formula1a()), 2, description='Test formula 1a implementation')
grader.addHiddenPart('1b-1-hidden', lambda : checkFormula('1b', submission.formula1b()), 2, description='Test formula 1b implementation')
grader.addHiddenPart('1c-1-hidden', lambda : checkFormula('1c', submission.formula1c()), 2, description='Test formula 1c implementation')
grader.addHiddenPart('1d-1-hidden', lambda : checkFormula('1d', submission.formula1d()), 2, description='Test formula 1d implementation')
grader.addHiddenPart('1e-1-hidden', lambda : checkFormula('1e', submission.formula1e()), 3, description='Test formula 1e implementation')

grader.grade()
