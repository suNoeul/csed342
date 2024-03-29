#!/usr/bin/python

import random

import numpy as np

import graderUtil
from util import *

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


def test_correct(func_name, assertion=lambda pred: True, equal=lambda x, y: x == y):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(equal(pred, answer))
    return test

def test_wrong(func_name, assertion=lambda pred: True):
    def test():
        pred = getattr(submission, func_name)()
        assert pred is None or assertion(pred)
        if solution_exist:
            answer = getattr(solution, func_name)()
            grader.requireIsTrue(pred != answer and pred is not None)
    return test

############################################################
# Problem 1: hinge loss
############################################################

def veceq(vec1, vec2):
    def veclen(vec):
        return sum(1 for k, v in vec.items() if v != 0)
    if veclen(vec1) != veclen(vec2):
        return False
    else:
        return all(v == vec2.get(k, 0) for k, v in vec1.items())


def assertion(vec):
    words = 'so interesting great plot bored not'.split()
    return all((k in words and
                (isinstance(v, int) or isinstance(v, float)))
               for k, v in vec.items())

grader.addHiddenPart('1a-1-hidden', test_correct('problem_1a', assertion, veceq), 2, maxSeconds=1)

############################################################
# Problem 2: sentiment classification
############################################################

### 2a

# Basic sanity check for feature extraction
def test2a0():
    ans = {"a":2, "b":1}
    grader.requireIsEqual(ans, submission.extractWordFeatures("a b a"))
grader.addBasicPart('2a-0-basic', test2a0, maxSeconds=1, description="basic test")

def test2a1():
    fix_seed(SEED)

    def get_gen():
        for i in range(10):
            sentence = ' '.join([random.choice(['a', 'aa', 'ab', 'b', 'c']) for _ in range(100)])
            pred = submission.extractWordFeatures(sentence)
            if solution_exist:
                answer = solution.extractWordFeatures(sentence)
                yield grader.requireIsTrue(veceq(pred, answer))
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2a-1-hidden', test2a1, maxSeconds=1, description="test multiple instances of the same word in a sentence")

### 2b

def test2b0():
    trainExamples = (("hello world", 1), ("goodnight moon", -1))
    testExamples = (("hello", 1), ("moon", -1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)
    grader.requireIsGreaterThan(0, weights["hello"])
    grader.requireIsLessThan(0, weights["moon"])
grader.addBasicPart('2b-0-basic', test2b0, maxSeconds=1, description="basic sanity check for learning correct weights on two training and testing examples each")

def test2b1():
    trainExamples = (("hi bye", 1), ("hi hi", -1))
    testExamples = (("hi", -1), ("bye", 1))
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)
    grader.requireIsLessThan(0, weights["hi"])
    grader.requireIsGreaterThan(0, weights["bye"])
grader.addBasicPart('2b-1-basic', test2b1, maxSeconds=1, description="test correct overriding of positive weight due to one negative instance with repeated words")

def test2b2():
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, numIters=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print("Official: train error = %s, dev error = %s" % (trainError, devError))
    grader.requireIsEqual(0.0737198, trainError, 0.015)
    grader.requireIsEqual(0.2771525, devError, 0.02)
grader.addBasicPart('2b-2-basic', test2b2, maxPoints=6, maxSeconds=8, description="test classifier on real polarity dev dataset")

### 2c

def test2c0():
    sentence = "I am what I am"
    ans = {'I am': 2, 'am what': 1, 'what I': 1}
    grader.requireIsEqual(ans, submission.extractNgramFeatures(sentence, 2))
grader.addBasicPart('2c-0-basic', test2c0, maxSeconds=1, description="test basic ngram features")

def test2c1():
    fix_seed(SEED)

    def get_gen():
        for i in range(10):
            sentence = ' '.join([random.choice(['a','ab', 'b', 'c']) for _ in range(100)])
            length = random.randint(2,5)
            pred = submission.extractNgramFeatures(sentence, length)
            if solution_exist:
                answer = solution.extractNgramFeatures(sentence, length)
                yield grader.requireIsTrue(veceq(pred, answer))
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('2c-1-hidden', test2c1, 2, maxSeconds=1, description="test feature extraction on random sentence and random length")

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

### 3a (forward function)

def test3a0():
    X, _ = generate_xor()
    mlp = submission.MLPBinaryClassifier()
    answer = np.array([0.66606602, 0.70413772, 0.42144312, 0.47843014])
    grader.requireIsEqual(answer, mlp.forward(X), 0.0001)
grader.addBasicPart('3a-0-basic', test3a0, 1, maxSeconds=1, description="test forward function on XOR dataset")

def test3a1():
    fix_seed(SEED)
    mlp = submission.MLPBinaryClassifier()

    def get_gen():
        for _ in range(10):
            test_data = np.random.rand(100, 2)
            pred = mlp.forward(test_data)
            if solution_exist:
                answer = solution.MLPBinaryClassifier().forward(test_data)
                yield grader.requireIsEqual(pred, answer, 0.0001)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('3a-1-hidden', test3a1, 3, maxSeconds=1, description="test forward function on random data")

### 3b (loss and backward functions)
##### 3b-0 (loss function)
def test3b0():
    X, Y = generate_xor()
    mlp = submission.MLPBinaryClassifier()
    pred = mlp.forward(X)
    loss = mlp.loss(pred, Y)
    answer = np.array([1.09681198, 0.35078131, 0.86407047, 0.65091206])
    grader.requireIsEqual(answer, loss, 0.0001)
grader.addBasicPart('3b-0-basic', test3b0, maxSeconds=1, description="test loss function on XOR dataset")

##### 3b-1 (backward function)
def test3b1():
    X, Y = generate_xor()
    mlp = submission.MLPBinaryClassifier()
    pred = mlp.forward(X)
    gradients = mlp.backward(pred, Y)
    answer = {
        "W1": np.array([
            [-0.00017901,  0.11980031, 0.00514622, -0.06293203, -0.05433818, -0.01238167, -0.01461798,  0.01613166, 0.04126251, -0.01458624, -0.02073736,  0.030957  ,  0.01032425, 0.01136519, 0.01484782, 0.02837492 ],
            [ 0.00045162, -0.02365028, 0.02547818,  0.00610173,  0.02594611,  0.01474015,  0.05141377, -0.03852162, 0.01807816,  0.0240724 ,  0.00229582,  0.00426998, -0.03126768, 0.00806802, 0.01367728, -0.05103786]
        ]),
        "b1": np.array([
            [ 0.00106607, -0.0552891 ,  0.16018339,  0.04599698,  0.04563665, 0.02423702,  0.08237003, -0.05984813, -0.05154594,  0.03923213, 0.01710336, -0.0267356 , -0.04992217, -0.01620964,  0.10347403, -0.0792793 ]
        ]),
        "W2": np.array([
            [0.10873962],
            [0.08720741],
            [0.16945497],
            [0.13357918],
            [0.09288733],
            [0.1291229 ],
            [0.08088964],
            [0.16136131],
            [0.09679866],
            [0.12808811],
            [0.13931845],
            [0.0557765 ],
            [0.17144842],
            [0.07860855],
            [0.16975627],
            [0.0702388 ]
        ]),
        "b2": np.array([[0.270077]])
    }
    grader.requireIsEqual(answer, gradients, 0.0001)
grader.addBasicPart('3b-1-basic', test3b1, maxSeconds=1, description="test backward function on XOR dataset")

### 3b-2 (hidden)
def test3b2():
    fix_seed(SEED)

    def get_gen():
        for _ in range(10):
            test_X = np.random.rand(100, 2)
            test_Y = np.random.randint(0, 2, 100)

            mlp = submission.MLPBinaryClassifier()
            pred = mlp.forward(test_X)
            gradients = mlp.backward(pred, test_Y)
            if solution_exist:
                mlp_answer = solution.MLPBinaryClassifier()
                pred_answer = mlp_answer.forward(test_X)
                gradients_answer = mlp_answer.backward(pred_answer, test_Y)
                yield grader.requireIsEqual(gradients, gradients_answer, 0.0001)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('3b-2-hidden', test3b2, 6, maxSeconds=1, description="test backward function on random data")

### 3b-c  (update and train functions)
def test3c0():
    X, Y = generate_xor()
    mlp = submission.MLPBinaryClassifier()
    grader.requireIsEqual(0.02855617, mlp.train(X, Y, epochs=1000, learning_rate=0.1), 0.0001)
grader.addBasicPart('3c-0-basic', test3c0, maxSeconds=10, description="test train function on XOR dataset")

def test3c1():
    fix_seed(SEED)

    def get_gen():
        for _ in range(10):
            test_X = np.random.rand(100, 2)
            test_Y = np.random.randint(0, 2, 100)

            mlp = submission.MLPBinaryClassifier()
            loss = mlp.train(test_X, test_Y, epochs=10, learning_rate=0.1)
            if solution_exist:
                mlp_answer = solution.MLPBinaryClassifier()
                loss_answer = mlp_answer.train(test_X, test_Y, epochs=10, learning_rate=0.1)
                yield grader.requireIsEqual(loss, loss_answer, 0.0001)
            else:
                yield True
    all(get_gen())
grader.addHiddenPart('3c-1-hidden', test3c1, maxSeconds=10, description="test train function on random data")

grader.grade()
