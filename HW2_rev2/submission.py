#!/usr/bin/python
#20200703 SOONHO KIM

import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, interesting, great, plot, bored, not
    """

    # 사전 리뷰 데이터
    mini_reviews = [ ("so interesting", +1), ("great plot", +1), ("so bored", -1), ("not interesting", -1) ]

    # 학습률(step_size) & weights 정의
    weights = collections.defaultdict(int)
    step_size = 1
  
    for review, label in mini_reviews :
        words = review.split()
        feature_vector = collections.defaultdict(int)
        
        # feature vector 생성
        for word in words :
                feature_vector[word] += 1

        # Loss 계산 : w·ϕ(x)*y
        loss = (sum(weights[word]*count for word, count in feature_vector.items())) * label

        # Hinge loss
        hingeloss = max(0, 1-loss)

        # Hinge loss -> Weights update 
        if hingeloss > 0 :
            for word in feature_vector :
                weights[word] += step_size * (feature_vector[word] * label)
    
    return dict(weights)
   
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    
    words = x.split()
    feature_vector = collections.defaultdict(int)

    for word in words :
        feature_vector[word] += 1

    return feature_vector    

    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER

    for iteration in range(numIters):
        for x, y in trainExamples:  
            feature_vector = featureExtractor(x)

            # w·ϕ(x) 계산
            score = sum(weights.get(word, 0)*count for word, count in feature_vector.items())

            # σ(w⋅x) 계산
            pred = sigmoid(score)

            if y == 1 :
                gradient = pred - 1
            else :
                gradient = pred

            # Weights update
            for word, count in feature_vector.items() :
                # Loss_NLL의 도함수 : (σ(w⋅x)−y)⋅x
                weights[word] = weights.get(word, 0) - eta * gradient * count

        # # 현재 모델을 이용한 예측 함수 정의
        # def predictor(x):
        #     features = featureExtractor(x)
        #     score = sum(weights.get(feature, 0) * value for feature, value in features.items())
        #     return 1 if sigmoid(score) > 0.5 else -1

        # # 훈련 데이터셋과 테스트 데이터셋에 대한 오류율 계산
        # trainError = evaluatePredictor(trainExamples, predictor)
        # testError = evaluatePredictor(testExamples, predictor)

        # # 오류율 출력으로 모니터링
        # print(f"Iteration {iteration}: Train error = {trainError}, Test error = {testError}")

    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    words = x.split()
    consecutive_words = {}

    for i in range(len(words) - (n-1)) :
        ngram = ' '.join(words[i:i+n])
        consecutive_words[ngram] = consecutive_words.get(ngram, 0) + 1

    return consecutive_words

    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:

    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)    #  2 x 16
        self.b1 = np.zeros((1, self.hidden_size))                       #  1 x 16
        self.W2 = np.random.randn(self.hidden_size, self.output_size)   # 16 x 1
        self.b2 = np.zeros((1, self.output_size))                       #  1 x 1
        self.init_weights() 

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x) :
        temp = self.sigmoid(x) 
        return temp * (1 - temp)

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER

        self.x_input = x
        
        # 1. hidden layer input 계산 (Bx2 ⋅ 2x16 = Bx16)           
        self.h_input = np.dot(self.x_input, self.W1) + self.b1 

        # hidden layer output 계산 (Bx16)
        self.h_output = self.sigmoid(self.h_input)

        # ２. output layer input 계산 (Bx16 ⋅ 16x1 = Bx1)
        self.o_input = np.dot(self.h_output, self.W2) + self.b2

        # output(pred) 계산 (Bx1)
        pred = self.sigmoid(self.o_input).squeeze()

        return pred

        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER

        eps = 1e-12 # log연산 0 대입 방지

        # target(y=1 or y=0) 값에 따른 NLL Loss 연산
        loss = - (target * np.log(pred + eps)) - ((1-target) * np.log(1-pred + eps))

        return loss
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER      

        # ∂L/∂pred = ∂L/∂y' ⋅ ∂y'/∂pred [when y' = sigmoid(pred)] Bx1(B,)
        delta_Loss = pred - target                                      

        # gradient W2 : ∂L/∂W2 = ∂L/∂pred ⋅ ∂pred/∂W2 = delta_Loss ⋅ h_output [16xB ⋅ Bx1 = 16x1(16,)]
        gd_W2 = np.dot(self.h_output.T, delta_Loss)                     

        # gradient b2 : ∂L/∂b2 = ∂L/∂pred ⋅ ∂pred/∂b2 = delta_Loss ⋅ 1 [(1,)]
        gd_b2 = np.sum(delta_Loss, axis=0, keepdims=True)
        

        # ∂L/∂h1 = ∂L/∂pred ⋅ ∂pred/∂σ ⋅ ∂σ/∂h1 = delta_Loss ⋅ W2 ⋅ (h1(1-h1)) [(B,16)]
        delta_hidden = np.dot(delta_Loss[:, np.newaxis], self.W2.T) * self.sigmoid_derivative(self.h_input) 
        
        # W1(2, 16), b1(1, 16) 그래디언트 계산  
        gd_W1 = np.dot(self.x_input.T, delta_hidden)

        gd_b1 = np.sum(delta_hidden, axis=0, keepdims=True) 
        
        # 그래디언트 딕셔너리 반환
        gradients = {"W1": gd_W1, "b1": gd_b1, "W2": gd_W2[:, np.newaxis], "b2": gd_b2[:, np.newaxis]}

        # print(gd_W2.shape, gd_b2.shape, gd_W1.shape, gd_b1.shape)
        # print(self.W2.shape, self.b2.shape, self.W1.shape, self.b1.shape)
        return gradients
    
        # END_YOUR_ANSWER
    
    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        # BEGIN_YOUR_ANSWER

        for _ in range(epochs) :
            for x, y in zip(X, Y) :
                x = np.expand_dims(x, axis=0)
                y = np.array([y])

                pred = self.forward(x)
                gradients = self.backward(pred, y)
                self.update(gradients, learning_rate)

                loss = self.loss(pred, y)

        # final_pred = self.forward(X)
        # final_loss = self.loss(final_pred, Y)
        # return np.mean(final_loss)
        return loss

        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))