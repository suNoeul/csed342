#!/usr/bin/env python3

import random
import collections

import graderUtil
grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    SEED = solution.SEED
    solution_exists = True
except ModuleNotFoundError:
    SEED = 42
    solution_exists = False

grader.useSolution = solution_exists

############################################################
# Problem 1a: findAlphabeticallyFirstWord

grader.add_basic_part('1a-0-basic', lambda:
                      grader.require_is_equal('alphabetically', submission.find_alphabetically_first_word(
                        'which is the first word alphabetically'),),
                        max_points=2,
                      description='simple test case')

grader.add_basic_part('1a-1-basic',
                      lambda: grader.require_is_equal('cat', submission.find_alphabetically_first_word('cat sun dog')),
                      description='simple test case')

grader.add_basic_part('1a-2-basic', lambda: grader.require_is_equal('0', submission.find_alphabetically_first_word(
    ' '.join(str(x) for x in range(100000)))), description='big test case')

############################################################
# Problem 1b: findFrequentWords

def test1b0():
    grader.require_is_equal({'the', 'fox'}, submission.find_frequent_words('the quick brown fox jumps over the lazy fox',2))
grader.add_basic_part('1b-0-basic', test1b0, 2, description='simple test')

def test1b1(numTokens, numTypes):
    def get_gen():
        text = ' '.join(str(random.randint(0, numTypes)) for _ in range(numTokens))
        for i in range(20):
            freq = 90 + i
            pred = submission.find_frequent_words(text,freq)
            if solution_exists:
                answer = solution.find_frequent_words(text,freq)
                yield pred == answer
            else:
                yield True
    grader.require_is_true(all(get_gen()))
grader.add_hidden_part('1b-1-hidden', lambda : test1b1(10000, 100), 2, description='random trials')

############################################################
# Problem 1c: findNonsingletonWords

def test1c0():
    grader.require_is_equal({'the', 'fox'},
                            submission.find_nonsingleton_words('the quick brown fox jumps over the lazy fox'))


grader.add_basic_part('1c-0-basic', test1c0, description='simple test')


def test1c12(num_tokens, num_types):
    import random
    random.seed(SEED)
    text = ' '.join(str(random.randint(0, num_types)) for _ in range(num_tokens))
    ans2 = submission.find_nonsingleton_words(text)
    if solution_exists:
        grader.require_is_equal(ans2, solution.find_nonsingleton_words(text))


grader.add_hidden_part('1c-1-hidden', lambda: test1c12(1000, 10), max_points=1, description='random trials')
grader.add_hidden_part('1c-2-hidden', lambda: test1c12(10000, 100), max_points=1, description='random trials (bigger)')

############################################################
# Problem 2a: denseVectorDotProduct
grader.add_basic_part('2a-0-basic', lambda : grader.require_is_equal(11, submission.dense_vector_dot_product([1,2],[3,4])), 
                      max_points=2, description='simple test')

def randvec():
    v = random.sample(range(-10,10),10)
    return v

def test2a1():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1 = randvec()
            v2 = randvec()
            pred = submission.dense_vector_dot_product(v1, v2)
            if solution_exists:
                answer = solution.dense_vector_dot_product(v1, v2)
                yield pred == answer
            else:
                yield True
    grader.require_is_true(all(get_gen()))

grader.add_hidden_part('2a-1-hidden', test2a1, 2, description='random trials')

############################################################
# Problem 2b: incrementDenseVector
grader.add_basic_part('2b-0-basic', lambda : grader.require_is_equal([13,17], submission.increment_dense_vector([1,2],3,[4,5])), 
                      max_points=2, description='simple test')

def test2b1():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v1 = randvec()
            v2 = randvec()
            s = random.randint(-10, 10)/2
            pred = submission.increment_dense_vector(v1, s, v2)
            if solution_exists:
                answer = solution.increment_dense_vector(v1, s, v2)
                yield pred==answer
            else:
                yield True
    grader.require_is_true(all(get_gen()))

grader.add_hidden_part('2b-1-hidden', test2b1, 2, description='random trials')

############################################################
# Problem 2c: dense2sparseVector
grader.add_basic_part('2c-0-basic', lambda : grader.require_is_equal(collections.defaultdict(float, {0:1, 3:2}), submission.dense_to_sparse_vector([1,0,0,2])), 
                      max_points=2, description='simple test')

def test2c1():
    random.seed(SEED)
    def get_gen():
        for _ in range(10):
            v = randvec()
            pred = submission.dense_to_sparse_vector(v)
            if solution_exists:
                answer = solution.dense_to_sparse_vector(v)
                yield pred==answer
            else:
                yield True
    grader.require_is_true(all(get_gen()))

grader.add_hidden_part('2c-1-hidden', test2c1, 2, description='random trials')



############################################################
# Problem 2d: dotProduct

def test2d0():
    grader.require_is_equal(15, submission.sparse_vector_dot_product(collections.defaultdict(float, {'a': 5}),
                                                                     collections.defaultdict(float, {'b': 2, 'a': 3})))


grader.add_basic_part('2d-0-basic', test2d0, max_points=2, description='simple test')


def randvec():
    v = collections.defaultdict(float)
    for _ in range(10):
        v[random.randint(0, 10)] = random.randint(0, 10) + 5
    return v


def test2d1():
    random.seed(SEED)
    for _ in range(10):
        v1 = randvec()
        v2 = randvec()
        ans2 = submission.sparse_vector_dot_product(v1, v2)
        if solution_exists:
            grader.require_is_equal(ans2, solution.sparse_vector_dot_product(v1, v2))


grader.add_hidden_part('2d-1-hidden', test2d1, max_points=2, description='random trials')


############################################################
# Problem 2e: incrementSparseVector

def test2e0():
    v = collections.defaultdict(float, {'a': 5})
    submission.increment_sparse_vector(v, 2, collections.defaultdict(float, {'b': 2, 'a': 3}))
    grader.require_is_equal(collections.defaultdict(float, {'a': 11, 'b': 4}), v)


grader.add_basic_part('2e-0-basic', test2e0, max_points=2, description='simple test')


def test2e1():
    random.seed(SEED)
    for _ in range(10):
        v1a = randvec()
        v1b = v1a.copy()
        v2 = randvec()
        submission.increment_sparse_vector(v1b, 4, v2)
        for key in list(v1b):
            if v1b[key] == 0:
                del v1b[key]
        if solution_exists:
            v1c = v1a.copy()
            submission.increment_sparse_vector(v1c, 4, v2)
            for key in list(v1c):
                if v1c[key] == 0:
                    del v1c[key]
            grader.require_is_equal(v1b, v1c)


grader.add_hidden_part('2e-1-hidden', test2e1, max_points=2, description='random trials')

############################################################
# Problem 2f: euclideanDistance

grader.add_basic_part('2f-0-basic', lambda: grader.require_is_equal(5, submission.euclidean_distance((1, 5), (4, 1))),
                      description='simple test case')


def test2f1():
    random.seed(SEED)
    for _ in range(100):
        x1 = random.randint(0, 10)
        y1 = random.randint(0, 10)
        x2 = random.randint(0, 10)
        y2 = random.randint(0, 10)
        ans2 = submission.euclidean_distance((x1, y1), (x2, y2))
        if solution_exists:
            grader.require_is_equal(ans2, solution.euclidean_distance((x1, y1), (x2, y2)))
        


grader.add_hidden_part('2f-1-hidden', test2f1, max_points=2, description='100 random trials')


############################################################
# Problem 3a: mutateSentences

def test3a0():
    grader.require_is_equal(sorted(['a a a a a']), sorted(submission.mutate_sentences('a a a a a')))
    grader.require_is_equal(sorted(['the cat']), sorted(submission.mutate_sentences('the cat')))
    grader.require_is_equal(
        sorted(['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']),
        sorted(submission.mutate_sentences('the cat and the mouse')))


grader.add_basic_part('3a-0-basic', test3a0, max_points=2, description='simple test')


def gen_sentence(alphabet_size, length):
    return ' '.join(str(random.randint(0, alphabet_size)) for _ in range(length))


def test3a1():
    random.seed(SEED)
    for _ in range(10):
        sentence = gen_sentence(3, 5)
        ans2 = submission.mutate_sentences(sentence)
        if solution_exists:
            grader.require_is_equal(ans2, solution.mutate_sentences(sentence))


grader.add_hidden_part('3a-1-hidden', test3a1, max_points=2, description='random trials')


def test3a2():
    random.seed(SEED)
    for _ in range(10):
        sentence = gen_sentence(25, 10)
        ans2 = submission.mutate_sentences(sentence)
        if solution_exists:
            grader.require_is_equal(ans2, solution.mutate_sentences(sentence))


grader.add_hidden_part('3a-2-hidden', test3a2, max_points=2, description='random trials (bigger)')

############################################################
grader.grade()
