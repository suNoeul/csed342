#!/usr/bin/python3

import graderUtil
import util
import random
import sys
import wordsegUtil
import copy

grader = graderUtil.Grader()
submission = grader.load('submission')

try:
    import solution
    grader.addHiddenPart = grader.addBasicPart
    SEED = solution.SEED
    solution_exist = True
    random.seed(SEED)
except ModuleNotFoundError:
    SEED = 42
    solution_exist = False


# -------------------------------
# For Text reconstruction problem
# -------------------------------

QUERIES_SEG = [
    'ThestaffofficerandPrinceAndrewmountedtheirhorsesandrodeon',
    'hellothere officerandshort erprince',
    'howdythere',
    'The staff officer and Prince Andrew mounted their horses and rode on.',
    'whatsup',
    'duduandtheprince',
    'duduandtheking',
    'withoutthecourtjester',
    'lightbulbneedschange',
    'imagineallthepeople',
    'thisisnotmybeautifulhouse',
]

QUERIES_INS = [
    'strng',
    'pls',
    'hll thr',
    'whats up',
    'dudu and the prince',
    'frog and the king',
    'ran with the queen and swam with jack',
    'light bulbs need change',
    'ffcr nd prnc ndrw',
    'ffcr nd shrt prnc',
    'ntrntnl',
    'smthng',
    'btfl',
]

QUERIES_BOTH = [
    'stff',
    'hllthr',
    'thffcrndprncndrw',
    'ThstffffcrndPrncndrwmntdthrhrssndrdn',
    'whatsup',
    'ipovercarrierpigeon',
    'aeronauticalengineering',
    'themanwiththegoldeneyeball',
    'lightbulbsneedchange',
    'internationalplease',
    'comevisitnaples',
    'somethingintheway',
    'itselementarymydearwatson',
    'itselementarymyqueen',
    'themanandthewoman',
    'nghlrdy',
    'jointmodelingworks',
    'jointmodelingworkssometimes',
    'jointmodelingsometimesworks',
    'rtfclntllgnc',
]

CORPUS = 'leo-will.txt'

_realUnigramCost, _realBigramCost, _possibleFills = None, None, None

def getRealCosts():
    global _realUnigramCost, _realBigramCost, _possibleFills

    if _realUnigramCost is None:
        sys.stdout.write('Training language cost functions [corpus: %s]... ' % CORPUS)
        sys.stdout.flush()

        _realUnigramCost, _realBigramCost = wordsegUtil.makeLanguageModels(CORPUS)
        _possibleFills = wordsegUtil.makeInverseRemovalDictionary(CORPUS, 'aeiou')

        print('Done!')
        print('')

    return _realUnigramCost, _realBigramCost, _possibleFills

# --------------


def add_parts_1a(grader, submission):
    if grader.selectedPartName in ['1a-2-basic', '1a-3-hidden', '1a-4-hidden', None]:  # avoid timeouts
        unigramCost, _, _ = getRealCosts()

    def t_1a_1():
        def unigramCost(x):
            if x in ['and', 'two', 'three', 'word', 'words']:
                return 1.0
            else:
                return 1000.0

        grader.requireIsEqual('', submission.segmentWords('', unigramCost))
        grader.requireIsEqual('word', submission.segmentWords('word', unigramCost))
        grader.requireIsEqual('two words', submission.segmentWords('twowords', unigramCost))
        grader.requireIsEqual('and three words', submission.segmentWords('andthreewords', unigramCost))

    grader.addBasicPart('1a-1-basic', t_1a_1, maxPoints=1, maxSeconds=2, description='simple test case using hand-picked unigram costs')

    def t_1a_2():
        grader.requireIsEqual('word', submission.segmentWords('word', unigramCost))
        grader.requireIsEqual('two words', submission.segmentWords('twowords', unigramCost))
        grader.requireIsEqual('and three words', submission.segmentWords('andthreewords', unigramCost))

    grader.addBasicPart('1a-2-basic', t_1a_2, maxPoints=1, maxSeconds=2, description='simple test case using unigram cost from the corpus')

    def t_1a_3():
        # Word seen in corpus
        solution1 = submission.segmentWords('pizza', unigramCost)

        # Even long unseen words are preferred to their arbitrary segmentations
        solution2 = submission.segmentWords('qqqqq', unigramCost)
        solution3 = submission.segmentWords('z' * 100, unigramCost)

        # But 'a' is a word
        solution4 = submission.segmentWords('aa', unigramCost)

        # With an apparent crossing point at length 6->7
        solution5 = submission.segmentWords('aaaaaa', unigramCost)
        solution6 = submission.segmentWords('aaaaaaa', unigramCost)

        if solution_exist:
            grader.requireIsEqual(solution1, solution.segmentWords('pizza', unigramCost))
            grader.requireIsEqual(solution2, solution.segmentWords('qqqqq', unigramCost))
            grader.requireIsEqual(solution3, solution.segmentWords('z' * 100, unigramCost))
            grader.requireIsEqual(solution4, solution.segmentWords('aa', unigramCost))
            grader.requireIsEqual(solution5, solution.segmentWords('aaaaaa', unigramCost))
            grader.requireIsEqual(solution6, solution.segmentWords('aaaaaaa', unigramCost))


    grader.addHiddenPart('1a-3-hidden', t_1a_3, maxPoints=3, maxSeconds=3, description='simple hidden test case')

    def t_1a_4():
        for query in QUERIES_SEG:
            query = wordsegUtil.cleanLine(query)
            parts = wordsegUtil.words(query)
            pred = [submission.segmentWords(part, unigramCost) for part in parts]

            if solution_exist:
                grader.requireIsEqual(pred, [solution.segmentWords(part, unigramCost) for part in parts])

    grader.addHiddenPart('1a-4-hidden', t_1a_4, maxPoints=5, maxSeconds=3, description='hidden test case for all queries in QUERIES_SEG')


def add_parts_1b(grader, submission):
    if grader.selectedPartName in ['1b-2-hidden', '1b-4-hidden', None]:  # avoid timeouts
        _, bigramCost, possibleFills = getRealCosts()

    def t_1b_1():
        def bigramCost(a, b):
            corpus = [wordsegUtil.SENTENCE_BEGIN] + 'beam me up scotty'.split()
            if (a, b) in list(zip(corpus, corpus[1:])):
                return 1.0
            else:
                return 1000.0

        def possibleFills(x):
            fills = {
                'bm'   : set(['beam', 'bam', 'boom']),
                'm'    : set(['me', 'ma']),
                'p'    : set(['up', 'oop', 'pa', 'epe']),
                'sctty': set(['scotty']),
            }
            return fills.get(x, set())

        grader.requireIsEqual(
            '',
            submission.insertVowels([], bigramCost, possibleFills)
        )
        grader.requireIsEqual( # No fills
            'zz$z$zz',
            submission.insertVowels(['zz$z$zz'], bigramCost, possibleFills)
        )
        grader.requireIsEqual(
            'beam',
            submission.insertVowels(['bm'], bigramCost, possibleFills)
        )
        grader.requireIsEqual(
            'me up',
            submission.insertVowels(['m', 'p'], bigramCost, possibleFills)
        )
        grader.requireIsEqual(
            'beam me up scotty',
            submission.insertVowels('bm m p sctty'.split(), bigramCost, possibleFills)
        )

    grader.addBasicPart('1b-1-basic', t_1b_1, maxPoints=1, maxSeconds=2, description='simple test case')

    def t_1b_2():
        solution1 = submission.insertVowels([], bigramCost, possibleFills)
        # No fills
        solution2 = submission.insertVowels(['zz$z$zz'], bigramCost, possibleFills)
        solution3 = submission.insertVowels([''], bigramCost, possibleFills)
        solution4 = submission.insertVowels('wld lk t hv mr lttrs'.split(), bigramCost, possibleFills)
        solution5 = submission.insertVowels('ngh lrdy'.split(), bigramCost, possibleFills)

        if solution_exist:
            grader.requireIsEqual(solution1, solution.insertVowels([], bigramCost, possibleFills))
            grader.requireIsEqual(solution2, solution.insertVowels(['zz$z$zz'], bigramCost, possibleFills))
            grader.requireIsEqual(solution3, solution.insertVowels([''], bigramCost, possibleFills))
            grader.requireIsEqual(solution4, solution.insertVowels('wld lk t hv mr lttrs'.split(), bigramCost, possibleFills))
            grader.requireIsEqual(solution5, solution.insertVowels('ngh lrdy'.split(), bigramCost, possibleFills))


    grader.addHiddenPart('1b-2-hidden', t_1b_2, maxPoints=3, maxSeconds=2, description='simple hidden test case')

    def t_1b_3():
        SB = wordsegUtil.SENTENCE_BEGIN

        # Check for correct use of SENTENCE_BEGIN
        def bigramCost(a, b):
            if (a, b) == (SB, 'cat'):
                return 5.0
            elif a != SB and b == 'dog':
                return 1.0
            else:
                return 1000.0

        solution1 = submission.insertVowels(['x'], bigramCost, lambda x: set(['cat', 'dog']))
    
        if solution_exist:
            grader.requireIsEqual(solution1, solution.insertVowels(['x'], bigramCost, lambda x: set(['cat', 'dog'])))

        # Check for non-greediness

        def bigramCost(a, b):
            # Dog over log -- a test poem by rf
            costs = {
                (SB, 'cat'):      1.0,  # Always start with cat

                ('cat', 'log'):   1.0,  # Locally prefer log
                ('cat', 'dog'):   2.0,  # rather than dog

                ('log', 'mouse'): 3.0,  # But dog would have been
                ('dog', 'mouse'): 1.0,  # better in retrospect
            }
            return costs.get((a, b), 1000.0)

        def fills(x):
            return {
                'x1': set(['cat', 'dog']),
                'x2': set(['log', 'dog', 'frog']),
                'x3': set(['mouse', 'house', 'cat'])
            }[x]

        solution2 = submission.insertVowels('x1 x2 x3'.split(), bigramCost, fills)

        if solution_exist:
            grader.requireIsEqual(solution2, solution.insertVowels('x1 x2 x3'.split(), bigramCost, fills))

        # Check for non-trivial long-range dependencies
        def bigramCost(a, b):
            # Dogs over logs -- another test poem by rf
            costs = {
                (SB, 'cat'):        1.0,  # Always start with cat

                ('cat', 'log1'):    1.0,  # Locally prefer log
                ('cat', 'dog1'):    2.0,  # Rather than dog

                ('log20', 'mouse'): 1.0,  # And this might even
                ('dog20', 'mouse'): 1.0,  # seem to be okay
            }
            for i in range(1, 20):       # But along the way
            #                               Dog's cost will decay
                costs[('log' + str(i), 'log' + str(i+1))] = 0.25
                costs[('dog' + str(i), 'dog' + str(i+1))] = 1.0 / float(i)
            #                               Hooray
            return costs.get((a, b), 1000.0)

        def fills(x):
            f = {
                'x0': set(['cat', 'dog']),
                'x21': set(['mouse', 'house', 'cat']),
            }
            for i in range(1, 21):
                f['x' + str(i)] = set(['log' + str(i), 'dog' + str(i), 'frog'])
            return f[x]

        solution3 = submission.insertVowels(['x' + str(i) for i in range(0, 22)], bigramCost, fills)

        if solution_exist:
            grader.requireIsEqual(solution3, solution.insertVowels(['x' + str(i) for i in range(0, 22)], bigramCost, fills))


    grader.addHiddenPart('1b-3-hidden', t_1b_3, maxPoints=3, maxSeconds=3, description='simple hidden test case')

    def t_1b_4():
        for query in QUERIES_INS:
            query = wordsegUtil.cleanLine(query)
            ws = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(query)]
            pred = submission.insertVowels(copy.deepcopy(ws), bigramCost, possibleFills)

            if solution_exist:
                grader.requireIsEqual(pred, solution.insertVowels(copy.deepcopy(ws), bigramCost, possibleFills))

    grader.addHiddenPart('1b-4-hidden', t_1b_4, maxPoints=3, maxSeconds=3, description='hidden test case for all queries in QUERIES_INS')


def add_parts_1c(grader, submission):
    if grader.selectedPartName in ['1c-2-basic', '1c-3-hidden', '1c-5-hidden', None]:  # avoid timeouts
        unigramCost, bigramCost, possibleFills = getRealCosts()

    def t_1c_1():
        def bigramCost(a, b):
            if b in ['and', 'two', 'three', 'word', 'words']:
                return 1.0
            else:
                return 1000.0

        fills_ = {
            'nd': set(['and']),
            'tw': set(['two']),
            'thr': set(['three']),
            'wrd': set(['word']),
            'wrds': set(['words']),
        }
        fills = lambda x: fills_.get(x, set())

        grader.requireIsEqual('', submission.segmentAndInsert('', bigramCost, fills))
        grader.requireIsEqual('word', submission.segmentAndInsert('wrd', bigramCost, fills))
        grader.requireIsEqual('two words', submission.segmentAndInsert('twwrds', bigramCost, fills))
        grader.requireIsEqual('and three words', submission.segmentAndInsert('ndthrwrds', bigramCost, fills))

    grader.addBasicPart('1c-1-basic', t_1c_1, maxPoints=1, maxSeconds=2, description='simple test case with hand-picked bigram costs and possible fills')

    def t_1c_2():
        bigramCost = lambda a, b: unigramCost(b)

        fills_ = {
            'nd': set(['and']),
            'tw': set(['two']),
            'thr': set(['three']),
            'wrd': set(['word']),
            'wrds': set(['words']),
        }
        fills = lambda x: fills_.get(x, set())

        grader.requireIsEqual(
            'word',
            submission.segmentAndInsert('wrd', bigramCost, fills))
        grader.requireIsEqual(
            'two words',
            submission.segmentAndInsert('twwrds', bigramCost, fills))
        grader.requireIsEqual(
            'and three words',
            submission.segmentAndInsert('ndthrwrds', bigramCost, fills))

    grader.addBasicPart('1c-2-basic', t_1c_2, maxPoints=1, maxSeconds=2, description='simple test case with unigram costs as bigram costs')

    def t_1c_3():
        bigramCost = lambda a, b: unigramCost(b)
        fills_ = {
            'nd': set(['and']),
            'tw': set(['two']),
            'thr': set(['three']),
            'wrd': set(['word']),
            'wrds': set(['words']),
            # Hah!  Hit them with two better words
            'th': set(['the']),
            'rwrds': set(['rewards']),
        }
        fills = lambda x: fills_.get(x, set())

        solution1 = submission.segmentAndInsert('wrd', bigramCost, fills)
        solution2 = submission.segmentAndInsert('twwrds', bigramCost, fills)
        # Waddaya know
        solution3 = submission.segmentAndInsert('ndthrwrds', bigramCost, fills)

        if solution_exist:
            grader.requireIsEqual(solution1, solution.segmentAndInsert('wrd', bigramCost, fills))
            grader.requireIsEqual(solution2, solution.segmentAndInsert('twwrds', bigramCost, fills))
            grader.requireIsEqual(solution3, solution.segmentAndInsert('ndthrwrds', bigramCost, fills))


    grader.addHiddenPart('1c-3-hidden', t_1c_3, maxPoints=3, maxSeconds=3, description='hidden test case with unigram costs as bigram costs and additional possible fills')

    def t_1c_4():
        def bigramCost(a, b):
            corpus = [wordsegUtil.SENTENCE_BEGIN] + 'beam me up scotty'.split()
            if (a, b) in list(zip(corpus, corpus[1:])):
                return 1.0
            else:
                return 1000.0

        def possibleFills(x):
            fills = {
                'bm'   : set(['beam', 'bam', 'boom']),
                'm'    : set(['me', 'ma']),
                'p'    : set(['up', 'oop', 'pa', 'epe']),
                'sctty': set(['scotty']),
                'z'    : set(['ze']),
            }
            return fills.get(x, set())

        # Ensure no non-word makes it through
        solution1 = submission.segmentAndInsert('zzzzz', bigramCost, possibleFills)
        solution2 = submission.segmentAndInsert('bm', bigramCost, possibleFills)
        solution3 = submission.segmentAndInsert('mp', bigramCost, possibleFills)
        solution4 = submission.segmentAndInsert('bmmpsctty', bigramCost, possibleFills)

        if solution_exist:
            grader.requireIsEqual(solution1, solution.segmentAndInsert('zzzzz', bigramCost, possibleFills))
            grader.requireIsEqual(solution2, solution.segmentAndInsert('bm', bigramCost, possibleFills))
            grader.requireIsEqual(solution3, solution.segmentAndInsert('mp', bigramCost, possibleFills))
            grader.requireIsEqual(solution4, solution.segmentAndInsert('bmmpsctty', bigramCost, possibleFills))


    grader.addHiddenPart('1c-4-hidden', t_1c_4, maxPoints=5, maxSeconds=3, description='hidden test case with hand-picked bigram costs and possible fills')

    def t_1c_5():
        smoothCost = wordsegUtil.smoothUnigramAndBigram(unigramCost, bigramCost, 0.2)
        for query in QUERIES_BOTH:
            query = wordsegUtil.cleanLine(query)
            parts = [wordsegUtil.removeAll(w, 'aeiou') for w in wordsegUtil.words(query)]
            pred = [submission.segmentAndInsert(part, smoothCost, possibleFills) for part in parts]

            if solution_exist:
                grader.requireIsEqual(pred, [solution.segmentAndInsert(part, smoothCost, possibleFills) for part in parts])

    grader.addHiddenPart('1c-5-hidden', t_1c_5, maxPoints=5, maxSeconds=3, description='hidden test case for all queries in QUERIES_BOTH with bigram costs and possible fills from the corpus')

def add_parts_2a(grader, submission):
    import numpy
    def t_2a_1():
        map = numpy.array([
            [1,1,1,1,1,1],
            [1,0,0,0,0,1],
            [1,0,0,0,0,1],
            [1,1,1,0,0,1],
            [1,0,0,0,0,1],
            [1,1,1,1,1,1]
        ], dtype=int)
      
        moveCost = util.CalculateMoveCost(map)
        possibleMoves = util.FindPossibleMoves(map)

        grader.requireIsEqual(3, submission.UCSMazeSearch((1, 1), (1, 4), moveCost, possibleMoves))
        grader.requireIsEqual(6, submission.UCSMazeSearch((1, 1), (4, 4), moveCost, possibleMoves))
        grader.requireIsEqual(7, submission.UCSMazeSearch((1, 1), (4, 1), moveCost, possibleMoves))

    grader.addBasicPart('2a-1-basic', t_2a_1, maxPoints=1, maxSeconds=1, description='simple test case with small maze')
    
    def t_2a_2():
        map = numpy.array([
            [1,1,1,1,1,1],
            [1,0,0,0,0,1],
            [1,0,0,0,0,1],
            [1,1,1,0,0,1],
            [1,0,0,0,0,1],
            [1,1,1,1,1,1]
        ], dtype=int)
        
        moveCost = util.CalculateMoveCost(map, moveCost={'LEFT': 1, 'RIGHT': 2, 'UP': 1, 'DOWN': 2})
        possibleMoves = util.FindPossibleMoves(map)

        grader.requireIsEqual(6, submission.UCSMazeSearch((1, 1), (1, 4), moveCost, possibleMoves))
        grader.requireIsEqual(12, submission.UCSMazeSearch((1, 1), (4, 4), moveCost, possibleMoves))
        grader.requireIsEqual(12, submission.UCSMazeSearch((1, 1), (4, 1), moveCost, possibleMoves))

    grader.addBasicPart('2a-2-basic', t_2a_2, maxPoints=1, maxSeconds=1, description='simple test case with small maze')
     
    
    def t_2a_3():
        map = numpy.zeros((11, 100), dtype=int)
        map[ 0, :] = 1
        map[-1, :] = 1
        map[:,  0] = 1
        map[:, -1] = 1
        moveCost = util.CalculateMoveCost(map)
        possibleMoves = util.FindPossibleMoves(map)

        solution1 = submission.UCSMazeSearch((5, 40), (5, 90), moveCost, possibleMoves)
        solution2 = submission.UCSMazeSearch((5, 20), (5, 90), moveCost, possibleMoves)
        
        if solution_exist:
            grader.requireIsEqual(solution1, solution.UCSMazeSearch((5, 40), (5, 90), moveCost, possibleMoves))
            grader.requireIsEqual(solution2, solution.UCSMazeSearch((5, 20), (5, 90), moveCost, possibleMoves))
    
    grader.addHiddenPart('2a-3-hidden', t_2a_3, maxPoints=1, maxSeconds=1, description='hidden test case where UCS becomes inefficient')
    
    
def add_parts_2b(grader, submission):
    import numpy
    def t_2b_1():
        # First maze in pdf, 2.b
        map = numpy.zeros((7, 20), dtype=int)
        map[ 0, :] = 1
        map[-1, :] = 1
        map[:,  0] = 1
        map[:, -1] = 1

        # map[2, 9:12] = 1
        # map[3, 11]   = 1
        # map[4, 9:12] = 1
        # cost = 12

        moveCost = util.CalculateMoveCost(map)
        possibleMoves = util.FindPossibleMoves(map)

        grader.requireIsEqual(8, submission.AStarMazeSearch((3, 6), (3, 14), moveCost, possibleMoves))

    grader.addBasicPart('2b-1-basic', t_2b_1, maxPoints=1, maxSeconds=1, description='simple test case with small maze')

    def t_2b_2():
        # Maze with a long wall
        map = numpy.zeros((50, 50), dtype=int)
        map[ 0, :] = 1
        map[-1, :] = 1
        map[:,  0] = 1
        map[:, -1] = 1
        
        map[-3, 2:-2] = 1
        map[2:-2, -3] = 1

        moveCost = util.CalculateMoveCost(map)
        possibleMoves = util.FindPossibleMoves(map)

        tm = graderUtil.TimeMeasure()

        time_cmp_list = []

        for _ in range(10):
            tm.check()
            pred_ucs = submission.UCSMazeSearch((46, 46), (48, 48), moveCost, possibleMoves)
            time_sub_ucs = tm.elapsed()

            tm.check()
            pred_astar = submission.AStarMazeSearch((46, 46), (48, 48), moveCost, possibleMoves)
            time_sub_astar = tm.elapsed()
            time_cmp_list.append(time_sub_ucs > time_sub_astar)

            grader.requireIsEqual(pred_ucs, pred_astar)

            if solution_exist:
                tm.check()
                answer_ucs = solution.AStarMazeSearch((46, 46), (48, 48), moveCost, possibleMoves)
                time_sol_astar = tm.elapsed()
                grader.requireIsEqual(answer_ucs, pred_astar)

        # A* should be faster than UCS at least in 80% of comparisons
        grader.requireIsTrue((sum(time_cmp_list) / len(time_cmp_list)) >= 0.8)

    grader.addHiddenPart('2b-2-hidden', t_2b_2, maxPoints=3, maxSeconds=3, description='hidden test case for maze search with A*')

    def t_2b_3():
        # Maze where UCS becomes inefficient
        map = numpy.zeros((11, 100), dtype=int)
        map[ 0, :] = 1
        map[-1, :] = 1
        map[:,  0] = 1
        map[:, -1] = 1
        moveCost = util.CalculateMoveCost(map)
        possibleMoves = util.FindPossibleMoves(map)

        tm = graderUtil.TimeMeasure()

        time_cmp_list = []

        for _ in range(10):
            tm.check()
            pred_ucs = submission.UCSMazeSearch((5, 40), (5, 90), moveCost, possibleMoves)
            time_sub_ucs = tm.elapsed()

            tm.check()
            pred_astar = submission.AStarMazeSearch((5, 40), (5, 90), moveCost, possibleMoves)
            time_sub_astar = tm.elapsed()
            time_cmp_list.append(time_sub_ucs > time_sub_astar)

            grader.requireIsEqual(pred_ucs, pred_astar)

            if solution_exist:
                tm.check()
                answer_ucs = solution.AStarMazeSearch((5, 40), (5, 90), moveCost, possibleMoves)
                time_sol_astar = tm.elapsed()
                grader.requireIsEqual(answer_ucs, pred_astar)

        # A* should be faster than UCS at least in 80% of comparisons
        grader.requireIsTrue((sum(time_cmp_list) / len(time_cmp_list)) >= 0.8)

    grader.addHiddenPart('2b-3-hidden', t_2b_3, maxPoints=4, maxSeconds=3, description='hidden test case for maze search with A*, where UCS becomes inefficient')


    def t_2b_4():
        # Maze where UCS becomes inefficient
        map = numpy.zeros((11, 100), dtype=int)
        map[ 0, :] = 1
        map[-1, :] = 1
        map[:,  0] = 1
        map[:, -1] = 1

        moveCost = util.CalculateMoveCost(map, moveCost={'LEFT': 2, 'RIGHT': 1, 'UP': 2, 'DOWN': 1})
        possibleMoves = util.FindPossibleMoves(map)
       
        tm = graderUtil.TimeMeasure()

        time_cmp_list = []

        for _ in range(10):
            tm.check()
            pred_ucs = submission.UCSMazeSearch((5, 40), (5, 90), moveCost, possibleMoves)
            time_sub_ucs = tm.elapsed()

            tm.check()
            pred_astar = submission.AStarMazeSearch((5, 40), (5, 90), moveCost, possibleMoves)
            time_sub_astar = tm.elapsed()
            time_cmp_list.append(time_sub_ucs > time_sub_astar)

            grader.requireIsEqual(pred_ucs, pred_astar)

            if solution_exist:
                tm.check()
                answer_ucs = solution.AStarMazeSearch((5, 40), (5, 90), moveCost, possibleMoves)
                time_sol_astar = tm.elapsed()
                grader.requireIsEqual(answer_ucs, pred_astar)

        # A* should be faster than UCS at least in 80% of comparisons
        grader.requireIsTrue((sum(time_cmp_list) / len(time_cmp_list)) >= 0.8)

    grader.addHiddenPart('2b-4-hidden', t_2b_4, maxPoints=4, maxSeconds=3, description='hidden test case for modified maze search with A*, where UCS becomes inefficient')


add_parts_1a(grader, submission)
add_parts_1b(grader, submission)
add_parts_1c(grader, submission)
add_parts_2a(grader, submission)
add_parts_2b(grader, submission)
grader.grade()
