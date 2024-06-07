#20200703 김순호
from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a():
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Implies(And(Summer, California), Not(Rain))
    # END_YOUR_CODE

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b():
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Equiv(Wet, Or(Rain, Sprinklers))
    # END_YOUR_CODE

# Sentence: "Either it's day or night (but not both)."
def formula1c():
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Or(And(Day, Not(Night)), And(Not(Day), Night))
    # END_YOUR_CODE

# Sentence: "One can access campus server only if she (or he) is a computer science major or not a freshman."
def formula1d():
    # Predicates to use:
    Access = Atom('Access')     # whether one can acess campus server
    Computer = Atom('Computer') # whether one is computer science major
    Freshman = Atom('Freshman') # whether one is freshman
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return Implies(Access, Or(Computer, Not(Freshman)))
    # END_YOUR_CODE

# Sentence: "There are 10 students and they all pass artificial intelligence course."
def formula1e():
    # Predicates to use:
    StudentNum = 10
    def PassAI(i): return Atom('PassAI' + str(i)) # whether student i pass AI course
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return AndList([PassAI(i+1) for i in range(0, StudentNum)])
    # END_YOUR_CODE
