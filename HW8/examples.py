from logic import *

# Sentence: "If it's rains, it's wet".
def rainWet():
    Rain = Atom('Rain') # whether it's raining
    Wet = Atom('Wet') # whether it's wet
    return Implies(Rain, Wet)
