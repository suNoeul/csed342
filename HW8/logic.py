# Simple logical inference system: resolution and model checking for propositional logic.

# Recursively apply str inside map
def rstr(x):
    if isinstance(x, tuple): return str(tuple(map(rstr, x)))
    if isinstance(x, list): return str(list(map(rstr, x)))
    if isinstance(x, set): return str(set(map(rstr, x)))
    if isinstance(x, dict):
        newx = {}
        for k, v in list(x.items()):
            newx[rstr(k)] = rstr(v)
        return str(newx)
    return str(x)

class Expression:
    # Helper functions used by subclasses.
    def ensureType(self, arg, wantedType):
        if not isinstance(arg, wantedType):
            raise Exception('%s: wanted %s, but got %s' % (self.__class__.__name__, wantedType, arg))
        return arg
    def ensureFormula(self, arg): return self.ensureType(arg, Formula)
    def isa(self, wantedType): return isinstance(self, wantedType)

    def __eq__(self, other): return str(self) == str(other)
    def __hash__(self): return hash(str(self))
    # Cache the string to be more efficient
    def __repr__(self):
        if not self.strRepn: self.strRepn = self.computeStrRepn()
        return self.strRepn

# A Formula represents a truth value.
class Formula(Expression): pass

# Predicate symbol (must be capitalized) applied to arguments.
# Example: LivesIn(john, palo_alto)
class Atom(Formula):
    def __init__(self, name):
        if not name[0].isupper(): raise Exception('Predicates must start with a uppercase letter, but got %s' % name)
        self.name = name
        self.strRepn = None
    def computeStrRepn(self):
        return self.name

AtomFalse = False
AtomTrue = True

# Example: Not(Rain)
class Not(Formula):
    def __init__(self, arg):
        self.arg = self.ensureFormula(arg)
        self.strRepn = None
    def computeStrRepn(self): return 'Not(' + str(self.arg) + ')'

# Example: And(Rain,Snow)
class And(Formula):
    def __init__(self, arg1, arg2):
        self.arg1 = self.ensureFormula(arg1)
        self.arg2 = self.ensureFormula(arg2)
        self.strRepn = None
    def computeStrRepn(self): return 'And(' + str(self.arg1) + ',' + str(self.arg2) + ')'

# Example: Or(Rain,Snow)
class Or(Formula):
    def __init__(self, arg1, arg2):
        self.arg1 = self.ensureFormula(arg1)
        self.arg2 = self.ensureFormula(arg2)
        self.strRepn = None
    def computeStrRepn(self): return 'Or(' + str(self.arg1) + ',' + str(self.arg2) + ')'

# Example: Implies(Rain,Wet)
class Implies(Formula):
    def __init__(self, arg1, arg2):
        self.arg1 = self.ensureFormula(arg1)
        self.arg2 = self.ensureFormula(arg2)
        self.strRepn = None
    def computeStrRepn(self): return 'Implies(' + str(self.arg1) + ',' + str(self.arg2) + ')'

# Take a list of conjuncts / disjuncts and return a formula
def AndList(forms):
    result = AtomTrue
    for form in forms:
        result = And(result, form) if result != AtomTrue else form
    return result
def OrList(forms):
    result = AtomFalse
    for form in forms:
        result = Or(result, form) if result != AtomFalse else form
    return result

# Return list of conjuncts of |form|.
# Example: And(And(A, Or(B, C)), Not(D)) => [A, Or(B, C), Not(D)]
def flattenAnd(form):
    if form.isa(And): return flattenAnd(form.arg1) + flattenAnd(form.arg2)
    else: return [form]

# Return list of disjuncts of |form|.
# Example: Or(Or(A, And(B, C)), D) => [A, And(B, C), Not(D)]
def flattenOr(form):
    if form.isa(Or): return flattenOr(form.arg1) + flattenOr(form.arg2)
    else: return [form]

# Syntactic sugar
def Equiv(a, b): return And(Implies(a, b), Implies(b, a))
def Xor(a, b): return And(Or(a, b), Not(And(a, b)))

############################################################
# Simple inference rules

# A Rule takes a sequence of argument Formulas and produces a set of result
# Formulas (possibly [] if the rule doesn't apply).
class Rule:
    pass

class UnaryRule(Rule):
    def applyRule(self, form): raise Exception('Override me')

class BinaryRule(Rule):
    def applyRule(self, form1, form2): raise Exception('Override me')
    # Override if rule is symmetric to save a factor of 2.
    def symmetric(self): return False

############################################################
# Unification

# Return whether unification was successful
# Assume forms are in CNF.
# Note: we don't do occurs check because we don't have function symbols.
def unify(form1, form2):
    if form1.isa(Atom):
        return form2.isa(Atom) and form1.name == form2.name
    if form1.isa(Not):
        return form2.isa(Not) and unify(form1.arg, form2.arg)
    if form1.isa(And):
        return form2.isa(And) and unify(form1.arg1, form2.arg1) and unify(form1.arg2, form2.arg2)
    if form1.isa(Or):
        return form2.isa(Or) and unify(form1.arg1, form2.arg1) and unify(form1.arg2, form2.arg2)
    raise Exception('Unhandled: %s' % form1)

############################################################
# Convert to CNF, Resolution rules

def withoutElementAt(items, i): return items[0:i] + items[i+1:]

def negateFormula(item):
    return item.arg if item.isa(Not) else Not(item)

# Given a list of Formulas, return a new list with:
# - If A and Not(A) exists, return [AtomFalse] for conjunction, [AtomTrue] for disjunction
# - Remove duplicates
# - Sort the list
def reduceFormulas(items, mode):
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            if negateFormula(items[i]) == items[j]:
                if mode == And: return [AtomFalse]
                elif mode == Or: return [AtomTrue]
                else: raise Exception("Invalid mode: %s" % mode)
    items = sorted(set(items), key=str)
    return items

# Generate a list of all subexpressions of a formula.
# Example:
# - Input: And(Atom('A'), Atom('B'))
# - Output: [And(Atom('A'), Atom('B')), Atom('A'), Atom('B')]
def allSubexpressions(form):
    subforms = []
    def recurse(form):
        subforms.append(form)
        if form.isa(Atom): pass
        elif form.isa(Not): recurse(form.arg)
        elif form.isa(And): recurse(form.arg1); recurse(form.arg2)
        elif form.isa(Or): recurse(form.arg1); recurse(form.arg2)
        elif form.isa(Implies): recurse(form.arg1); recurse(form.arg2)
        else: raise Exception("Unhandled: %s" % form)
    recurse(form)
    return subforms

class ToCNFRule(UnaryRule):
    def applyRule(self, form):
        newForm = form

        # Step 1: remove implications
        def removeImplications(form):
            if form.isa(Atom): return form
            if form.isa(Not): return Not(removeImplications(form.arg))
            if form.isa(And): return And(removeImplications(form.arg1), removeImplications(form.arg2))
            if form.isa(Or): return Or(removeImplications(form.arg1), removeImplications(form.arg2))
            if form.isa(Implies): return Or(Not(removeImplications(form.arg1)), removeImplications(form.arg2))
            raise Exception("Unhandled: %s" % form)
        newForm = removeImplications(newForm)

        # Step 2: push negation inwards (de Morgan)
        def pushNegationInwards(form):
            if form.isa(Atom): return form
            if form.isa(Not):
                if form.arg.isa(Not):  # Double negation
                    return pushNegationInwards(form.arg.arg)
                if form.arg.isa(And):  # De Morgan
                    return Or(pushNegationInwards(Not(form.arg.arg1)), pushNegationInwards(Not(form.arg.arg2)))
                if form.arg.isa(Or):  # De Morgan
                    return And(pushNegationInwards(Not(form.arg.arg1)), pushNegationInwards(Not(form.arg.arg2)))
                return form
            if form.isa(And): return And(pushNegationInwards(form.arg1), pushNegationInwards(form.arg2))
            if form.isa(Or): return Or(pushNegationInwards(form.arg1), pushNegationInwards(form.arg2))
            if form.isa(Implies): return Or(Not(pushNegationInwards(form.arg1)), pushNegationInwards(form.arg2))
            raise Exception("Unhandled: %s" % form)
        newForm = pushNegationInwards(newForm)

        # Step 3: distribute Or over And (want And on the outside): Or(And(A,B),C) becomes And(Or(A,C),Or(B,C))
        def distribute(form):
            if form.isa(Atom): return form
            if form.isa(Not): return Not(distribute(form.arg))
            if form.isa(And): return And(distribute(form.arg1), distribute(form.arg2))
            if form.isa(Or):
                # First need to distribute as much as possible
                f1 = distribute(form.arg1)
                f2 = distribute(form.arg2)
                if f1.isa(And):
                    return And(distribute(Or(f1.arg1, f2)), distribute(Or(f1.arg2, f2)))
                if f2.isa(And):
                    return And(distribute(Or(f1, f2.arg1)), distribute(Or(f1, f2.arg2)))
                return Or(f1, f2)
            raise Exception("Unhandled: %s" % form)
        newForm = distribute(newForm)

        # Post-processing: break up conjuncts into conjuncts and sort the disjuncts in each conjunct
        # Remove instances of A and Not(A)
        conjuncts = [OrList(reduceFormulas(flattenOr(f), Or)) for f in flattenAnd(newForm)]
        #print rstr(form), rstr(conjuncts)
        assert len(conjuncts) > 0
        if any(x == AtomFalse for x in conjuncts): return [AtomFalse]
        if all(x == AtomTrue for x in conjuncts): return [AtomTrue]
        conjuncts = [x for x in conjuncts if x != AtomTrue]
        results = reduceFormulas(conjuncts, And)
        if len(results) == 0: results = [AtomFalse]
        #print 'CNF', form, rstr(results)
        return results

class ResolutionRule(BinaryRule):
    # Assume formulas are in CNF
    # Assume A and Not(A) don't both exist in a form (taken care of by CNF conversion)
    def applyRule(self, form1, form2):
        items1 = flattenOr(form1)
        items2 = flattenOr(form2)
        results = []
        for i, item1 in enumerate(items1):
            for j, item2 in enumerate(items2):
                if unify(negateFormula(item1), item2):
                    newItems1 = withoutElementAt(items1, i)
                    newItems2 = withoutElementAt(items2, j)
                    newItems = newItems1 + newItems2

                    if len(newItems) == 0:  # Contradiction: False
                        results = [AtomFalse]
                        break

                    result = OrList(reduceFormulas(newItems, Or))

                    # Don't add redundant stuff
                    if result == AtomTrue: continue
                    if result in results: continue

                    results.append(result)
            if results == [AtomFalse]: break

        return results
    def symmetric(self): return True

############################################################
# Model checking

# Return the set of models
def performModelChecking(allForms, findAll, verbose=0):
    if verbose >= 3:
        print('performModelChecking', rstr(allForms))

    allForms = list(set(allForms) - set([AtomTrue, AtomFalse]))
    if verbose >= 3:
        print('All Forms:', rstr(allForms))

    if allForms == []: return [set()]  # One model
    if allForms == [AtomFalse]: return []  # No models

    # Atoms are the variables
    atoms = set()
    for form in allForms:
        for f in allSubexpressions(form):
            if f.isa(Atom): atoms.add(f)
    atoms = list(atoms)

    if verbose >= 3:
        print('Atoms:', rstr(atoms))
        print('Constraints:', rstr(allForms))

    # For each atom, list the set of formulas
    # atom index => list of formulas
    atomForms = [
        (atom, [form for form in allForms if atom in allSubexpressions(form)]) \
        for atom in atoms \
    ]
    # Degree heuristic
    atomForms.sort(key = lambda x : -len(x[1]))
    atoms = [atom for atom, form in atomForms]

    # Keep only the forms for an atom if it only uses atoms up until that point.
    atomPrefixForms = []
    for i, (atom, forms) in enumerate(atomForms):
        prefixForms = []
        for form in forms:
            useAtoms = set(x for x in allSubexpressions(form) if x.isa(Atom))
            if useAtoms <= set(atoms[0:i+1]):
                prefixForms.append(form)
        atomPrefixForms.append((atom, prefixForms))

    if verbose >= 3:
        print('Plan:')
        for atom, forms in atomForms:
            print("  %s: %s" % (rstr(atom), rstr(forms)))
    assert sum(len(forms) for atom, forms in atomPrefixForms) == len(allForms)

    # Build up an interpretation
    N = len(atoms)
    models = []  # List of models which are true
    model = set()  # Set of true atoms, mutated over time
    def recurse(i): # i: atom index
        if not findAll and len(models) > 0: return
        if i == N:  # Found a model on which the formulas are true
            models.append(set(model))
            return
        atom, forms = atomPrefixForms[i]

        if interpretForms(forms, model): recurse(i+1)
        model.add(atom)
        if interpretForms(forms, model): recurse(i+1)
        model.remove(atom)
    recurse(0)

    if verbose >= 5:
        print('Models:')
        for model in models:
            print("  %s" % rstr(model))

    return models

# A model is a set of atoms.
def printModel(model):
    for x in sorted(map(str, model)):
        print('*', x, '=', 'True')
    print('*', '(other atoms if any)', '=', 'False')

def interpretForm(form, model):
    if form.isa(Atom): return form in model
    if form.isa(Not): return not interpretForm(form.arg, model)
    if form.isa(And): return interpretForm(form.arg1, model) and interpretForm(form.arg2, model)
    if form.isa(Or): return interpretForm(form.arg1, model) or interpretForm(form.arg2, model)
    if form.isa(Implies): return not interpretForm(form.arg1, model) or interpretForm(form.arg2, model)
    raise Exception("Unhandled: %s" % form)

# Conjunction
def interpretForms(forms, model):
    return all(interpretForm(form, model) for form in forms)

############################################################

# A Derivation is a tree where each node corresponds to the application of a rule.
# For any Formula, we can extract a set of categories.
# Rule arguments are labeled with category.
class Derivation:
    def __init__(self, form, children, cost, derived):
        self.form = form
        self.children = children
        self.cost = cost
        self.permanent = False  # Marker for being able to extract.
        self.derived = derived  # Whether this was derived (as opposed to added by the user).
    def __repr__(self): return 'Derivation(%s, cost=%s, permanent=%s, derived=%s)' % (self.form, self.cost, self.permanent, self.derived)

# Possible responses to queries to the knowledge base
ENTAILMENT = "ENTAILMENT"
CONTINGENT = "CONTINGENT"
CONTRADICTION = "CONTRADICTION"

# A response to a KB query
class KBResponse:
    # query: what the query is (just a string description for printing)
    # modify: whether we modified the knowledge base
    # status: one of the ENTAILMENT, CONTINGENT, CONTRADICTION
    # trueModel: if available, a model consistent with the KB for which the the query is true
    # falseModel: if available, a model consistent with the KB for which the the query is false
    def __init__(self, query, modify, status, trueModel, falseModel):
        self.query = query
        self.modify = modify
        self.status = status
        self.trueModel = trueModel
        self.falseModel = falseModel

    def show(self, verbose=1):
        padding = '>>>>>'
        print(padding, self.responseStr())
        if verbose >= 1:
            print('Query: %s[%s]' % ('TELL' if self.modify else 'ASK', self.query))
            if self.trueModel:
                print('An example of a model where query is TRUE:')
                printModel(self.trueModel)
            if self.falseModel:
                print('An example of a model where query is FALSE:')
                printModel(self.falseModel)

    def responseStr(self):
        if self.status == ENTAILMENT:
            if self.modify: return 'I already knew that.'
            else: return 'Yes.'
        elif self.status == CONTINGENT:
            if self.modify: return 'I learned something.'
            else: return 'I don\'t know.'
        elif self.status == CONTRADICTION:
            if self.modify: return 'I don\'t buy that.'
            else: return 'No.'
        else:
            raise Exception("Invalid status: %s" % self.status)

    def __repr__(self): return self.responseStr()

def showKBResponse(response, verbose=1):
    if isinstance(response, KBResponse):
        response.show(verbose)
    else:
        items = [(obj, r.status) for ((var, obj), r) in list(response.items())]
        print('Yes: %s' % rstr([obj for obj, status in items if status == ENTAILMENT]))
        print('Maybe: %s' % rstr([obj for obj, status in items if status == CONTINGENT]))
        print('No: %s' % rstr([obj for obj, status in items if status == CONTRADICTION]))

# A KnowledgeBase is a set collection of Formulas.
# Interact with it using
# - addRule: add inference rules
# - tell: modify the KB with a new formula.
# - ask: query the KB about 
class KnowledgeBase:
    def __init__(self, standardizationRule, rules, modelChecking, verbose=0):
        # Rule to apply to each formula that's added to the KB (None is possible).
        self.standardizationRule = standardizationRule

        # Set of inference rules
        self.rules = rules

        # Use model checking as opposed to applying rules.
        self.modelChecking = modelChecking

        # For debugging
        self.verbose = verbose 

        # Formulas that we believe are true (used when not doing model checking).
        self.derivations = {}  # Map from Derivation key (logical form) to Derivation

    # Add a formula |form| to the KB if it doesn't contradict.  Returns a KBResponse.
    def tell(self, form):
        return self.query(form, modify=True)

    # Ask whether the logical formula |form| is True, False, or unknown based
    # on the KB.  Returns a KBResponse.
    def ask(self, form):
        return self.query(form, modify=False)

    def dump(self):
        print('==== Knowledge base [%d derivations] ===' % len(self.derivations))
        for deriv in list(self.derivations.values()):
            print(('-' if deriv.derived else '*'), deriv if self.verbose >= 2 else deriv.form)

    ####### Internal functions

    # Returns a KBResponse.
    def query(self, form, modify):
        formStr = '%s, standardized: %s' % (form, rstr(self.standardize(form)))

        # Models to serve as supporting evidence
        falseModel = None  # Makes the query false
        trueModel = None  # Makes the query true

        # Add Not(form)
        if not self.addAxiom(Not(form)):
            self.removeTemporary()
            status = ENTAILMENT
        else:
            # Inconclusive...
            falseModel = self.consistentModel
            self.removeTemporary()

            # Add form
            if self.addAxiom(form):
                if modify:
                    self.makeTemporaryPermanent()
                else:
                    self.removeTemporary()
                trueModel = self.consistentModel
                status = CONTINGENT
            else:
                self.removeTemporary()
                status = CONTRADICTION

        return KBResponse(query = formStr, modify = modify, status = status, trueModel = trueModel, falseModel = falseModel)

    # Apply the standardization rule to |form|.
    def standardize(self, form):
        if self.standardizationRule:
            return self.standardizationRule.applyRule(form)
        return [form]

    # Return whether adding |form| is consistent with the current knowledge base.
    # Add |form| to the knowledge base if we can.  Note: this is done temporarily!
    # Just calls addDerivation.
    def addAxiom(self, form):
        self.consistentModel = None
        for f in self.standardize(form):
            if f == AtomFalse: return False
            if f == AtomTrue: continue
            deriv = Derivation(f, children = [], cost = 0, derived = False)
            if not self.addDerivation(deriv): return False
        return True

    # Return whether the Derivation is consistent with the KB.
    def addDerivation(self, deriv):
        # Derived a contradiction
        if deriv.form == AtomFalse: return False

        key = deriv.form
        oldDeriv = self.derivations.get(key)
        maxCost = 100
        if oldDeriv == None and deriv.cost <= maxCost:
            # Something worth updating
            self.derivations[key] = deriv
            if self.verbose >= 3: print('add %s [%s derivations]' % (deriv, len(self.derivations)))

            if self.modelChecking:
                allForms = [deriv.form for deriv in list(self.derivations.values())]
                models = performModelChecking(allForms, findAll=False, verbose=self.verbose)
                if len(models) == 0: return False
                else: self.consistentModel = models[0]

            # Apply rules forward
            if not self.applyUnaryRules(deriv): return False
            for key2, deriv2 in list(self.derivations.items()):
                if not self.applyBinaryRules(deriv, deriv2): return False
                if not self.applyBinaryRules(deriv2, deriv): return False

        return True

    # Raise an exception if |formulas| is not a list of Formulas.
    def ensureFormulas(self, rule, formulas):
        if isinstance(formulas, list) and all(formula == False or isinstance(formula, Formula) for formula in formulas):
            return formulas
        raise Exception('Expected list of Formulas, but %s returned %s' % (rule, formulas))

    # Return whether everything is okay (no contradiction).
    def applyUnaryRules(self, deriv):
        for rule in self.rules:
            if not isinstance(rule, UnaryRule): continue
            for newForm in self.ensureFormulas(rule, rule.applyRule(deriv.form)):
                if not self.addDerivation(Derivation(newForm, children = [deriv], cost = deriv.cost + 1, derived = True)):
                    return False
        return True

    # Return whether everything is okay (no contradiction).
    def applyBinaryRules(self, deriv1, deriv2):
        for rule in self.rules:
            if not isinstance(rule, BinaryRule): continue
            if rule.symmetric() and str(deriv1.form) >= str(deriv2.form): continue  # Optimization
            for newForm in self.ensureFormulas(rule, rule.applyRule(deriv1.form, deriv2.form)):
                if not self.addDerivation(Derivation(newForm, children = [deriv1, deriv2], cost = deriv1.cost + deriv2.cost + 1, derived = True)):
                    return False
        return True

    # Remove all the temporary derivations from the KB.
    def removeTemporary(self):
        for key, value in list(self.derivations.items()):
            if not value.permanent:
                del self.derivations[key]

    # Mark all the derivations marked temporary to permanent.
    def makeTemporaryPermanent(self):
        for deriv in list(self.derivations.values()):
            deriv.permanent = True

# Create an empty knowledge base equipped with the usual inference rules.
def createResolutionKB():
    return KnowledgeBase(standardizationRule = ToCNFRule(), rules = [ResolutionRule()], modelChecking = False)

def createModelCheckingKB():
    return KnowledgeBase(standardizationRule = None, rules = [], modelChecking = True)
