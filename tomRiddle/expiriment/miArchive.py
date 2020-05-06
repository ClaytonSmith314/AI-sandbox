# holds all of mi's old setup information

from mu import heap, records, data, context, stack, f_context
import random


# --------------- #
# control Methods #
# --------------- #

def event():
    # f_manage()
    construct()
    effectuate()
    # define()
    # record()
    schedule()


def f_manage():  # replacing method with one that fits better
    """recusively loops through all echo context"""
    for key, instances in f_context.items():
        for symbol, info in instances.items():
            native[key](symbol, info)


# current management setup doesn't work well
# proposed control scheme:
#   we have four mane processes to manage. Each process works with specific resources and causes them to act in
#   a certain way. We need to make sure that all the right information gets the right focus from each method.
#   One idea is a lower level context with only information about where each process needs to be applied. This
#   would provide the flexability needed to opperate with lots of information, but would require that the
#   f-context be updated reliably and constantly to avoid dead information.

# -------------------------- #
# Main Symbol Theory Methods #
# -------------------------- #


# this stage of the learning cycle is analogous to the encoding of DNA into proteans

def construct():
    for symbol in f_context['construct']:
        construct_symbol(symbol)


def construct_symbol(symbol):
    """step one of learning cycle"""
    # TODO: setup the script to create the heap
    print('constructing ' + symbol)
    if symbol in heap:  # we need to make sure that all the symbols are in the heap, or it doesn't work
        definition = heap[symbol]  # the definition of all symbols is stored in the heap
        reader = StringReader(definition)  # create the object that will create the construction
        construction = reader.read()  # build the construction from the string
        context[symbol] = construction  # put that construction in context
        f_context['efectuate'][symbol] = 'init'  # this last step tells that the context needs to be effectuated
    else:
        print("\terror: symbol " + symbol + " cannot be constructed")


# this process is what we can describe as acting, or thinking.
# It is how we take the context that describes how we are acting and put it into action
def effectuate():
    """step two of learning cycle"""
    # take context information
    # read syntax according to given expression notation
    # run each peace as a step. change stages when complete
    # edit data from
    # look at the interactions between situational data and what we can and are doing
    context_reader.evaluate()  # all functionality is offloaded to context_reader, since it works better anyway


# this constitutes what we call our memory. Any time we notice something
# or act, we record what happens or what we realize and remember it.
def record(info):
    """step three of learning cycle"""
    recorder.mark(info)  # the recorder needs to use this to create definitions
    records.append(info)  # add it to the records
    print(info)  # keep track what is being done so we can see it

    # the heap constitutes our basic knowledge base. This is the process where we get new knowlegde from what we remember.
    # def define(symbol, info):
    """step 4 of learning cycle"""
    # start by keeping track of begining and ends of symbol definitions
    #       we need to know what snippits of operations code for a certain kind of symbol
    #       snippets should start by declaring a variable and end by declaring the variables definition
    # take resulting opperation and encode it into a context expression
    # generate a half random symbol. Make sure it doesn't match any other generated symbols.
    # add symbol and definition to heap
    # let the program know that the new variable needs to be tested


def schedule():
    """takes data and uses it to decide what heap functions need to be loaded"""
    # this one is actually a part of a smaller loop different from the learning cycle, called the action loop
    # look at a new instance of data
    # determine its form and find a suitable referance in heap
    # add that referance to the list of heap definitions that need to be constructed


# -------------------------------- #
# Objects for opperating on the AI #
# -------------------------------- #


class StringReader:
    """this class is designed to read expressions and create active constructions from those expressions"""

    # we're probably going to have to redo all of this. Work out how this works!

    # --------------------------------- #
    # STRING READER
    # String --> Construction
    # Used for creating context structures from heap definitions
    # --------------------------------- #

    def __init__(self, string, construction=None):
        if construction is None:
            construction = {}
        self.construction = construction
        self.stack = []
        self.path = []
        if type(string).__name__ == 'string':
            self.string = string.split(' ')
        else:
            self.string = string

    def read(self):
        for codon in self.string:
            # do some opperation. Don't forget we have to create context for the next symbol read
            print(codon)
            self.run(codon)
        return self.construction

    def run(self, codon):  # takes a codon and runs its specific opperation
        if codon in StringReader.codon_definitions:  # if the codon is a native definition, then we run it as a method
            self.codon_definitions[codon]()
        else:  # if not, then it is a composite codon, and we need to find it in heap and read its definition
            if codon in heap:
                definition = heap[codon]
                self.read[definition]
            else:
                print("Error: codon " + codon + " could not be read")

    def my_await(self, codon):
        self.stack.append(codon)

    def write(self, codon):
        """writes a specific expression to the open directory in the construct"""

    def open(self, path):
        """opens a certain part of the construct tree by storing it to path"""

    codon_definitions = {
        'await': my_await,
        'open': open,
        'write': write,
    }


class Constructor:
    # queue = []

    def __init__(self):
        self.queue = []

    def constructall(self):
        """ """
        # take everything from the queue and use string reader to evaluate it
        # note: it might not be the case that the heap codes for context all the time
        # infact, I think I'll have to redo stringreader to just read the heap and do whatever action it wants
        # regardless of whether it updates context or data or even something else.
        # Heap is supposed to be like a definition of words. Some words are verbs, those that code for context,
        # but their are also nouns and the like...


class ContextReader:
    def __init__(self):
        self.Active_List = []

    def evaluate(self):
        self.__recur__(context)

    def __recur__(self, d):
        for sub, value in d:
            if type(value).__name__ == 'dictionary':  # This action is one explicatly described in context
                self.__recur__(value)  # treat it like any context dictionary
            if type(value) == 'string':  # this type is a state. Their are many ways we can process states.
                print(sub + ' is a state')
                # if it's a stage in a cycle, run the cycle and pull the next value
                # if it's an expression, we need to run it some way aswell
                # Honestly, I'm still trying to figure out exactly what needs to happen hear
            if type(value) == 'function':  # this is a native method and needs to be run like one
                value()  # run the native call


class Recorder:
    """"keeps track of all possible definitions in records"""

    # we need opening and closing of data
    # when data is opened, it starts a definition
    # when data is closed, it ends a definition
    # all context run inbetween opening and closing of data that acts on the data is added to the definition
    # note, this allows for implicate definitions, using unknown context and variables
    # we could add that context into the definition. That'd help.

    def __init__(self):
        self.potentials = {}
        self.path = ""

    def mark(self, info):
        content = info.split(" ")
        action = content[0]
        if action == 'open':
            # new definitions occur when we open and exit directories.

            # first, we need to define the old difiniton, unless we are opening a sub definition
            if self.path in self.potentials.keys():  # if we have a valid path
                name = rand_string(5)  # make a new name for the symbol
                heap[name] = self.potentials[self.path]  # set heap definition of new random symbol to the new potential
                del self.potentials[self.path]  # remove from potentials, so potentials doesn't get cluttered

            # part two: add the new definition
            self.path = content[1]
            if not self.path in self.potentials.keys():
                self.potentials[self.path] = []

        # TODO: update to work woth multiple paths from different contexts
        if action == 'write':  # we need to change the difinition aswell
            subject = content[1]
            self.potentials[self.path].append(subject)
        # no other actions apply to this part


def rand_string(n):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    x = 0
    key = ''.join(random.choice(letters) for i in range(n))
    return key


# --------------------- #
# Native def dictionary #
# --------------------- #

context_reader = ContextReader()

recorder = Recorder()

native = {
    'contruct': construct,
    'effectuate': effectuate,
    'record': record,
}


##################################################################


import types

from mu import heap, records, data, context, stack
import random

# TODO: this context is a mess! It's like a disfunctional programming language. YOu have to rethink it!
# context needs to use expression notation. we need to have data and opperations

# --------------- #
# control Methods #
# --------------- #

def initialize():
    print("initializing AI")
    context['foo'] = {'open': ''}
    # data['foo'] = 'foobar'


def event():
    construct()
    effectuate()
    schedule()


# -------------------------- #
# Main Symbol Theory Methods #
# -------------------------- #

def construct():
    # print("construct:")
    constructor.construct()


def construct_symbol(symbol):
    print('\tconstructing ' + symbol)
    if symbol in heap:  # we need to make sure that all the symbols are in the heap, or it doesn't work
        definition = heap[symbol]  # the definition of all symbols is stored in the heap
        reader = StringReader(definition)  # create the object that will create the construction
        construction = reader.read()  # build the construction from the string
        context[symbol] = construction  # put that construction in context
    else:
        print("\t\terror: symbol "+symbol+" cannot be constructed")


def effectuate():
    # print("effectuat:")
    context_reader.effectuate()


def record(info):
    """step three of learning cycle"""
    recorder.mark(info)  # the recorder needs to use this to create definitions
    records.append(info)  # add it to the records
    print(info)  # keep track what is being done so we can see it


def schedule():
    # print('Schedule:')
    scheduler.schedule()


def define(expression):
    name = rand_string(5)  # make a new name for the symbol
    heap[name] = expression  # set heap definition of new random symbol to the new potential
    print(name + ' =  ' + expression)

# ------------------- #
# Objects and enzymes #
# ------------------- #


############################################


class StringReader:
    """this class is designed to read expressions and create active constructions from those expressions"""

    def __init__(self, string, construction=None):
        if construction is None:
            construction = {}
        self.construction = construction
        self.stack = []
        self.path = []
        if isinstance(string, str):
            self.string = string.split(' ')
        else:
            self.string = string

    def read(self):
        for codon in self.string:
            # do some opperation. Don't forget we have to create context for the next symbol read
            print(codon)
            self.run(codon)
        return self.construction

    def run(self, codon):  # takes a codon and runs its specific opperation
        if codon in StringReader.codon_definitions:  # if the codon is a native definition, then we run it as a method
            self.codon_definitions[codon]()
        else:  # if not, then it is a composite codon, and we need to find it in heap and read its definition
            if codon in heap:
                definition = heap[codon]
                self.read[definition]
            else:
                print("Error: codon " + codon + " could not be read")

    def my_await(self, codon):
        self.stack.append(codon)

    def write(self, codon):
        """writes a specific expression to the open directory in the construct"""

    def open(self, path):
        """opens a certain part of the construct tree by storing it to path"""

    codon_definitions = {
        'await': my_await,
        'open': open,
        'write': write,
    }


############################################


class Constructor:
    def __init__(self):
        self.queue = []

    def append(self, symbol):
        self.queue.append(symbol)

    def construct(self):
        """ """
        for symbol in self.queue:
            construct_symbol(symbol)
        self.queue = []


constructor = Constructor()


############################################


class Scheduler:
    def __init__(self):
        self.queue = []

    def append(self, symbol):
        self.queue.append(symbol)

    def schedule(self):  # determines what contextual responce needs to be called from a symbol
        for symbol in self.queue:
            """ """
            # constructor.append(symbol)  # this is not the ways it should work, but it should get it doing something


scheduler = Scheduler()


############################################


class ContextReader:
    def __init__(self):
        self.Active_List = []

    def effectuate(self):
        self.__recur__(context)

    def __recur__(self, d):  # runs for each dictionary in context
        for sub, value in d.items():  # loop through the dictionary
            # TODO: finish symbol interpretation
            method = ''  # method is an empty string
            print("unpacking "+sub)
            if isinstance(value, dict):  # This action is one explicatly described in context
                self.__recur__(value)  # treat it like any context dictionary
                method = 'run ' + sub
            if isinstance(value, str):  # this type is a state. Their are many ways we can process states.
                print(sub + ' is a state')
                # if it's a stage in a cycle, run the cycle and pull the next value
                # if it's an expression, we need to run it some way aswell
                # Honestly, I'm still trying to figure out exactly what needs to happen hear
            if isinstance(value, types.FunctionType):  # this is a native method and needs to be run like one
                value()  # run the native call
            if value is None:
                print(sub+' has no value')

            if not method == '':
                record(method)  # sets up the next step


context_reader = ContextReader()


############################################


class Recorder:

    def __init__(self):
        self.potentials = {}
        self.path = ""

    def mark(self, info):
        content = info.split(" ")
        action = content[0]
        if action == 'open':
            # new definitions occur when we open and exit directories.

            # first, we need to define the old difiniton, unless we are opening a sub definition
            if self.path in self.potentials.keys():  # if we have a valid path
                define(self.poitentials[self.path])
                del self.potentials[self.path]  # remove from potentials, so potentials doesn't get cluttered
                define(self.poitentials[self.path])

            # part two: add the new definition
            self.path = content[1]
            if self.path not in self.potentials.keys():
                self.potentials[self.path] = []

        # TODO: update to work woth multiple paths from different contexts
        if action == 'write':  # we need to change the difinition as well
            subject = content[1]
            self.potentials[self.path].append(subject)
        # no other actions apply to this part


recorder = Recorder()


############################################


# --------------------- #
# Native def dictionary #
# --------------------- #

native = {
    'contruct': construct,
    'effectuate': effectuate,
    'record': record,
}


# ----- #
# Other #
# ----- #

def rand_string(n):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    x = 0
    key = ''.join(random.choice(letters) for i in range(n))
    return key

#####################################################################################

import types

from mu import heap, records, data, context, stack
import random


# TODO: this context is a mess! It's like a disfunctional programming language. You have to rethink it!
# context needs to use expression notation. we need to have data and opperations and referance system now


# --------------- #
# control Methods #
# --------------- #

def test():
    print("initializing AI")
    print(interpretor.read('ans="fff ans "ans f=ans'))
    print(data)
    print(interpretor.evaluate('scramble("abc,)'))
    print(interpretor.evaluate('g=concat(ans,"arm,)'))
    print(interpretor.evaluate('concat(g,scramble(concat("sqr,g,),),)'))

def initialize():
    test()
    context['root'] = []


def event():
    construct()
    effectuate()
    schedule()
    expiriment()


# -------------------------- #
# Main Symbol Theory Methods #
# -------------------------- #

def construct():
    constructor.construct()


def effectuate():
    context_reader.effectuate()


def record(info):
    recorder.mark(info)  # the recorder needs to use this to create definitions
    records.append(info)  # add it to the records
    print(info)  # keep track what is being done so we can see it


def schedule():
    # print('Schedule:')
    rationalmanager.schedule()


def define(expression):
    name = rand_string(5)  # make a new name for the symbol
    heap[name] = expression  # set heap definition of new random symbol to the new potential
    print(name + ' =  ' + expression)


def expiriment():  # this can be done better, but for now, this works
    if not len(heap) == 0:
        test = random.choice(list(heap.keys()))
        constructor.append(test)


# ------------------- #
# Objects and enzymes #
# ------------------- #


############################################


class Interpretor:  # this class is responsible for interpreting all synatic language in the AI system
    def __init__(self):
        self.result = ''

    def read(self, expression, parameters=None, position='null'):
        """The intepritor has to be used to put """
        self.result = []
        if isinstance(expression, str):
            expression = expression.split()  # split expression into spaces. NO SPACES ADDED INSIDE SYMBOLS

        for symbol in expression:
            """take each symbol one at a time and run it"""
            self.result.append(self.evaluate(symbol, parameters, position))

        return self.result

    def evaluate(self, symbol, parameters={}, position='null'):

        if parameters is None:
            parameters = {}
        # TODO: this needs to be an ordered evaluation. Think PEMDAS.
        # pos = position
        # I need double parintheses. How in the world do I do that?
        print('evaluating ' + symbol)

        # this first set is the situational opperators
        if symbol.startswith('"'):  # code for a literal. only one ("arx), no double ("arx")
            return symbol.lstrip('"')
        if symbol.startswith('>'):  # executes the result of the expression. >"fn() <==> fn()
            return self.evaluate(self.evaluate(symbol.lstrip('>')))
        if ':' in symbol:  # resets the referance frame of the expression.
            pos, exp = symbol.split(':', 1)
            if pos.isalpha():  # NOTE: only returns if the front is symbol. Prevents it from running inside a function
                return self.evaluate(exp, parameters, pos)  # might want to change to avoid stack overhead, but okay...

        # second phase is action opperators
        if ('=' or '<=' or '<-') in symbol:
            print(symbol + ' is opperator')
            if '=' in symbol:
                var, exp = symbol.split('=', 1)
                if var.isalpha():
                    data[var] = self.evaluate(exp, parameters, position)
                    return data[var]
            if "<=" in symbol:
                var, exp = symbol.split('<=', 1)
                if var.isalpha():
                    context_reader.instants[var] = self.evaluate(exp, parameters, position)
                    return context_reader.instants[var]
            if "<-" in symbol:
                var, exp = symbol.split('<-', 1)
                if var.isalpha():
                    heap[var] = exp  # TODO: add parameters, and what if we want the result to be put in heap?
                    #                   also, what if this is just a temp function or the function's already defined?
        else:
            # last we have the referance opperators
            if '(' in symbol:  # this is an implicate function call. These are controlled by heap
                # TODO: work out all the kinks with functions and running functions.
                fn, p = symbol.split('(', 1)
                par = p.split(',')
                parstack = [[]]
                fnstack = [fn]
                for exp in par:
                    if '(' in exp:  # fn() or fn(a,)
                        if exp.endswith('()'):
                            fnstack[-1].append(constructor.execute(exp.rstrip('()')))
                        else:
                            nfn, npar = exp.split('(', 1)
                            parstack.append([npar])
                            fnstack.append[nfn]
                    else:
                        # note that all closing parintheses have to be separated by commas ie. fn(a,b,c,) or fn(a,b(gg,),)
                        if exp == ')':
                            aexp = parstack.pop()
                            if not parstack:
                                return constructor.execute(fnstack.pop(), aexp)
                            else:
                                parstack[-1].append(constructor.execute(fnstack.pop(), aexp))
                        else:
                            parstack[-1].append(exp)

            else:  # two types left - direct and relative
                print('ref ' + symbol)
                if ('.' or '/') in symbol:  # this is a relative referance
                    """ """  # TODO: make relative notation. Not yet though. See if what we have works first.
                else:  # two big types of direct referances - native and constructed
                    print('finding ' + symbol)
                    if symbol.isalpha():  # if it's made of letters, the symbol is a constructed referance. 3 scopes
                        if symbol in parameters:  # parameter scope.
                            return parameters[symbol]  # this is a copy referance. Should we use key referance instead?
                        else:
                            if symbol in context_reader.instants:  # instants scope
                                return context_reader.instants[symbol]
                            else:
                                if symbol in data:  # data scope
                                    return data[symbol]


interpretor = Interpretor()


# nynder<="iifei need<=ner:fuwner(gqw(asd),nwoe(uer),owt.wur--nytn)

############################################


class Constructor:  # this is the low level way that ideas are constructed
    def __init__(self):
        self.queue = []  # here is a list of

    def append(self, symbol):
        self.queue.append(symbol)  # maybe add some parameters...

    def construct(self):  # takes everything in the queue and construct it
        for symbol in self.queue:  # the queue is the list of
            self.execute(symbol)
        self.queue = []

    def execute(self, symbol, parlist={}):  # how use parameters? should self be a parameter?
        print(parlist)
        print(heap[symbol][0])
        parameters = {}
        for key, value in zip(heap[symbol][0], parlist):
            parameters[key] = interpretor.evaluate(value)
        print(parameters)
        for expression in heap[symbol][1]:
            if isinstance(expression, str):
                return interpretor.read(expression, parameters)
            else:
                if isinstance(expression, types.FunctionType):
                    return expression(parameters)


constructor = Constructor()


############################################

# TODO: remake scheduler to pair data directly to context and not wait for
class RationalManager:
    def __init__(self):
        self.queue = []

    def append(self, symbol):
        self.queue.append(symbol)

    def schedule(self):  # determines what contextual responce needs to be called from a symbol
        for symbol in self.queue:
            """ """
            # constructor.append(symbol)  # this is not the ways it should work, but it should get it doing something


rationalmanager = RationalManager()


############################################


def run(symbol):
    for exp in context[symbol]:
        interpretor.read(exp)


class ContextReader:
    def __init__(self):
        self.Active_List = []
        self.instants = {}

    def effectuate(self):
        self.act()
        self.update()

    def act(self):
        """ expressions. We need to read expressions """
        self.run('root')

    def run(self, symbol):
        for exp in context[symbol]:
            interpretor.read(exp)

    def update(self):
        """ """


context_reader = ContextReader()


############################################


class Recorder:

    def __init__(self):
        self.potentials = {}
        self.path = ""

    def mark(self, info):
        """ """


recorder = Recorder()


############################################

# this class creates combinations of heap definitions and sets them to be run
class Expirimenter:
    def __init__(self):
        """ """

    def expiriment(self):
        """ """


expirimenter = Expirimenter()

############################################


# --------------------- #
# Native def dictionary #
# --------------------- #

native = {
    'contruct': construct,
    'effectuate': effectuate,
    'record': record,
}


# ----- #
# Other #
# ----- #

def rand_string(n):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    key = ''.join(random.choice(letters) for i in range(n))
    return key
