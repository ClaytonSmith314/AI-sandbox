import types

from mu import heap, records, data, context, stack
import random


# --------------- #
# control Methods #
# --------------- #

def initialize():
    print("initializing AI")
    context['foo'] = {'foobar': 'ugh'}


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

    def __recur__(self, d):
        for sub, value in d.items():
            # TODO: finish symbol interpretation
            print("unpacking "+sub)
            if isinstance(value, dict):  # This action is one explicatly described in context
                self.__recur__(value)  # treat it like any context dictionary
            if isinstance(value, str):  # this type is a state. Their are many ways we can process states.
                print(sub + ' is a state')
                # if it's a stage in a cycle, run the cycle and pull the next value
                # if it's an expression, we need to run it some way aswell
                # Honestly, I'm still trying to figure out exactly what needs to happen hear
            if isinstance(value, types.FunctionType):  # this is a native method and needs to be run like one
                value()  # run the native call


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
                name = rand_string(5)  # make a new name for the symbol
                heap[name] = self.potentials[self.path]  # set heap definition of new random symbol to the new potential
                del self.potentials[self.path]  # remove from potentials, so potentials doesn't get cluttered

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
