import types

from mu import heap, records, data, context, stack
import random


# TODO: this context is a mess! It's like a disfunctional programming language. You have to rethink it!
# context needs to use expression notation. we need to have data and opperations and referance system now


# --------------- #
# control Methods #
# --------------- #

def initialize():
    context['root'] = []


def event():
    """ """

# -------------------------- #
# Main Symbol Theory Methods #
# -------------------------- #


def record(info):
    recorder.mark(info)  # the recorder needs to use this to create definitions
    records.append(info)  # add it to the records
    print(info)  # keep track what is being done so we can see it


def define(expression):
    name = rand_string(5)  # make a new name for the symbol
    heap[name] = expression  # set heap definition of new random symbol to the new potential
    print(name + ' =  ' + expression)


def expiriment():  # this can be done better, but for now, this works
    if not len(heap) == 0:
        test = random.choice(list(heap.keys()))


# ------------------- #
# Objects and enzymes #
# ------------------- #


class ContextReader:
    def __init__(self):
        self.Active_List = []
        self.instants = {}

    def effectuate(self):
        self.act()
        self.update()

    def act(self):
        """ """

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

# ----- #
# Other #
# ----- #

def rand_string(n):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    key = ''.join(random.choice(letters) for i in range(n))
    return key
