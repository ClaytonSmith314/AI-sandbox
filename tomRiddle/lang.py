
from mu import heap


# I'll set this up as a blocking call for now, but we can change it later
# ex sentance: remove the panel and get the green cube

class Interpreter:  # tasked with interpreting a string of words
    def __init__(self, text):
        self.text = text  # this will make it easy to iterate through
        self.construction = {}
        self.history = []
        self.pointer = 0

    def read(self):  # reads multple statements in a text
        self.construction = {}
        for statement in self.text.split('.'):
            running = self.evaluate(statement)  # I could easily make this a generator function...
            next(running)  # work out ...
        yield 'done', self.construction

    def evaluate(self, statement):  # reads a statement word by word
        self.history = []
        self.pointer = 0
        for word in statement.split():
            ex = execute()
            status, value = next(ex)

            # rather than if statements, we can yield from a handeler generator
            if status == 'await':  # status is 'under what condition are you yielding control'
                x = yield 'await', value
                ex.send(x)
            if status == 'sus':
                """ """
            if status == 'done':
                """ """

            self.pointer = self.pointer + 1
            # TODO: add stack waiting part here

def execute(self, word):  # takes a word, finds its definition, and executes the definition
    # yield directly from method or whatever
    if word in heap:  # this word is defined word.
        ex = Interpreter(heap[word])  # ex is another interpreter. This creates a rich kind of recursion
        read = ex.read()
        yield from read  # the executer yields whatever it reads. This causes read to run to get the execute value
    else:
        if word in native:  # this word is a function word
            ex = native[word]()  # ex here is a generator function. This creates a halting point for the inherent recurssion
            yield from ex  # the executer yields whatever it reads
        else:  # this word is a referance word. We need to find its data
            """ """  # note, some words have both definitions and data. These are object words. Makes sure we can make them


# all native functions will be generators
# all operating functions would be generators aswell

def returnword():
    yield 'done', 'return'


def nextword():  # really think about this one. It's opperation could reviel a lot about how to get things to cooperate on different levels
    n = yield 'await', 'next'  # next suspends to get the parent statements value
    yield 'done', n
                           # TODO: find out how to get next to return the actual next value


def lastword(interpreter):
    return 'done', 'last'


def isword(interpreter):
    """ """


native = {
    'return': returnword,
    'next': nextword,
}


handelers = {
    # we can use handelers to take care of different statuses
}

# native methods could be classes or methods. which one do I want. probibaly want both honestly
