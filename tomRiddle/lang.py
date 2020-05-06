
from mu import heap, context, data, records


# I'll set this up as a blocking call for now, but we can change it later
# ex sentance: remove the panel and get the green cube

class Interpreter:  # tasked with interpreting a string of words
    def __init__(self, rawstring):
        self.string = rawstring.split()  # this will make it easy to iterate through
        self.pointer = 0  # start at the begining
        # self.active = {}  # all the words running their definitions
        self.stack = []  # all the words waiting on other words
        self.history = []  # each word has a result, which can be used by later words. That (or a referance to it) goes here

    def read(self):  # blocking read of string
        while self.pointer < len(self.string):  # run until string is completly interpreted
            self.run(Executer(self, self.string[self.active]))  # create a new Executer to run the words code
            self.pointer = self.pointer + 1  # incriment one to run next word

        return self.history

    def run(self, executer):
        status, result = executer.execute()  # execute the word and get the result and status
        if status == 'done':  # if the executer is done, we can save the result in history and run the last stack
            self.history[self.pointer] = result
            if self.stack is not None:
                self.run(self.stack.pop())  # note: this only works if we only suspend for the next word, but we can suspend twice...
        if status == 'suspended':  # if the executer needs to wait on the next word, it is suspended and added to the stack
            self.stack.append(executer)  # for now, I'll assume it suspends only until next return


# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- # -- #


class Executer:
    def __init__(self, interpretor, word):
        self.word = word
        self.interpretor = interpretor
        if word in heap:
            self.definition = heap[word]
        else:
            if word in native:  # use a native method call to execute word
                """ """
        self.pointer = 0
        self.status = 'active'

    # words with an object property should return the address of their object allong with their return type
    def execute(self):
        if self.word in heap:

            while self.pointer < len(self.definition):
                """for each line in the definition, interpret it."""
                line = self.definition(self.pointer)
                reader = Interpreter(line)
                results = reader.read()  # normally, reader doesn't return any usefull results, but sometimes, it changes opperations
                if results[0] == 'return':
                    return 'done', results[1]  # might want to edit to return more than one value...
                if results[0] == 'next':
                    return 'suspended', ''
                self.pointer = self.pointer + 1  # move on to next word
        else:
            if self.word in native:  # native words have their own functions
                return native[self.word](self.interpretor)  # we can run the function and directly get the result


# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- # -- #


# all native functions will be generators

def returnword(interpreter):
    return 'done', 'return'


def nextword(interpreter):
    return 'done', 'next'  # next suspends the parent opperation, but it itself does not suspend
                           # TODO: find out how to get next to return the actual next value

def lastword(interpreter):
    """ """


def isword(interpreter):
    """ """

native = {
    'return': returnword,
    'next': nextword,
}

# native methods could be classes or methods. which one do I want. probibaly want both honestly
