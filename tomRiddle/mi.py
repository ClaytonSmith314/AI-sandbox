
from mu import heap, records, data, context, stack, f_context


# --------------- #
# control Methods #
# --------------- #

def event():
    # f_manage()
    construct()
    effectuate()
    # record()
    # define()
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
        print("\terror: symbol "+symbol+" cannot be constructed")


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
def record():
    """step three of learning cycle"""
    recorder.



# the heap constitutes our basic knowledge base. This is the process where we get new knowlegde from what we remember.
def define(symbol, info):
    """step 4 of learning cycle"""
    # start by keeping track of begining and ends of symbol definitions
    #       we need to know what snippits of operations code for a certain kind of symbol
    #       snippets should start by declaring a variable and end by declaring the variables definition
    # take resulting opperation and encode it into a context expression
    # generate a half random symbol. Make sure it doesn't match any other generated symbols.
    # add symbol and definition to heap
    # let the program know that the new variable needs to be tested
    print('defining ' + symbol)


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
    def __init__(self):
        self.recordQueue = []
        self.focus = {}

    def add_info(self, info):
        records.append(info)
        self.focus['...'] = info



# --------------------- #
# Native def dictionary #
# --------------------- #

context_reader = ContextReader()

recorder = Recorder()

native = {
    'contruct': construct,
    'effectuate': effectuate,
    'record': record,
    'define': define,
}

