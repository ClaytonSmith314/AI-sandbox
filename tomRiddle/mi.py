
from mu import heap, record, data, context, stack, f_context


# --------------- #
# control Methods #
# --------------- #

def event():
    f_manage()


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

def construct(symbol, info):
    """step one of learning cycle"""
    # take heap data
    # structure it in some way      # need to define a running structure. But what?
    # put it into context
    # label it as ready to solve
    print('constructing ' + symbol)
    

def effectuate(symbol, info):
    """step two of learning cycle"""
    # take context information
    # read syntax according to given expression notation
    # run each peace as a step. change stages when complete
    # edit data from
    print('effectuating ' + symbol)


def record(symbol, info):
    """step three of learning cycle"""
    # take opperation done
    # create an expression for the opperation
    # add expression to end of record list
    print('recording ' + symbol)

def define(symbol, info):
    """step 4 of learning cycle"""
    # start once the record has completed a symbol
    # take resulting opperation and encode it into a cycle or expression
    # generate a half random symbol. Make sure it doesn't match any other generated symbols.
    # add symbol and definition to heap
    print('defining ' + symbol)


# --------------------- #
# Native def dictionary #
# --------------------- #

native = {
    'contruct': construct,
    'effectuate': effectuate,
    'record': record,
    'define': define,
}
