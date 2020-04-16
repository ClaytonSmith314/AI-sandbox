
from mu import heap, record, data, context, stack, f_context


# --------------- #
# control Methods #
# --------------- #

def event():
    f_recur(f_context, 'main')


def f_recur(xdir, name):  # this method doesn't seem to fit well with the Learning Cycle
    """recusively loops through all echo context"""
    stack.append(name)
    print(stack)
    for key, pair in xdir.items():
        # TODO: figure out how to perform opperations in responce to the presence of symbols
        if type(pair).__name__ == 'dict':
            f_recur(pair, key)
    stack.pop()
    print(stack)


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

def initiate():
    """step one of learning cycle"""
    # take heap data
    # structure it in some way
    # put it into context
    # label it as ready to solve
    


def effectuate():
    """step two of learning cycle"""
    # take context information
    # read syntax according to given expression notation
    # run each peace as a step. change stages when complete
    # edit data from


def record():
    """step three of learning cycle"""
    # take opperation done
    # create an expression for the opperation
    # add expression to end of record list


def define():
    """step 4 of learning cycle"""
    # start once the record has completed a symbol
    # take resulting opperation and encode it into a cycle or expression
    # generate a half random symbol. Make sure it doesn't match any other generated symbols.
    # add symbol and definition to heap
