
import Memory

# this is the location of all main data structures used in the program


# ---------------------------- #
# Symbol level Data Structures #
# ---------------------------- #

records = []

data = {}

context = {}

heap = {}


# -------------------------------- #
#  f-scheme level data structures  #
# -------------------------------- #

stack = []

f_context = {
    'construct': [],
    'effectuate': {},
    'record': {},
    'define': {},
}


# the main event loop that runs all actions in program

async def event_loop():
    import mi
    global heap, context

    heap = Memory.load('heap')
    print(heap)

    """this loop runs all the tasks in the task list and checks the context for what to do"""
    awake = 'true'
    while awake == 'true':
        mi.event()
        # awake = 'false'  # temorarily stops loop from running forever
