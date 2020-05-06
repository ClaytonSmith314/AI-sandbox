
import Memory
import time

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


# the main event loop that runs all actions in program

def event_loop():
    import mi, native
    global heap, context

    heap = Memory.load('heap')
    native.build()

    mi.initialize()
    """this loop runs all the tasks in the task list and checks the context for what to do"""
    awake = True
    waiting = True  # helps to see eaach cycle pass by one at a time
    while awake:
        mi.event()
        if(waiting):
            time.sleep(1)
            print('...')

