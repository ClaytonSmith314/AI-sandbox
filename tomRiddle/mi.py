
from mu import heap, Objectives_q, Objectives, Context_q, Context, \
    Results_q, Results, Situation_q, Situation, awake, Info, rules
import random
import threading
import time

# --------------- #
# control Methods #
# --------------- #

def run():
    select_thread = threading.Thread(target=select)
    restructure_thread = threading.Thread(target=restructure)
    actuate_thread = threading.Thread(target=actuate)
    record_thread = threading.Thread(target=record)
    learning_thread = threading.Thread(target=learn)

    select_thread.start()
    restructure_thread.start()
    actuate_thread.start()
    record_thread.start()
    learning_thread.start()

    # while(awake):
    #    time.sleep(1)
    #    print('...')

# -------------------------------

def select():
    # SELECT IS DONE for now...
    desires = {}
    while awake:
        print('selecting')
    # accepting phase
            # we need to check to see if new data
        while len(Situation_q) > 0:
            signal, parameters = Situation_q.pop(0)
            Situation[signal] = parameters
            # check to see if matches desired results
            # two phases - 1: does info match the TYPE of info wanted?
            #              2: do the results match the desired results?
            if signal in desires:  # note- some desires have conditions for them to be resolved
                del desires[signal]
            if signal in rules:  # rules map situations to desires
                desire = rules[signal]
                desires[signal] = desire  # note: desires doesn't do anything here...
                if desire in heap:  # if we have a response for that desire
                    Objectives_q.append((desire, parameters))  # what about parameters?

    # running phase
        # so far, we've done everthing in the accpeting phase. We can come back to this and add something here later


    # returning phase
        time.sleep(.02)  # sleep to aviod waisting system resources


def restructure():  # restructure needs to turn objectives into context commands
    while awake:
        print('restructuring')
        # accepting phase
        while len(Objectives_q) > 0:
            desire, parameters = Objectives_q.pop(0)
            Objectives[desire] = parameters
            response = heap[desire]
            Context_q.append((response, parameters))

        # running phase

        # returning phase
        time.sleep(.02)  # sleep to aviod waisting system resources


def actuate():
    while awake:
        print('acuating')
        # accepting phase
        while len(Context_q) > 0:
            signal, parameters = Context_q.pop(0)
            Context[signal] = parameters
            # run each one on the data provided???

            # two options. run each context as it comes in here
            # or run parts of each until done...
            # results are going to be context based. We need the results of a context
            # to return based in that context

        # running phase
        # returning phase
        time.sleep(.01)  # sleep to aviod wasting system resources


def record():
    while awake:
        print('recording')
        # accepting phase
        while len(Results_q) > 0:
            name, info = Results_q.pop(0)
            Results[name] = info


        # running phase

        # returning phase
        time.sleep(.02)  # sleep to aviod waisting system resources


def learn():
    while awake:
        print('learning')




############################################


def define(expression):
    name = rand_string(5)  # make a new name for the symbol
    heap[name] = expression  # set heap definition of new random symbol to the new potential
    print(name + ' =  ' + expression)



############################################


class Selector:

    def __init__(self):
        """ """

    def select(self):
        """ """


selector = Selector()


############################################


class Constructor:

    def __init__(self):
        """ """

    def construct(self):
        """ """


constructor = Constructor()


############################################


class Actuator:

    def __init__(self):
        """ """

    def actuate(self):
        """ """


actuator = Actuator()


############################################


class Recorder:

    def __init__(self):
        self.potentials = {}
        self.path = ""

    def record(self):
        """ """

    def mark(self, statement):
        """ """
        # add code to add statement correctly to potentials

    def define(self):
        """ """
        # loop through potentials. find ones ready to define (they have a closing statement)


recorder = Recorder()



# ----- #
# Other #
# ----- #

def rand_string(n):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    key = ''.join(random.choice(letters) for i in range(n))
    return key
