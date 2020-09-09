
import objects as ob
import util
import numpy as np

# vars
running = False
MEMORY_DIR = 'project2/LangModel/memory'
BASE_SIZE = 128

# data structures

# contains all learned data, including functional and structural models
models = {}

state = {}

tracks = {}

metadata = {}


# pre_loading

# ... need to load designations and other stuff

# execution

print(id(29))

# setup
firstTime = True
if firstTime:
    methods = util.stems()


# event loop

while running:
    pass
    # stage 1: data acception and designation
    # stage 2: association anylisis
    # stage 3: seed formulation (building model/entity pairs)
    # stage 4: method tracing and model application <--NOTE: might need an explore step first, to know what to trace
    # stage 5: trace evaluation/resolution with gradent_tape
    # stage 6: data comparison and loss generation
    # stage 7: loss propogation, gradent calculation and aplication
    # stage 8: Update memory. Refresh state





