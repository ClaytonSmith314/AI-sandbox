
import objects as ob
import util
import numpy as np

# vars
running = False
MEMORY_DIR = 'project2/LangModel/memory'

# static data structures
input_devices = {}
output_devices = {}

# data structures

models = {}

metadata = {}


# queues/voltile sets
subject_data = {}
past_data = {}
delta_d = {}


# pre_loading methods

def register_devices():
    print("Registering Devices...")
    inputs = {}
    outputs = {}
    # --DEVICE REGISTRIES GO HERE-- #

    # ----------------------------- #
    print("Device Registration Complete")
    return inputs, outputs

def load_memory():
    pass


# setup

input_devices, output_devices = register_devices()

# event loop

while running:
# stage 1: data acception and designation
    # step 1: get info from devices
    for device in input_devices:
        subject_data[device.designation] = device.get()

    # step 2: cycle the subject data, which is meant to be a voltile dataset
    past_data = subject_data
    subject_data = {}


# stage 2: association analysis
    # step 1 calculate variation
    for designation, data_0 in subject_data:
        if designation in past_data.keys():
            data_1 = past_data[designation]
            delta = data_0 - data_1
            delta_d[designation] = delta

    # step 2 create "graphs", or relationships between information
        # ideally, each designated set should have 1 relationship with every other designated set
        # however, every set can be split into multiple smaller pieces of data
        # the problem to solve here is how to we combine data to compare it to other data
        # the method needs to have these properties: iterative, builds on itself, efficient, sufficient

    # step 3 graph analysis
        # for each graph, once chosen, we need do some analysis on it for the later programm
            # the most important one is coorelation. Coorelation can tell us wether data is related
            # before we even build a model with it.
            # we also need test's to see what kind of data we are getting (discrete, spatial, ordered, or symbolic)
            # this will determine the type of model we use:
            #       discrete    <==>    Dense NN
            #       spatial     <==>    Convolutional NN
            #       ordered     <==>    Recurent NN
            #       symbolic    <==>    Mapping
        # this is the

    # step 4 generate norms
        # norms are associative patterns that are common.
    # all the information gathered here will be stored in metadata

# stage 3: norms - seed formulation (building model/entity pairs)
# stage 4: method tracing and model application <--NOTE: might need an explore step first, to know what to trace
# stage 5: trace evaluation/resolution and declaration with gradient_tape
# stage 6: data comparison and loss generation
# stage 7: loss propagation, gradient calculation and application
# stage 8: output calculation and setting
# stage 9: Update memory. Refresh state






