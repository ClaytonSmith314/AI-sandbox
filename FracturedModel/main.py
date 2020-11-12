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