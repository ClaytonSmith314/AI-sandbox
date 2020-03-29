
import json

# TODO: update Memory to work with multiple files efectively
# TODO: Update Memory to be asynchornous
# handles interaction between program and memory files

root = "C:/Users/CSmith/Documents/GitHub/AI-sandbox/Memory/"
mfile = "expmemory.json"

memory = {

}


def load(path=root+mfile):
    with open(path) as f:
        global memory
        memory = json.load(f)


def dump(path=root+mfile):
    with open(path, 'w') as f:
        json.dump(memory, f, indent=1)


def clear_memory(path=root+mfile):
    global memory
    memory = {}
    dump(path)


def update(symbol, value, path=root+mfile):
    global memory
    memory[symbol] = value
    dump(path)
