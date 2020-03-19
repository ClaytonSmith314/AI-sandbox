
import json

# handles interaction between program and memory files

root = "C:/Users/CSmith/Documents/GitHub/AI-sandbox/Memory/"
mfile = "memory.json"

memory = {

}


def load(path=root+mfile):
    with open(path) as f:
        global memory
        memory = json.load(f)


def dump(path=root+mfile):
    with open(path, 'w') as f:
        json.dump(memory, f)
