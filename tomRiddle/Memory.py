
import json

# TODO: update lang to work with multiple files efectively
# TODO: Update lang to be asynchornous
# handles interaction between program and lang files

root = "C:/Users/CSmith/Documents/GitHub/AI-sandbox/Memory/"

files = {
}


def import_file(file_id, file):
    files[file_id] = file


def load(file_id):
    path = root + files[file_id]
    with open(path) as f:
        return json.load(f)


def dump(d, file_id):
    path = root + files[file_id]
    with open(path, 'w') as f:
        json.dump(d, f, indent=1)


def clear_memory(d, file_id):
    path = root + files[file_id]
    d = {}
    dump(d)
    return {}


import_file('heap', 'heap.json')
