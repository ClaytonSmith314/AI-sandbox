
from random import Random as rand
import random
from LangModel import objects as ob
import tensorflow as tf

BASE_SIZE = 128

# holds utility methods needed for the model


# memory
def load_all(directory):
    pass


def save_all(directory):
    pass


# creation
def rand_string(n=16, letters='abcdefghijklmnopqrstuvwxyz0123456789'):
    key = ''.join(random.choice(letters) for i in range(n))
    return key


# utility classes

class Device:
    def __init__(self, assigned_designation, record=False, has_in=True, has_out=True):
        self.designation = assigned_designation
        self.record = record
        self.has_in = has_in
        self.has_out = has_out

    def get(self):
        pass

    def set(self, arg):
        pass

    def register(self, inputs, outputs):
        if self.has_in:
            inputs[self.designation] = self
        if self.has_out:
            inputs[self.designation] = self
