import Memory
import random

def fillRand(k):
    letters = 'abcdefghijklmnopqrstuvwxyz '
    x = 0
    for x in range(1000):
        key = ''.join(random.choice(letters) for i in range(8))
        data = ''.join(random.choice(letters) for i in range(20))
        Memory.memory[key] = data

