import Memory
import random


async def fillrand(n):
    letters = 'abcdefghijklmnopqrstuvwxyz '
    x = 0
    for x in range(n):
        key = ''.join(random.choice(letters) for i in range(8))
        data = ''.join(random.choice(letters) for i in range(20))
        Memory.memory[key] = data
