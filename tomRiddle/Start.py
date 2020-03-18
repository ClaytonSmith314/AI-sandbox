
import Memory
import random
import setup

setup.boot()

print(
    "\n\n"
    "//***********************************\\\\\n"
    "||                                   ||\n"
    "||           Tom Riddle AI           ||\n"
    "||                                   ||\n"
    "\\\\***********************************//\n"
)

# put AI initializing code here

letters = 'abcdefghijklmnopqrstuvwxyz '
x = 0
for x in range(1000):
    key = ''.join(random.choice(letters) for i in range(8))
    data = ''.join(random.choice(letters) for i in range(20))
    Memory.memory[key] = data

print(Memory.memory)
print(len(Memory.memory))


setup.close()
