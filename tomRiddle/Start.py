
import Memory
import setup
from expiriment import memexp

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

print(Memory.memory)
print(len(Memory.memory))


def finish():
    setup.close()


finish()
