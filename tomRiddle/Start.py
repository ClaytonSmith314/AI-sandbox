import Memory
import mu
import mi


print(
    "\n\n"
    "//***********************************\\\\\n"
    "||                                   ||\n"
    "||           Tom Riddle AI           ||\n"
    "||                                   ||\n"
    "\\\\***********************************//\n"
)

print('loading memory')
mu.heap = Memory.load('heap')

print('initializing threads')
mi.run()
print('start exited')