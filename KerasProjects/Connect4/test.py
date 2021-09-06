import numpy as np
import math

x = np.arange(0,5)
print(x)

y = np.array([10,40,0,5,5])
print(y)
print(y/np.sum(y))

for n in range(0,20):
    print(np.random.choice(x, p=y/np.sum(y)))

size=7*6
score=1
decay_rate = .99
for n in range(size):
    print((score - .5) * math.pow(decay_rate, n) + .5)