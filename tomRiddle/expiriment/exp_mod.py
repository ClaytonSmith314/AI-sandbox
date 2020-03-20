
import asyncio
import Memory
from expiriment import memexp

mystring = "MU"


def mymethod(append):
    global mystring
    mystring = mystring + append

async def count_down(n):
    while n >= 0:
        print(n)
        await asyncio.sleep(1)
        n -= 1

async def exampleCoroutine():
    t = asyncio.create_task(count_down(10))
    await asyncio.sleep(2.5)
    await memexp.fillrand(10)
    print(Memory.memory)
    print(len(Memory.memory))
    await asyncio.sleep(5)
    print("tea time!")
    await t