
import asyncio
import Memory
from expiriment import memexp
import Op_Space

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
    b = asyncio.create_task(fibo(34))
    await memexp.fillrand(10)
    print(Memory.memory)
    print(len(Memory.memory))
    await asyncio.sleep(5)
    print("tea time!")
    print(await b)
    await t


# don't ever run these. This will break the computer
async def paradox1():
    Op_Space.tasks["paradox2"] = asyncio.create_task(paradox2())
    print("start paradox1")
    await Op_Space.tasks["paradox2"]
    print("paradox1 resolved")


async def paradox2():
    print("start paradox2")
    await Op_Space.tasks["paradox1"]
    print("paradox2 resolved")


def recur_fibo(n):
    if n <= 1:
        return n
    else:
        r = recur_fibo(n-1) + recur_fibo(n-2)
        if r > 500:
            print(".")
        return r


async def fibo(n):
    return recur_fibo(n)
