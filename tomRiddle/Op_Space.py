
import asyncio
import setup
from expiriment import memexp
import Memory

# here's where other coroutines go


async def count_down(n):
    while n >= 0:
        print(n)
        await asyncio.sleep(1)
        n -= 1




# main coroutine. only runs once.
async def main():
    print("entering main function")
    t = asyncio.create_task(count_down(10))
    await asyncio.sleep(2.5)
    await memexp.fillrand(10)
    print(Memory.memory)
    print(len(Memory.memory))
    await asyncio.sleep(5)
    print("tea time!")
    await t
    print("exiting main")
    setup.close()


def start():
    asyncio.run(main())
