
import asyncio
import setup
from expiriment import memexp
import Memory

# here's where other coroutines go






# main coroutine. only runs once.
async def main():
    print("entering main function")
    await asyncio.sleep(1)
    await memexp.fillrand(10)
    print(Memory.memory)
    await asyncio.sleep(5)
    print("exiting main")
    setup.close()


def start():
    asyncio.run(main())
