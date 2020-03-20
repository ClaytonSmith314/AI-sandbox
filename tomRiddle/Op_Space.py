
import asyncio
import setup
from expiriment import exp_mod
from expiriment import memexp
import Memory

# here's where other coroutines go

running = {
}


# main coroutine. only runs once.
async def main():
    print("entering main function")
    running[asyncio.create_task(exp_mod.exampleCoroutine())] = {}
    for task in running.keys():
        await task
    print("exiting main")
    setup.close()


def start():
    asyncio.run(main())
