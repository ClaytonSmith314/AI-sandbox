
# this module has become obelite. Either find a way to make it usefull, or get rid of it

import asyncio
import setup
import mu
import threading

# this will be list of all the tasks that are running at any given time ["name": task]
tasks = {}

# this will be a tree of what tasks are AWAITING other futures {"name": ["future"]}
# technicly, we could break the hierarhy and create unbreakable await loops - Good or bad???
# -- it's bad. We could break the system

stack_tree = {}


# main coroutine. only runs once.
async def main():
    #global tasks
    print("entering main function")

    # this is where all the logic that is running goes
    await mu.event_loop()

    print("exiting main")


def start():
    asyncio.run(main())


async def run(thread):
    thread.start()

