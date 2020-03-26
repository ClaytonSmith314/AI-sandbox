
import asyncio
import setup
from expiriment import exp_mod
from expiriment import memexp
import Memory

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

    print("exiting main")
    setup.close()


def start():
    asyncio.run(main())
