
import asyncio
from subsystem import Subsystem

# here is where everything that makes this an AI will go


async def mu():
    print("mu running")


class mu(Subsystem):
    def __init__(self):
        super.__init__()