
class Subsystem:
    def __init__(self):
        self.data = {}
        self.state = {}
        self.run()

    async def run(self):
        self.data["x"] = "zen"
