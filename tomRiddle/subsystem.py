
class Subsystem:
    def __init__(self):
        self.data = {}
        self.state = {}
        self.context_state = {}
        self.run()

    async def run(self):
        self.data["x"] = "zen"


class DataSubsystem(Subsystem):
    def __init__(self):
        super.__init__()
