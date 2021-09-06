



class Rule:
    def __init__(self, searcher, validater, asserter):
        self.searcher = searcher
        self.validater = validater
        self.asserter = asserter


class Object:
    def __init__(self, referances, data):
        self.references = referances
        self.data = data


