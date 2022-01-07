
import os

class test():
    def __init__(self, id_, name):
        self.id = id_
        self.name = name

    def printf(self):
        print(self.id)
        print(self.name)

id1 = test(1, "name")
id1.printf()