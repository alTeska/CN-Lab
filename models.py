'''
Collection of models to be used during Computational Neuroscience classes at MSNE TUM
'''


class HHModel:
    def __init__(self, name, age, major):
        self.name = name
        self.age = age
        self.major = major

    def is_old(self):
        return self.age > 100
