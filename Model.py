from os import listdir


class Model:
    def __init__(self, path, cl):
        self.files = listdir(path + "/" + cl)
        self.wordList = {}
