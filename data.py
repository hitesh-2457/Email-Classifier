import re


class dataParser:
    def __init__(self):
        self.pattern = "[a-zA-Z]+"

    def parse_file(self, fileName):
        f = open(fileName, "r")
        myList = re.findall(self.pattern, f.read().lower())
        f.close()
        return myList
