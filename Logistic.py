import math
import copy

from Model import Model
from data import dataParser


class Logistic:
    def __init__(self, trainPath, testPath, stopWordFile, learnRate, regFactor, iterations):
        self.trainPath = trainPath
        self.testPath = testPath
        self.learnRate = learnRate
        self.regFactor = regFactor
        self.iterations = iterations

        self.stopWords = dataParser().parse_file(stopWordFile)

        self.classes = ['ham', 'spam']

        self.trainData = {}
        for c in self.classes:
            self.trainData[c] = Model(self.trainPath, c)

        self.testData = {}
        for c in self.classes:
            self.testData[c] = Model(self.testPath, c)

        self.vocab = set()
        self.weightList = {}

    def addToList(self, wordList, dataList):
        for word in wordList:
            self.vocab.add(word)
            if word in dataList:
                dataList[word] += 1
            else:
                dataList[word] = 1

    def removeSWTrain(self, dataSet):
        self.vocab = set()
        for c in self.classes:
            self.rmvSWTest(dataSet[c].wordList)

            self.vocab.update(dataSet[c].wordList.keys())

    def rmvSWTest(self, dataSet):
        for w in self.stopWords:
            if w in dataSet:
                dataSet.pop(w)

    def build_vocab(self):
        parser = dataParser()

        self.vocab = set()
        for c in self.classes:
            for file in self.trainData[c].files:
                wordList = parser.parse_file(self.trainPath + "/" + c + "/" + file)
                self.vocab.update(wordList)

        if self.stopWordFlag:
            for w in self.stopWords:
                if w in self.vocab:
                    self.vocab.remove(w)

    def build_wordFreq(self, parser, totalDocs):
        for c in self.classes:
            totalDocs += len(self.trainData[c].files)
            for v in self.vocab:
                self.trainData[c].wordList[v] = 0
            for file in self.trainData[c].files:
                wordList = parser.parse_file(self.trainPath + "/" + c + "/" + file)
                self.addToList(wordList=wordList, dataList=self.trainData[c].wordList)
        return totalDocs

    def sum_series(self, wordList):
        sum = self.weightList["weight_zero"]
        for w in wordList:
            sum += (self.weightList[w] * wordList[w])
        return sum

    def calc_prob(self, wordList):
        try:
            exp = math.exp(self.sum_series(wordList))
            return ((float)(exp) / (1 + exp))
        except:
            return 0

    def tot_count(self, word):
        xi = 0
        for c in self.classes:
            xi += self.trainData[c].wordList[word]
        return xi

    def countWords(self):
        parser = dataParser()
        self.countMatrix = {}
        for c in self.classes:
            self.countMatrix[c] = {}
            for file in self.trainData[c].files:
                wordList = parser.parse_file(self.trainPath + "/" + c + "/" + file)
                wordListFreq = {}
                self.addToList(wordList, wordListFreq)
                if self.stopWordFlag:
                    self.rmvSWTest(wordListFreq)
                self.countMatrix[c][file] = copy.deepcopy(wordListFreq)

    def updateWeights(self):
        tempWeights = copy.deepcopy(self.weightList)
        for i in self.weightList:
            term = 0
            for c in self.classes:
                for file in self.trainData[c].files:
                    if i in self.countMatrix[c][file]:
                        term += ((self.classes.index(c) - self.calc_prob(self.countMatrix[c][file])) *
                                 self.countMatrix[c][file][i])
            tempWeights[i] += ((term - (self.regFactor * self.weightList[i])) * self.learnRate)
        self.weightList = copy.deepcopy(tempWeights)

    def trainLogistic(self, stopWordFlag):
        self.stopWordFlag = stopWordFlag

        self.build_vocab()
        totalDocs = 0
        self.build_wordFreq(parser=dataParser(), totalDocs=totalDocs)

        initial_weights = 0.01
        self.weightList["weight_zero"] = initial_weights
        for w in self.vocab:
            self.weightList[w] = initial_weights
        self.countWords()
        for i in range(1, self.iterations):
            self.updateWeights()

    def testLogistic(self):
        parser = dataParser()
        hits = 0
        totalDocs = 0
        for c in self.classes:
            totalDocs += len(self.testData[c].files)
            for file in self.testData[c].files:
                wordList = parser.parse_file(self.testPath + "/" + c + "/" + file)
                wordListFreq = {}
                self.addToList(wordList, wordListFreq)
                pred = (int)(round(self.calc_prob(wordListFreq)))
                if c == self.classes[pred]:
                    hits += 1
        return ((float)(hits) * 100) / totalDocs
