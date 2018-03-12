import math

import copy

from Model import Model
from data import dataParser


class NB:
    def __init__(self, trainPath, testPath, stopWordFile):
        self.trainPath = trainPath
        self.testPath = testPath

        self.stopWords = dataParser().parse_file(stopWordFile)

        self.classes = ['ham', 'spam']

        self.trainData = {}
        for c in self.classes:
            self.trainData[c] = Model(self.trainPath, c)

        self.testData = {}
        for c in self.classes:
            self.testData[c] = Model(self.testPath, c)

        self.vocab = set()
        self.prior = {}
        self.condProb = {}

    def addToList(self, wordList, dataList):
        for word in wordList:
            if word in self.vocab:
                dataList[word] += 1

    def removeSWTrain(self, dataSet):
        self.vocab = set()
        for c in self.classes:
            self.rmvSWTest(dataSet[c].wordList)
            self.vocab.update(dataSet[c].wordList.keys())

    def rmvSWTest(self, dataSet):
        for w in self.stopWords:
            if w in dataSet:
                dataSet.pop(w)

    def trainNB(self, stopWordFlag):
        self.stopWordFlag = stopWordFlag
        parser = dataParser()
        totalDocs = 0
        self.build_vocab()
        totalDocs = self.build_wordFreq(parser, totalDocs)
        self.calc_cond_prob(totalDocs)

    def calc_cond_prob(self, totalDocs):
        for c in self.classes:
            self.prior[c] = (len(self.trainData[c].files) * 1.0) / totalDocs
            for t in self.vocab:
                if t in self.trainData[c].wordList:
                    wordCount = self.trainData[c].wordList[t]
                    if t not in self.condProb:
                        self.condProb[t] = {}
                    self.condProb[t][c] = (wordCount * 1.0 + 1) / (
                            sum(self.trainData[c].wordList.values()) + len(self.vocab))

    def build_wordFreq(self, parser, totalDocs):
        for c in self.classes:
            totalDocs += len(self.trainData[c].files)
            for v in self.vocab:
                self.trainData[c].wordList[v] = 0
            for file in self.trainData[c].files:
                wordList = parser.parse_file(self.trainPath + "/" + c + "/" + file)
                self.addToList(wordList=wordList, dataList=self.trainData[c].wordList)
        return totalDocs

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

    def applyNB(self, file):
        parser = dataParser()
        score = {}
        wordList = parser.parse_file(file)
        for cl in self.classes:
            score[cl] = math.log(self.prior[cl])
            for w in wordList:
                if w in self.vocab and cl in self.condProb[w]:
                    score[cl] += math.log10(self.condProb[w][cl])

        return "spam" if score["spam"] > score["ham"] else "ham"

    def testNB(self):
        hits = 0
        totalFiles = 0
        for c in self.classes:
            totalFiles += len(self.testData[c].files)
            for file in self.testData[c].files:
                pred = self.applyNB(self.testPath + "/" + c + "/" + file)
                if pred == c:
                    hits += 1
        return ((hits * 1.0) / totalFiles) * 100
