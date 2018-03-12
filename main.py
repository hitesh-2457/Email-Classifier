import sys
from data import dataParser
from NaiveBayes import NB
from Logistic import Logistic


class mailClassification:
    def __init__(self, learnRate, regFactor, iterations):
        self.NBayes = NB(trainPath="train", testPath="test", stopWordFile="stopWords.txt")
        self.Logistic = Logistic(trainPath="train", testPath="test", stopWordFile="stopWords.txt", learnRate=learnRate,
                                 regFactor=regFactor, iterations=iterations)

    def trainNB(self, stopWordFlag):
        self.NBayes.trainNB(stopWordFlag)
        print "Accuracy of Naive Bayes':", self.NBayes.testNB()

    def trainLogistic(self,stopWordFlag):
        self.Logistic.trainLogistic(stopWordFlag)
        print "Accuracy of Logistic Regression:", self.Logistic.testLogistic(),
        print "with Learning Rate:", self.Logistic.learnRate,
        print "Regularization Factor:", self.Logistic.regFactor,
        print "Iterations:", self.Logistic.iterations


def main():
    learnRate = (float)(sys.argv[1])
    regFactor = (float)(sys.argv[2])
    iterations = (int)(sys.argv[3])
    classify = mailClassification(learnRate=learnRate, regFactor=regFactor, iterations=iterations)
    print "Stop Words excluded: "
    classify.trainNB(stopWordFlag=True)
    classify.trainLogistic(stopWordFlag=True)
    print "Without excluding Removing Stop Words: "
    classify.trainNB(stopWordFlag=False)
    classify.trainLogistic(stopWordFlag=False)


if __name__ == "__main__":
    main()
