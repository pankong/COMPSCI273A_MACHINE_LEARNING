# -*- coding: utf-8 -*-
import numpy as np
import random
from numpy.random import multivariate_normal

class PerceptronLearning():   
    def __init__(self, dimension=9, poolSize=1000):
        self.poolSize = poolSize
        self.sigma = np.eye(dimension)
        self.mu = [0 for _ in range(dimension)]
        self.mvn_large = multivariate_normal(self.mu, self.sigma, poolSize)

    def drawMeans(self):
        poolIndxes = [i for i in range(self.poolSize)]
        [mu1_indx, mu2_indx] = np.random.choice(poolIndxes, 2, replace=False)
        self.mu1 = self.mvn_large[mu1_indx]
        self.mu2 = self.mvn_large[mu2_indx]
        
    def drawMeansForToyOCR(self):
        size = 3 * 3
        k = random.choice(range(3,size-2))
        pixels = [1 for _ in range(k)] + [0 for _ in range(size - k)]
        random.shuffle(pixels)
        self.mu1 = pixels[:]
        while self.mu1 == pixels:
            random.shuffle(pixels)
        self.mu2 = pixels

    def createMVNs(self, alpha, n):
        mvn1 = multivariate_normal(self.mu1, self.sigma * alpha, n)
        mvn2 = multivariate_normal(self.mu2, self.sigma * alpha, n)
        return mvn1, mvn2
    
    def fisherLDA(self, mvn1, mvn2):
        mu1 = np.mean(mvn1, axis=0)
        mu2 = np.mean(mvn2, axis=0)
        #compute Sw
        dif1 = np.subtract(mvn1, mu1)
        dif2 = np.subtract(mvn2, mu2)
        Sw = np.add(np.dot(dif1.T, dif1), np.dot(dif2.T, dif2))
        #compute w
        w = np.dot(np.linalg.inv(Sw), np.subtract(mu2, mu1).T)
        w = np.matrix(w)
        return w
        
    def fisherLDAerrorCalculation(self, alpha, N, ToyOCR):
        #Repeat the fisher LDA for 1000 times to get classification correct rate
        repeats = 1000
        results = []
        for _ in range(repeats):
            #draw means for the two groups
            if ToyOCR:
                self.drawMeansForToyOCR()
            else:
                self.drawMeans()
            #Create training sets and derive w using the training set
            mvn_train1, mvn_train2 = self.createMVNs(alpha, N)
            w = self.fisherLDA(mvn_train1, mvn_train2)
            #Create test sets based on the same mean and alpha, 500 samples in each group
            testSize = 500
            mvn_test1, mvn_test2 = self.createMVNs(alpha, testSize)
            y_test1 = mvn_test1 * w.T
            y_test2 = mvn_test2 * w.T
            mu1 = np.mean(mvn_test1, axis=0)
            mu2 = np.mean(mvn_test2, axis=0)
            #Calculate the classification threshould
            threshold = np.dot(np.add(mu1, mu2), 0.5) * w.T
            #Record classification correct rate for the test sets
            errors = 0
            for y in np.nditer(y_test1):
                if y > threshold:
                    errors += 1
            for y in np.nditer(y_test2):
                if y <= threshold:
                    errors += 1
            rate = (2 * testSize - errors) * 1.0 / (2 * testSize)
            results.append(rate)
        ave = sum(results) / repeats
        std = (sum([(res - ave) ** 2 for res in results]) / (repeats - 1)) ** 0.5
        return ave, std
        
    def varyAlphaN(self, alphas, Ns, ToyOCR=False):
        #Systematically vary alpha and N, and report classification correct rates
        result = {}
        for alpha in alphas:
            for N in Ns:
                ave, std = self.fisherLDAerrorCalculation(alpha, N, ToyOCR)
                result[(alpha, N)] = (ave, std)
                print 'classfication correct rate(%)',
                print 'for alpha=%.2f, N=%5d is: %.2f ± %.2f' %(alpha, N, ave*100, std*100)
        return result
   
    def perceptronLearning(self, X, y, iterations):
        w = np.zeros((1, X.shape[1]))
        for _ in range(iterations):
            for i in range(X.shape[0]):
                x = X[i]
                if np.dot(x, w.T) * y[i] <= 0:
                    w += x * y[i]
        return w
        
    def perceptronErrorCalculation(self, iterations, testSize, ToyOCR):
        #Repeat the perceptron learning for 1000 times to get classification correct rate
        alpha = 0.9
        N = 10
        repeats = 1000
        results = []
        for i in range(repeats):
            if ToyOCR:
                self.drawMeansForToyOCR()
            else:
                self.drawMeans()
            #Create traning sets (10 samples in each group)
            mvn_train1, mvn_train2 = self.createMVNs(alpha, N)
            mvn_train = np.concatenate((mvn_train1, mvn_train2), axis=0)
            x0 = np.ones((mvn_train.shape[0], 1))
            mvn_train = np.concatenate((x0, mvn_train), axis=1)
            y_train1 = np.ones((mvn_train1.shape[0], 1))
            y_train2 = np.ones((mvn_train2.shape[0], 1)) * -1
            y_train = np.concatenate((y_train1, y_train2), axis=0)
            #Derive w using perceptron learning with the given iterations
            w = self.perceptronLearning(mvn_train, y_train, iterations)
            #Create test sets based on the given test set size
            mvn_test1, mvn_test2 = self.createMVNs(alpha, testSize/2)
            y_test1 = np.ones((testSize/2, 1))
            y_test2 = np.ones((testSize/2, 1)) * -1
            mvn_test = np.concatenate((mvn_test1, mvn_test2), axis=0)
            x0_test = np.ones((testSize, 1))
            mvn_test = np.concatenate((x0_test, mvn_test), axis=1)
            y_test = np.concatenate((y_test1, y_test2), axis=0)
            #Record classification correct rate for the test sets
            errors = 0
            for i in range(testSize):
                if np.dot(mvn_test[i], w.T) * y_test[i] <= 0:
                    errors += 1
            rate = (testSize - errors) * 1.0 / testSize
            results.append(rate)
        ave = sum(results) / repeats
        std = (sum([(res - ave) ** 2 for res in results]) / (repeats - 1)) ** 0.5
        return ave, std
        
    def varyIterationsTestSize(self, iterations, testSizes, ToyOCR=False):
        #Systematically vary iterations and size, and report the calculate the misclassification rate
        result = {}
        for i in iterations:
            for testSize in testSizes:
                ave, std = self.perceptronErrorCalculation(i, testSize, ToyOCR)
                result[(i, testSize)] = (ave, std)
                print 'classfication correct rate(%)',
                print 'for iteration=%5d, test set size=%6d is: %.2f ± %.2f' %(i, testSize, ave*100, std*100)
        return result

def varyDimensions():
    #vary the dimensions of the Gaussian and report the classication correct rate
    dimensions = range(1,10)
    results = {}
    for d in dimensions:
        pl = PerceptronLearning(d)
        alpha = 0.3
        N = 100
        ave, std = pl.fisherLDAerrorCalculation(alpha, N, False)
        results[d] = (ave, std)
        print 'classfication correct rate(%)',
        print 'for %d dimension data sets(alpha=%f, N=%d) is: %.2f ± %.2f' %(d, alpha, N, ave*100, std*100)                        
                                                                        
def main():
    """
    The average correct classificate rate is calculated by repeating the process for 1000 times.
    It takes approximately 15 minutes to compute all the results for 1b, 1c, 2b, 2c.
    """
    pl = PerceptronLearning()
    #Define variables for 1b, 2b
    alphas = [0.03, 0.1, 0.3, 0.9, 0.99]
    Ns = [10, 30, 100, 300, 1000]
    #Define variables for 1c, 2c
    iterations = [1, 3, 10, 30, 100]
    testSizes = [10, 30, 100, 300, 1000]
    #Result for 1b
    print "printing results for problem 1b..."
    result_1b = pl.varyAlphaN(alphas, Ns)
    #Result for 1c
    print "printing results for problem 1c..."
    result_1c = pl.varyIterationsTestSize(iterations, testSizes)   
    #Result for 2b
    print "printing results for problem 2b..."
    result_2b = pl.varyAlphaN([0.3], Ns, True)
    #Result for 2c
    print "printing results for problem 2c..."
    result_2c = pl.varyIterationsTestSize(iterations, testSizes, True)
    print "printing results for observation about dimension..."
    varyDimensions()

if __name__ == "__main__":
    main()                     