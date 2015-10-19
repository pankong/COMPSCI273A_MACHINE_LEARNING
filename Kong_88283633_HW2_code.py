# -*- coding: utf-8 -*-
from __future__ import division
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.special import expit

#total size for the training and testing data
dataSetSize = 400

"""
backbone fuctions for the artificial neural network questions
"""
class ArtificalNeuralNetwork():
    def __init__(self, dimension, N_hidden, classification):
        #classification is a boolean value which means whether the question is a classification problem
        self.dimension = dimension
        self.N_hidden = N_hidden
        self.classification = classification
        
    def generateW(self, l1, l2):
        #randomly initialize W around 0 with the size of layer 1 and layer 2
        mu = [0 for _ in range(l1 + 1)]
        cov = np.eye(l1 + 1)
        W = multivariate_normal(mu, cov, l2)
        return W
    
    def backPropogation(self, X, Y, W12, W23):
        #calculate the gradients for W using back propagation
        m = X.shape[0]
        #forward propagation
        A1 = np.concatenate((np.ones((m, 1)), X), axis=1)
        Z2 = np.dot(A1, W12.T)
        A2 = expit(Z2)
        A2 = np.concatenate((np.ones((m, 1)), A2), axis=1)
        Z3 = np.dot(A2, W23.T)
        Y_hat = expit(Z3)
        #back propagation
        delta3 = Y_hat - Y
        delta2 = np.multiply(np.dot(delta3, W23)[:,1:], np.multiply(expit(Z2), 1 - expit(Z2)))
        Delta1 = np.dot(delta2.T, A1)
        Delta2 = np.dot(delta3.T, A2)
        #calculate gradients for W12 and W23
        W12_grad = Delta1 / m
        W23_grad = Delta2 / m
        #calculate regularizations for W12 and W23
        W12_reg = np.concatenate((np.zeros((self.N_hidden, 1)), W12[:,1:]), axis=1) / m
        W23_reg = np.concatenate((np.zeros((Y.shape[1], 1)), W23[:, 1:]), axis=1) / m
        return W12_grad, W23_grad, W12_reg, W23_reg
        
    def learning(self, X, Y, learningRate, iteration, regWeight, repeats=100, plotW=False):
        #learn Ws based on the given learning rate, iteration and regularization weight
        train_results = []
        test_results = []
        #randomly initialize Ws
        W12 = self.generateW(self.dimension, self.N_hidden)
        W23 = self.generateW(self.N_hidden, Y.shape[1])
        #divide the data set into training set and testing set
        X_train = X[:dataSetSize/2]
        Y_train = Y[:dataSetSize/2]
        X_test = X[dataSetSize/2:]
        Y_test = Y[dataSetSize/2:]
        #repeat the neural network learning process to get statistics
        for _ in range(repeats):
            for _ in range(iteration):
                W12_grad, W23_grad, W12_reg, W23_reg = self.backPropogation(X_train, Y_train, W12, W23)
                W12 = W12 - learningRate * W12_grad - regWeight * W12_reg
                W23 = W23 - learningRate * W23_grad - regWeight * W23_reg
            if self.classification:
                train_result = self.calculateCorrectRate(X_train, Y_train, W12, W23)
                test_result = self.calculateCorrectRate(X_test, Y_test, W12, W23)
            else:
                train_result = self.calculateSquaredError(X_train, Y_train, W12, W23)
                test_result = self.calculateSquaredError(X_test, Y_test, W12, W23)
            train_results.append(train_result)
            test_results.append(test_result)
        #return W12 and W23 for hinton plots
        if plotW:
            return W12, W23
        #calculate average and standard deviation
        train_ave = sum(train_results) / repeats
        train_std = (sum([(res - train_ave) ** 2 for res in train_results]) / (repeats - 0)) ** 0.5
        test_ave = sum(test_results) / repeats
        test_std = (sum([(res - test_ave) ** 2 for res in test_results]) / (repeats - 0)) ** 0.5
        return train_ave, train_std, test_ave, test_std
    
    def varyHyperParameters(self, X, Y, learningRates, iterations, regWeights):
        #change the hyper parameters to see effects on neural network learning
        for learningRate in learningRates:
            for iteration in iterations:
                for regWeight in regWeights:
                    train_ave, train_std, test_ave, test_std = self.learning(X, Y, learningRate, iteration, regWeight)
                    if self.classification:
                        print "Classification correct rate(%) on the training set ",
                        print "for learningRate=%6.3f, Niteration=%4d, regWeight=%.2f is %.2f ± %.2f" %(learningRate, iteration, regWeight, train_ave * 100, train_std * 100)
                        print "Classification correct rate(%) on the test set ",
                        print "for learningRate=%6.3f, Niteration=%4d, regWeight=%.2f is %.2f ± %.2f" %(learningRate, iteration, regWeight, test_ave * 100, test_std * 100)
                    else:    
                        print "Averaged squared error on the training set ",
                        print "for learningRate=%6.3f, Niteration=%4d, regWeight=%.2f is %.2f ± %.2f" %(learningRate, iteration, regWeight, train_ave * 100, train_std * 100)
                        print "Averaged squared error on the testing set ",
                        print "for learningRate=%6.3f, Niteration=%4d, regWeight=%.2f is %.2f ± %.2f" %(learningRate, iteration, regWeight, test_ave * 100, test_std * 100)
                    sys.stdout.flush()
                    
    def calculateCorrectRate(self, X, Y, W1, W2, threshold=0.5):
        m = X.shape[0]
        A1 = np.concatenate((np.ones((m, 1)), X), axis=1)
        A2 = expit(np.dot(A1, W1.T))
        A2 = np.concatenate((np.ones((m, 1)), A2), axis=1)
        Y_hat = expit(np.dot(A2, W2.T))
        Y_hat = 1 * (Y_hat > threshold)
        wrongs = int(sum(np.absolute(Y_hat - Y)))
        return (Y.shape[0] - wrongs) / Y.shape[0]
        
    def calculateSquaredError(self, X, Y, W1, W2):
        m = X.shape[0]
        A1 = np.concatenate((np.ones((m, 1)), X), axis=1)
        A2 = expit(np.dot(A1, W1.T))
        A2 = np.concatenate((np.ones((m, 1)), A2), axis=1)
        Y_hat = expit(np.dot(A2, W2.T))
        squaredError = np.square(Y_hat - Y)
        return sum(sum(squaredError)) / squaredError.shape[0]
    
    def plotW(self, W):
        plt.pcolor(np.array(W), cmap=plt.cm.seismic, vmin=W.min(), vmax=W.max())
        plt.colorbar()
        plt.show()

"""
Solution for problem 1a auto encoder
"""
class AutoEncoder(ArtificalNeuralNetwork):
    def __init__(self, dimension, N_hidden, classification=False):
        ArtificalNeuralNetwork.__init__(self, dimension, N_hidden, classification)
    
    def generateAutoencoderData(self, n):
        np.random.seed(2222)
        n_dim_full = self.dimension
        n_dim_limited = n_dim_full - 10
        eigenvals_big = np.random.randn(n_dim_limited) + 3
        eigenvals_small = np .abs(np.random.randn(n_dim_full - n_dim_limited)) * .1
        eigenvals = np.concatenate([eigenvals_big, eigenvals_small])
        diag = np.diag(eigenvals)
        q , r = np.linalg.qr(np.random.randn(n_dim_full, n_dim_full))
        cov_mat = q.dot(diag).dot(q.T)
        mu = np.zeros(n_dim_full)
        X = np.random.multivariate_normal(mu, cov_mat, n)
        X = expit(X)
        return X, X.copy()
        
    def run(self, learningRates, iterations, regWeights):
        X, Y = self.generateAutoencoderData(dataSetSize)
        self.varyHyperParameters(X, Y, learningRates, iterations, regWeights)
    
    def plot(self, learningRate, iteration, regWeight):
        X, Y = self.generateAutoencoderData(dataSetSize)
        W12, W23 = self.learning(X, Y, learningRate, iteration, regWeight, repeats=10, plotW=True)
        self.plotW(W12)
        #self.plotW(W23)


"""
Solution for problem 1b exactly-one-on
"""
class ExactlyOneOn(ArtificalNeuralNetwork):
    def __init__(self, dimension, N_hidden, classification=True):
        ArtificalNeuralNetwork.__init__(self, dimension, N_hidden, classification)
        
    def generateExactlyOneOnData(self, n):
        #create X with the pattern of normal distribution and Y with exactly-one-on
        np.random.seed(1234)
        X = np.random.rand(n, self.dimension)
        threshold = 1 / self.dimension
        X_binary = 1 * (X < threshold)
        Y = np.matrix(1 * (np.sum(X_binary, 1) == 1))
        return X_binary, Y.T 

    def run(self, learningRates, iterations, regWeights):
        X, Y = self.generateExactlyOneOnData(dataSetSize)
        self.varyHyperParameters(X, Y, learningRates, iterations, regWeights)
    
    def plot(self, learningRate, iteration, regWeight):
        X, Y = self.generateExactlyOneOnData(dataSetSize)
        W12, W23 = self.learning(X, Y, learningRate, iteration, regWeight, repeats=10, plotW=True)
        self.plotW(W12)
        #self.plotW(W23)
      
            
"""
Solution for problem 1c OCR
"""
class OCR(ArtificalNeuralNetwork):
    def __init__(self, dimension, N_hidden, classification=True):
        ArtificalNeuralNetwork.__init__(self, dimension, N_hidden, classification)
    
    def generateOCRData(self, n):
        np.random.seed(2222)
        k = random.choice(range(int(self.dimension/3), int(self.dimension/1.5)))
        pixels = [1 for _ in range(k)] + [0 for _ in range(self.dimension - k)]
        random.shuffle(pixels)
        mu1 = pixels[:]
        while mu1 == pixels:
            random.shuffle(pixels)
        mu2 = pixels
        X1 = multivariate_normal(mu1, 0.5 * np.eye(self.dimension), int(n/2))
        Y1 = np.ones((int(n/2), 1))
        X2 = multivariate_normal(mu2, 0.5 * np.eye(self.dimension), int(n/2))
        Y2 = np.zeros((int(n/2), 1))
        X = np.concatenate((np.concatenate((X1[:int(n/4)], X2[:int(n/4)]), axis=0), np.concatenate((X1[int(n/4):], X2[int(n/4):]), axis=0)), axis=0)
        Y = np.concatenate((np.concatenate((Y1[:int(n/4)], Y2[:int(n/4)]), axis=0), np.concatenate((Y1[int(n/4):], Y2[int(n/4):]), axis=0)), axis=0)  
        return X, Y  
        
    def run(self, learningRates, iterations, regWeights):
        X, Y = self.generateOCRData(dataSetSize)
        self.varyHyperParameters(X, Y, learningRates, iterations, regWeights)
        
    def plot(self, learningRate, iteration, regWeight):
        X, Y = self.generateOCRData(dataSetSize)
        W12, W23 = self.learning(X, Y, learningRate, iteration, regWeight, repeats=10, plotW=True)
        #self.plotW(W12)
        self.plotW(W23)
        self.varyHyperParameters(X, Y, [0.3], [1000], [0.1])

"""
total running time for all the three questions will be around 45 mins
"""        
def main():
    
    #problem 1a autoencoder
    autoencoder = AutoEncoder(40, 20)
    print "printing results for problem 1 autoencoder..."
    learningRates = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10]
    Niterations = [1, 3, 10, 30, 100, 300, 1000]
    autoencoder.run(learningRates, Niterations, [0])
    autoencoder.plot(0.3, 1000, 0)    
    
    #problem 1b exactly-one-on
    exactlyoneon = ExactlyOneOn(10, 10)
    print "printing results for problem 1 exactly-one-on..."
    Niterations = [30, 100, 300, 1000, 3000, 10000]
    exactlyoneon.run([0.3], Niterations, [0.1])
    learningRates = [0.01, 0.03, 0.1, 0.2, 0.3, 0.9]
    exactlyoneon.run(learningRates, [1000], [0.1])
    exactlyoneon.plot(0.3, 300, 0.1)    
    
    #problem 2 OCR
    ocr = OCR(35, 20)
    print "printing results for problem 2 OCR..."
    learningRates = [0.01, 0.03, 0.1, 0.3, 1.0]
    Niterations = [1, 3, 10, 30, 100, 300, 1000]
    ocr.run(learningRates, Niterations, [0.1]) 
    regWeights = [0.01, 0.03, 0.1]
    ocr.run([0.01], [100], regWeights) 
    ocr.plot(0.3, 1000, 0.1)
             
if __name__ == "__main__":
    main()   