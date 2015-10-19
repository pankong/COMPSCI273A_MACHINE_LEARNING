# COMPSCI273A_MACHINE_LEARNING
python codes for the homework

Homework 1:
1. Perceptron learning
(a) Create 9-dimensional input vectors in two classes by drawing two cluster centers from a
spherical Gaussian distribution, then scaling down the width of that Gaussian by a factor of
alpha in (0,1) to create N cluster members for each cluster by adding in noise.
(b) Use a formula to devise a single perceptron to discriminate between the classes.
Numerically evaluate its accuracy, by averaging over draws of cluster centers and cluster
members, and by systematically varying alpha and N (make tables or plots). All reported
results should have numerically estimated error bars arising from stated probability
distributions.
(c) Use the Perceptron Learning Algorithm to do the same, and evaluate its results the same
way but fix alpha and N from 1(b) and instead vary as parameters the number of iterations
of the algorithm and the size of the test set. (Report performance on a test set, not the
training set.)
2. Toy OCR.
(a) create two patterns of 3x3 binary images with the same number of on and off pixels. Add
real-valued noise in a controlled way as in 1a, independently for each pixel.
(b) as in 1b.
(c) as in 1c.

Homework 2:
1. Implement and test a feed-forward artificial neural network (ANN or MLP = multilayer
perceptron) with two layers of weights (i.e one layer of hidden units) and logistic transfer
functions, using back-propagation of error (online or batch) as the learning algorithm.
Parameters to vary include learning rate, and any parameters of the stopping criterion (eg.
number of iterations and/or target accuracy); optionally also, the strength of a weight decay
term (section 5.5 in Bishop). For each value of the parameters, report average error of the
algorithm on both a training and a test set, along with statistical error bars on those
quantities. For the problems that the ANN should solve, choose or invent two easy
problems, e.g. (a) autoencoder and (b) some logic function on binary vectors such as
exactly-one-on (coincides with xor/parity for 2 inputs, but is easier for more inputs For details
on one way to approach these particular options, see forthcoming hints sheet from the TA).
Show a visual representation (similar to a heat map) of the final trained weights for your best
network. Probability distributions to use in reporting a single average error measurement
includes a distribution (eg a narrow Gaussian around zero) over starting weight arrays, and
a distribution on input patterns from which you draw your training and test sets; just say
what distributions you chose.
2. Apply your network of #1 to a simple image classification problem, e.g. that of HW1 or more
realistic, e.g. with a larger image such as 5 x 7. Report performance as above and again
visualize the final trained weights, but now in an image-centric format for the first layer.
Extra credit:
- Add weight decay (section 5.5 in Bishop) to the exploration of problem 1 or 2 above.
- Or, generalize the Backpropagation of Error algorithm to include weight-sharing, and use that
variant to obtain (perhaps!) better performance on #2 due to translation invariant features.
- Or, invent your own variation of the foregoing problems.
