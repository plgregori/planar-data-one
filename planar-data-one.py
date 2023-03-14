# In this program, we implement a 2-class classification neural network with a single hidden layer.

import numpy as np
import copy
import matplotlib.pyplot as plt
from testCases_v2 import *
from public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# We import the usual packages, plus sklearn, which provides simple and efficient tools for data mining and data analysis.
# We then import the dataset:

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# We can plot the dataset to visualize it.

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}


dataset = "noisy_circles"


X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y%2

# The following is the default dataset (flower shape), comment the next line if you want to choose one of the above.

X, Y = load_planar_dataset()

# We can visualize the dataset by plotting it:

plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
plt.show()

# The goal of the program is to determine to define regions as either red or blue.
# The dataset comprises an array of coordinates X, which contains the coordinates (x1,x2) of 400 dots, and an array of colors (red:0, blue:1) Y.
# We start by determining the size of the dataset:

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

# We have shape_X = (2,400); shape_Y = (1,400); m = 400.
# We now implement a simple logistic regression to this problem, to see how it performs. Luckily, sklearn already contains logistic regression:

clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

# We can also plot the decision boundary for logistic regression:

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# And compute the accuracy:

LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
plt.show()

# Which yields: "Accuracy of logistic regression: 47 % (percentage of correctly labelled datapoints)"
# Not great!
# We now implement a neural network with a single hidden layer.
# We start by defining the neural network structure:

def layer_sizes(X, Y):

    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x, n_h, n_y)

# Here, we hard-coded the size of the hidden layer to be 4.
# We now implement a function which initializes the model's parameters:

def initialize_parameters(n_x, n_h, n_y):
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# As we learned, we initialize the weight matrices to random values, and the bias vectors to zero. Both need to be of the appropriate size.
# We now define the usual forward propagation:

def forward_propagation(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

# Here we used the hyperbolic tangent for the hidden layer and the (imported) sigmoid function for the output layer.
# We now compute the cost function for the output:

def compute_cost(A2, Y):
   
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = -(1/m)* np.sum(logprobs) 
    
    cost = float(np.squeeze(cost)) 
    
    return cost

# The line before "return cost" makes sure cost is the dimension we expect (i.e. a float). For example, it turns [[17]] into 17
# np.squeeze removes redundant dimensions. For example, it turns [[[1 2]]] of size (1,1,2) into [1,2] of size (2,).
# We can now implement the backward propagation:

def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1 - np.power(A1, 2)))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

# And the gradient descent:

def update_parameters(parameters, grads, learning_rate = 1.2):
        
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W1 = copy.deepcopy(W1)
    W2 = copy.deepcopy(W2)
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# Finally, we merge everything in our model function:

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
   
    for i in range(0, num_iterations):
     
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
       
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

# np.random.seed(x) simply let's us choose the seed of randomly generated numbers. If we always choose the same seed, we end up always generating the same random numbers.
# We are now ready to test the model:

def predict(parameters, X):
   
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

# Here we used the simple line of code "X_new = (X > threshold)" which from a matrix X creates a new matrix whose entries are all zeros and ones depending on whether the entries of X were larger than the threshold.

# We can now build a model with a n_h-dimensional hidden layer:

parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# And plot the decision boundary:

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# We can also print the accuracy of our model:

predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# We now want to test for different hidden layer sizes:

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

plt.show()
