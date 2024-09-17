import numpy as numpy
class LinearRegression:
    def __init__(self,
                learning_rate=0.01,
                epochs=100
                ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        return
    
    def predict(self, X):
        return numpy.dot(X, self.weights) + self.bias

    def fit(self, X, y):
        # Define rows (m) and columns/features (n)
        m, n = X.shape

        # Define weights, initially we don't care about it, start with random value
        # Create a single random weight for each feature
        self.weights = numpy.random.rand(n, 1)
        # Bias is a scalar
        self.bias = numpy.random.rand(1)

        # Reshape
        y = y.reshape(m, 1)

        # Save losses
        losses_list = list()
        # Save bias
        bias_list = list()
        # Save weights
        weights_list = list()

        # Go over each epoch/iterations

        for epoch in range(self.epochs):

            # calculate prediction
            y_predict = numpy.dot(X, self.weights) + self.bias

            # get current loss - L - J
            # Mean Square Error, this is how bad our prediction is
            loss = numpy.mean((y - y_predict)**2)
            # Append the loss
            losses_list.append(loss)

            # calculate gradient
            dw = (-2 / m) * numpy.dot(X.T, (y-y_predict))
            db = (-2 / m) * numpy.sum((y-y_predict))

            # update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            weights_list.append(self.weights)
            bias_list.append(self.bias)

            #print(f'epoch:{epoch:#d} loss:{loss} weights:{self.weights} bias:{self.bias}')

        return self.weights, self.bias, losses_list, bias_list, weights_list