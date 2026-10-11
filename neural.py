import random
import math
# Functions
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
def grad_sigmoid(x):
    return math.exp(-x) / ((1 + math.exp(-x))**2)
def relu(x):
    return (x >= 0) * x
def grad_relu(x):
    return (x >= 0) * 1.0

# Main neural network class
class NeuralNetwork:
    def __init__(self, modes=10):
        # Glorot Initialization
        limit = math.sqrt(6.0/(modes + 2))
        self.weights1 = [random.uniform(-limit, limit) for _ in range(modes)]
        self.weights2 = [random.uniform(-limit, limit) for _ in range(modes)]
        self.biases1 = [random.uniform(-limit, limit) for _ in range(modes)]
        self.biases2 = [random.uniform(-limit, limit)]
        self.modes = modes

    def predict(self, x):
        hidden_layer1 = 0.0
        for i in range(self.modes):
            hidden_layer1 += self.weights2[i] * relu(self.weights1[i] * x + self.biases1[i])
        value = sigmoid(self.biases2[0] + hidden_layer1)
        return value

    def gradient_biases2(self, x):
        hidden_layer1 = 0.0
        for i in range(self.modes):
            hidden_layer1 += self.weights2[i] * relu(self.weights1[i] * x + self.biases1[i])
        return [grad_sigmoid(self.biases2[0] + hidden_layer1)]

    def gradient_biases1(self, x):
        hidden_layer1 = 0.0
        grad_vector = []
        for i in range(self.modes):
            hidden_layer1 += self.weights2[i] * relu(self.weights1[i] * x + self.biases1[i])
        for index in range(self.modes):
            grad_vector.append(grad_sigmoid(self.biases2[0] + hidden_layer1) * self.weights2[index] * grad_relu(self.weights1[index] * x + self.biases1[index]))
        return grad_vector

    def gradient_weights1(self, x):
        hidden_layer1 = 0.0
        grad_vector = []
        for i in range(self.modes):
            hidden_layer1 += self.weights2[i] * relu(self.weights1[i] * x + self.biases1[i])
        for index in range(self.modes):
            grad_vector.append(grad_sigmoid(self.biases2[0] + hidden_layer1) * self.weights2[index] * x * grad_relu(self.weights1[index] * x + self.biases1[index]))
        return grad_vector

    def gradient_weights2(self, x):
        hidden_layer1 = 0.0
        grad_vector = []
        for i in range(self.modes):
            hidden_layer1 += self.weights2[i] * relu(self.weights1[i] * x + self.biases1[i])
        for index in range(self.modes):
            grad_vector.append(grad_sigmoid(self.biases2[0] + hidden_layer1) * relu(self.weights1[index] * x + self.biases1[index]))
        return grad_vector


