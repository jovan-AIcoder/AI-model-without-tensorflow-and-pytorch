import math 
import neural as nn
import joblib
# function definitions
def add_vector(vec1, vec2):
    vec3 = []
    for i in range(len(vec1)):
        vec3.append(vec1[i] + vec2[i])
    return vec3
def multiply_scalar(k, vec1):
    vec2 = []
    for i in range(len(vec1)):
        vec2.append(k * vec1[i])
    return vec2
# load dataset
with open('dataset.txt', 'r') as file:
    data = file.read()
dataset = data.split('\n')
df = []
for i in range(len(dataset)):
    df.append(dataset[i].split(','))
X = []
y = []
for i in range(len(df) - 1):
    X.append(float(df[i][0]))
    y.append(float(df[i][1]))

# training
mode = int(input('Number of neurons on layer1: '))
model = nn.NeuralNetwork(mode)
epochs = int(input('Epochs: '))
lr = float(input('Learning Rate: '))
M = len(X)
for iter in range(1, epochs + 1):
    # avoid gradient accumulation per epochs
    loss = 0.0
    mae = 0.0
    grad_w1 = [0.0 for _ in range(mode)]
    grad_w2 = [0.0 for _ in range(mode)]
    grad_bias1 = [0.0 for _ in range(mode)]
    grad_bias2 = [0.0]

    # add up gradients for all data points
    for data in range(M):
        y_hat = model.predict(X[data])
        loss += (1.0/M) * ((y[data] - y_hat)**2.0)
        mae += (1.0/M) * abs(y[data] - y_hat)
        buf_grad_w1 = model.gradient_weights1(X[data])
        buf_grad_w2 = model.gradient_weights2(X[data])
        buf_grad_b1 = model.gradient_biases1(X[data])
        buf_grad_b2 = model.gradient_biases2(X[data])
        grad_w1 = add_vector(grad_w1, multiply_scalar((2.0/M)*(y_hat - y[data]), buf_grad_w1))
        grad_w2 = add_vector(grad_w2, multiply_scalar((2.0/M)*(y_hat - y[data]), buf_grad_w2))
        grad_bias1 = add_vector(grad_bias1, multiply_scalar((2.0/M)*(y_hat - y[data]), buf_grad_b1))
        grad_bias2 = add_vector(grad_bias2, multiply_scalar((2.0/M)*(y_hat - y[data]), buf_grad_b2))

    print(f'Epoch {iter}/{epochs} --> loss: {loss} - mae: {mae}')
    # backpropagation
    model.weights1 = add_vector(model.weights1, multiply_scalar(-lr, grad_w1))
    model.weights2 = add_vector(model.weights2, multiply_scalar(-lr, grad_w2))
    model.biases1 = add_vector(model.biases1, multiply_scalar(-lr, grad_bias1))
    model.biases2 = add_vector(model.biases2, multiply_scalar(-lr, grad_bias2))

# save model
joblib.dump(model, 'model.pkl')
print('Done!')
