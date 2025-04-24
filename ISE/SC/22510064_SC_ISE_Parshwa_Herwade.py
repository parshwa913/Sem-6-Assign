import numpy as np

training_data = [
    ([0.4, -0.7], 0.1),
    ([0.3, -0.5], 0.05),
    ([0.6,  0.1], 0.3),
    ([0.2,  0.4], 0.25),
    ([0.1,  0.2], 0.12)
]

X = np.array([x for (x, _) in training_data], dtype=np.float32)
y = np.array([t for (_, t) in training_data], dtype=np.float32)

input_neurons = 2
hidden_neurons = 2
output_neurons = 1
eta = 0.5

def sigmoid(z):
    # Sigmoid activation function
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    # Derivative of sigmoid function
    return a * (1 - a)

W_input_hidden = np.random.uniform(-0.5, 0.5, (input_neurons, hidden_neurons))
W_hidden_output = np.random.uniform(-0.5, 0.5, (hidden_neurons, output_neurons))
b_hidden = np.random.uniform(-0.5, 0.5, (hidden_neurons,))
b_output = np.random.uniform(-0.5, 0.5, (output_neurons,))

epochs = 10000

for epoch in range(epochs):
    total_error = 0.0
    for i in range(len(X)):
        net_hidden = np.dot(X[i], W_input_hidden) + b_hidden
        out_hidden = sigmoid(net_hidden)
        net_output = np.dot(out_hidden, W_hidden_output) + b_output
        out_output = sigmoid(net_output)
        target = y[i]
        error = 0.5 * (target - out_output) ** 2
        total_error += error
        delta_output = (out_output - target) * sigmoid_deriv(out_output)
        delta_hidden = delta_output * W_hidden_output[:, 0] * sigmoid_deriv(out_hidden)
        W_hidden_output[:, 0] -= eta * delta_output * out_hidden
        b_output -= eta * delta_output
        for j in range(hidden_neurons):
            W_input_hidden[:, j] -= eta * delta_hidden[j] * X[i]
        b_hidden -= eta * delta_hidden
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Error: {total_error}")

print("\nTraining complete.")
for i in range(len(X)):
    net_hidden = np.dot(X[i], W_input_hidden) + b_hidden
    out_hidden = sigmoid(net_hidden)
    net_output = np.dot(out_hidden, W_hidden_output) + b_output
    out_output = sigmoid(net_output)
    print(f"Input: {X[i]}, Predicted: {out_output}, Target: {y[i]}")
