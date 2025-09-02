#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0,z)

class Neuron:
    def __init__(self, weights, bias, activation="sigmoid"):
        self.weights = np.array(weights)
        self.bias = bias
        self.activation = activation
        
    def forward(self, input):
        z = np.dot(input, self.weights) + self.bias 
        if self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "tanh":
            return tanh(z)
        elif self.activation == "relu":
            return relu(z)
        else:
            raise ValueError("Unknown activation function")

inputs = np.array([0.5, -1.2, 3.0])
weights = [0.4, -0.6, 0.3]
bias = 0.1

for act in ["sigmoid", "tanh", "relu"]:
    neuron = Neuron(weights, bias, activation=act)
    output = neuron.forward(inputs)
    print(f"Activation = {act:7s} --> Output = {output:.4f}")

z_values = np.linspace(-5, 5, 200)

activations = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu
}

for act_name, act_func in activations.items():
    plt.figure(figsize=(6, 4))
    plt.plot(z_values, act_func(z_values), label=f"{act_name} Activation")
    plt.title(f"{act_name.capitalize()} Activation Function")
    plt.xlabel("Input z")
    plt.ylabel("Activation Output")
    plt.grid(True)
    plt.legend()
    plt.show()

