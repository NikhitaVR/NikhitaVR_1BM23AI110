#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def perceptron_output(x, w, b, activation_func):
    """
    x: input
    w: weight
    b: bias
    activation_func: activation function (linear, sigmoid, tanh)
    """
    z = w * x + b
    return activation_func(z)

x_vals = np.linspace(-10, 10, 200)

weight = 1.0
bias = 0.0

linear_outputs = perceptron_output(x_vals, weight, bias, linear_activation)
sigmoid_outputs = perceptron_output(x_vals, weight, bias, sigmoid_activation)
tanh_outputs = perceptron_output(x_vals, weight, bias, tanh_activation)

plt.figure(figsize=(10, 6))

plt.plot(x_vals, linear_outputs, label="Linear Activation", color='blue')
plt.plot(x_vals, sigmoid_outputs, label="Sigmoid Activation", color='green')
plt.plot(x_vals, tanh_outputs, label="Tanh Activation", color='red')

plt.title("Perceptron Output with Different Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.ylim(-2,2)
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Activation Functions
def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)

def softmax_activation(x):
    # Numerically stable softmax for each element as if in a batch
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def perceptron_output(x, w, b, activation_func):
    """
    x: input
    w: weight
    b: bias
    activation_func: activation function (linear, sigmoid, tanh, relu, etc.)
    """
    z = w * x + b
    return activation_func(z)

x_vals = np.linspace(-10, 10, 200)

weight = 1.0
bias = 0.0

linear_outputs = perceptron_output(x_vals, weight, bias, linear_activation)
sigmoid_outputs = perceptron_output(x_vals, weight, bias, sigmoid_activation)
tanh_outputs = perceptron_output(x_vals, weight, bias, tanh_activation)
relu_outputs = perceptron_output(x_vals, weight, bias, relu_activation)
leaky_relu_outputs = perceptron_output(x_vals, weight, bias, leaky_relu_activation)
softmax_outputs = softmax_activation(x_vals)  # Special case, applied directly

# Plotting
plt.figure(figsize=(12, 7))

plt.plot(x_vals, linear_outputs, label="Linear", color='blue')
plt.plot(x_vals, sigmoid_outputs, label="Sigmoid", color='green')
plt.plot(x_vals, tanh_outputs, label="Tanh", color='red')
plt.plot(x_vals, relu_outputs, label="ReLU", color='orange')
plt.plot(x_vals, softmax_outputs, label="Softmax (normalized)", color='brown')

plt.title("Perceptron Output with Various Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.ylim(-2, 2)
plt.show()


# In[ ]:




