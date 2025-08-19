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


# In[ ]:




