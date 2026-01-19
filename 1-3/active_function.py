import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def relu(x):
    return np.maximum(0, x)

def step_function(x):
    return np.array(x > 0, dtype=np.int32)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x

    return y