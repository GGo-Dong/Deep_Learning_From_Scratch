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
    c = np.max(x)
    exp_x = np.exp(x-c) # 오버플로우를 막기 위해 c를 빼줌
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    
    return y