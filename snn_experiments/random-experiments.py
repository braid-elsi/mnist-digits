'''
Leaky integration and fire equation:
    X: The input signal
    U: Membrane potential

Note that in SNNTorch, this function is already implemented. Example:
    lif = snn.Leaky(beta=0.9, threshold=1) # initialize Neuron class

'''
def lif_implemented(X, U):
    beta = 0.9 # set decay rate (how the neuron loses its charge)
    W = 0.5 # learnable parameter 
    theta = 1 # set threshold
    S = 0 # initialize output spike
    U = beta * U + W*X - S * theta # iterate over one time step of Eq. 4 
    S = int(S > theta) # Eq. 5
    return S, U

'''
From page 9
'''
def helper_output(X, S, U):
    print('-' * 46)
    print(f'{"S (Spikes)":>10} | {"X (Inputs)":>10} | {"U (Membrane Potential)":>20}')
    print('-' * 46)
    for i in range(len(X)):
        print(f'{S[i]:10.0f} | {X[i]:10.4f} | {U[i]:20.4f}')

import torch
import snntorch as snn
import time

'''
-------------------------------------------------------
snn.Leaky
-------------------------------------------------------
Initializes the neuron function based on input parameters:
    * beta: the decay rate
    * threshold: when to fire
Returns two vectors:
    * S: A vector of spikes (1 or 0)
    * U: A vector of new membrane potentials
'''
lif = snn.Leaky(beta=0.9, threshold=1) 
X = torch.rand(10) # vector of 10 random inputs
U = torch.zeros(10) # initialize hidden states of 10 neurons to 0 V
infinite_loop = True
while infinite_loop:
    S, U = lif(X, U) # forward-pass of leaky integrate-and-fire neuron
    helper_output(X, S, U)
    time.sleep(1)