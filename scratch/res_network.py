import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):


    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        '''
        TODO: Initialize the class inheriting from scratch.network.Network.
        The method should check whether the residual network is properly initialized.
        '''
        pass
        
        


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''
        pass



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
        The method should also account for the residual connection in the hidden layer.

        '''
        pass


