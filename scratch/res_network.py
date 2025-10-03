import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):


    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        '''
        TODO: Initialize the class inheriting from scratch.network.Network.
        The method should check whether the residual network is properly initialized.

        Initialize residual network by inheriting from Network.
        Simply call the parent constructor.
        '''
        super().__init__(sizes, epochs=epochs, learning_rate=learning_rate, random_state=random_state)

    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''
        x = x_train.reshape(-1, 1)  # column vector

        Z1 = self.params['W1'] @ x
        A1 = self.activation_func(Z1)

        # Residual connection: add A1 to pre-activation of second layer
        Z2 = self.params['W2'] @ A1 + A1
        A2 = self.activation_func(Z2)

        Z3 = self.params['W3'] @ A2
        A3 = self.output_func(Z3)

        self.cache = {
            'x': x,
            'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2,
            'Z3': Z3, 'A3': A3
        }
        return A3

    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
        The method should also account for the residual connection in the hidden layer.

        '''
        y = y_train.reshape(-1, 1)
        A3 = output
        A2 = self.cache['A2']
        A1 = self.cache['A1']
        x = self.cache['x']

        # Derivative of cost wrt output
        dA3 = self.cost_func_deriv(y, A3)

        # Output layer
        dZ3 = dA3 * self.output_func_deriv(self.cache['Z3'])
        dW3 = dZ3 @ A2.T

        # Hidden layer 2 with residual
        dA2 = self.params['W3'].T @ dZ3
        dZ2 = dA2 * self.activation_func_deriv(self.cache['Z2'])
        dW2 = dZ2 @ A1.T

        # Backprop to A1: comes from two paths — through W2 and directly through residual
        dA1 = self.params['W2'].T @ dZ2 + dZ2
        dZ1 = dA1 * self.activation_func_deriv(self.cache['Z1'])
        dW1 = dZ1 @ x.T

        grads = {'dW1': dW1, 'dW2': dW2, 'dW3': dW3}
        return grads


