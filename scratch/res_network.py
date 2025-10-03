import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):
    """
    Feed-forward network with a residual connection from hidden layer 1 to hidden layer 2.
    Inherits from Network in scratch.network.
    """

    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        super().__init__(sizes, epochs=epochs, learning_rate=learning_rate, random_state=random_state)

    def _forward_pass(self, x_train):
        """
        Forward propagation with residual connection.
        Adds hidden layer 1 activation (A1) to hidden layer 2 pre-activation (Z2).
        """
        x = x_train.reshape(-1, 1)  # column vector

        # Layer 1
        Z1 = self.params['W1'] @ x
        A1 = self.activation_func(Z1)

        # Layer 2 with residual connection
        Z2 = self.params['W2'] @ A1 + A1  # residual connection
        A2 = self.activation_func(Z2)

        # Output layer
        Z3 = self.params['W3'] @ A2
        A3 = self.output_func(Z3)

        # Cache for backprop
        self.cache = {
            'x': x,
            'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2,
            'Z3': Z3, 'A3': A3
        }
        return A3

    def _backward_pass(self, y_train, output):
        """
        Backpropagation including residual connection.
        Computes gradients of weights for all layers.
        """
        y = y_train.reshape(-1, 1)
        A3 = output
        A2 = self.cache['A2']
        A1 = self.cache['A1']
        x = self.cache['x']

        # Output layer
        dA3 = self.cost_func_deriv(y, A3)
        dZ3 = dA3 * self.output_func_deriv(self.cache['Z3'])
        dW3 = dZ3 @ A2.T

        # Hidden layer 2 with residual
        dA2 = self.params['W3'].T @ dZ3
        dZ2 = dA2 * self.activation_func_deriv(self.cache['Z2'])
        dW2 = dZ2 @ A1.T

        # Backprop to A1: two paths (through W2 and through residual)
        dA1 = self.params['W2'].T @ dZ2 + dZ2
        dZ1 = dA1 * self.activation_func_deriv(self.cache['Z1'])
        dW1 = dZ1 @ x.T

        return {'dW1': dW1, 'dW2': dW2, 'dW3': dW3}
