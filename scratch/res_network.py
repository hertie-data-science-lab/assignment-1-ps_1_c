import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        super().__init__(sizes, epochs=epochs, learning_rate=learning_rate, random_state=random_state)
        
        # Residual connection: input -> hidden layer 2
        input_layer = self.sizes[0]
        hidden_layer_2 = self.sizes[2]

        self.params['W_res'] = np.random.rand(hidden_layer_2, input_layer) - 0.5

    def _forward_pass(self: "ResNetwork", x_train: np.ndarray) -> np.ndarray:
        '''
        Forward propagation algorithm.

        Args:
            x_train (np.ndarray): Input data for the forward pass.
        Returns:
            np.ndarray: Output of the network after forward pass.
        '''
        # Reshape input
        x = x_train.reshape(-1, 1)

        # Layer 1
        z1 = self.params['W1'] @ x
        a1 = self.activation_func(z1)

        # Layer 2 with residual connection
        z2 = self.params['W2'] @ a1 + self.params['W_res'] @ x # skip connection from input → hidden layer 2
        a2 = self.activation_func(z2)

        # Output layer
        z3 = self.params['W3'] @ a2
        a3 = self.output_func(z3)

        # Store for backprop
        self.cache = {
            'x': x, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3
        }

        return a3


    def _backward_pass(self: "ResNetwork", y_train: np.ndarray, output: np.ndarray) -> dict:
        '''
        Backpropagation algorithm (responsible for updating the weights of the neural network).

        Args:
            y_train (np.ndarray): True labels for the input data.
            output (np.ndarray): Output from the forward pass.
        Returns:
            dict: Gradients of weights including residual connection.
        '''
        # Reshape target
        y = y_train.reshape(-1, 1)
        a3 = output
        a2 = self.cache['a2']
        a1 = self.cache['a1']
        x = self.cache['x']

        # Output layer error
        dz3 = self.cost_func_deriv(y, a3) * self.output_func_deriv(self.cache['z3'])
        dW3 = dz3 @ a2.T

        # Hidden layer 2 error (includes residual)
        dz2 = (self.params['W3'].T @ dz3) * self.activation_func_deriv(self.cache['z2'])
        dW2 = dz2 @ a1.T
        dW_res = dz2 @ x.T   # residual gradient

        # Hidden layer 1 error
        dz1 = (self.params['W2'].T @ dz2) * self.activation_func_deriv(self.cache['z1'])
        dW1 = dz1 @ x.T

        return {'dW1': dW1, 'dW2': dW2, 'dW3': dW3, 'dW_res': dW_res}


