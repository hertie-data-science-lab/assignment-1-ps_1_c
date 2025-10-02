import numpy as np
from scratch.network import Network

class ResNetwork(Network):
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        super().__init__(sizes, epochs=epochs, learning_rate=learning_rate, random_state=random_state)

    def _forward_pass(self, x_train):
        self.x_input = x_train

    # Hidden layer 1
        self.a1 = np.dot(self.params['W1'], x_train)
        self.h1 = self.activation_func(self.a1)

    # Hidden layer 2 (residual connection adds h1 back in)
        self.a2 = np.dot(self.params['W2'], self.h1)
        self.h2 = self.activation_func(self.a2) + self.h1

    # Output layer
        self.a3 = np.dot(self.params['W3'], self.h2)
        self.output = self.output_func(self.a3)

        return self.output

    def _backward_pass(self, y_train, output):
        # Error at output
        error = self.cost_func_deriv(y_train, output)

        # Gradient for W3
        delta3 = error * self.output_func_deriv(self.a3)
        grad_W3 = np.outer(delta3, self.h2)

        # Gradient for W2
        delta2 = np.dot(self.params['W3'].T, delta3) * self.activation_func_deriv(self.a2)
        grad_W2 = np.outer(delta2, self.h1)

        # Gradient for W1 (with residual connection)
        delta1 = np.dot(self.params['W2'].T, delta2) * self.activation_func_deriv(self.a1)
        grad_W1 = np.outer(delta1, self.x_input)  # residual connection derivative = 1

        return {'W1': grad_W1, 'W2': grad_W2, 'W3': grad_W3}
