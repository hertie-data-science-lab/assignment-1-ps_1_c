import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing


class Network():
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv

        self.params = self._initialize_weights()

        # optional
        self.train_accuracies = []
        self.val_accuracies = []


    def _initialize_weights(self):
        # number of neurons in each layer
        input_layer = self.sizes[0]
        hidden_layer_1 = self.sizes[1]
        hidden_layer_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # random initialization of weights
        np.random.seed(self.random_state)
        params = {
            'W1': np.random.rand(hidden_layer_1, input_layer) - 0.5,
            'W2': np.random.rand(hidden_layer_2, hidden_layer_1) - 0.5,
            'W3': np.random.rand(output_layer, hidden_layer_2) - 0.5,
        }

        return params


    def _forward_pass(self: "Network", x_train: np.ndarray) -> np.ndarray:
        '''
        Forward propagation algorithm (Computes predictions by passing input data through the network layers.)

        Args:
            x_train (np.ndarray): Input data for the forward pass.
        
        Returns:
            np.ndarray: Output of the network after forward pass.
        '''
        x = x_train.reshape(-1, 1)  # reshape input to column vector
        
        # Layer 1
        z1 = self.params['W1'] @ x # linear transformation
        a1 = self.activation_func(z1) # apply activation function

        # Layer 2
        z2 = self.params['W2'] @ a1 # linear transformation
        a2 = self.activation_func(z2) # apply activation function

        # Output layer
        z3 = self.params['W3'] @ a2 # linear transformation
        a3 = self.output_func(z3) # apply output activation function

        # Store for backpropagation
        self.cache = {
            'x': x, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'a3': a3
        }

        # Return the output
        return a3


    def _backward_pass(self: "Network", y_train: np.ndarray, output: np.ndarray) -> dict:
        """
        Backpropagation algorithm (Computes gradients of loss with respect to weights using chain rule.)

        Args:
            y_train (np.ndarray): True labels for the input data.
            output (np.ndarray): Output from the forward pass.
        
        Returns:
            dict: Gradients of weights and biases.
        """
        y = y_train.reshape(-1, 1) # Reshape target
        a3 = output # network output from forward pass
        a2 = self.cache['a2'] # activations from layer 2
        a1 = self.cache['a1'] # activations from layer 1
        x = self.cache['x'] # input data

        # Error at output layer
        error = self.cost_func_deriv(y, a3)
        dz3 = error * self.output_func_deriv(self.cache['z3'])
        
        # Compute weight gradient
        grad_W3 = dz3 @ a2.T

        # Hidden layer 2 error
        dz2 = (self.params['W3'].T @ dz3) * self.activation_func_deriv(self.cache['z2'])
        grad_W2 = dz2 @ a1.T

        # Hidden layer 1 error
        dz1 = (self.params['W2'].T @ dz2) * self.activation_func_deriv(self.cache['z1'])
        
        # Compute weight gradient
        grad_W1 = dz1 @ x.T

        # Store gradients
        return {'dW1': grad_W1, 'dW2': grad_W2, 'dW3': grad_W3}


    def _update_weights(self: "Network", weights_gradient: dict, learning_rate: float) -> None:
        """
        Update weights and biases.

        Args:
            weights_gradient (dict): Gradients of weights and biases.
            learning_rate (float): Learning rate for weight updates.

        Returns: None
        """
        for key in self.params:
            grad_key = 'd' + key
            if grad_key in weights_gradient:
                self.params[key] -= learning_rate * weights_gradient[grad_key]



    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)

        # Append accuracies to lists for plotting later
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def compute_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self: "Network", x: np.ndarray) -> np.ndarray:
        '''
        Make a prediction for a single input sample.

        Args:
            x (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Index of the most likely output class.
        '''
        output = self._forward_pass(x)
        return np.argmax(output)


    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                
                if cosine_annealing_lr:
                    learning_rate = cosine_annealing(self.learning_rate, 
                                                     iteration, 
                                                     len(x_train), 
                                                     self.learning_rate)
                else: 
                    learning_rate = self.learning_rate
                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                
                self._update_weights(weights_gradient, learning_rate=learning_rate)

            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
