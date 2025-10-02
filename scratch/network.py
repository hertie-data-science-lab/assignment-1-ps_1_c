import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing
import math


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


    def _forward_pass(self, x_train):
    # store the input so we can use it in backprop
        self.x_input = x_train
    
    # Hidden layer 1
        self.a1 = np.dot(self.params['W1'], x_train)
        self.h1 = self.activation_func(self.a1)
    
    # Hidden layer 2
        self.a2 = np.dot(self.params['W2'], self.h1)
        self.h2 = self.activation_func(self.a2)

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

    # Gradient for W1 (now uses self.x_input instead of undefined self.a_values)
        delta1 = np.dot(self.params['W2'].T, delta2) * self.activation_func_deriv(self.a1)
        grad_W1 = np.outer(delta1, self.x_input)

        return {'W1': grad_W1, 'W2': grad_W2, 'W3': grad_W3}



    def _update_weights(self, weights_gradient, learning_rate):
        # Update each weight matrix by subtracting (learning rate * gradient)
        for key in self.params:
            self.params[key] -= learning_rate * weights_gradient[key]

    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
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


    def predict(self, x):
        # Run forward pass
        output = self._forward_pass(x)
        # Return index of largest probability which is biggest one 
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
