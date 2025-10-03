import time

import torch
import torch.nn as nn
import torch.optim as optim


class TorchNetwork(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(self.random_state)

        self.linear1 = nn.Linear(sizes[0], sizes[1])
        self.linear2 = nn.Linear(sizes[1], sizes[2])
        self.linear3 = nn.Linear(sizes[2], sizes[3])

        self.activation_func = torch.sigmoid
        self.output_func = torch.softmax
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        # optional
        self.train_accuracies = []
        self.val_accuracies = []


    def _forward_pass(self: "TorchNetwork", x_train: torch.Tensor) -> torch.Tensor:
        '''
        Forward propagation algorithm.

        Args:
            x_train (torch.Tensor): Input data for the forward pass.
        
        Returns:
            torch.Tensor: Output of the network after forward pass.
        '''
        # Pass through layer 1
        x = self.activation_func(self.linear1(x_train)) # Apply activation function and linear transformation
        
        # Pass through layer 2
        x = self.activation_func(self.linear2(x)) # Apply activation function and linear transformation

        # Output layer
        x = self.output_func(self.linear3(x), dim=1) # Apply output function and linear transformation
        return x


    def _backward_pass(self: "TorchNetwork", y_train: torch.Tensor, output: torch.Tensor) -> float:
        '''
        Backward propagation algorithm responsible for computing the gradients.

        Args:
            y_train (torch.Tensor): True labels.
            output (torch.Tensor): Output from the forward pass.

        Returns:
            float: Computed loss value.
        '''
        loss = self.loss_func(output, y_train.float())
        loss.backward()
        return loss.item()


    def _update_weights(self: "TorchNetwork"):
        '''Update the network weights according to stochastic gradient descent.'''
        self.optimizer.step()


    def _flatten(self, x):
        return x.view(x.size(0), -1)       


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)

        # Optional store accuracies for plotting later
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def predict(self: "TorchNetwork", x: torch.Tensor) -> torch.Tensor:
        '''
        Make a prediction for a single input sample.

        Args:
            x (torch.Tensor): Input data for prediction.
        Returns:
            torch.Tensor: Index of the most likely output class.
        '''
        x = self._flatten(x)
        with torch.no_grad():
            logits = self._forward_pass(x)
            probs = self.output_func(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds


    def fit(self, train_loader, val_loader):
        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10).float() # NOTE had to add this here
                self.optimizer.zero_grad()


                output = self._forward_pass(x)
                self._backward_pass(y, output)
                self._update_weights()

            self._print_learning_progress(start_time, iteration, train_loader, val_loader)



    def compute_accuracy(self, data_loader):
        correct = 0
        for x, y in data_loader:
            pred = self.predict(x)
            correct += torch.sum(torch.eq(pred, y))

        return correct / len(data_loader.dataset)
