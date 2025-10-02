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



    def _forward_pass(self, x_train):
        # Pass through hidden layer 1
        h1 = self.activation_func(self.linear1(x_train))
        # Hidden layer 2
        h2 = self.activation_func(self.linear2(h1))
        # Output layer (softmax over 10 classes)
        output = self.output_func(self.linear3(h2), dim=1)
        return output


    def _backward_pass(self, y_train, output):
        # Compute the loss (BCEWithLogitsLoss)
        loss = self.loss_func(output, y_train.float())
        # Backpropagate (compute gradients automatically)
        loss.backward()
        return loss


    def _update_weights(self):
        self.optimizer.step()


    def _flatten(self, x):
        return x.view(x.size(0), -1)       


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def predict(self, data_loader):
        correct = 0
        total = 0
        self.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x = self._flatten(x)
                output = self._forward_pass(x)
                _, predicted = torch.max(output, dim=1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total  # <-- scalar float


    def fit(self, train_loader, val_loader):
        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in train_loader:
                x = self._flatten(x)
                y = nn.functional.one_hot(y, 10)
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
