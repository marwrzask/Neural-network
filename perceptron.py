import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
import os
from datetime import datetime


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):  # __init__ constructor type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(3)  # input weights including bias


    def activation_function(self, summation):  # unit_step
        return np.where(summation >= 0, 1, 0)

    def adder(self, input_X_values, weights):
        return np.dot(input_X_values, weights)

    def fit(self, X_input, y_goal):
        self.X_input = X_input
        self.y_goal = y_goal

        for epochs in range(self.epochs):
            print("--" * 10)
            print(f"for epoch >> {epochs}")
            print("--" * 10)

            X_with_bias = inputs_with_bias[:, :3]
            summation = self.adder(X_with_bias, self.weights)
            y_predicted = self.activation_function(summation)
            print(f"predicted value after forward pass: \n{y_predicted}")

            self.error = self.y_goal - y_predicted
            print(f"error: \n{self.error}")
            if sum(self.error) == 0:
                print(f'Errors in total 0 in {epochs} epochs')
                break

            self.weights = self.weights + self.learning_rate * np.dot(X_with_bias.T, self.error)
            print(f"updated weights after epoch: {epochs + 1}/{self.epochs}: \n{self.weights}")
            print("##" * 10)

    def predict(self):
        X_with_bias = inputs_with_bias[:, :3]
        summation = self.adder(X_with_bias, self.weights)
        return self.activation_function(summation)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"\ntotal loss: {total_loss}\n")
        return total_loss


X = scipy.io.loadmat('dane_perceptron.mat')
X_dane = np.transpose(X['dane'])
inputs_with_bias = np.c_[np.ones((len(X_dane), 1)), X_dane]
X_input = X_dane[:, 0:2]
y_goal = X_dane[:, 2]


if __name__ == '__main__':

    time_scores = []
    for i in range(5):
        start_time = datetime.now()
        model_1 = Perceptron(learning_rate=0.003, epochs=200)
        model_1.fit(X_input, y_goal)
        end_time = datetime.now()
        _ = model_1.total_loss()
        time_scores.append(end_time-start_time)
    print(time_scores)
