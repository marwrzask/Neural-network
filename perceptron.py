import numpy as np
import scipy.io
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


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
            print(f"predicted value: \n{y_predicted}")
            print(f"To co otrzymuje z wagi i input: \n{summation}")

            self.error = self.y_goal - y_predicted
            print(f"error: \n{self.error}")
            if sum(abs(self.error)) == 0:
                print(f'Errors in total 0 in {epochs} epochs')
                break

            self.weights = self.weights + self.learning_rate * np.dot(X_with_bias.T, self.error)
            print(f"updated weights after epoch: {epochs + 1}/{self.epochs}: \n{self.weights}")
            print("##" * 10)

    def predict(self):
        X_with_bias = np.c_[np.ones((len(X_dane), 1)), X_dane]
        summation = self.adder(X_with_bias, self.weights)
        return self.activation_function(summation)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"\ntotal loss: {total_loss}\n")
        return total_loss

    def predict(self, X_dane):
        X_with_bias = np.c_[np.ones((len(X_dane), 1)), X_dane]
        summation = self.adder(X_with_bias, self.weights)
        return self.activation_function(summation)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"\ntotal loss: {total_loss}\n")
        return total_loss

    def base_plot(self, input_value):
       
        fig, ax = plt.subplots()
        scatter = ax.scatter(input_value[0], input_value[1], c=y_goal,  s=50, cmap = "coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)

    def _plot_decision_regions(self, X, classifier, resolution=0.005):
        #colors = ("cyan", "lightgreen")
        #cmap = ListedColormap(colors)

        x1 = X[:, 0]
        x2 = X[:, 1]

        x1_min, x1_max = x1.min() - 0.25, x1.max() + 0.25
        x2_min, x2_max = x2.min() - 0.25, x2.max() + 0.25

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution)
                               )
        y_line = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_line = y_line.reshape(xx1.shape)

        plt.contourf(xx1, xx2, y_line, alpha=0.35, s=50, cmap='coolwarm')
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.xlabel('x1 - input values')
        plt.ylabel('x2 - input values')
        plt.title('Perceptron results')
        plt.plot()


if __name__ == '__main__':

    X = scipy.io.loadmat('dane_perceptron.mat')
    X_dane = np.transpose(X['dane'])
    df = pd.DataFrame(X_dane)
    inputs_with_bias = np.c_[np.ones((len(X_dane), 1)), X_dane]
    X_input = X_dane[:, 0:2]
    y_goal = X_dane[:, 2]

    """
    X_input = np.array([[0,1],[0,0],[1,1],[1,0]])
    inputs_with_bias = np.c_[np.ones((4, 1)), X_input]
    y_goal = np.array([1,1,0,1])
    y_goal.reshape(4,1)
    df = pd.DataFrame(X_input)
    """

    time_scores = []
    for i in range(5):
        start_time = datetime.now()
        model_1 = Perceptron(learning_rate=0.1, epochs=1000)
        model_1.fit(X_input, y_goal)
        end_time = datetime.now()
        _ = model_1.total_loss()
        time_scores.append(end_time-start_time)
        __ = model_1.base_plot(df)
        ___ = model_1._plot_decision_regions(X_input, model_1)
    print(time_scores)




