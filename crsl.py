import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Завантаження даних
def load_data(filepath):
    data = pd.read_excel(filepath)
    X = data.iloc[:, :-1].values  # Вхідні вектори
    y = data.iloc[:, -1].values   # Мітки класів
    return X, y

# Поділ даних на навчальні, валідаційні та тестові
def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * len(indices))
    val_end = int(val_ratio * len(indices)) + train_end
    
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]

# Активаційна функція
def activation(x):
    return 1 if x >= 0 else 0

# Персептрон
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, alpha=0.1, tolerance=1e-3):
        self.weights = np.random.rand(input_size + 1) * 2 - 1  # Випадкові ваги (з урахуванням зсуву)
        self.lr = learning_rate
        self.alpha = alpha
        self.tolerance = tolerance

    def predict(self, X):
        X_bias = np.c_[X, np.ones(X.shape[0])]  # Додаємо зсув (bias)
        return np.array([activation(np.dot(x, self.weights)) for x in X_bias])

    def train(self, X, y, max_epochs=1000):
        X_bias = np.c_[X, np.ones(X.shape[0])]  # Додаємо зсув
        epochs = 0
        errors = []

        for epoch in range(max_epochs):
            error_count = 0
            for i in range(len(X)):
                prediction = activation(np.dot(X_bias[i], self.weights))
                error = y[i] - prediction
                if error != 0:
                    self.weights += self.lr * error * X_bias[i]
                    error_count += 1

            errors.append(error_count)
            epochs += 1
            if error_count <= self.tolerance:
                break

        return epochs, errors

# Візуалізація залежностей
def plot_metrics(metric_values, param_values, xlabel, ylabel, title):
    plt.plot(param_values, metric_values, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()

# Візуалізація класифікації
def visualize_decision_boundary(perceptron, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = perceptron.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.5, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title("Розподіл класів")
    plt.show()

# Основна функція
def main():
    # Завантаження даних
    filepath = 'Лаб1_15.xlsx'  # Заміни на шлях до файлу
    X, y = load_data(filepath)

    # Поділ даних
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Ініціалізація персептрона
    perceptron = Perceptron(input_size=X_train.shape[1], learning_rate=0.01, alpha=0.1, tolerance=1e-3)

    # Навчання
    epochs, errors = perceptron.train(X_train, y_train)
    print(f"Навчання завершено за {epochs} епох")

    # Візуалізація метрик
    plot_metrics(errors, list(range(1, epochs + 1)), "Епохи", "Помилки", "Залежність помилок від епох")

    # Візуалізація класифікації
    visualize_decision_boundary(perceptron, X_test, y_test)

if __name__ == "__main__":
    main()
