import numpy as np
import matplotlib.pyplot as plt


def visualize_data(x_train, y_train, x_test, y_test, weights):
    plt.scatter(x_train, y_train, color='orange', label='Training data')
    plt.scatter(x_test, y_test, color='purple', label='Testing data')

    x_values = np.linspace(min(np.min(x_train), np.min(x_test)), max(np.max(x_train), np.max(x_test)))
    y_values = weights[0] + weights[1] * x_values
    plt.plot(x_values, y_values, color='black', label='Fitted line')

    plt.legend()
    plt.show()


def gradient_descent(x_train, y_train, convergence, learning_rate, epochs):
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    np.random.seed(0)
    weights = np.random.randn(x_train.shape[1], 1)
    loss_new = float('inf')
    i = 0
    while i < epochs:
        y_predicted = np.dot(x_train, weights)

        gradient = -2 * np.dot(x_train.T, (y_train - y_predicted)) / x_train.shape[0]
        weights -= learning_rate * gradient
        loss = np.mean((y_train - y_predicted) ** 2)

        if np.abs(loss_new - loss) < convergence:
            print("Convergence reached")
            break
        loss_new = loss

        print(f'Epoch {i}: Loss = {loss}, Weights = {weights.flatten()}')
        i = i + 1
    return weights


def compute_error(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)


def split_data(file_name):
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    x_vals = data[:, 1].reshape(-1, 1)
    y_vals = data[:, 2].reshape(-1, 1)
    return x_vals, y_vals


def main():
    train_data_X, train_data_y = split_data('lab_1_test.csv')
    test_data_X, test_data_y = split_data('lab_1_train.csv')

    optimal_weights = gradient_descent(train_data_X, train_data_y, 0.00001, 0.1, 10000)

    visualize_data(train_data_X, train_data_y, test_data_X, test_data_y, optimal_weights)

    test_data_X = np.hstack((np.ones((test_data_X.shape[0], 1)), test_data_X))
    predicted_test_y = np.dot(test_data_X, optimal_weights)
    test_error = compute_error(test_data_y, predicted_test_y)
    print(f'Test Error: {test_error}')

if __name__ == "__main__":
    main()