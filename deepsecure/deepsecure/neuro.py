import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib
import matplotlib.pyplot as plt

# Set Matplotlib backend to 'Agg' to avoid interactive issues
matplotlib.use('Agg')

# Load dataset from CSV file
data_file_path = "employee_danger_level_dataset.csv"
data = pd.read_csv(data_file_path)

# Split the dataset into features and labels
X = data.iloc[:, :-1].values  # Features (all columns except the last)
y = data.iloc[:, -1].values   # Danger level (labels, last column)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Manual implementation of the neural network with Batch Gradient Descent and Momentum
class DangerLevelModel:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.weights = []
        self.biases = []
        self.velocities_w = []
        self.velocities_b = []

        # Initialize weights, biases, and velocities with Xavier initialization
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
            self.velocities_w.append(np.zeros_like(weight))
            self.velocities_b.append(np.zeros_like(bias))

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = []
        self.z_values = []

        current_activation = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_activation = self.leaky_relu(z)
            self.activations.append(current_activation)

        # Output layer
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)

        return output

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true.astype(int) - 1])
        return np.sum(log_probs) / m

    def backward(self, X, y_true):
        m = X.shape[0]
        y_true_one_hot = np.zeros_like(self.activations[-1])
        y_true_one_hot[range(m), y_true.astype(int) - 1] = 1

        # Gradient of the output layer
        delta = self.activations[-1] - y_true_one_hot
        self.d_weights = []
        self.d_biases = []

        for i in reversed(range(len(self.weights))):
            dw = np.dot(self.activations[i - 1].T if i > 0 else X.T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            self.d_weights.insert(0, dw)
            self.d_biases.insert(0, db)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.leaky_relu_derivative(self.z_values[i - 1])

    def update_parameters(self, learning_rate, momentum):
        for i in range(len(self.weights)):
            self.velocities_w[i] = momentum * self.velocities_w[i] - learning_rate * self.d_weights[i]
            self.velocities_b[i] = momentum * self.velocities_b[i] - learning_rate * self.d_biases[i]

            self.weights[i] += self.velocities_w[i]
            self.biases[i] += self.velocities_b[i]

    def train(self, X, y, epochs, learning_rate, batch_size, momentum):
        m = X.shape[0]
        self.loss_history = []
        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, m, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch)
                self.update_parameters(learning_rate, momentum)

            y_pred_full = self.forward(X)
            loss = self.compute_loss(y, y_pred_full)
            self.loss_history.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1) + 1

# Increase hidden layers, use batch gradient descent with momentum, and add more epochs
manual_model = DangerLevelModel(input_size=X_train.shape[1], hidden_layer_sizes=[256, 128, 64], output_size=3)
manual_model.train(X_train, y_train, epochs=200, learning_rate=0.003, batch_size=32, momentum=0.9)

# Evaluate the manual model
y_pred_manual = manual_model.predict(X_test)
manual_accuracy = np.mean(y_pred_manual == y_test)
precision_manual = precision_score(y_test, y_pred_manual, average="weighted")
recall_manual = recall_score(y_test, y_pred_manual, average="weighted")
f1_manual = f1_score(y_test, y_pred_manual, average="weighted")

print(f"Manual Neural Network Test Accuracy: {manual_accuracy:.2f}")
print(f"Precision: {precision_manual:.2f}")
print(f"Recall: {recall_manual:.2f}")
print(f"F1 Score: {f1_manual:.2f}")

# Train a neural network using sklearn for comparison
sklearn_model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=2000, random_state=42)
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
precision_sklearn = precision_score(y_test, y_pred_sklearn, average="weighted")
recall_sklearn = recall_score(y_test, y_pred_sklearn, average="weighted")
f1_sklearn = f1_score(y_test, y_pred_sklearn, average="weighted")

print(f"Sklearn Neural Network Test Accuracy: {sklearn_accuracy:.2f}")
print(f"Precision: {precision_sklearn:.2f}")
print(f"Recall: {recall_sklearn:.2f}")
print(f"F1 Score: {f1_sklearn:.2f}")

# Plot the loss history for the manual model
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(manual_model.loss_history) + 1), manual_model.loss_history, label="Manual Model Loss")
plt.title("Loss History During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("manual_loss_history.png")

# Compare precision, recall, and F1 score between manual and sklearn models
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
manual_metrics = [manual_accuracy, precision_manual, recall_manual, f1_manual]
sklearn_metrics = [sklearn_accuracy, precision_sklearn, recall_sklearn, f1_sklearn]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, manual_metrics, width, label="Manual Model")
plt.bar(x + width/2, sklearn_metrics, width, label="Sklearn Model")
plt.xticks(x, metrics)
plt.ylabel("Scores")
plt.title("Comparison of Metrics: Manual vs Sklearn Model")
plt.legend()
plt.savefig("model_comparison.png")
