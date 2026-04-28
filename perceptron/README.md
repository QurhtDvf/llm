
# Perceptron Implementation in Python

This repository details a step-by-step implementation of a simple Perceptron, a fundamental building block of neural networks, from scratch using Python and NumPy. The goal is to understand how a Perceptron learns to classify linearly separable data, such as basic logical gates like AND and OR.

## Project Status

### 1. Perceptron Implementation (Completed)

We have successfully implemented a Perceptron model capable of learning basic logical gates. This implementation covers:

*   **`Perceptron` Class Definition**:
    *   **Constructor (`__init__`)**: Initializes the Perceptron with a specified number of input features and a learning rate. Weights and biases are randomly initialized with small values.
        ```python
        def __init__(self, num_inputs, learning_rate=0.1):
            self.weights = np.random.rand(num_inputs)
            self.bias = np.random.rand(1)[0]
            self.learning_rate = learning_rate
        ```
    *   **Activation Function (`_activation_function`)**: A simple step function is used, returning 1 if the input is greater than 0, otherwise 0.
        ```python
        def _activation_function(self, x):
            return 1 if x > 0 else 0
        ```
    *   **Prediction Method (`predict`)**: Calculates the weighted sum of inputs, adds the bias, and applies the activation function to produce an output (0 or 1).
        ```python
        def predict(self, inputs):
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            return self._activation_function(weighted_sum)
        ```
    *   **Training Method (`train`)**: Implements the Perceptron learning rule. It iterates over the training data for a specified number of epochs, adjusting weights and bias whenever a misclassification occurs. The adjustment is proportional to the error, input, and learning rate.
        ```python
        def train(self, training_inputs, labels, epochs):
            for epoch in range(epochs):
                errors = 0
                for inputs, label in zip(training_inputs, labels):
                    prediction = self.predict(inputs)
                    error = label - prediction
                    if error != 0:
                        self.weights += self.learning_rate * error * inputs
                        self.bias += self.learning_rate * error
                        errors += 1
                if errors == 0:
                    print(f"Epoch {epoch + 1}: No errors. Perceptron converged.")
                    break
        ```

*   **AND/OR Gate Training and Verification**:
    *   The Perceptron is trained on standard input patterns for AND and OR gates (e.g., `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`).
    *   After training, the model's predictions are verified against the expected outputs for these logical gates, demonstrating its ability to learn and correctly classify linearly separable problems.

The full implementation can be found in `perceptron.py`.

## Getting Started

To run the `perceptron.py` code:

1.  Ensure you have `numpy` installed:
    ```bash
    pip install numpy
    ```
2.  Execute the Python script:
    ```bash
    python perceptron.py
    ```
    This script will:
    *   Initialize and train a Perceptron for the AND gate.
    *   Verify its predictions.
    *   Initialize and train a Perceptron for the OR gate.
    *   Verify its predictions.

## Files

*   `perceptron.py`: Contains the `Perceptron` class and example code for training and testing AND/OR gates.
