
import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)[0]
        self.learning_rate = learning_rate

    def _activation_function(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self._activation_function(weighted_sum)

    def train(self, training_inputs, labels, epochs):
        print(f"
Starting training for {epochs} epochs...")
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
            # print(f"Epoch {epoch + 1}: Errors: {errors}. Current weights: {self.weights}, bias: {self.bias}")
        print("Training finished.")

if __name__ == '__main__':
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    labels_and = np.array([0, 0, 0, 1])
    labels_or = np.array([0, 1, 1, 1])

    print("--- Training Perceptron for AND Gate ---")
    perceptron_and = Perceptron(num_inputs=2, learning_rate=0.1)
    perceptron_and.train(training_inputs, labels_and, epochs=10)
    
    print("
--- Verifying AND Gate Perceptron ---")
    print("Input | Actual AND | Predicted AND")
    print("----------------------------------")
    for inputs, label in zip(training_inputs, labels_and):
        prediction = perceptron_and.predict(inputs)
        print(f"{inputs} | {label}          | {prediction}")

    print("
--- Training Perceptron for OR Gate ---")
    perceptron_or = Perceptron(num_inputs=2, learning_rate=0.1)
    perceptron_or.train(training_inputs, labels_or, epochs=10)

    print("
--- Verifying OR Gate Perceptron ---")
    print("Input | Actual OR | Predicted OR")
    print("---------------------------------")
    for inputs, label in zip(training_inputs, labels_or):
        prediction = perceptron_or.predict(inputs)
        print(f"{inputs} | {label}         | {prediction}")
