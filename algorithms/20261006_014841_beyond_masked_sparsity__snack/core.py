from __future__ import annotations
from typing import List, Tuple, Dict, Iterable
import random

class SparseMatrix:
    def __init__(self, shape: Tuple[int, int], entries: Iterable[Tuple[int, int, float]] | None = None):
        self.shape = shape
        self.entries: Dict[Tuple[int, int], float] = {}
        if entries:
            for i, j, v in entries:
                if v != 0.0:
                    self.entries[(i, j)] = v

    def dot(self, vector: List[float]) -> List[float]:
        assert len(vector) == self.shape[1]
        result = [0.0] * self.shape[0]
        for (i, j), v in self.entries.items():
            result[i] += v * vector[j]
        return result

    def add_grad(self, grad: Dict[Tuple[int, int], float], lr: float) -> None:
        for (i, j), g in grad.items():
            self.entries[(i, j)] = self.entries.get((i, j), 0.0) - lr * g

class SparseDenseLayer:
    def __init__(self, input_dim: int, output_dim: int, sparsity: float = 0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = SparseMatrix((output_dim, input_dim))
        self.bias = [0.0] * output_dim
        for i in range(output_dim):
            for j in range(input_dim):
                if random.random() < sparsity:
                    self.weight.entries[(i, j)] = random.uniform(-0.1, 0.1)

    def forward(self, input_vec: List[float]) -> List[float]:
        self.input = input_vec
        z = self.weight.dot(input_vec)
        for i in range(self.output_dim):
            z[i] += self.bias[i]
        self.output = z
        return z

    def backward(self, delta: List[float], lr: float) -> List[float]:
        grad_w: Dict[Tuple[int, int], float] = {}
        for i in range(self.output_dim):
            for j in range(self.input_dim):
                if (i, j) in self.weight.entries:
                    grad_w[(i, j)] = delta[i] * self.input[j]
        self.weight.add_grad(grad_w, lr)
        for i in range(self.output_dim):
            self.bias[i] -= lr * delta[i]
        prev_delta = [0.0] * self.input_dim
        for (i, j), v in self.weight.entries.items():
            prev_delta[j] += v * delta[i]
        return prev_delta

class SparseNeuralNetwork:
    def __init__(self, layer_dims: List[int], sparsity: float = 0.5):
        self.layers: List[SparseDenseLayer] = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(SparseDenseLayer(layer_dims[i], layer_dims[i + 1], sparsity))
        self.activations: List[List[float]] = []

    def forward(self, x: List[float]) -> List[float]:
        out = x
        self.activations = [x]
        for i, layer in enumerate(self.layers):
            z = layer.forward(out)
            if i < len(self.layers) - 1:
                out = [max(0.0, v) for v in z]
            else:
                out = z
            self.activations.append(out)
        return out

    def backward(self, y_pred: List[float], y_true: List[float], lr: float) -> None:
        delta = [(yp - yt) for yp, yt in zip(y_pred, y_true)]
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                act = self.activations[i + 1]
                delta = [d * (1.0 if a > 0 else 0.0) for d, a in zip(delta, act)]
            delta = self.layers[i].backward(delta, lr)

    def train(self, data: List[Tuple[List[float], List[float]]], epochs: int, lr: float) -> None:
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in data:
                y_pred = self.forward(x)
                loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y)) / len(y)
                total_loss += loss
                self.backward(y_pred, y, lr)
            avg_loss = total_loss / len(data)
            print(f"Epoch {epoch + 1}/{epochs} loss={avg_loss:.6f}")
