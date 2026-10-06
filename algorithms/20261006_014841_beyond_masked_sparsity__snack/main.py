import unittest
import time
from core import SparseNeuralNetwork

class TestSparseNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.data = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ]

    def test_forward_shape(self):
        net = SparseNeuralNetwork([2, 2, 1], sparsity=0.5)
        out = net.forward([0.5, -0.5])
        self.assertEqual(len(out), 1)

    def test_backward_reduces_loss(self):
        net = SparseNeuralNetwork([2, 2, 1], sparsity=0.5)
        x, y = self.data[0]
        y_pred = net.forward(x)
        loss_before = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y)) / len(y)
        net.backward(y_pred, y, lr=0.1)
        y_pred_after = net.forward(x)
        loss_after = sum((yp - yt) ** 2 for yp, yt in zip(y_pred_after, y)) / len(y)
        self.assertLess(loss_after, loss_before)

    def test_training_improves_loss(self):
        net = SparseNeuralNetwork([2, 2, 1], sparsity=0.5)
        losses = []
        for epoch in range(50):
            total_loss = 0.0
            for x, y in self.data:
                y_pred = net.forward(x)
                loss = sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y)) / len(y)
                total_loss += loss
                net.backward(y_pred, y, lr=0.1)
            losses.append(total_loss / len(self.data))
        self.assertLess(losses[-1], losses[0])

def benchmark_training():
    net = SparseNeuralNetwork([2, 4, 1], sparsity=0.5)
    data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    start = time.perf_counter()
    net.train(data, epochs=200, lr=0.1)
    end = time.perf_counter()
    print(f"Training time: {end - start:.4f} seconds")

def demo():
    net = SparseNeuralNetwork([2, 4, 1], sparsity=0.5)
    data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    net.train
