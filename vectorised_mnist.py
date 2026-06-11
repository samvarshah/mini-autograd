import numpy as np
import math

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = []

    def __repr__(self):
        return f"Tensor(shape={self.data.shape})"

    # -------- ops --------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        out = Tensor(self.data + other.data)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        out._prev = [self, other]
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s)

        def _backward():
            self.grad += (s * (1 - s)) * out.grad

        out._backward = _backward
        out._prev = [self]
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data))

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        out._prev = [self]
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:
                    build(p)
                topo.append(t)

        build(self)

        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            t._backward()



def softmax(x):
    exp = np.exp(x.data - np.max(x.data))
    return exp / np.sum(exp)


def cross_entropy(pred, target):
    loss = -np.log(pred[target] + 1e-9)
    out = Tensor(loss)

    def _backward():
        grad = pred.copy()
        grad[target] -= 1
        out.grad = 1
        return grad

    return loss

class Linear:
    def __init__(self, in_f, out_f):
        self.w = Tensor(np.random.randn(in_f, out_f) * 0.01)
        self.b = Tensor(np.zeros((1, out_f)))

    def __call__(self, x):
        return Tensor(x.data @ self.w.data + self.b.data)


class MLP:
    def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def __call__(self, x):
        x = self.l1(x)
        x = Tensor(np.maximum(0, x.data))  # ReLU
        x = self.l2(x)
        return x


from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data[:2000].values / 255.0
y = mnist.target[:2000].astype(int).values


model = MLP()
lr = 0.01

for epoch in range(5):

    correct = 0
    loss_sum = 0

    print(f"\n--- Epoch {epoch} started ---")

    for i in range(2000):

        # progress indicator (so you KNOW it's running)
        if i % 500 == 0:
            print(f"processing sample {i}")

        # -------- forward --------
        x = X[i].reshape(1, 784)

        h = np.maximum(0, x @ model.l1.w.data + model.l1.b.data)   # (1,128)
        logits = h @ model.l2.w.data + model.l2.b.data             # (1,10)

        # softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        target = y[i]

        loss = -np.log(probs[0, target] + 1e-9)
        loss_sum += loss

        # -------- gradient --------
        grad = probs.copy()
        grad[0, target] -= 1   # shape (1,10)

        # -------- backward L2 --------
        model.l2.w.grad = h.T @ grad          # (128,1) @ (1,10) = (128,10)
        model.l2.b.grad = grad

        # -------- hidden gradient --------
        dh = grad @ model.l2.w.data.T         # (1,10) @ (10,128) = (1,128)
        dh[h <= 0] = 0  # ReLU derivative

        # -------- backward L1 --------
        model.l1.w.grad = x.T @ dh            # (784,1) @ (1,128)
        model.l1.b.grad = dh

        # -------- update --------
        model.l1.w.data -= lr * model.l1.w.grad
        model.l2.w.data -= lr * model.l2.w.grad
        model.l1.b.data -= lr * model.l1.b.grad
        model.l2.b.data -= lr * model.l2.b.grad

        # -------- accuracy --------
        if np.argmax(probs) == target:
            correct += 1

    # -------- epoch summary --------
    print(f"epoch {epoch} done")
    print(f"loss: {loss_sum/2000:.4f}")
    print(f"acc : {correct/2000:.3f}")
