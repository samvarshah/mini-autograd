import math
import random

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data:.4f}, grad={self.grad:.4f})"

    # -------- ops --------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    # -------- activation --------
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    # -------- backprop --------
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

def bce(pred, target):
    eps = 1e-7
    p = min(max(pred.data, eps), 1 - eps)

    loss = -(target.data * math.log(p) +
             (1 - target.data) * math.log(1 - p))

    out = Tensor(loss, (pred,), 'bce')

    def _backward():
        pred.grad += ((-target.data / p) +
                      (1 - target.data) / (1 - p)) * out.grad

    out._backward = _backward
    return out

X = [(0,0),(0,1),(1,0),(1,1)]
y = [0,0,0,1]

w1 = Tensor(random.uniform(-1,1))
w2 = Tensor(random.uniform(-1,1))
b  = Tensor(0.0)

lr = 0.1

for epoch in range(300):
    total_loss = 0

    for (x1,x2), target in zip(X,y):

        x1 = Tensor(x1)
        x2 = Tensor(x2)

        z = x1*w1 + x2*w2 + b
        pred = z.sigmoid()

        loss = bce(pred, Tensor(target))
        total_loss += loss.data

        loss.backward()

        # update
        w1.data -= lr * w1.grad
        w2.data -= lr * w2.grad
        b.data  -= lr * b.grad

        # reset grads
        w1.grad = w2.grad = b.grad = 0

    if epoch % 30 == 0:
        print(f"epoch {epoch}, loss {total_loss:.4f}")

class Neuron:
    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Tensor(0.0)

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        return act.sigmoid()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


class MLP:
    def __init__(self):
        self.l1 = Layer(784, 64)
        self.l2 = Layer(64, 10)

    def __call__(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data[:1000] / 255.0
y = mnist.target[:1000].astype(int)

def softmax(logits):
    exps = [math.exp(l.data) for l in logits]
    s = sum(exps)
    return [e/s for e in exps]

class Adam:
    def __init__(self, params, lr=0.001):
        self.params = params
        self.lr = lr
        self.m = {p:0 for p in params}
        self.v = {p:0 for p in params}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            g = p.grad

            self.m[p] = 0.9*self.m[p] + 0.1*g
            self.v[p] = 0.999*self.v[p] + 0.001*(g*g)

            m_hat = self.m[p] / (1 - 0.9**self.t)
            v_hat = self.v[p] / (1 - 0.999**self.t)

            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + 1e-8)
            p.grad = 0
