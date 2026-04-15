import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        if isinstance(other, Tensor):  # allows you to do x + 5
            other = other
        else:
            other = Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):  # allows you to do x * 5
            other = other
        else:
            other = Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def matmul(self, other):
        if isinstance(other, Tensor):
            other = other
        else:
            other = Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def backward(self):
        if not self.requires_grad:
            return

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for t in reversed(topo):
            t._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, op={self._op}, requires_grad={self.requires_grad})"

    def history(self):
        return {
            "data": self.data,
            "op": self._op,
            "parents": [repr(p) for p in self._prev],
        }


x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)
z = x * y + x

z.backward()
print("z.data =", z.data)
print("x.grad =", x.grad)
print("y.grad =", y.grad)
