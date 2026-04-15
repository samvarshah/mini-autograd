import numpy as np


class Tensor:
    """
    requiring gradient is a default parameter set to false
    data -> actual value
    zeros_like -> array to track grad set to 0
    _prev -> parent input
    op -> operation
    """
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None
        self._prev = set(_children)
        self._op = _op

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
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )
        return out

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

print(z)
print(z.history())
