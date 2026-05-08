# Micrograd — From Scratch

A ground-up implementation of Andrej Karpathy's **micrograd**: a tiny scalar-valued autograd engine with a PyTorch-like API, written in pure Python with no external dependencies beyond the standard library.

---

## What is Micrograd?

Micrograd is a minimalist automatic differentiation engine. It lets you build mathematical expressions out of scalar `Value` objects, then call `.backward()` to automatically compute the gradient of any output with respect to every input — exactly like PyTorch, but in ~100 lines of Python.

This makes it a perfect tool for understanding how neural networks and backpropagation actually work under the hood.

---

## Contents

```
micrograd_from_scratch.ipynb   ← Main notebook (all implementation + experiments)
README.md                      ← This file
```

---

## Notebook Structure

| Section | Topic                                                           |
| ------- | --------------------------------------------------------------- |
| 1       | Imports                                                         |
| 2       | Numerical differentiation (1-var and multi-var)                 |
| 3       | The `Value` object — scalar autograd engine                     |
| 4       | Computation graph visualisation with Graphviz                   |
| 5       | Backpropagation through a single neuron (`tanh` fused + manual) |
| 6       | Debugging gradient accumulation bugs                            |
| 7       | Cross-validation against PyTorch                                |
| 8       | Building `Neuron` → `Layer` → `MLP` from scratch                |
| 9       | Full training loop (MSE loss + gradient descent)                |
| 10      | Summary table and key insights                                  |

---

## The Core: `Value`

```python
from micrograd_from_scratch import Value   # or run the notebook cell

a = Value(2.0)
b = Value(3.0)
c = a * b + a ** 2   # builds computation graph
c.backward()          # fills a.grad and b.grad automatically

print(a.grad)   # ∂c/∂a = b + 2a = 3 + 4 = 7
print(b.grad)   # ∂c/∂b = a = 2
```

Every arithmetic operation (`+`, `-`, `*`, `/`, `**`) and activation function (`tanh`, `exp`) records how it was computed and registers a `_backward` closure that knows how to propagate gradients one step back. Calling `.backward()` on the final output triggers a topological sort of the graph and fires every closure in reverse order — this is backpropagation.

---

## The Neural Network

Built from three composable classes:

```
Neuron(nin)          →  o = tanh(w·x + b)
Layer(nin, nout)     →  nout neurons in parallel
MLP(nin, nouts)      →  stack of layers, e.g. MLP(3, [4, 4, 1])
```

### Training example

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5],
      [0.5, 1.0,  1.0], [1.0,  1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

for k in range(20):
    ypred = [n(x) for x in xs]
    loss  = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)], Value(0.0))

    for p in n.parameters():
        p.grad = 0.0   # zero gradients before backward
    loss.backward()

    for p in n.parameters():
        p.data -= 0.05 * p.grad

    print(k, loss.data)
```

After 20 steps the loss converges close to zero and predictions match targets.

---

## Key Concepts Demonstrated

**Automatic differentiation via computation graphs**
Every `Value` object records the operation and operands that created it. This forms a directed acyclic graph (DAG). `.backward()` traverses this graph in reverse topological order and applies the chain rule at each node.

**Gradient accumulation**
When the same node appears in multiple branches of the graph, its gradient contributions must be _added together_ (`+=`), not overwritten. A classic bug is using `=` instead of `+=` in `_backward`.

**Zeroing gradients**
Gradients accumulate across multiple `.backward()` calls. In a training loop, always zero `.grad` before each backward pass.

**Chain rule**
Each `_backward` closure implements the local derivative rule for its operation. The chain rule is applied automatically by passing `out.grad` through each closure.

---

## Requirements

```
python >= 3.9
numpy
matplotlib
graphviz          (pip install graphviz  +  system graphviz binary)
torch             (optional — only for the cross-validation section)
```

Install dependencies:

```bash
pip install numpy matplotlib graphviz torch
```

---

## Running the Notebook

```bash
jupyter notebook micrograd_from_scratch.ipynb
```

Run cells top to bottom. All classes (`Value`, `Neuron`, `Layer`, `MLP`) and helper functions (`draw_dot`) are defined in order and reused in later sections.

---

## Credits

Inspired by and based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and the accompanying YouTube lecture _"The spelled-out intro to neural networks and backpropagation"_.

Re-implemented and extended by **Muhammad Zulqarnain Abdullah** (24-CS-19).
