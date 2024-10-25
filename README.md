# My Package

This package solves ODEs for chemical reactor design.

## Installation
```bash
pip install ReactSolver
```

## Usage
```python
from ReactSolver import ReactorSolver
solver = ReactorSolver(params={})
solution = solver.solve_odes(odes, t_span, y0)
```
