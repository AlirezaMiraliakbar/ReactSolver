# ReactSolver

This package solves for chemical reactor design. The reactor models covered so far:
- CSTR
- PFR

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
