![ReactSolver Logo](https://github.com/AlirezaMiraliakbar/ReactSolver/blob/main/docs/acc/logo.png)

# ReactSolver

This package solves for chemical reactor design. The reactor models covered so far:
- CSTR
- PFR

## Folder Structure

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
## Validation
