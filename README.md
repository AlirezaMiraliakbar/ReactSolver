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
## Verfication

The models and mathematical methods used in ReactSolver are verified using commertial softwares of COMSOL Multiphysics v6.1 and MATLAB R2022b under the academic license provided to University of Connecticut.
