![ReactSolver Logo](https://github.com/AlirezaMiraliakbar/ReactSolver/blob/main/docs/acc/logo.png)

# ReactSolver

This package solves for chemical reactor design. The reactor models covered so far:
- CSTR
- PFR

## Folder Structure
Root Directory
├── ReactSolver
│   └── molebalance_init
├── docs
│   └── logo
├── tests
│   └── models init
├── .DS_Store
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
└── test_notebook.ipynb
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
