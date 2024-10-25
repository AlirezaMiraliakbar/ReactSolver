import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from ReactSolver.models import CSTR


def test_reactor_solver():
    solver = CSTR(params={"some_param": 1.0})
    # Add assertions to test the solver's functionality
    assert solver is not None
