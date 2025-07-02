"""
This file makes the alchemy_workflow directory a Python package
and exposes its main components at the top level.
"""

# Import the main class and function to make them directly accessible
# from the package.
from .geometry_generator import GeometryGenerator
from .data_processor import create_training_set
from .hessian import *

print("alchemy_workflow package initialized.")

