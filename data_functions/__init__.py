"""
data_functions package.

This package automatically adds the project root to sys.path when imported.
Import it normally: from data_functions import stitch_data
"""
import sys
from pathlib import Path

# Automatically add project root to Python path
# This allows importing data_functions from any location
_project_root = Path(__file__).parent.parent
_project_root_str = str(_project_root)

if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Define the __all__ variable
__all__ = ["stitch_data"]

# Import the functions from submodules
from .stitch_data import stitch_data
