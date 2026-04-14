"""Helpers for making ASP imports work from any working directory."""

from pathlib import Path
import sys

def add_project_root(anchor_file=__file__):
    project_root = Path(anchor_file).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root