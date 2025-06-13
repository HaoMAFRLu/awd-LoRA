"""Collection of useful functions.
"""
import os
from pathlib import Path
import numpy as np
from datetime import datetime

def mkdir(path: Path) -> None:
    """Check if the folder exists and create it
    if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_parent_path(lvl: int=0) -> Path:
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def get_folder_name() -> str:
    """Return a folder name based on the current time.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S") 