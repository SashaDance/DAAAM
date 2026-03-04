"""
Describe Anything, Anywhere, at Any Moment (DAAAM) Package
"""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

ROOT_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

if not (ROOT_DIR / "output").exists():
    os.makedirs(ROOT_DIR / "output")

load_dotenv()
