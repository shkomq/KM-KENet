"""Make the project root importable (`common`, `step1_...`) when pytest runs from any folder."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
