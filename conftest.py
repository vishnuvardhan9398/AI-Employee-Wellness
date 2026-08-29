import sys
from pathlib import Path

# Ensure the project root is importable as `src.*` regardless of the
# directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
