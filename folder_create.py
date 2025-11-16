import sys
from pathlib import Path

testing_path = Path(sys.argv[1])

testing_path.mkdir(parents=True, exist_ok=True)