import sys
from pathlib import Path

pickle_path = Path(sys.argv[1])

# Check if the path exists and is a file with .pickle extension
if pickle_path.exists() and pickle_path.is_file() and pickle_path.suffix == ".pickle":
    response = input(f"The pickle file {pickle_path} already exists. Do you want to replace it? (y/n): ")

    if response.lower() == "y":
        pickle_path.unlink()
    else:
        # Output export command for bash script to evaluate
        print("export SKIP_PROCESSING=true")
        sys.exit(1)
