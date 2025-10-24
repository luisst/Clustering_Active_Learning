import sys
from pathlib import Path

testing_path = Path(sys.argv[1])

if testing_path.exists() and any(testing_path.iterdir()):
    response = input(f"The folder {testing_path} already exists and contains files. Do you want to overwrite them? (y/n): ")

    if response.lower() == "y":
        for item in testing_path.glob('*'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
    else:
        # Output export command for bash script to evaluate
        print("export SKIP_PROCESSING=true")
        sys.exit(1)

testing_path.mkdir(parents=True, exist_ok=True)