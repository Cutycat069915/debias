from pathlib import Path

# Get all items inside the logs directory
paths = Path("./logs").glob("*")

for path in paths:
    # Check if the path is actually a directory before adding subdirectories
    if path.is_dir():
        try:
            # Create subdirectories; exist_ok=True prevents errors if they already exist
            (path / "similarity").mkdir(parents=True, exist_ok=True)
            (path / "accuracy").mkdir(parents=True, exist_ok=True)
            print(f"Successfully updated: {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")
